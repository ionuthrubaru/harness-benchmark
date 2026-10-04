"""
benchmarks/mongodb/naive/q1_revenue.py — MongoDB Naive: Q1
===========================================================
Q1: Monthly revenue by subscription tier, last 12 months.
    Includes marketplace invoices attributed to the user's active
    subscription tier at the time of purchase.
    Temporal JOIN on subscription_tier_pricing captures the correct
    monthly price in effect at invoice creation time.

Naive implementation notes:
    All collections are flat mirrors of the PostgreSQL schema — no
    embedding, no denormalisation. All stored values are strings
    (dates, numbers) as loaded from CSV, so numeric aggregation and
    date arithmetic are performed in Python after fetching.

    The implementation requires four MongoDB reads plus Python-side
    joining and aggregation:

      1. Fetch all paid invoices in the last 12 months  [timed]
      2. Fetch all subscriptions to resolve tier_id per invoice
      3. Fetch all subscription_tier_pricing rows for temporal lookup
      4. Fetch subscription_tiers for tier names

    Steps 2–4 are static reference data that do not change between
    iterations. They are pre-loaded once before the benchmark loop —
    the same pattern used by the PostgreSQL benchmarks for ID pool
    pre-fetching. Only the invoice fetch and Python-side aggregation
    (step 1 + join logic) are timed per iteration.

    Marketplace invoice attribution requires finding the most recently
    started subscription for the invoice's user as of the invoice date —
    done with Python-side filtering since MongoDB has no LATERAL JOIN.

    This multi-step Python orchestration is intentionally awkward.
    It demonstrates the engine-effect cost of running a relational
    temporal query against a naive document store with no schema
    optimisation. The awkwardness is academically valuable.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive MongoDB result.
    This file measures the naive (engine-only) side of that decomposition.

Usage:
    cd benchmarks/mongodb/naive
    python q1_revenue.py                   # 1000 iterations, save results
    python q1_revenue.py --iterations 100  # quick smoke test
    python q1_revenue.py --dry-run         # run once, print result sample
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── date helpers ──────────────────────────────────────────────────────────────
# All dates in the naive collections are stored as ISO 8601 strings.

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def twelve_months_ago_str() -> str:
    """Return an ISO 8601 string for 12 months before now."""
    return (now_utc() - timedelta(days=365)).isoformat()


def parse_dt(s: str) -> datetime:
    """Parse an ISO 8601 string to a timezone-aware datetime."""
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

# ── one-time reference data loaders ──────────────────────────────────────────
# These are called once before the benchmark loop, not inside the timed fn.

def load_subs_by_id(db) -> dict:
    """
    Return {subscription_id: doc} for all subscriptions.
    Used to resolve tier_id for subscription invoices.
    """
    docs = db["subscriptions"].find(
        {}, {"_id": 1, "tier_id": 1, "user_id": 1, "started_at": 1}
    )
    return {d["_id"]: d for d in docs}


def load_subs_by_user(db) -> dict:
    """
    Return {user_id: [subscription docs sorted by started_at desc]}.
    Used to attribute marketplace invoices to the user's active tier.
    Pre-sorted so attribution is a simple linear scan with early exit.
    """
    docs = db["subscriptions"].find(
        {}, {"_id": 1, "tier_id": 1, "user_id": 1, "started_at": 1}
    )
    by_user = defaultdict(list)
    for d in docs:
        by_user[d["user_id"]].append(d)
    for uid in by_user:
        by_user[uid].sort(key=lambda s: s["started_at"], reverse=True)
    return dict(by_user)


def load_pricing(db) -> list[dict]:
    """Return all subscription_tier_pricing rows."""
    return list(db["subscription_tier_pricing"].find(
        {}, {"_id": 0, "tier_id": 1, "valid_from": 1,
             "valid_to": 1, "monthly_price_usd": 1}
    ))


def load_tiers(db) -> dict:
    """Return {tier_id: tier_name}."""
    return {d["_id"]: d["name"]
            for d in db["subscription_tiers"].find({}, {"_id": 1, "name": 1})}

# ── temporal price lookup ─────────────────────────────────────────────────────

def resolve_price(tier_id: str, invoice_dt: datetime,
                  pricing_rows: list[dict]) -> float | None:
    """
    Find the pricing row valid at invoice_dt for the given tier.
    Mirrors the SQL temporal JOIN:
        stp.valid_from <= invoice_dt
        AND (stp.valid_to IS NULL OR stp.valid_to > invoice_dt)
    """
    for row in pricing_rows:
        if row["tier_id"] != tier_id:
            continue
        valid_from = parse_dt(row["valid_from"])
        valid_to   = parse_dt(row["valid_to"]) if row.get("valid_to") else None
        if valid_from <= invoice_dt and (valid_to is None or valid_to > invoice_dt):
            return float(row["monthly_price_usd"])
    return None

# ── core Q1 logic (timed portion) ─────────────────────────────────────────────

def run_q1(db, subs_by_id: dict, subs_by_user: dict,
           pricing: list[dict], tiers: dict) -> list[dict]:
    """
    Fetch invoices from MongoDB and aggregate revenue by month and tier.
    Lookup tables are passed in pre-loaded — only the invoice fetch and
    aggregation logic run inside the timed loop.

    Returns rows matching the PostgreSQL Q1 output format:
        [{month, tier_name, price_in_effect_usd,
          invoice_count, total_revenue_usd}]
    """
    cutoff = twelve_months_ago_str()

    # ── timed step: fetch paid invoices in 12-month window ────────────────
    invoices = list(db["invoices"].find(
        {
            "status":     "paid",
            "created_at": {"$gte": cutoff},
        },
        {
            "_id": 1, "user_id": 1, "invoice_type": 1,
            "total_usd": 1, "created_at": 1, "subscription_id": 1,
        }
    ))

    # ── timed step: resolve tier + aggregate ──────────────────────────────
    groups: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "revenue": 0.0})

    for inv in invoices:
        inv_dt   = parse_dt(inv["created_at"])
        inv_type = inv.get("invoice_type", "")
        tier_id  = None

        if inv_type == "subscription":
            sub = subs_by_id.get(inv.get("subscription_id", ""))
            if sub:
                tier_id = sub["tier_id"]

        elif inv_type == "marketplace":
            for sub in subs_by_user.get(inv.get("user_id", ""), []):
                if parse_dt(sub["started_at"]) <= inv_dt:
                    tier_id = sub["tier_id"]
                    break

        if tier_id is None:
            continue

        price = resolve_price(tier_id, inv_dt, pricing)
        if price is None:
            continue

        key = (month_key(inv_dt), tier_id, price)
        groups[key]["count"]   += 1
        groups[key]["revenue"] += float(inv.get("total_usd", 0))

    rows = []
    for (month, tier_id, price), agg in groups.items():
        rows.append({
            "month":               month,
            "tier_name":           tiers.get(tier_id, tier_id),
            "price_in_effect_usd": price,
            "invoice_count":       agg["count"],
            "total_revenue_usd":   round(agg["revenue"], 2),
        })

    rows.sort(key=lambda r: (r["month"], r["tier_name"]))
    return rows

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, subs_by_id, subs_by_user, pricing, tiers):
    """
    Return a zero-argument callable for the harness.
    Lookup tables are captured in the closure — loaded once, reused
    across all iterations.
    """
    def _run():
        run_q1(db, subs_by_id, subs_by_user, pricing, tiers)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, subs_by_id, subs_by_user, pricing, tiers):
    print("\n  DRY RUN — MongoDB Naive Q1 result sample (first 10 rows):\n")
    rows = run_q1(db, subs_by_id, subs_by_user, pricing, tiers)
    if not rows:
        print("  ⚠  Query returned no rows — is the database populated?")
        return
    headers = ["month", "tier_name", "price_in_effect_usd",
                "invoice_count", "total_revenue_usd"]
    col_w = 22
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("─" * col_w for _ in headers))
    for row in rows[:10]:
        print("  " + "  ".join(str(row[h]).ljust(col_w) for h in headers))
    print(f"\n  ... {len(rows)} total rows returned")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q1 — monthly revenue benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Execute once and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q1.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q1.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q1 Monthly Revenue Benchmark")
    print("=" * 55)

    db = get_db()

    print("  Pre-loading reference data (subscriptions, tiers, pricing)...")
    subs_by_id   = load_subs_by_id(db)
    subs_by_user = load_subs_by_user(db)
    pricing      = load_pricing(db)
    tiers        = load_tiers(db)
    print(f"  Loaded {len(subs_by_id):,} subscriptions, "
          f"{len(pricing)} pricing rows, {len(tiers)} tiers.\n")

    if args.dry_run:
        dry_run(db, subs_by_id, subs_by_user, pricing, tiers)
        return

    run_benchmark(
        query_fn=make_query_fn(db, subs_by_id, subs_by_user, pricing, tiers),
        db="mongodb_naive",
        query_id="Q1",
        label=(
            "Monthly revenue by subscription tier (last 12 months). "
            "Naive: invoice fetch from MongoDB + Python-side temporal "
            "JOIN against pre-loaded subscriptions, tiers, and pricing. "
            "No embedding or denormalisation — direct port of PostgreSQL schema."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()