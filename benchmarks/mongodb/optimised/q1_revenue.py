"""
benchmarks/mongodb/optimised/q1_revenue.py — MongoDB Optimised: Q1
====================================================================
Q1: Monthly revenue by subscription tier, last 12 months.

Optimised schema changes vs naive:
  - total_usd, monthly_price_usd are native floats → no float() conversion
  - subscriptions have compound index (user_id, started_at DESC) →
    faster pre-load for per-user attribution lookups
  - subscription_tier_pricing.monthly_price_usd is a native float →
    no string parsing in resolve_price()

Query structure is the same as naive (Python-side tier attribution):
  1. Fetch paid invoices in 12-month window  [timed]
  2. Python-side tier resolution via pre-loaded subscription dicts
  3. Python-side temporal pricing lookup (same as naive)

The engine + schema gain here is modest: native numeric types eliminate
float() conversion overhead, and the compound subscription index reduces
pre-load time. Q2 and Q7 show the larger schema effects.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive (native types + better indexes).

Usage:
    python q1_revenue.py
    python q1_revenue.py --iterations 100
    python q1_revenue.py --dry-run
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

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def twelve_months_ago_str() -> str:
    return (now_utc() - timedelta(days=365)).isoformat()

def parse_dt(s: str) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

# ── one-time reference data loaders ──────────────────────────────────────────

def load_subs_by_id(db) -> dict:
    docs = db["subscriptions"].find(
        {}, {"_id": 1, "tier_id": 1, "user_id": 1, "started_at": 1}
    )
    return {d["_id"]: d for d in docs}

def load_subs_by_user(db) -> dict:
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
    """
    In the optimised schema, monthly_price_usd is a native float.
    No float() conversion needed in resolve_price().
    """
    return list(db["subscription_tier_pricing"].find(
        {}, {"_id": 0, "tier_id": 1, "valid_from": 1,
             "valid_to": 1, "monthly_price_usd": 1}
    ))

def load_tiers(db) -> dict:
    return {d["_id"]: d["name"]
            for d in db["subscription_tiers"].find({}, {"_id": 1, "name": 1})}

# ── temporal price lookup ─────────────────────────────────────────────────────

def resolve_price(tier_id: str, invoice_dt: datetime,
                  pricing_rows: list[dict]) -> float | None:
    for row in pricing_rows:
        if row["tier_id"] != tier_id:
            continue
        valid_from = parse_dt(row["valid_from"])
        valid_to   = parse_dt(row["valid_to"]) if row.get("valid_to") else None
        if valid_from <= invoice_dt and (valid_to is None or valid_to > invoice_dt):
            # Optimised: monthly_price_usd is already a float — no conversion
            return row["monthly_price_usd"]
    return None

# ── core Q1 logic (timed portion) ─────────────────────────────────────────────

def run_q1(db, subs_by_id: dict, subs_by_user: dict,
           pricing: list[dict], tiers: dict) -> list[dict]:
    cutoff = twelve_months_ago_str()

    invoices = list(db["invoices"].find(
        {"status": "paid", "created_at": {"$gte": cutoff}},
        {"_id": 1, "user_id": 1, "invoice_type": 1,
         "total_usd": 1, "created_at": 1, "subscription_id": 1},
    ))

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
        # Optimised: total_usd is already a float — no float() conversion
        groups[key]["revenue"] += inv.get("total_usd", 0)

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
    def _run():
        run_q1(db, subs_by_id, subs_by_user, pricing, tiers)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, subs_by_id, subs_by_user, pricing, tiers):
    print("\n  DRY RUN — MongoDB Optimised Q1 result sample (first 10 rows):\n")
    rows = run_q1(db, subs_by_id, subs_by_user, pricing, tiers)
    if not rows:
        print("  ⚠  Query returned no rows — is the optimised DB populated?")
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
        description="MongoDB Optimised Q1 — monthly revenue benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q1.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q1 Monthly Revenue Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

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
        db="mongodb_optimised",
        query_id="Q1",
        label=(
            "Monthly revenue by subscription tier (last 12 months). "
            "Optimised: native float total_usd/monthly_price_usd (no float() "
            "conversion), compound (user_id, started_at DESC) subscription index. "
            "Python-side tier attribution retained — schema effect for Q1 is "
            "primarily numeric type gain; larger gains visible in Q2/Q7."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()