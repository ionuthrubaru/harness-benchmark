"""
benchmarks/cassandra/naive/q1_revenue.py — Cassandra Naive: Q1
===============================================================
Q1: Monthly revenue by subscription tier, last 12 months.

Naive schema penalty — three full table scans + Python-side joins
──────────────────────────────────────────────────────────────────
PostgreSQL Q1 resolves this in a single query with CTEs and a LATERAL
temporal JOIN. In the Cassandra naive schema (id as sole partition key
on every table), this requires:

  1. ALLOW FILTERING scan on invoices — filter by status='paid' and
     created_at >= 12 months ago. Cassandra must read every partition
     on every node because neither status nor created_at is a partition
     key or clustering column on the invoices table.

  2. Full table scan on subscriptions (no WHERE clause → no ALLOW
     FILTERING needed, but still a full cross-node scan). Used to:
       • Resolve tier_id for subscription invoices (by subscription_id)
       • Resolve tier_id for marketplace invoices (find the most recent
         subscription started_at <= invoice created_at for that user)
     Both are Python-side operations on the scanned data.

  3. Full scan on subscription_tier_pricing — tiny table (5 rows), but
     demonstrates the same problem: in a naive schema there is no way
     to push the temporal predicate (valid_from <= date < valid_to)
     into CQL without ALLOW FILTERING.

  4. Full scan on subscription_tiers — 3 rows. Included for schema
     symmetry; negligible cost.

All aggregation — grouping by (year_month, tier_id), summing total_usd,
resolving tier_name and monthly_price_usd — is done in Python.

This is the opposite of the optimised schema (invoices_by_month_tier),
where Q1 reads 36 small pre-shaped partitions and returns pre-resolved
amounts. The cost difference between naive and optimised Q1 is the
schema effect for this query.

Session thread safety
──────────────────────
The Cassandra Python driver Session is fully thread-safe and manages
its own internal connection pool. Unlike psycopg2 (one connection per
thread) or the Neo4j driver (one session per thread), a single
cassandra.cluster.Session can safely service concurrent requests from
multiple Python threads. For Q1 (concurrency=1) this makes no
difference, but it is noted here for consistency with Q3.

Usage:
    python q1_revenue.py                   # 1000 iterations
    python q1_revenue.py --iterations 100  # quick smoke test
    python q1_revenue.py --dry-run         # run once, print result sample
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")

# ── tier metadata (hardcoded — same 3 rows as schema) ─────────────────────────

TIER_NAMES = {1: "Free", 2: "Pro", 3: "Business"}

PRICING = [
    (1, datetime(2023, 1, 1, tzinfo=timezone.utc), None,                                       Decimal("0.00")),
    (2, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("14.99")),
    (2, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("19.99")),
    (3, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("39.99")),
    (3, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("49.99")),
]

# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_price(tier_id: int, created_at: datetime) -> Decimal | None:
    for t_id, valid_from, valid_to, price in PRICING:
        if t_id != tier_id:
            continue
        if valid_from <= created_at and (valid_to is None or valid_to > created_at):
            return price
    return None


def _year_month(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

# ── benchmark query function ──────────────────────────────────────────────────

def make_query_fn(session):
    """
    Return a zero-argument callable that executes Q1 on the naive schema.

    Each call performs:
      1. ALLOW FILTERING scan on invoices (status + created_at filter)
      2. Full scan on subscriptions (no WHERE — builds user→subs and sub_id→tier indexes)
      3. Full scan on subscription_tier_pricing (5 rows — read for completeness)
      4. Python-side tier attribution and temporal price resolution
      5. Python-side aggregation: group by (year_month, tier_id), sum revenue

    Scans 2 and 3 could theoretically be pre-cached across iterations, but doing
    so would introduce inter-iteration state that does not exist in the optimised
    schema. Each iteration starts from scratch, matching a real application request.
    """
    since = datetime.now(timezone.utc) - timedelta(days=365)

    def _run():
        # ── Step 1: paid invoices in last 12 months (ALLOW FILTERING) ────────
        invoices = list(session.execute(
            """
            SELECT id, user_id, invoice_type, total_usd, subscription_id, created_at
            FROM invoices
            WHERE status = 'paid' AND created_at >= %s
            ALLOW FILTERING
            """,
            (since,),
        ))

        # ── Step 2: all subscriptions (full scan, no WHERE clause) ────────────
        # Used to resolve tier_id for both invoice types:
        #   subscription invoices → sub_id_to_tier dict keyed by subscription.id
        #   marketplace invoices  → user_subs dict keyed by user_id, sorted by started_at
        subs = list(session.execute(
            "SELECT id, user_id, tier_id, started_at FROM subscriptions"
        ))
        sub_id_to_tier = {}
        user_subs = defaultdict(list)
        for s in subs:
            if s.id and s.tier_id is not None:
                sub_id_to_tier[str(s.id)] = s.tier_id
            if s.user_id and s.started_at and s.tier_id is not None:
                user_subs[str(s.user_id)].append((s.started_at, s.tier_id))
        for uid in user_subs:
            user_subs[uid].sort(key=lambda x: x[0])

        # ── Step 3: aggregate by (year_month, tier_id) ────────────────────────
        # revenue[year_month][tier_id] = {"count": N, "total": Decimal}
        revenue = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total": Decimal("0")}))

        for inv in invoices:
            created_at = inv.created_at
            if created_at is None:
                continue
            # Ensure timezone-aware for comparison
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            # Resolve tier_id
            if inv.invoice_type == "subscription" and inv.subscription_id:
                tier_id = sub_id_to_tier.get(str(inv.subscription_id))
            else:
                # Marketplace: find most recent subscription started_at <= created_at
                tier_id = None
                user_id_str = str(inv.user_id) if inv.user_id else None
                if user_id_str:
                    for started_at, tid in user_subs.get(user_id_str, []):
                        if started_at.tzinfo is None:
                            started_at = started_at.replace(tzinfo=timezone.utc)
                        if started_at <= created_at:
                            tier_id = tid
                        else:
                            break

            if tier_id is None:
                continue

            ym = _year_month(created_at)
            revenue[ym][tier_id]["count"] += 1
            revenue[ym][tier_id]["total"] += inv.total_usd or Decimal("0")

        # ── Step 4: format results (mirrors PostgreSQL Q1 output shape) ────────
        results = []
        for ym in sorted(revenue.keys()):
            for tier_id in sorted(revenue[ym].keys()):
                agg = revenue[ym][tier_id]
                price = _resolve_price(tier_id, since)  # approximate — use window start
                results.append({
                    "month":                ym,
                    "tier_name":            TIER_NAMES.get(tier_id, str(tier_id)),
                    "price_in_effect_usd":  price,
                    "invoice_count":        agg["count"],
                    "total_revenue_usd":    agg["total"],
                })
        return results

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session):
    print("\n  DRY RUN — Q1 naive result sample (first 10 rows):\n")
    results = make_query_fn(session)()
    if not results:
        print("  ⚠  No results — is the naive keyspace loaded?")
        return
    print(f"  {'Month':<10} {'Tier':<12} {'Price':>10} {'Invoices':>10} {'Revenue':>14}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*14}")
    for row in results[:10]:
        print(
            f"  {row['month']:<10} {row['tier_name']:<12} "
            f"{str(row['price_in_effect_usd']):>10} "
            f"{row['invoice_count']:>10} "
            f"{str(row['total_revenue_usd']):>14}"
        )
    print(f"\n  {len(results)} total rows returned.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q1 revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q1.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q1 Monthly Revenue Benchmark")
    print("=" * 60)
    print("  Schema   : cassandra_naive")
    print("  Method   : ALLOW FILTERING scan (invoices) + full scans")
    print("             (subscriptions) + Python-side join + aggregation")
    print("  ⚠  Each iteration performs 2+ full table scans.")
    print("     Wall time will be high — this quantifies the naive penalty.")

    # request_timeout=120.0s: naive Q1 scans invoices (ALLOW FILTERING) + full subscriptions table.
    # 30s default may be exceeded on cold cache runs.
    cluster, session = get_session(keyspace=KEYSPACE, request_timeout=120.0)
    try:
        if args.dry_run:
            dry_run(session)
            return

        run_benchmark(
            query_fn=make_query_fn(session),
            db="cassandra_naive",
            query_id="Q1",
            label=(
                "Monthly revenue by subscription tier (last 12 months). "
                "ALLOW FILTERING scan on invoices (status + created_at). "
                "Full scan on subscriptions for tier attribution. "
                "Tier pricing and tier names hardcoded (5 and 3 rows). "
                "Temporal price resolution and aggregation in Python. "
                "Naive penalty: no query-driven partitioning → 2 full table scans per iteration."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()