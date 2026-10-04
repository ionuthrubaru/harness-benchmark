"""
benchmarks/cassandra/optimised/q1_revenue.py — Cassandra Optimised: Q1
=======================================================================
Q1: Monthly revenue by subscription tier, last 12 months.

Optimised schema — 36-partition fan-out, no ALLOW FILTERING
─────────────────────────────────────────────────────────────
Table: invoices_by_month_tier
PK:    ((year_month, tier_id), invoice_id)

Tier attribution and temporal price resolution were pre-computed at
load time. Each partition holds all paid invoices for one (month, tier)
combination. Q1 fans out to up to 36 partitions (12 months × 3 tiers),
reads each, counts invoices and sums total_usd, then aggregates in Python.

Schema effect vs naive:
  Naive  : ALLOW FILTERING scan of all invoices + full subscriptions scan
           + Python temporal attribution every iteration → minutes per call
  Optimised : 36 small partition reads, pre-resolved values, Python sum
              → milliseconds per call

The fan-out of 36 reads is O(months × tiers), not O(dataset size).
Regardless of how many invoices are in the dataset, Q1 always reads
exactly the same number of partitions.

Usage:
    python q1_revenue.py                   # 1000 iterations
    python q1_revenue.py --iterations 100
    python q1_revenue.py --dry-run
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

KEYSPACE   = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")
TIER_IDS   = [1, 2, 3]
TIER_NAMES = {1: "Free", 2: "Pro", 3: "Business"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _months_back(n: int) -> list[str]:
    """Return last n 'YYYY-MM' strings ending with the current month."""
    now = datetime.now(timezone.utc)
    months = []
    for i in range(n - 1, -1, -1):
        dt = datetime(now.year, now.month, 1, tzinfo=timezone.utc) - timedelta(days=i * 30)
        months.append(dt.strftime("%Y-%m"))
    # Deduplicate preserving order (timedelta rounding can produce duplicates)
    seen = set()
    return [m for m in months if not (m in seen or seen.add(m))]

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session):
    """
    Fan out to up to 36 partitions (12 months × 3 tiers).
    Each partition read is a small sequential scan — no ALLOW FILTERING.
    Aggregation (count, sum) is done in Python over the pre-resolved rows.
    """
    def _run():
        months = _months_back(12)
        results = []

        for ym in months:
            for tier_id in TIER_IDS:
                rows = list(session.execute(
                    "SELECT invoice_id, invoice_type, total_usd, "
                    "tier_name, monthly_price_usd_at_time, created_at "
                    "FROM invoices_by_month_tier "
                    "WHERE year_month = %s AND tier_id = %s",
                    (ym, tier_id),
                ))
                if not rows:
                    continue
                total_revenue = sum(r.total_usd for r in rows if r.total_usd)
                results.append({
                    "month":                  ym,
                    "tier_name":              TIER_NAMES[tier_id],
                    "price_in_effect_usd":    rows[0].monthly_price_usd_at_time,
                    "invoice_count":          len(rows),
                    "total_revenue_usd":      total_revenue,
                })

        return sorted(results, key=lambda r: (r["month"], r["tier_name"]))

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session):
    print("\n  DRY RUN — Q1 optimised result sample (first 10 rows):\n")
    results = make_query_fn(session)()
    if not results:
        print("  ⚠  No results — is the optimised keyspace loaded?")
        return
    print(f"  {'Month':<10} {'Tier':<12} {'Price':>10} {'Invoices':>10} {'Revenue':>14}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*14}")
    for row in results[:10]:
        print(
            f"  {row['month']:<10} {row['tier_name']:<12} "
            f"{str(row['price_in_effect_usd']):>10} "
            f"{row['invoice_count']:>10} "
            f"{float(row['total_revenue_usd']):>14.2f}"
        )
    print(f"\n  {len(results)} total rows. Partitions read: {len(results)} "
          f"(≤36 = 12 months × 3 tiers).")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q1 revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q1.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q1 Monthly Revenue Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_optimised (invoices_by_month_tier)")
    print("  Method : 36-partition fan-out (12 months × 3 tiers)")
    print("           Pre-resolved tier + price — Python sum only")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        if args.dry_run:
            dry_run(session)
            return
        run_benchmark(
            query_fn=make_query_fn(session),
            db="cassandra_optimised",
            query_id="Q1",
            label=(
                "Monthly revenue by tier (last 12 months). "
                "Table: invoices_by_month_tier PK ((year_month, tier_id), invoice_id). "
                "Fan-out: 12 months × 3 tiers = up to 36 partition reads per iteration. "
                "Tier attribution and price resolution pre-computed at load time. "
                "Python aggregates count + sum per partition. No ALLOW FILTERING."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()