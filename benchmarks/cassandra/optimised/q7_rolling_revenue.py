"""
benchmarks/cassandra/optimised/q7_rolling_revenue.py — Cassandra Optimised: Q7
===============================================================================
Q7: 7-day rolling average of daily revenue per subscription tier over
    a 6-month window, with gap-filling for days with zero activity.

Optimised schema — 3 partition reads, CQL date range
──────────────────────────────────────────────────────
Table: invoices_by_tier
PK:    ((tier_id), paid_date ASC, invoice_id ASC)

Three partitions total, one per tier. paid_date is a Cassandra `date`
(LocalDate) extracted from invoice created_at at load time. Tier
attribution was pre-computed at load time.

A CQL range predicate on paid_date (a clustering column) is native CQL —
no ALLOW FILTERING needed. Cassandra returns only the rows within the
date window, sorted by paid_date ASC.

Python then:
  1. Aggregates by day (sums total_usd per date)
  2. Gap-fills missing days with zero revenue
  3. Computes 7-day rolling average

All three are O(window_days × tiers) operations — trivial cost compared
to the 3 partition reads.

Schema effect vs naive:
  Naive  : ALLOW FILTERING full scan of invoices (date range)
           + full subscriptions scan (tier attribution)
           + Python tier attribution every iteration
           → 2 full table scans per iteration
  Optimised : 3 small partition reads with CQL date range
              + pre-resolved tier attribution
              + Python aggregation only (no join, no attribution)

Note on Cassandra vs TimescaleDB for Q7:
  TimescaleDB uses time_bucket_gapfill() + continuous aggregates —
  the gap-fill and rolling average run server-side on pre-materialised
  data. Cassandra does these in Python. Both the 3-partition read and
  the Python computation are included in the measured latency, which
  is the correct comparison: the full end-to-end cost of answering Q7.

Window design
──────────────
Date range bounds are computed once at startup from the actual min/max
paid_date in invoices_by_tier, avoiding hardcoded dates.

Usage:
    python q7_rolling_revenue.py                   # 1000 iterations
    python q7_rolling_revenue.py --iterations 100
    python q7_rolling_revenue.py --dry-run
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE     = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")
WINDOW_DAYS  = 182   # ~6 months
ROLLING_DAYS = 7
TIER_IDS     = [1, 2, 3]
TIER_NAMES   = {1: "Free", 2: "Pro", 3: "Business"}

# ── startup helpers ───────────────────────────────────────────────────────────

def _to_pydate(d) -> date:
    """
    Convert cassandra.util.Date to Python datetime.date.
    Cassandra's `date` column returns a cassandra.util.Date object which
    does not support Python arithmetic operators. Converting via isoformat
    gives a plain datetime.date that supports subtraction and timedelta.
    """
    if isinstance(d, date):
        return d
    # cassandra.util.Date has an isoformat() method returning 'YYYY-MM-DD'
    return date.fromisoformat(str(d))


def fetch_date_range(session) -> tuple[date, date]:
    """
    Find min and max paid_date across all three tier partitions.
    Three small partition scans — startup cost only, not benchmarked.
    """
    all_dates = []
    for tier_id in TIER_IDS:
        rows = list(session.execute(
            "SELECT paid_date FROM invoices_by_tier WHERE tier_id = %s",
            (tier_id,),
        ))
        all_dates.extend(_to_pydate(r.paid_date) for r in rows if r.paid_date)
    if not all_dates:
        raise RuntimeError("No rows in invoices_by_tier — run cassandra_optimised_loader.py first.")
    return min(all_dates), max(all_dates)

# ── helpers ───────────────────────────────────────────────────────────────────

def _gap_fill(daily_totals: dict, start_d: date, end_d: date) -> list[tuple]:
    result = []
    current = start_d
    while current <= end_d:
        result.append((current, daily_totals.get(current, Decimal("0"))))
        current += timedelta(days=1)
    return result

def _rolling_avg(daily_series: list[Decimal], window: int) -> list[Decimal]:
    avgs = []
    for i in range(len(daily_series)):
        chunk = daily_series[max(0, i - window + 1): i + 1]
        avgs.append(sum(chunk) / Decimal(len(chunk)))
    return avgs

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, data_min: date, data_max: date):
    """
    3 partition reads (one per tier_id) with CQL range predicate on paid_date.
    paid_date is a clustering column — range predicates are native CQL, no
    ALLOW FILTERING needed. Python does daily aggregation, gap-fill, rolling avg.
    """
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)

    def _run():
        offset  = random.randint(0, max_start) if max_start > 0 else 0
        start_d = data_min + timedelta(days=offset)
        end_d   = min(start_d + timedelta(days=WINDOW_DAYS - 1), data_max)

        results = []
        for tier_id in TIER_IDS:
            # CQL range on clustering column — no ALLOW FILTERING
            rows = list(session.execute(
                "SELECT paid_date, total_usd, tier_name "
                "FROM invoices_by_tier "
                "WHERE tier_id = %s AND paid_date >= %s AND paid_date <= %s",
                (tier_id, start_d, end_d),
            ))

            # Daily aggregation — convert cassandra.util.Date to Python date
            daily = defaultdict(Decimal)
            for r in rows:
                if r.paid_date and r.total_usd:
                    daily[_to_pydate(r.paid_date)] += r.total_usd

            # Gap-fill + rolling average
            filled  = _gap_fill(daily, start_d, end_d)
            totals  = [t for _, t in filled]
            rolling = _rolling_avg(totals, ROLLING_DAYS)

            for i, (d, total) in enumerate(filled):
                results.append({
                    "day":           str(d),
                    "tier_name":     TIER_NAMES[tier_id],
                    "daily_revenue": total,
                    "rolling_7d_avg": rolling[i],
                })

        return results

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, data_min, data_max):
    print(f"\n  DRY RUN — Q7 optimised rolling revenue\n")
    fn = make_query_fn(session, data_min, data_max)
    results = fn()
    if not results:
        print("  ⚠  No results.")
        return
    # Show first non-zero rows for readability
    non_zero = [r for r in results if r["daily_revenue"] > 0]
    print(f"  {len(results)} total rows. {len(non_zero)} with non-zero revenue.")
    print(f"\n  {'Day':<12} {'Tier':<12} {'Daily':>12} {'7d Avg':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for r in (non_zero[:10] if non_zero else results[:10]):
        print(
            f"  {r['day']:<12} {r['tier_name']:<12} "
            f"{float(r['daily_revenue']):>12.2f} "
            f"{float(r['rolling_7d_avg']):>12.2f}"
        )
    print(f"\n  3 partition reads (one per tier_id), CQL date range on clustering column.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q7 rolling revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q7 Rolling Revenue Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_optimised (invoices_by_tier)")
    print("  Method : 3 partition reads, CQL range on paid_date (clustering col)")
    print("           Python: daily aggregation + gap-fill + 7-day rolling avg")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        print("\n  Fetching date range from invoices_by_tier (startup scan)...")
        data_min, data_max = fetch_date_range(session)
        print(f"  Date range: {data_min} → {data_max}")

        if args.dry_run:
            dry_run(session, data_min, data_max)
            return

        run_benchmark(
            query_fn=make_query_fn(session, data_min, data_max),
            db="cassandra_optimised",
            query_id="Q7",
            label=(
                "7-day rolling revenue average per tier, 6-month window. "
                "Table: invoices_by_tier PK ((tier_id), paid_date ASC, invoice_id). "
                "3 partition reads with CQL range on paid_date (clustering column) — "
                "no ALLOW FILTERING. Tier attribution pre-computed at load time. "
                "Python: daily aggregation, gap-fill (zero for missing days), "
                "7-day rolling average. "
                f"Window: {WINDOW_DAYS} days, randomly placed in "
                f"({data_min} → {data_max})."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()