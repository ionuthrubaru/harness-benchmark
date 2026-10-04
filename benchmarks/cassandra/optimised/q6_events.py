"""
benchmarks/cassandra/optimised/q6_events.py — Cassandra Optimised: Q6
======================================================================
Q6: Retrieve all activity events for a specific user in a 30-day
    window, ordered by time.

This is Cassandra's killer query on the correct schema.

Optimised schema — 1-2 partition reads, no full table scan
───────────────────────────────────────────────────────────
Table: events_by_user_month
PK:    ((user_id, year_month), occurred_at DESC, id ASC)

Every event for a given user in a given calendar month is in one
partition, physically contiguous on disk, pre-sorted by occurred_at
DESC. A 30-day window may straddle two calendar months — Q6 queries
at most 2 partitions and merges the results Python-side.

Each partition read uses a native CQL range predicate on occurred_at
(a clustering column) — no ALLOW FILTERING, no full table scan. Cost
is proportional only to the number of events for that user in those
months, completely independent of the total dataset size.

Schema effect vs naive:
  Naive  : Paginated full table scan of ~1M events (entire table),
           Python-side filter, Python sort → 30-90 seconds per call.
           ALLOW FILTERING with WHERE clause additionally causes server
           ReadTimeout at this dataset scale regardless of configuration.
  Optimised : 1-2 partition reads, clustering-ordered, milliseconds per call.

This is the most dramatic schema effect in the entire Cassandra chapter
— potentially 10,000× or more. The naive schema cannot run Q6 at all
with a server-side WHERE predicate (ReadTimeout); even the paginated
workaround takes tens of seconds. The optimised schema runs in
low milliseconds because it reads only what it needs.

Anchor pool
────────────
(user_id, occurred_at) pairs are sampled directly from events_by_user_month.
The year_month for each anchor is computed from occurred_at; the 30-day
window centred on the anchor guarantees at least one event per iteration.
Windows straddling a month boundary are handled by querying both months.

Usage:
    python q6_events.py                   # 1000 iterations
    python q6_events.py --iterations 100
    python q6_events.py --dry-run
    python q6_events.py --pool-size 500
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE    = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")
WINDOW_DAYS = 30

# ── helpers ───────────────────────────────────────────────────────────────────

def _year_month(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def _anchor_window(anchor_dt: datetime) -> tuple[datetime, datetime]:
    """30-day window centred ±15 days on anchor. Identical to PostgreSQL Q6 methodology."""
    if anchor_dt.tzinfo is None:
        anchor_dt = anchor_dt.replace(tzinfo=timezone.utc)
    return anchor_dt - timedelta(days=15), anchor_dt + timedelta(days=15)

def _months_in_range(start: datetime, end: datetime) -> list[str]:
    """Return deduplicated list of 'YYYY-MM' strings covering start→end."""
    months = []
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    end_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while current <= end_month:
        months.append(current.strftime("%Y-%m"))
        # Advance by one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_anchor_pool(session, pool_size: int) -> list[tuple]:
    """
    Sample (user_id, year_month, occurred_at) from events_by_user_month.
    No WHERE clause — Cassandra returns from the first token ranges it encounters.
    Shuffle in Python for variety.
    """
    rows = list(session.execute(
        f"SELECT user_id, year_month, occurred_at FROM events_by_user_month LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No rows in events_by_user_month — run cassandra_optimised_loader.py first.")
    pairs = [
        (r.user_id, r.occurred_at)
        for r in rows if r.user_id and r.occurred_at
    ]
    random.shuffle(pairs)
    print(f"  Anchor pool: {len(pairs):,} (user_id, occurred_at) pairs loaded.")
    return pairs

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, pairs: list[tuple]):
    """
    Query 1-2 partitions of events_by_user_month using the clustering column
    occurred_at for the date range — native CQL range predicate, no ALLOW FILTERING.
    Results come back pre-sorted DESC by the clustering order; Python merge-sorts
    only when the window straddles two months (rare, trivial cost).
    """
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        if anchor_dt.tzinfo is None:
            anchor_dt_tz = anchor_dt.replace(tzinfo=timezone.utc)
        else:
            anchor_dt_tz = anchor_dt
        start, end = _anchor_window(anchor_dt_tz)
        months = _months_in_range(start, end)

        all_rows = []
        for ym in months:
            rows = list(session.execute(
                "SELECT id, event_type, occurred_at, product_id, session_id, metadata "
                "FROM events_by_user_month "
                "WHERE user_id = %s AND year_month = %s "
                "AND occurred_at >= %s AND occurred_at < %s",
                (user_id, ym, start, end),
            ))
            all_rows.extend(rows)

        # Merge-sort only needed when two months were queried.
        # Each partition is already sorted DESC by clustering order.
        if len(months) > 1:
            all_rows.sort(
                key=lambda r: r.occurred_at or datetime.min,
                reverse=True,
            )

        return all_rows

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    if anchor_dt.tzinfo is None:
        anchor_dt = anchor_dt.replace(tzinfo=timezone.utc)
    start, end = _anchor_window(anchor_dt)
    months = _months_in_range(start, end)

    print(f"\n  DRY RUN — Q6 optimised for user {user_id}")
    print(f"  Window    : {start} → {end}")
    print(f"  Partitions: {months} ({len(months)} read)\n")

    fn = make_query_fn(session, [(user_id, anchor_dt)])
    rows = fn()
    if not rows:
        print("  ⚠  No events in this window.")
        return
    print(f"  {len(rows)} event(s) — from {len(months)} partition(s), pre-sorted:\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<32} {'Product ID'}")
    print(f"  {'─'*4} {'─'*25} {'─'*32} {'─'*36}")
    for i, r in enumerate(rows[:15], 1):
        print(
            f"  {i:<4} {str(r.event_type):<25} "
            f"{str(r.occurred_at):<32} "
            f"{str(r.product_id) if r.product_id else 'N/A'}"
        )
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more rows")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q6 events benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q6.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q6 User Events Benchmark")
    print("=" * 60)
    print("  Schema  : cassandra_optimised (events_by_user_month)")
    print("  Method  : 1-2 partition reads, CQL range on clustering column")
    print(f"  Window  : {WINDOW_DAYS} days centred on anchor event")
    print("  ✔  Cassandra's killer query on the correct schema.")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pairs = fetch_anchor_pool(session, args.pool_size)
        if args.dry_run:
            dry_run(session, pairs)
            return
        run_benchmark(
            query_fn=make_query_fn(session, pairs),
            db="cassandra_optimised",
            query_id="Q6",
            label=(
                f"All events for a user in a {WINDOW_DAYS}-day window. "
                "Table: events_by_user_month PK ((user_id, year_month), occurred_at DESC, id). "
                "1-2 partition reads — CQL range predicate on clustering column occurred_at. "
                "No ALLOW FILTERING. Results pre-sorted by clustering order. "
                "Python merge-sort only when window straddles two calendar months. "
                f"Window centred ±15 days on sampled anchor event. "
                f"Pool of {args.pool_size} (user_id, anchor) pairs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()