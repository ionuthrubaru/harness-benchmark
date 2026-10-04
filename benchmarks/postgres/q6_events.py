"""
benchmarks/postgres/q6_events.py — PostgreSQL Baseline: Q6
===========================================================
Q6: Retrieve all activity events for a specific user in a
    30-day window, ordered by time.

    This is the PostgreSQL baseline for the Cassandra comparison (Q6).
    Cassandra will serve the same result via a single wide-row partition
    scan with (user_id, month) as the composite partition key and
    occurred_at as the clustering column.

Killer feature demonstrated (Cassandra side):
    Wide-row partition scan at 5M+ rows — Cassandra stores all events
    for a (user_id, month) partition physically contiguous on disk.
    A time-range scan within a partition is a sequential read with no
    index traversal. PostgreSQL must use a B-tree index scan across a
    heap table, which involves random I/O for large result sets and
    lock overhead on the shared buffer pool.

PostgreSQL baseline design notes
──────────────────────────────────
The composite index idx_events_user_time on (user_id, occurred_at DESC)
covers both the WHERE and ORDER BY clauses, giving PostgreSQL an
index-only scan path in most cases.

Window anchoring
────────────────
(user_id, occurred_at) pairs are sampled directly from the events table.
Each pair becomes an anchor: the 30-day window is centred ±15 days around
it, guaranteeing the window always contains at least one real event.

This replaces the previous approach of picking a random date within a
hardcoded DATASET_START/DATASET_END range, which returned zero rows
whenever the random window missed the user's sparse event history.

Usage:
    python q6_events.py                   # 1000 iterations
    python q6_events.py --iterations 100  # quick smoke test
    python q6_events.py --explain         # EXPLAIN ANALYZE
    python q6_events.py --dry-run         # run once, print result sample
    python q6_events.py --pool-size 500   # pre-fetch 500 anchor pairs
"""

import argparse
import os
import random
import sys
from datetime import timedelta

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

WINDOW_DAYS = 30

# -- connection ----------------------------------------------------------------

from pg_conn import get_connection

# -- query --------------------------------------------------------------------

Q6_SQL = """
SELECT
    e.id            AS event_id,
    e.event_type,
    e.occurred_at,
    e.product_id,
    e.session_id,
    e.metadata
FROM events e
WHERE e.user_id     = %s
  AND e.occurred_at >= %s
  AND e.occurred_at <  %s
ORDER BY e.occurred_at DESC;
"""

Q6_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q6_SQL

# -- anchor pool --------------------------------------------------------------

def fetch_anchor_pool(conn, pool_size: int) -> list[tuple]:
    """
    Sample (user_id, occurred_at) pairs directly from the events table.
    The 30-day window is centred on occurred_at so every iteration is
    guaranteed to return at least one row — no empty scans.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, occurred_at
            FROM events
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            (pool_size,),
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            "No events found in the database. "
            "Run the data loader before benchmarking."
        )

    print(f"  Loaded pool of {len(rows)} (user_id, anchor) pairs from events.")
    return list(rows)

def anchor_window(anchor_dt) -> tuple:
    """
    30-day window centred on the anchor event.
    Anchor falls ~15 days in — always inside the window.
    """
    start = anchor_dt - timedelta(days=15)
    end   = anchor_dt + timedelta(days=15)
    return start, end

# -- benchmark factory --------------------------------------------------------

def make_query_fn(conn, pairs: list[tuple]):
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start, end = anchor_window(anchor_dt)
        with conn.cursor() as cur:
            cur.execute(Q6_SQL, (user_id, start, end))
            cur.fetchall()
    return _run

# -- helper modes -------------------------------------------------------------

def dry_run(conn, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    start, end = anchor_window(anchor_dt)

    print(f"\n  DRY RUN -- Q6 events for user {user_id}")
    print(f"  Window: {start} to {end}\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q6_SQL, (user_id, start, end))
        rows = cur.fetchall()

    if not rows:
        print("  ⚠  Still no events — check the loader ran correctly.")
        return

    print(f"  {len(rows)} event(s) returned:\n")
    print(
        f"  {'#':<4} {'Event type':<25} {'Occurred at':<28} "
        f"{'Product ID':<38} {'Session ID'}"
    )
    print(f"  {'─'*4} {'─'*25} {'─'*28} {'─'*38} {'─'*20}")
    for i, row in enumerate(rows[:20], 1):
        print(
            f"  {i:<4} {str(row['event_type']):<25} "
            f"{str(row['occurred_at']):<28} "
            f"{str(row['product_id'] or 'N/A'):<38} "
            f"{str(row['session_id'] or 'N/A')}"
        )
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more rows")


def explain(conn, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    start, end = anchor_window(anchor_dt)
    print(f"\n  EXPLAIN ANALYZE -- Q6 (user {user_id}, {start} to {end}):\n")
    with conn.cursor() as cur:
        cur.execute(Q6_EXPLAIN_SQL, (user_id, start, end))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# -- entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Q6 user events benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of anchor pairs to pre-fetch (default: 1000)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch events for one user in one window then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q6_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q6_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL -- Q6 User Events Benchmark")
    print("=" * 50)
    print(f"  Window size : {WINDOW_DAYS} days (centred on anchor event)")

    conn = get_connection()

    try:
        pairs = fetch_anchor_pool(conn, args.pool_size)

        if args.explain:
            explain(conn, pairs)
            return

        if args.dry_run:
            dry_run(conn, pairs)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, pairs),
            db="postgres",
            query_id="Q6",
            label=(
                f"All events for a user in a {WINDOW_DAYS}-day window, ordered "
                "by occurred_at DESC. Composite index on (user_id, occurred_at). "
                "Window centred on a sampled real event — guaranteed non-empty. "
                f"Pool of {args.pool_size} (user_id, anchor) pairs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()