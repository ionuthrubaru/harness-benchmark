"""
benchmarks/timescaledb/naive/q6_events.py — TimescaleDB Naive: Q6
==================================================================
Q6: Retrieve all activity events for a specific user in a
    30-day window, ordered by time.

This is Cassandra's killer query — and the first query where the
TimescaleDB naive schema shows a genuine engine effect over PostgreSQL.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
The same WHERE clause runs unchanged:
    WHERE e.user_id = %s
      AND e.occurred_at >= %s
      AND e.occurred_at <  %s
    ORDER BY e.occurred_at DESC

Engine effect for Q6 (naive) — chunk pruning
─────────────────────────────────────────────
events is a hypertable partitioned by occurred_at with 7-day chunks
(the naive default). A 30-day window query touches at most 5 consecutive
7-day chunks. TimescaleDB prunes all other chunks at the query planning
stage — they are never opened, never scanned, and never considered.

PostgreSQL uses the composite B-tree index idx_events_user_time on
(user_id, occurred_at DESC) to achieve a similar effect. Both approaches
avoid a full table scan, but they work differently:

  PostgreSQL: B-tree index traversal → random I/O into the heap for
              matching rows → back to index for next matching row.
              Cost scales with the number of matching index entries.

  TimescaleDB: Chunk pruning → only 4-5 chunk files are opened.
               Within those chunks, the same composite index is used.
               The key advantage is that the chunk files are smaller
               (7 days of data each) and more likely to fit in the
               OS page cache as a unit.

Whether this translates to lower latency than PostgreSQL at this dataset
scale is the engine effect measurement. With ~1M events over 2 years,
each 7-day chunk holds ~10,000 rows — small enough to be cache-friendly.

Window anchoring
─────────────────
Identical to PostgreSQL Q6: (user_id, occurred_at) pairs sampled from the
events table. The 30-day window centred ±15 days on the anchor guarantees
at least one result per iteration.

Usage:
    cd benchmarks/timescaledb/naive
    python q6_events.py                   # 1000 iterations
    python q6_events.py --iterations 100  # quick smoke test
    python q6_events.py --explain         # EXPLAIN ANALYZE (shows chunk pruning)
    python q6_events.py --dry-run         # run once, print result sample
    python q6_events.py --pool-size 500   # smaller anchor pool
"""

import argparse
import os
import random
import sys
from datetime import timedelta

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

WINDOW_DAYS = 30

# ── query — identical to PostgreSQL Q6 ────────────────────────────────────────

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

# ── anchor pool ───────────────────────────────────────────────────────────────

def fetch_anchor_pool(conn, pool_size: int) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, occurred_at FROM events ORDER BY RANDOM() LIMIT %s",
            (pool_size,),
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No events found — run timescaledb_naive_loader.py first.")
    print(f"  Anchor pool: {len(rows):,} (user_id, occurred_at) pairs loaded.")
    return list(rows)


def anchor_window(anchor_dt) -> tuple:
    start = anchor_dt - timedelta(days=15)
    end   = anchor_dt + timedelta(days=15)
    return start, end

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, pairs: list[tuple]):
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start, end = anchor_window(anchor_dt)
        with conn.cursor() as cur:
            cur.execute(Q6_SQL, (user_id, start, end))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    start, end = anchor_window(anchor_dt)
    print(f"\n  DRY RUN — Q6 naive for user {user_id}")
    print(f"  Window: {start} → {end}\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q6_SQL, (user_id, start, end))
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No events in this window.")
        return
    print(f"  {len(rows)} event(s) returned:\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<28} {'Product ID':<36}")
    print(f"  {'─'*4} {'─'*25} {'─'*28} {'─'*36}")
    for i, row in enumerate(rows[:20], 1):
        print(
            f"  {i:<4} {str(row['event_type']):<25} "
            f"{str(row['occurred_at']):<28} "
            f"{str(row['product_id'] or 'N/A'):<36}"
        )
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more rows")


def explain(conn, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    start, end = anchor_window(anchor_dt)
    print(f"\n  EXPLAIN ANALYZE — Q6 naive (user {user_id}, {start} → {end}):\n")
    print("  Note: look for 'Chunks excluded by runtime exclusion' in the plan.")
    print("        This confirms TimescaleDB chunk pruning is active.\n")
    with conn.cursor() as cur:
        cur.execute(Q6_EXPLAIN_SQL, (user_id, start, end))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q6 events benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q6.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q6 User Events Benchmark")
    print("=" * 60)
    print("  Schema : naive (events hypertable, 7-day chunks)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : chunk pruning on occurred_at — 30-day window")
    print("           touches at most 5 of the ~104 7-day chunks")
    print(f"  Window : {WINDOW_DAYS} days (centred on anchor event)")

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
            db="timescaledb_naive",
            query_id="Q6",
            label=(
                f"All events for a user in a {WINDOW_DAYS}-day window, "
                "ordered by occurred_at DESC. "
                "SQL identical to PostgreSQL baseline. "
                "TimescaleDB engine effect: chunk pruning on events hypertable — "
                "30-day window touches ≤5 of ~104 7-day chunks. "
                "Naive schema: 7-day chunks (default). "
                "Window centred ±15 days on sampled anchor event — guaranteed non-empty. "
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