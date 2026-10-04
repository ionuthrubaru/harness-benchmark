"""
benchmarks/timescaledb/naive/q3_session.py — TimescaleDB Naive: Q3
===================================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
TimescaleDB adds nothing for Q3 — sessions is a regular PostgreSQL table
(not a hypertable). The query, indexes, and connection-per-thread model
are identical to the PostgreSQL baseline. Latency should be comparable,
confirming TimescaleDB does not penalise non-time-series workloads.

One connection per thread — same rationale as PostgreSQL Q3
─────────────────────────────────────────────────────────────
Each thread opens its own psycopg2 connection via thread-local storage,
mirroring a real application connection pool. This is necessary because
psycopg2 connections are not thread-safe. The connection is created once
per thread and reused across all iterations in that thread.

Usage:
    cd benchmarks/timescaledb/naive
    python q3_session.py                   # 1000 iterations, 50 threads
    python q3_session.py --iterations 100  # quick smoke test
    python q3_session.py --concurrency 1   # single-threaded sanity check
    python q3_session.py --explain         # EXPLAIN ANALYZE
    python q3_session.py --dry-run         # run once, print result
    python q3_session.py --pool-size 500   # smaller pool
"""

import argparse
import os
import random
import sys
import threading

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

# ── query — identical to PostgreSQL Q3 ────────────────────────────────────────

Q3_SQL = """
SELECT
    s.id                AS session_id,
    s.user_id,
    s.cart,
    s.ip_address,
    s.user_agent,
    s.created_at,
    s.last_active_at,
    s.expires_at
FROM sessions s
WHERE s.user_id = %s
ORDER BY s.last_active_at DESC
LIMIT 1;
"""

Q3_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q3_SQL

# ── ID pool ────────────────────────────────────────────────────────────────────

def fetch_user_id_pool(conn, pool_size: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id FROM (
                SELECT DISTINCT user_id FROM sessions
            ) AS distinct_users
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            (pool_size,),
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No sessions found — run timescaledb_naive_loader.py first.")
    ids = [str(row[0]) for row in rows]
    print(f"  User ID pool: {len(ids):,} entries loaded.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(user_ids: list[str]):
    """
    One thread-local connection per thread, reused across iterations.
    Identical pattern to PostgreSQL Q3.
    """
    local = threading.local()

    def _get_or_create_conn():
        if not getattr(local, "conn", None) or local.conn.closed:
            local.conn = get_connection()
        return local.conn

    def _run():
        conn = _get_or_create_conn()
        user_id = random.choice(user_ids)
        with conn.cursor() as cur:
            cur.execute(Q3_SQL, (user_id,))
            cur.fetchone()

    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Q3 naive for user {user_id}:\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q3_SQL, (user_id,))
        row = cur.fetchone()
    if not row:
        print("  ⚠  No session found for this user.")
        return
    print(f"  Session ID   : {row['session_id']}")
    print(f"  Last active  : {row['last_active_at']}")
    print(f"  Expires      : {row['expires_at']}")
    cart = row["cart"]
    if not cart:
        print("  Cart         : (empty)")
    else:
        print(f"  Cart ({len(cart)} item(s)):")
        for item in cart:
            print(f"    product_id={item.get('product_id','N/A')}  "
                  f"qty={item.get('quantity','?')}  "
                  f"price={item.get('price_usd','?')}")


def explain(conn, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q3 naive (user {user_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q3_EXPLAIN_SQL, (user_id,))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q3 session benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q3.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q3 Session & Cart Benchmark")
    print("=" * 60)
    print("  Schema      : naive (sessions is a regular table, not hypertable)")
    print("  SQL         : identical to PostgreSQL baseline")
    print(f"  Concurrency : {args.concurrency} threads, one connection each")

    conn = get_connection()
    try:
        pool = fetch_user_id_pool(conn, args.pool_size)
        if args.explain:
            explain(conn, pool)
            return
        if args.dry_run:
            dry_run(conn, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(pool),
            db="timescaledb_naive",
            query_id="Q3",
            label=(
                "Active session + cart retrieval by user_id under "
                f"{args.concurrency} concurrent threads. "
                "One psycopg2 connection per thread (thread-local). "
                "sessions is a plain PostgreSQL table (not hypertable). "
                "SQL identical to PostgreSQL baseline. "
                f"Pool of {len(pool):,} user IDs."
            ),
            iterations=args.iterations,
            concurrency=args.concurrency,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()