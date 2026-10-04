"""
benchmarks/postgres/q3_session.py — PostgreSQL Baseline: Q3
============================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

    This is the PostgreSQL baseline for the Redis comparison (Q3).
    Redis will serve the same data from a single key lookup at
    O(1) with no connection overhead per request.

Killer feature demonstrated (Redis side):
    Single-key lookup under high concurrency — no JOINs, no query
    planner, no lock contention. Redis serves the full session +
    cart as one in-memory GET under thousands of concurrent clients.

PostgreSQL baseline design notes:
    - Each thread holds its own connection — this mirrors how a
      real application connection pool allocates one connection
      per concurrent request. It means we are measuring the true
      cost of concurrent relational session lookups, not a
      single-connection bottleneck.
    - A pool of user IDs is pre-fetched at startup. Each thread
      picks randomly to avoid cache bias (same logic as Q2).
    - The query fetches the session row + cart JSONB in one
      SELECT — no secondary query for cart contents. This gives
      PostgreSQL its best chance against Redis.
    - Sessions are looked up by user_id (not session token) to
      mirror the realistic application pattern: "give me the
      active session for this authenticated user".
    - concurrency=50 is passed to the harness, matching the
      dissertation spec exactly.

Usage:
    python q3_session.py                   # 1000 iterations, 50 threads
    python q3_session.py --iterations 100  # quick smoke test
    python q3_session.py --concurrency 1   # single-threaded sanity check
    python q3_session.py --explain         # print EXPLAIN ANALYZE, no benchmark
    python q3_session.py --dry-run         # run once, print result sample
    python q3_session.py --pool-size 500   # pre-fetch 500 user IDs (default: 1000)
"""

import argparse
import os
import random
import sys
import threading

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

# ── connection factory ────────────────────────────────────────────────────────
#
# Q3 uses a connection-per-thread model rather than a single shared connection.
# This is intentional: each concurrent request in a real application gets its
# own connection from a pool. Sharing one connection across 50 threads would
# serialise all queries through a single socket — not what we are measuring.

from pg_conn import get_connection

# ── query ─────────────────────────────────────────────────────────────────────
#
# Design notes
# ────────────
# We fetch by user_id + status = 'active', ordered by last_active_at DESC,
# LIMIT 1 — this is the realistic application pattern. A user may have
# multiple expired sessions; we want the current one.
#
# The cart column is JSONB and is returned as-is. In the Redis version,
# the equivalent is: GET session:{user_id} → deserialise JSON.
# Both paths materialise the same data; we are measuring retrieval cost.
#
# idx_sessions_user_id covers the WHERE clause.
# idx_sessions_last_active covers the ORDER BY.
# Together they allow an index-only scan in most cases.

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

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_user_id_pool(conn, pool_size: int) -> list[str]:
    """
    Pre-fetch a random sample of user IDs that have at least one session.
    We do not filter by expires_at because the dataset was generated for
    2025 — all sessions are technically expired relative to the current
    date. The benchmark measures retrieval performance, not expiry logic,
    so we sample from all sessions unconditionally.
    """
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
        raise RuntimeError(
            "No sessions found in the database. "
            "Run the data loader before benchmarking."
        )

    ids = [str(row[0]) for row in rows]
    print(f"  Loaded pool of {len(ids)} user IDs with active sessions.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────
#
# Each thread receives its own dedicated connection opened at thread-start
# time (inside the closure). This matches a connection-pool model and avoids
# the GIL + socket contention of sharing one psycopg2 connection.
#
# The connection is opened once per thread and reused across all of that
# thread's iterations — consistent with how a long-lived worker process
# (gunicorn worker, FastAPI worker) would behave.

def make_query_fn(user_ids: list[str]):
    """
    Return a zero-argument callable.
    Each call opens a thread-local connection on first use, then executes
    Q3 against a randomly selected user ID.
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
            cur.fetchone()   # LIMIT 1 — fetchone mirrors real app usage

    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, user_ids: list[str]):
    """Fetch one session and print the result for sanity checking."""
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Q3 result for user {user_id}:\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q3_SQL, (user_id,))
        row = cur.fetchone()

    if not row:
        print("  ⚠  No active session found for this user.")
        return

    print(f"  Session ID     : {row['session_id']}")
    print(f"  User ID        : {row['user_id']}")
    print(f"  IP Address     : {row['ip_address']}")
    print(f"  Last active    : {row['last_active_at']}")
    print(f"  Expires at     : {row['expires_at']}")

    cart = row["cart"]
    if not cart:
        print(f"\n  Cart           : (empty)")
    else:
        print(f"\n  Cart ({len(cart)} item(s)):")
        print(f"  {'Product ID':<38} {'Name':<30} {'Qty':>4} {'Price':>10}")
        print(f"  {'─'*38} {'─'*30} {'─'*4} {'─'*10}")
        for item in cart:
            print(
                f"  {str(item.get('product_id', 'N/A')):<38} "
                f"{str(item.get('product_name', 'N/A')):<30} "
                f"{str(item.get('quantity', '?')):>4} "
                f"{str(item.get('price_usd', '?')):>10}"
            )


def explain(conn, user_ids: list[str]):
    """Print EXPLAIN ANALYZE for one session fetch."""
    user_id = user_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q3 (user {user_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q3_EXPLAIN_SQL, (user_id,))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Q3 session/cart benchmark")
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations per thread (default: 1000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Number of concurrent threads (default: 50)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of user IDs to pre-fetch for random sampling (default: 1000)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch one session and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q3_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q3_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q3 Session & Cart Benchmark")
    print("=" * 50)

    # Single connection used only for pool fetch, dry-run, and explain.
    # The benchmark itself opens per-thread connections inside make_query_fn.
    conn = get_connection()

    try:
        user_ids = fetch_user_id_pool(conn, args.pool_size)

        if args.explain:
            explain(conn, user_ids)
            return

        if args.dry_run:
            dry_run(conn, user_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(user_ids),
            db="postgres",
            query_id="Q3",
            label=(
                "Active session + cart retrieval by user_id under "
                f"{args.concurrency} concurrent threads. "
                "One connection per thread (connection-pool model). "
                f"Random user sampled from pool of {len(user_ids)} IDs per iteration."
            ),
            iterations=args.iterations,
            concurrency=args.concurrency,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()