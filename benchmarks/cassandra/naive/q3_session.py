"""
benchmarks/cassandra/naive/q3_session.py — Cassandra Naive: Q3
===============================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

Naive schema penalty — ALLOW FILTERING on sessions.user_id
────────────────────────────────────────────────────────────
The naive sessions table has id (the session token, a text hash) as its
sole partition key. Querying by user_id — the realistic application
pattern — requires ALLOW FILTERING: Cassandra scans every partition
on every node to find rows where user_id matches.

In the optimised schema this is eliminated entirely: sessions_by_user
partitions on user_id, so Q3 becomes a single-partition LIMIT 1 read.
The difference between naive (ALLOW FILTERING across all session
partitions) and optimised (single-partition LIMIT 1) is the schema
effect for Q3.

No ORDER BY possible in naive CQL
───────────────────────────────────
CQL ORDER BY is only valid on clustering columns. The naive sessions
table has no clustering column — id is the sole partition key. To find
the most recently active session for a user, the benchmark fetches all
sessions for that user (ALLOW FILTERING) and sorts by last_active_at
DESC in Python. This adds client-side sort overhead on top of the
ALLOW FILTERING scan cost.

Cassandra Session thread safety
────────────────────────────────
The Cassandra Python driver Session is fully thread-safe. Its internal
connection pool handles concurrent requests from multiple threads
transparently. Unlike psycopg2 (one connection per thread) or the Neo4j
driver (one session per thread), a single cassandra.cluster.Session
safely services all 50 concurrent benchmark threads. This is an
important architectural difference documented in the methodology:
Cassandra's driver model is inherently connection-pool-based, not
connection-per-request.

Concurrency = 50 — identical to the PostgreSQL Q3 baseline. All
specialised DBs run Q3 at the same concurrency so the only variable is
the database engine and schema.

Usage:
    python q3_session.py                   # 1000 iterations, 50 threads
    python q3_session.py --iterations 100  # quick smoke test
    python q3_session.py --dry-run         # run once for one user, print result
    python q3_session.py --pool-size 500   # smaller user pool
"""

import argparse
import os
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_user_id_pool(session, pool_size: int) -> list:
    """
    Fetch user_ids that have at least one session.
    SELECT user_id FROM sessions has no WHERE clause — no ALLOW FILTERING.
    The pool is shuffled so threads pick from diverse users.
    """
    rows = list(session.execute(f"SELECT user_id FROM sessions LIMIT {pool_size}"))
    if not rows:
        raise RuntimeError("No sessions found — run cassandra_naive_loader.py first.")
    ids = [r.user_id for r in rows if r.user_id is not None]
    random.shuffle(ids)
    print(f"  User ID pool: {len(ids):,} entries loaded.")
    return ids

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, user_ids: list):
    """
    Each call:
      1. ALLOW FILTERING scan for all sessions by user_id
      2. Python-side sort by last_active_at DESC → take first row
    The sort is necessary because CQL ORDER BY requires a clustering
    column; the naive schema has none.
    """
    def _run():
        user_id = random.choice(user_ids)
        rows = list(session.execute(
            "SELECT id, user_id, cart, ip_address, user_agent, "
            "created_at, last_active_at, expires_at "
            "FROM sessions WHERE user_id = %s ALLOW FILTERING",
            (user_id,),
        ))
        if not rows:
            return None
        # Sort Python-side — CQL ORDER BY not available on non-clustering columns
        rows.sort(key=lambda r: r.last_active_at or 0, reverse=True)
        return rows[0]

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, user_ids: list):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Q3 naive session for user {user_id}\n")
    fn = make_query_fn(session, [user_id])
    row = fn()
    if not row:
        print("  ⚠  No session found for this user.")
        return
    print(f"  Session ID    : {row.id[:20]}...")
    print(f"  Last active   : {row.last_active_at}")
    print(f"  Expires       : {row.expires_at}")
    cart_preview = str(row.cart)[:80] if row.cart else "[]"
    print(f"  Cart (preview): {cart_preview}")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q3 session benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q3.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q3 Session Fetch Benchmark")
    print("=" * 60)
    print("  Schema      : cassandra_naive")
    print("  Method      : ALLOW FILTERING on sessions.user_id + Python sort")
    print("  Concurrency : 50 threads (shared thread-safe Session)")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pool = fetch_user_id_pool(session, args.pool_size)

        if args.dry_run:
            dry_run(session, pool)
            return

        run_benchmark(
            query_fn=make_query_fn(session, pool),
            db="cassandra_naive",
            query_id="Q3",
            label=(
                "Most recent session for a user. ALLOW FILTERING scan on sessions "
                "by user_id (user_id is not a partition key). Python-side sort by "
                "last_active_at DESC to find most recent session (CQL ORDER BY not "
                "available without clustering column). Concurrency=50 threads "
                "sharing one thread-safe Cassandra Session. "
                f"Pool of {len(pool):,} user IDs."
            ),
            iterations=args.iterations,
            concurrency=50,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()