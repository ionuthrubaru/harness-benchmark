"""
benchmarks/cassandra/optimised/q3_session.py — Cassandra Optimised: Q3
=======================================================================
Q3: Retrieve a user's most recently active session and full cart
    contents under 50 concurrent requests.

Optimised schema — single-partition LIMIT 1
────────────────────────────────────────────
Table: sessions_by_user
PK:    ((user_id), last_active_at DESC, id ASC)

user_id is the partition key. last_active_at DESC is the first clustering
column, so the most recently active session is physically the first row
in the partition. LIMIT 1 returns it in a single page read with no
filtering and no Python-side sort.

Schema effect vs naive:
  Naive  : ALLOW FILTERING full scan on sessions.user_id
           + Python sort by last_active_at DESC → one full table scan
  Optimised : single-partition LIMIT 1 read → returns in O(1)

Usage:
    python q3_session.py                   # 1000 iterations, 50 threads
    python q3_session.py --iterations 100
    python q3_session.py --dry-run
    python q3_session.py --pool-size 500
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

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_user_id_pool(session, pool_size: int) -> list:
    rows = list(session.execute(
        f"SELECT user_id FROM sessions_by_user LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No rows in sessions_by_user — run cassandra_optimised_loader.py first.")
    ids = list({r.user_id for r in rows})
    random.shuffle(ids)
    print(f"  User ID pool: {len(ids):,} unique users loaded.")
    return ids

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, user_ids: list):
    """
    Single-partition LIMIT 1 read.
    Clustering order is last_active_at DESC so the first row is the
    most recently active session — no Python sort needed.
    """
    def _run():
        user_id = random.choice(user_ids)
        row = session.execute(
            "SELECT id, user_id, cart, ip_address, user_agent, "
            "created_at, last_active_at, expires_at "
            "FROM sessions_by_user WHERE user_id = %s LIMIT 1",
            (user_id,),
        ).one()
        return row
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, user_ids: list):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Q3 optimised session for user {user_id}\n")
    row = make_query_fn(session, [user_id])()
    if not row:
        print("  ⚠  No session found.")
        return
    print(f"  Session ID   : {str(row.id)[:20]}...")
    print(f"  Last active  : {row.last_active_at}")
    print(f"  Expires      : {row.expires_at}")
    print(f"  Cart preview : {str(row.cart)[:80] if row.cart else '[]'}")
    print(f"  Method       : LIMIT 1 on partition — most recent session first")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q3 session benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q3.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q3 Session Fetch Benchmark")
    print("=" * 60)
    print("  Schema      : cassandra_optimised (sessions_by_user)")
    print("  Method      : Single-partition LIMIT 1, clustering DESC")
    print("  Concurrency : 50 threads (shared thread-safe Session)")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pool = fetch_user_id_pool(session, args.pool_size)
        if args.dry_run:
            dry_run(session, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(session, pool),
            db="cassandra_optimised",
            query_id="Q3",
            label=(
                "Most recent session for a user. "
                "Table: sessions_by_user PK ((user_id), last_active_at DESC, id). "
                "LIMIT 1 returns most recent session as first row in partition — "
                "no ALLOW FILTERING, no Python sort. "
                f"Pool of {len(pool):,} user IDs. Concurrency=50."
            ),
            iterations=args.iterations,
            concurrency=50,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()