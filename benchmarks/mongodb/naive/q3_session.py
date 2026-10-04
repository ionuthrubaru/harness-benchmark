"""
benchmarks/mongodb/naive/q3_session.py — MongoDB Naive: Q3
===========================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

Naive implementation notes:
    Sessions are stored as flat documents mirroring the PostgreSQL schema.
    The cart field is a JSON string (not a native array) — matching the
    naive constraint that no MongoDB-idiomatic data structures are used.

    Each iteration:
      1. find_one("sessions", {user_id: user_id}, sort by last_active_at DESC)
      2. The cart field is returned as a raw string — no deserialisation
         is performed (we measure retrieval cost, not parsing cost,
         consistent with the PostgreSQL baseline which returns JSONB as-is)

    Concurrency model:
      pymongo's MongoClient is thread-safe by design — a single client
      instance manages an internal connection pool and is shared safely
      across all threads. This differs from psycopg2 (which requires one
      connection per thread) and reflects the idiomatic pymongo pattern.
      The concurrency pressure on MongoDB is real: 50 threads issue
      concurrent find_one calls through the shared client.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive result.
    This file measures the naive (engine-only) side.

Usage:
    cd benchmarks/mongodb/naive
    python q3_session.py                   # 1000 iterations, 50 threads
    python q3_session.py --iterations 100  # quick smoke test
    python q3_session.py --concurrency 1   # single-threaded sanity check
    python q3_session.py --dry-run         # run once, print result sample
    python q3_session.py --pool-size 500   # use 500 user IDs (default: 1000)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_user_id_pool(db, pool_size: int) -> list[str]:
    """
    Pre-fetch a random sample of user IDs that have at least one session.
    We do not filter by expires_at — all sessions in the dataset are
    technically expired relative to today. The benchmark measures retrieval
    performance, not expiry logic, consistent with the PostgreSQL baseline.
    """
    pipeline = [
        {"$group": {"_id": "$user_id"}},
        {"$sample": {"size": pool_size}},
    ]
    ids = [doc["_id"] for doc in db["sessions"].aggregate(pipeline)]
    if not ids:
        raise RuntimeError(
            "No sessions found in MongoDB. "
            "Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} user IDs with sessions.")
    return ids

# ── core Q3 logic ─────────────────────────────────────────────────────────────

def run_q3(db, user_id: str) -> dict | None:
    """
    Fetch the most recently active session for a user.
    Cart is returned as a raw string — no deserialisation.
    Mirrors the PostgreSQL Q3 pattern:
        SELECT ... FROM sessions WHERE user_id = %s
        ORDER BY last_active_at DESC LIMIT 1
    """
    return db["sessions"].find_one(
        {"user_id": user_id},
        sort=[("last_active_at", -1)],
    )

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, user_ids: list[str]):
    """
    Return a zero-argument callable for the harness.
    pymongo MongoClient is thread-safe — the shared db object is used
    directly across all concurrent threads without any thread-local wrappers.
    """
    def _run():
        user_id = random.choice(user_ids)
        run_q3(db, user_id)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — MongoDB Naive Q3 result for user {user_id}:\n")
    row = run_q3(db, user_id)
    if not row:
        print("  ⚠  No session found for this user.")
        return
    print(f"  Session ID     : {row.get('_id')}")
    print(f"  User ID        : {row.get('user_id')}")
    print(f"  IP Address     : {row.get('ip_address')}")
    print(f"  Last active    : {row.get('last_active_at')}")
    print(f"  Expires at     : {row.get('expires_at')}")
    cart = row.get("cart", "[]")
    print(f"\n  Cart (raw string): {str(cart)[:120]}{'...' if len(str(cart)) > 120 else ''}")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q3 — session and cart benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
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
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch one session and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q3.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q3.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q3 Session & Cart Benchmark")
    print("=" * 55)

    db = get_db()
    user_ids = fetch_user_id_pool(db, args.pool_size)

    if args.dry_run:
        dry_run(db, user_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, user_ids),
        db="mongodb_naive",
        query_id="Q3",
        label=(
            f"Session + cart retrieval by user_id under {args.concurrency} "
            "concurrent threads. Naive: flat session document, cart stored "
            "as JSON string. Single find_one per iteration sorted by "
            "last_active_at DESC. pymongo thread-safe shared client. "
            f"Random user sampled from pool of {len(user_ids)} IDs."
        ),
        iterations=args.iterations,
        concurrency=args.concurrency,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()