"""
benchmarks/mongodb/optimised/q3_session.py — MongoDB Optimised: Q3
====================================================================
Q3: Retrieve a user's active session and full cart contents under
    concurrent load.

Optimised schema changes vs naive:
  - cart is stored as a native BSON array (list of subdocuments)
    instead of a JSON string → no json.loads() call on the Python side
  - metadata on related events is stored as a native BSON subdocument
  - is_active stored as bool, not string "True"/"False"

Revision 2026-11: the query is now "most recent session of a user"
(find_one by user_id sorted by last_active_at DESC) at 50 threads, the
same access path and concurrency as the naive driver and every other
engine. The earlier version looked the session up by its ID on a single
thread, which is not equivalent; it remains available as --by-session-id.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive, on the same access path.

Usage:
    python q3_session.py
    python q3_session.py --iterations 100
    python q3_session.py --dry-run
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── ID pools ──────────────────────────────────────────────────────────────────

def fetch_user_id_pool(db, pool_size: int) -> list[str]:
    """
    Equivalent access path (revision, 2026-11): a random sample of user IDs that
    have at least one session, exactly as in the naive driver and the PostgreSQL
    baseline. The query is "most recent session of this user".
    """
    pipeline = [{"$group": {"_id": "$user_id"}}, {"$sample": {"size": pool_size}}]
    ids = [doc["_id"] for doc in db["sessions"].aggregate(pipeline)]
    if not ids:
        raise RuntimeError("No sessions found in the optimised MongoDB database.")
    print(f"  Loaded pool of {len(ids)} user IDs with sessions.")
    return ids


def load_session_ids(db, sample_size: int = 500) -> list[str]:
    """Legacy pool (lookup by session ID), kept only for --by-session-id."""
    ids = [d["_id"] for d in db["sessions"].find({"is_active": True}, {"_id": 1}).limit(sample_size * 4)]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    if not ids:
        ids = [d["_id"] for d in db["sessions"].find({}, {"_id": 1}).limit(sample_size)]
    return ids

# ── core Q3 logic (timed portion) ─────────────────────────────────────────────

def run_q3(db, user_id: str) -> dict | None:
    """
    Most recent session of a user, served by the compound index
    (user_id ASC, last_active_at DESC) that the optimised loader creates.
    Optimised: doc["cart"] is already a Python list, no json.loads().
    """
    doc = db["sessions"].find_one({"user_id": user_id}, sort=[("last_active_at", -1)])
    if doc:
        _ = doc.get("cart", [])   # already a list; no deserialisation needed
    return doc


def run_q3_by_session_id(db, session_id: str) -> dict | None:
    """Legacy driver (lookup by session ID); not equivalent to the other engines."""
    doc = db["sessions"].find_one({"_id": session_id})
    if doc:
        _ = doc.get("cart", [])
    return doc

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, ids, by_session_id: bool = False):
    if by_session_id:
        def _run():
            run_q3_by_session_id(db, random.choice(ids))
    else:
        def _run():
            run_q3(db, random.choice(ids))
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, user_ids):
    print("\n  DRY RUN — MongoDB Optimised Q3 result sample:\n")
    doc = run_q3(db, random.choice(user_ids))
    if not doc:
        print("  ⚠  No session returned — is the optimised DB populated?")
        return

    cart = doc.get("cart", [])

    print(f"  Session ID      : {doc.get('_id')}")
    print(f"  User ID         : {doc.get('user_id')}")
    print(f"  Is active       : {doc.get('is_active')} (type: {type(doc.get('is_active')).__name__})")
    print(f"  Cart type       : {type(cart).__name__}  ← should be 'list', not 'str'")
    print(f"  Cart items      : {len(cart)}")
    if cart:
        item = cart[0]
        print(f"\n  First cart item :")
        print(f"    product_id   : {item.get('product_id')}")
        print(f"    product_name : {item.get('product_name')}")
        print(f"    quantity     : {item.get('quantity')}")
        print(f"    price_usd    : {item.get('price_usd')}")
    print(f"\n  Queries issued: 1 (find_one on sessions — cart is native BSON array)")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q3 — session + cart benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Number of concurrent threads (default: 50, as for every other engine)")
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--by-session-id", action="store_true", dest="by_session_id",
                        help="Legacy, non-equivalent driver: lookup by session ID")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q3.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q3 Session + Cart Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    if args.by_session_id:
        ids = load_session_ids(db)
        label = ("LEGACY, not equivalent: single find_one by session ID. Optimised: cart stored "
                 f"as native BSON array. Concurrency={args.concurrency}.")
    else:
        ids = fetch_user_id_pool(db, args.pool_size)
        label = (f"Session + cart retrieval by user_id under {args.concurrency} concurrent threads "
                 "(equivalent driver, revision 2026-11). find_one sorted by last_active_at DESC on "
                 "the compound index (user_id, last_active_at DESC). Optimised: cart stored as native "
                 f"BSON array, no json.loads. Random user sampled from pool of {len(ids)} IDs.")

    if args.dry_run:
        if args.by_session_id:
            print(run_q3_by_session_id(db, random.choice(ids)))
        else:
            dry_run(db, ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, ids, args.by_session_id),
        db="mongodb_optimised",
        query_id="Q3",
        label=label,
        iterations=args.iterations,
        concurrency=args.concurrency,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()