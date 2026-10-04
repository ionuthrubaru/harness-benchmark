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

Schema effect for Q3 is modest: the query is still a single find_one
by session ID (same as naive). The gain comes from eliminating the
json.loads() deserialisation step, which matters more under high
concurrency (Q3 is the concurrency benchmark). The real concurrency
story is Q3 run at 1/10/50 threads — see throughput chart.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive — eliminating json.loads() is
  the primary schema gain here; the single-key lookup path is identical.

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

# ── sample session IDs ────────────────────────────────────────────────────────

def load_session_ids(db, sample_size: int = 500) -> list[str]:
    ids = [
        d["_id"]
        for d in db["sessions"].find(
            {"is_active": True}, {"_id": 1}
        ).limit(sample_size * 4)
    ]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    if not ids:
        # fallback: any session, active flag may differ in naive vs optimised
        ids = [d["_id"] for d in db["sessions"].find({}, {"_id": 1}).limit(sample_size)]
    return ids

# ── core Q3 logic (timed portion) ─────────────────────────────────────────────

def run_q3(db, session_ids: list[str]) -> dict | None:
    """
    Single find_one by session ID.
    Optimised: doc["cart"] is already a Python list — no json.loads().
    Returns the full session document including the native cart list.
    """
    session_id = random.choice(session_ids)
    doc = db["sessions"].find_one({"_id": session_id})
    if doc:
        # In the optimised schema, cart is a BSON array — direct access,
        # no deserialisation step. This line is intentionally explicit for
        # documentation purposes.
        _ = doc.get("cart", [])   # already a list; no json.loads needed
    return doc

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, session_ids):
    def _run():
        run_q3(db, session_ids)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, session_ids):
    print("\n  DRY RUN — MongoDB Optimised Q3 result sample:\n")
    doc = run_q3(db, session_ids)
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

    print("  Sampling active session IDs...")
    session_ids = load_session_ids(db)
    print(f"  Loaded {len(session_ids):,} session IDs for random sampling.\n")

    if args.dry_run:
        dry_run(db, session_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, session_ids),
        db="mongodb_optimised",
        query_id="Q3",
        label=(
            "Active session + cart retrieval. Optimised: cart stored as native "
            "BSON array — no json.loads() deserialisation. Single find_one by "
            "session ID. Schema effect is elimination of string deserialisation; "
            "most significant under high concurrency."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()