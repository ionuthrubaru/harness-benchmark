"""
benchmarks/mongodb/optimised/q4_recommendations.py — MongoDB Optimised: Q4
============================================================================
Q4: Top 10 product recommendations based on co-purchase — given a seed
    product, find products most frequently bought in the same order.

Optimised schema changes vs naive:
  - orders embed an `items` array of subdocuments (order_items eliminated
    as a separate collection)
  - multikey index on `items.product_id` → MongoDB indexes every element
    of the embedded array, enabling efficient lookup of all orders
    containing a given product
  - order_items collection is ELIMINATED

Query structure vs naive:
  Naive:  2-pass — find order IDs for seed product (order_items),
          then find co-purchased products (second order_items query)
  Optimised: single $match → $unwind → $match → $group pipeline
          using the multikey index, all within the orders collection

Academic context:
  MongoDB cannot perform graph traversal natively (that is Neo4j's
  domain). Both naive and optimised implementations do co-purchase
  counting, not graph traversal. The schema effect here is:
    1. One collection instead of two (no lookup across collections)
    2. Multikey index eliminates full-collection scan on order_items

Usage:
    python q4_recommendations.py
    python q4_recommendations.py --iterations 100
    python q4_recommendations.py --dry-run
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

TOP_N = 10

# ── sample seed product IDs ───────────────────────────────────────────────────

def load_product_ids(db, sample_size: int = 200) -> list[str]:
    ids = [d["_id"] for d in db["products"].find({"is_active": True}, {"_id": 1}).limit(sample_size * 4)]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    return ids

# ── core Q4 logic (timed portion) ─────────────────────────────────────────────

def run_q4(db, product_ids: list[str]) -> list[dict]:
    """
    Single aggregation pipeline on orders collection.

    Uses multikey index on items.product_id for the first $match.

    Pipeline:
      1. $match orders containing the seed product (multikey index hit)
      2. $unwind items array
      3. $match: exclude the seed product itself
      4. $group: count co-occurrences per co-purchased product_id
      5. $sort: descending by count
      6. $limit: top N
    """
    seed = random.choice(product_ids)
    pipeline = [
        {"$match": {"items.product_id": seed}},
        {"$unwind": "$items"},
        {"$match": {"items.product_id": {"$ne": seed}}},
        {"$group": {
            "_id":            "$items.product_id",
            "co_buy_count":   {"$sum": 1},
            "product_name":   {"$first": "$items.product_name"},
            "product_type":   {"$first": "$items.product_type"},
        }},
        {"$sort":  {"co_buy_count": -1}},
        {"$limit": TOP_N},
    ]
    return list(db["orders"].aggregate(pipeline))

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, product_ids):
    def _run():
        run_q4(db, product_ids)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, product_ids):
    print("\n  DRY RUN — MongoDB Optimised Q4 result sample:\n")
    results = run_q4(db, product_ids)
    if not results:
        print("  ⚠  No recommendations returned — is the optimised DB populated?")
        return
    print(f"  {'Rank':<6} {'product_id':<38} {'product_name':<35} {'co_buy_count'}")
    print(f"  {'─'*6} {'─'*38} {'─'*35} {'─'*12}")
    for i, row in enumerate(results, 1):
        print(
            f"  {i:<6} {str(row.get('_id','')):<38} "
            f"{str(row.get('product_name','')):<35} {row.get('co_buy_count')}"
        )
    print(f"\n  Queries issued: 1 (aggregation pipeline on orders — multikey index)")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q4 — co-purchase recommendations benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q4.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q4 Recommendations Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    print("  Sampling product IDs...")
    product_ids = load_product_ids(db)
    print(f"  Loaded {len(product_ids):,} product IDs for random sampling.\n")

    if args.dry_run:
        dry_run(db, product_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, product_ids),
        db="mongodb_optimised",
        query_id="Q4",
        label=(
            f"Top {TOP_N} co-purchase recommendations for a seed product. "
            "Optimised: order_items collection eliminated — orders embed an "
            "items array with multikey index on items.product_id. Single "
            "$match→$unwind→$group pipeline; no cross-collection lookup."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()