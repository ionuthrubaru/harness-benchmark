"""
benchmarks/mongodb/naive/q4_recommendations.py — MongoDB Naive: Q4
===================================================================
Q4: Given a product, find the top 10 recommendations based on
    co-purchase patterns — "customers who bought this also bought..."

Naive implementation notes:
    Collections are flat mirrors of the PostgreSQL schema — no graph,
    no precomputed ALSO_BOUGHT edges, no relationship weights.

    Replicating the PostgreSQL 2-hop JOIN requires a multi-stage
    aggregation pipeline across order_items and orders:

      Stage 1 — filter order_items to confirmed orders containing
                the target product → collect matching order_ids
      Stage 2 — find all other products in those same orders
                (second pass over order_items)
      Stage 3 — group by product_id, count co-purchase frequency
      Stage 4 — sort descending, limit to top 10
      Stage 5 — lookup product details from products collection

    Because MongoDB has no LATERAL JOIN or cross-collection JOIN in a
    single pipeline pass, step 1 is executed as a separate query to
    collect the matching order_ids, then step 2 uses a $in filter.
    This two-query approach mirrors the PostgreSQL CTE structure.

    Only confirmed orders (status: confirmed, shipped, delivered)
    are used — consistent with the PostgreSQL baseline.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive result.
    This file measures the naive (engine-only) side.

Usage:
    cd benchmarks/mongodb/naive
    python q4_recommendations.py                   # 1000 iterations
    python q4_recommendations.py --iterations 100  # quick smoke test
    python q4_recommendations.py --dry-run         # run once, print results
    python q4_recommendations.py --pool-size 500   # use 500 product IDs
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── confirmed order statuses (mirrors PostgreSQL baseline) ────────────────────

CONFIRMED_STATUSES = ["confirmed", "shipped", "delivered"]

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_product_id_pool(db, pool_size: int) -> list[str]:
    """
    Pre-fetch IDs of products that appear in at least one confirmed order.
    Products with no order history return zero recommendations — timing
    empty results would not represent the realistic hot path.
    """
    pipeline = [
        {"$lookup": {
            "from":         "orders",
            "localField":   "order_id",
            "foreignField": "_id",
            "as":           "order",
        }},
        {"$unwind": "$order"},
        {"$match": {"order.status": {"$in": CONFIRMED_STATUSES}}},
        {"$group": {"_id": "$product_id"}},
        {"$sample": {"size": pool_size}},
    ]
    ids = [doc["_id"] for doc in db["order_items"].aggregate(pipeline)]
    if not ids:
        raise RuntimeError(
            "No products found in confirmed orders. "
            "Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} product IDs with co-purchase history.")
    return ids

# ── core Q4 logic ─────────────────────────────────────────────────────────────

def run_q4(db, product_id: str) -> list[dict]:
    """
    Find top 10 co-purchased products for a given product.

    Step 1: find all confirmed order_ids containing this product
    Step 2: find all other products in those orders, count frequency
    Step 3: lookup product details for the top 10
    """
    # ── step 1: collect confirmed order_ids containing the target product ──
    # We need to join order_items → orders to filter by order status.
    # Done as a separate aggregation to keep the logic readable and to
    # mirror the PostgreSQL CTE structure (orders_with_product).
    order_id_cursor = db["order_items"].aggregate([
        {"$match": {"product_id": product_id}},
        {"$lookup": {
            "from":         "orders",
            "localField":   "order_id",
            "foreignField": "_id",
            "as":           "order",
        }},
        {"$unwind": "$order"},
        {"$match": {"order.status": {"$in": CONFIRMED_STATUSES}}},
        {"$group": {"_id": "$order_id"}},
    ])
    order_ids = [doc["_id"] for doc in order_id_cursor]

    if not order_ids:
        return []

    total_orders = len(order_ids)

    # ── step 2: find co-purchased products in those orders ─────────────────
    co_purchase_cursor = db["order_items"].aggregate([
        {"$match": {
            "order_id":   {"$in": order_ids},
            "product_id": {"$ne": product_id},   # exclude the target itself
        }},
        {"$group": {
            "_id":              "$product_id",
            "co_purchase_count": {"$sum": 1},
        }},
        {"$sort":  {"co_purchase_count": -1}},
        {"$limit": 10},
    ])
    co_purchases = list(co_purchase_cursor)

    if not co_purchases:
        return []

    # ── step 3: resolve product details ───────────────────────────────────
    top_product_ids = [cp["_id"] for cp in co_purchases]
    products_by_id  = {
        p["_id"]: p
        for p in db["products"].find(
            {"_id": {"$in": top_product_ids}, "is_active": "True"},
            {"_id": 1, "name": 1, "product_type": 1, "price_usd": 1},
        )
    }

    results = []
    for cp in co_purchases:
        prod = products_by_id.get(cp["_id"])
        if not prod:
            continue
        count      = cp["co_purchase_count"]
        confidence = round(count / total_orders, 4) if total_orders else 0.0
        results.append({
            "product_id":        prod["_id"],
            "product_name":      prod.get("name"),
            "product_type":      prod.get("product_type"),
            "price_usd":         prod.get("price_usd"),
            "co_purchase_count": count,
            "confidence":        confidence,
        })

    return results

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, product_ids: list[str]):
    def _run():
        product_id = random.choice(product_ids)
        run_q4(db, product_id)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  DRY RUN — MongoDB Naive Q4 recommendations for product {product_id}:\n")

    source = db["products"].find_one(
        {"_id": product_id}, {"name": 1, "product_type": 1}
    )
    if source:
        print(f"  Source product : {source.get('name')} ({source.get('product_type')})\n")

    rows = run_q4(db, product_id)
    if not rows:
        print("  ⚠  No recommendations — product may have no co-purchase history.")
        return

    print(f"  Top {len(rows)} recommendations:\n")
    print(
        f"  {'#':<3} {'Product name':<35} {'Type':<16} "
        f"{'Price':>8} {'Co-purchases':>13} {'Confidence':>11}"
    )
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*13} {'─'*11}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row['product_name']):<35} "
            f"{str(row['product_type']):<16} "
            f"{str(row['price_usd']):>8} "
            f"{str(row['co_purchase_count']):>13} "
            f"{str(row['confidence']):>11}"
        )

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q4 — co-purchase recommendations benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of product IDs to pre-fetch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch recommendations for one product and print results then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q4.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q4.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q4 Co-Purchase Recommendations")
    print("=" * 55)

    db = get_db()

    print("  Pre-fetching product ID pool (requires order_items → orders join)...")
    product_ids = fetch_product_id_pool(db, args.pool_size)

    if args.dry_run:
        dry_run(db, product_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, product_ids),
        db="mongodb_naive",
        query_id="Q4",
        label=(
            "Top 10 co-purchase recommendations via two aggregation pipeline "
            "passes over order_items (mirroring PostgreSQL 2-hop JOIN). "
            "No precomputed graph edges — co-purchase counts computed per query. "
            f"Random product sampled from pool of {len(product_ids)} IDs."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()