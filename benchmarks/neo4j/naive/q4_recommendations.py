"""
benchmarks/neo4j/naive/q4_recommendations.py — Neo4j Naive: Q4
===============================================================
Q4: Top 10 co-purchase recommendations for a given product.

Naive schema: no precomputed ALSO_BOUGHT edges. Co-purchase counts
are computed at query time via a 2-hop Cypher traversal:

    (target:Product)<-[:FOR_PRODUCT]-(oi1:OrderItem)
    -[:IN_ORDER]->(:Order {status: confirmed/shipped/delivered})
    <-[:IN_ORDER]-(oi2:OrderItem)-[:FOR_PRODUCT]->(rec:Product)

This mirrors the PostgreSQL 2-hop JOIN through order_items exactly:
  orders_with_product CTE → co_purchased CTE → products JOIN

Both engines must scan and aggregate the full co-purchase history
on every query invocation. The academic point: Neo4j stores
relationships as direct pointers (index-free adjacency), so
traversal cost scales with neighbourhood size rather than total
dataset size — even naive Neo4j may outperform PostgreSQL here.
The optimised version eliminates the traversal entirely.

Usage:
    python q4_recommendations.py
    python q4_recommendations.py --iterations 100
    python q4_recommendations.py --dry-run
"""

import argparse
import os
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

# ── Cypher ────────────────────────────────────────────────────────────────────
#
# Step 1: find all confirmed orders containing the target product
# Step 2: from those orders, find all other products (co-purchased)
# Step 3: count, compute confidence, sort, return top 10
#
# Mirrors the PostgreSQL CTE structure exactly.
# ALSO_BOUGHT relationships do NOT exist in the naive schema.

Q4_CYPHER = """
MATCH (target:Product {id: $product_id})<-[:FOR_PRODUCT]-(oi1:OrderItem)
      <-[:CONTAINS]-(o:Order)
WHERE o.status IN ['confirmed', 'shipped', 'delivered']

MATCH (o)-[:CONTAINS]->(oi2:OrderItem)-[:FOR_PRODUCT]->(rec:Product)
WHERE rec.id <> target.id

WITH rec, count(DISTINCT o) AS co_purchase_count, count(DISTINCT oi1) AS total_orders
ORDER BY co_purchase_count DESC, rec.name
LIMIT 10

RETURN
    rec.id AS product_id,
    rec.name AS product_name,
    rec.product_type AS product_type,
    rec.price_usd AS price_usd,
    co_purchase_count,
    ROUND(toFloat(co_purchase_count)/total_orders,4) AS confidence
"""

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_product_id_pool(driver, pool_size: int) -> list[str]:
    """
    Pre-fetch IDs of products that appear in at least one confirmed order.
    Products with no OrderItem relationships would return zero recommendations.
    """
    cypher = """
    MATCH (o:Order)-[:CONTAINS]->(oi1:OrderItem)-[:FOR_PRODUCT]->(p:Product)
    WHERE o.status IN ['confirmed','shipped','delivered']
    
    MATCH (o)-[:CONTAINS]->(oi2:OrderItem)-[:FOR_PRODUCT]->(other:Product)
    WHERE other.id <> p.id
    
    WITH DISTINCT p, rand() AS r
    ORDER BY r
    LIMIT $pool_size
    RETURN p.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, pool_size=pool_size)
        ids = [row["id"] for row in result]

    if not ids:
        raise RuntimeError(
            "No products found in confirmed orders. "
            "Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} product IDs with co-purchase history.")
    return ids


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, product_ids: list[str]):
    def _run():
        product_id = random.choice(product_ids)
        with driver.session() as session:
            result = session.run(Q4_CYPHER, product_id=product_id)
            result.data()
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  DRY RUN — Neo4j Naive Q4 recommendations for product {product_id}:\n")

    # Show source product name
    with driver.session() as session:
        src = session.run(
            "MATCH (p:Product {id: $id}) RETURN p.name AS name, p.product_type AS type",
            id=product_id,
        ).single()
    if src:
        print(f"  Source product : {src['name']} ({src['type']})\n")

    with driver.session() as session:
        result = session.run(Q4_CYPHER, product_id=product_id)
        rows = result.data()

    if not rows:
        print("  ⚠  No recommendations — product may have no co-purchase history.")
        return

    print(f"  Top {len(rows)} recommendations:  [2-hop traversal — naive]\n")
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
    parser = argparse.ArgumentParser(description="Neo4j Naive Q4 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size",  type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_naive_Q4.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Naive — Q4 Co-Purchase Recommendations")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))

    try:
        product_ids = fetch_product_id_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, product_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, product_ids),
            db="neo4j_naive",
            query_id="Q4",
            label=(
                "Top 10 co-purchase recommendations via 2-hop Cypher traversal. "
                "No precomputed ALSO_BOUGHT edges — co-purchase counts aggregated "
                "at query time. Mirrors PostgreSQL 2-hop JOIN structure. "
                f"Random product from pool of {len(product_ids)} IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()