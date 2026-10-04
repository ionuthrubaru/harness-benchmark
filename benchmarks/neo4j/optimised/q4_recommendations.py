"""
benchmarks/neo4j/optimised/q4_recommendations.py — Neo4j Optimised: Q4
=======================================================================
Q4: Top 10 co-purchase recommendations for a given product.

Optimised schema: ALSO_BOUGHT relationships precomputed at load time.
Each (Product)-[:ALSO_BOUGHT {count, confidence}]->(Product) edge
stores the co-purchase frequency and confidence score as properties.

The query is a single-hop neighbour lookup — no aggregation, no
multi-step traversal, no JOIN through order_items at query time:

    MATCH (target:Product {id: $product_id})-[r:ALSO_BOUGHT]->(rec:Product)
    RETURN rec, r.count, r.confidence
    ORDER BY r.count DESC LIMIT 10

This is the structural argument for Neo4j in the dissertation:
the co-purchase graph is a first-class data structure, not a
query-time computation. Traversal cost is O(degree of node),
not O(order_items table size).

Schema effect vs naive: the entire 2-hop traversal + aggregation
is eliminated. Any latency is now dominated by I/O to fetch
the top 10 ALSO_BOUGHT neighbours, not computation.

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
# Single-hop traversal on precomputed ALSO_BOUGHT edges.
# count and confidence are stored as relationship properties —
# no aggregation happens at query time.

Q4_CYPHER = """
MATCH (target:Product {id: $product_id})-[r:ALSO_BOUGHT]->(rec:Product)
RETURN
    rec.id              AS product_id,
    rec.name            AS product_name,
    rec.product_type    AS product_type,
    rec.price_usd       AS price_usd,
    r.count             AS co_purchase_count,
    r.confidence        AS confidence
ORDER BY r.count DESC, rec.name
LIMIT 10
"""

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_product_id_pool(driver, pool_size: int) -> list[str]:
    """
    Pre-fetch IDs of products that have at least one ALSO_BOUGHT edge.
    Products with no edges would return zero results — not the hot path.
    """
    cypher = """
    MATCH (p:Product)-[:ALSO_BOUGHT]->(:Product)
    WITH p, rand() AS r
    ORDER BY r
    LIMIT $pool_size
    RETURN p.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, pool_size=pool_size)
        ids = [row["id"] for row in result]

    if not ids:
        raise RuntimeError(
            "No ALSO_BOUGHT edges found. "
            "Run the optimised loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} product IDs with ALSO_BOUGHT edges.")
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
    # pick a product that actually has ALSO_BOUGHT edges
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (p:Product)-[r:ALSO_BOUGHT]->(rec:Product)
            RETURN p.id AS id
            LIMIT 1
            """
        ).single()
        if not rec:
            print("⚠ No products with active ALSO_BOUGHT edges found in DB.")
            return
        product_id = rec["id"]

    print(f"\n  DRY RUN — Neo4j Optimised Q4 recommendations for product {product_id}:\n")

    # Show source product name
    with driver.session() as session:
        src = session.run(
            "MATCH (p:Product {id: $id}) RETURN p.name AS name, p.product_type AS type",
            id=product_id,
        ).single()
    if src:
        print(f"  Source product : {src['name']} ({src['type']})\n")

    # Fetch ALSO_BOUGHT recommendations
    with driver.session() as session:
        result = session.run(Q4_CYPHER, product_id=product_id)
        rows = result.data()

    if not rows:
        print("  ⚠  No ALSO_BOUGHT edges found for this product.")
        return

    print(f"  Top {len(rows)} recommendations:  [single-hop ALSO_BOUGHT — optimised]\n")
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
    parser = argparse.ArgumentParser(description="Neo4j Optimised Q4 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size",  type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_optimised_Q4.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Optimised — Q4 Co-Purchase Recommendations")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))

    try:
        product_ids = fetch_product_id_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, product_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, product_ids),
            db="neo4j_optimised",
            query_id="Q4",
            label=(
                "Top 10 co-purchase recommendations via single-hop ALSO_BOUGHT "
                "traversal. Edges precomputed at load time with count and "
                "confidence as relationship properties — zero aggregation at "
                "query time. Schema effect vs naive: eliminates full 2-hop "
                f"traversal. Random product from pool of {len(product_ids)} IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()