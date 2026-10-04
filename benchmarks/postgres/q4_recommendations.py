"""
benchmarks/postgres/q4_recommendations.py — PostgreSQL Baseline: Q4
====================================================================
Q4: Given a product, find the top 10 recommendations based on
    co-purchase patterns — "customers who bought this also bought..."

    This is the PostgreSQL baseline for the Neo4j comparison (Q4).
    Neo4j will serve the same result via a single-hop traversal on
    pre-computed ALSO_BOUGHT relationships with no JOIN overhead.

Killer feature demonstrated (Neo4j side):
    Graph traversal — relationships are first-class citizens stored
    as direct pointers. Neo4j's index-free adjacency means traversal
    cost is proportional to the neighbourhood size, not the total
    dataset size. PostgreSQL must JOIN through the full order_items
    table on every query regardless of how many co-purchases exist.

PostgreSQL baseline design notes
─────────────────────────────────
The co-purchase query is a 2-hop JOIN:

    product A
        → order_items (find all orders containing product A)
            → order_items (find all other products in those orders)
                → products (resolve product details)

This is NOT implemented as a recursive CTE deliberately. A recursive
CTE is for variable-depth traversal (e.g. "find all products reachable
within N hops"). Q4 is a fixed 2-hop pattern, which a multi-hop JOIN
expresses more naturally — and gives PostgreSQL the best chance since
the planner can optimise a known-depth JOIN better than a recursive CTE.

The academic point: even with this optimal SQL formulation, PostgreSQL
must still scan/join through order_items twice and aggregate. Neo4j
pre-computes the ALSO_BOUGHT edges at load time, reducing the query to
a single-hop neighbour lookup. This structural difference is the
dissertation's argument.

Only confirmed (status = 'confirmed', 'shipped', or 'delivered') orders
are used — cancelled and refunded orders should not influence
recommendations.

Usage:
    python q4_recommendations.py                   # 1000 iterations
    python q4_recommendations.py --iterations 100  # quick smoke test
    python q4_recommendations.py --explain         # EXPLAIN ANALYZE
    python q4_recommendations.py --dry-run         # run once, print results
    python q4_recommendations.py --pool-size 500   # pre-fetch 500 product IDs
"""

import argparse
import os
import random
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

# ── connection ────────────────────────────────────────────────────────────────

from pg_conn import get_connection

# ── query ─────────────────────────────────────────────────────────────────────
#
# CTE breakdown
# ─────────────
#
#   1. orders_with_product
#      Find every confirmed order that contains the target product.
#      Filters to meaningful orders only (excludes cancelled/refunded).
#
#   2. co_purchased
#      For each of those orders, collect every OTHER product that
#      appeared in the same order. Self-joins on order_id with a
#      != predicate excludes the target product from its own results.
#
#   3. ranked
#      Count how many distinct orders each co-purchased product
#      appeared in alongside the target product. This is the
#      co-purchase frequency — the edge weight Neo4j will store
#      as a relationship property on ALSO_BOUGHT edges.
#
# The final SELECT joins onto products to resolve names and prices,
# returning the top 10 by co-purchase frequency.

Q4_SQL = """
WITH orders_with_product AS (

    -- Step 1: find all confirmed orders containing the target product
    SELECT DISTINCT oi.order_id
    FROM   order_items oi
    JOIN   orders o ON o.id = oi.order_id
    WHERE  oi.product_id = %s
      AND  o.status IN ('confirmed', 'shipped', 'delivered')

),

co_purchased AS (

    -- Step 2: find every other product bought in those same orders
    SELECT
        oi.product_id,
        COUNT(DISTINCT oi.order_id) AS co_purchase_count
    FROM   order_items oi
    WHERE  oi.order_id   IN (SELECT order_id FROM orders_with_product)
      AND  oi.product_id != %s          -- exclude the target product itself
    GROUP BY oi.product_id

)

-- Step 3: resolve product details and return top 10
SELECT
    p.id                            AS product_id,
    p.name                          AS product_name,
    p.product_type,
    p.price_usd,
    cp.co_purchase_count,
    -- Confidence: proportion of target-product orders that also include
    -- this product. Stored as a relationship property in Neo4j (ALSO_BOUGHT).
    ROUND(
        cp.co_purchase_count::NUMERIC /
        NULLIF((SELECT COUNT(*) FROM orders_with_product), 0),
        4
    )                               AS confidence
FROM   co_purchased cp
JOIN   products p ON p.id = cp.product_id
WHERE  p.is_active = TRUE
ORDER BY cp.co_purchase_count DESC, p.name
LIMIT 10;
"""

Q4_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q4_SQL

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_product_id_pool(conn, pool_size: int) -> list[str]:
    """
    Pre-fetch IDs of products that appear in at least one confirmed order.
    Products with no order history would return zero recommendations —
    timing empty results would not represent the realistic hot path.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT product_id FROM (
                SELECT DISTINCT oi.product_id
                FROM   order_items oi
                JOIN   orders o ON o.id = oi.order_id
                WHERE  o.status IN ('confirmed', 'shipped', 'delivered')
            ) AS purchased_products
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            (pool_size,),
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            "No products found in confirmed orders. "
            "Run the data loader before benchmarking."
        )

    ids = [str(row[0]) for row in rows]
    print(f"  Loaded pool of {len(ids)} product IDs with co-purchase history.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, product_ids: list[str]):
    """
    Return a zero-argument callable that fetches recommendations for a
    randomly selected product on every call.
    Q4 passes the product_id twice — once for orders_with_product CTE
    and once for the != exclusion in co_purchased CTE.
    """
    def _run():
        product_id = random.choice(product_ids)
        with conn.cursor() as cur:
            cur.execute(Q4_SQL, (product_id, product_id))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, product_ids: list[str]):
    """Fetch recommendations for one product and print the result."""
    product_id = product_ids[0]
    print(f"\n  DRY RUN — Q4 recommendations for product {product_id}:\n")

    # Print the source product name first for context
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT name, product_type FROM products WHERE id = %s",
            (product_id,),
        )
        source = cur.fetchone()

    if source:
        print(f"  Source product : {source['name']} ({source['product_type']})\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q4_SQL, (product_id, product_id))
        rows = cur.fetchall()

    if not rows:
        print("  ⚠  No recommendations found — product may have no co-purchase history.")
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


def explain(conn, product_ids: list[str]):
    """Print EXPLAIN ANALYZE for one recommendation query."""
    product_id = product_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q4 (product {product_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q4_EXPLAIN_SQL, (product_id, product_id))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Q4 co-purchase recommendations benchmark"
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
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch recommendations for one product and print results then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q4_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q4_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q4 Co-Purchase Recommendations")
    print("=" * 50)

    conn = get_connection()

    try:
        product_ids = fetch_product_id_pool(conn, args.pool_size)

        if args.explain:
            explain(conn, product_ids)
            return

        if args.dry_run:
            dry_run(conn, product_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, product_ids),
            db="postgres",
            query_id="Q4",
            label=(
                "Top 10 co-purchase recommendations via 2-hop JOIN through "
                "order_items. Co-purchase count and confidence score computed "
                "per query — Neo4j will pre-compute these as ALSO_BOUGHT "
                f"relationship properties. Random product sampled from pool "
                f"of {len(product_ids)} IDs per iteration."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()