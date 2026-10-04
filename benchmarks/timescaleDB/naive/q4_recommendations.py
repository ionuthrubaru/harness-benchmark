"""
benchmarks/timescaledb/naive/q4_recommendations.py — TimescaleDB Naive: Q4
===========================================================================
Q4: Given a product, find the top 10 recommendations based on
    co-purchase patterns — "customers who bought this also bought..."

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
TimescaleDB adds nothing for Q4 — order_items and orders are plain
PostgreSQL tables (not hypertables). The 2-hop self-JOIN on order_items
runs identically to the PostgreSQL baseline. Latency should be comparable,
confirming TimescaleDB does not penalise relational graph-style queries.

Engine effect for Q4 (naive)
─────────────────────────────
None expected. Neither order_items nor orders is a hypertable, so no
chunk pruning applies. The query plan will be identical to PostgreSQL.
This is the correct baseline behaviour: TimescaleDB is PostgreSQL with
time-series extensions — non-hypertable tables behave exactly as in
plain PostgreSQL.

Usage:
    cd benchmarks/timescaledb/naive
    python q4_recommendations.py                   # 1000 iterations
    python q4_recommendations.py --iterations 100  # quick smoke test
    python q4_recommendations.py --explain         # EXPLAIN ANALYZE
    python q4_recommendations.py --dry-run         # run once, print results
    python q4_recommendations.py --pool-size 500   # smaller pool
"""

import argparse
import os
import random
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

# ── query — identical to PostgreSQL Q4 ────────────────────────────────────────

Q4_SQL = """
WITH orders_with_product AS (

    SELECT DISTINCT oi.order_id
    FROM   order_items oi
    JOIN   orders o ON o.id = oi.order_id
    WHERE  oi.product_id = %s
      AND  o.status IN ('confirmed', 'shipped', 'delivered')

),

co_purchased AS (

    SELECT
        oi.product_id,
        COUNT(DISTINCT oi.order_id) AS co_purchase_count
    FROM   order_items oi
    WHERE  oi.order_id   IN (SELECT order_id FROM orders_with_product)
      AND  oi.product_id != %s
    GROUP BY oi.product_id

)

SELECT
    p.id                            AS product_id,
    p.name                          AS product_name,
    p.product_type,
    p.price_usd,
    cp.co_purchase_count,
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

# ── ID pool ────────────────────────────────────────────────────────────────────

def fetch_product_id_pool(conn, pool_size: int) -> list[str]:
    """Products that appear in at least one confirmed order — same as PostgreSQL Q4."""
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
        raise RuntimeError("No products in confirmed orders — run timescaledb_naive_loader.py first.")
    ids = [str(row[0]) for row in rows]
    print(f"  Product ID pool: {len(ids):,} entries loaded.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, product_ids: list[str]):
    def _run():
        product_id = random.choice(product_ids)
        with conn.cursor() as cur:
            cur.execute(Q4_SQL, (product_id, product_id))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  DRY RUN — Q4 naive recommendations for product {product_id}:\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT name, product_type FROM products WHERE id = %s", (product_id,))
        src = cur.fetchone()
    if src:
        print(f"  Source product : {src['name']} ({src['product_type']})\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q4_SQL, (product_id, product_id))
        rows = cur.fetchall()

    if not rows:
        print("  ⚠  No co-purchase recommendations found for this product.")
        return
    print(f"  {'Product name':<40} {'Type':<15} {'Price':>8} {'Count':>8} {'Conf':>8}")
    print(f"  {'─'*40} {'─'*15} {'─'*8} {'─'*8} {'─'*8}")
    for row in rows:
        print(
            f"  {str(row['product_name'])[:40]:<40} "
            f"{str(row['product_type']):<15} "
            f"{str(row['price_usd']):>8} "
            f"{row['co_purchase_count']:>8} "
            f"{str(row['confidence']):>8}"
        )


def explain(conn, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q4 naive (product {product_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q4_EXPLAIN_SQL, (product_id, product_id))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q4 recommendations benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=500, dest="pool_size")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q4.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q4 Recommendations Benchmark")
    print("=" * 60)
    print("  Schema : naive (order_items and orders are plain tables)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : no chunk pruning — non-hypertable tables")

    conn = get_connection()
    try:
        pool = fetch_product_id_pool(conn, args.pool_size)
        if args.explain:
            explain(conn, pool)
            return
        if args.dry_run:
            dry_run(conn, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(conn, pool),
            db="timescaledb_naive",
            query_id="Q4",
            label=(
                "Top-10 co-purchase recommendations via 2-hop self-JOIN on order_items. "
                "SQL identical to PostgreSQL baseline. "
                "order_items and orders are plain PostgreSQL tables (not hypertables). "
                "No chunk pruning — TimescaleDB engine effect expected to be zero. "
                f"Pool of {len(pool):,} product IDs with co-purchase history."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()