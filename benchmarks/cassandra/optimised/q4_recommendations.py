"""
benchmarks/cassandra/optimised/q4_recommendations.py — Cassandra Optimised: Q4
===============================================================================
Q4: Top 10 product recommendations via co-purchase patterns.

Optimised schema — single partition read, pre-sorted
─────────────────────────────────────────────────────
Table: also_bought
PK:    ((product_id), co_purchase_count DESC, co_product_id ASC)

Co-purchase counts were pre-aggregated at load time from order_items.
Product details (name, type, price, is_active) are embedded in the row.
Clustering order is co_purchase_count DESC so the highest-count
recommendations appear first. LIMIT 10 returns the top 10 without any
Python-side sort or aggregation.

Schema effect vs naive:
  Naive  : ALLOW FILTERING scan (order_items by product_id)
           + N PK lookups (orders, status filter)
           + full order_items scan (Python-side join)
           + Python co-purchase aggregation + sort
           → multiple full table scans per iteration
  Optimised : single partition read LIMIT 10 → 1 round trip, pre-sorted

Usage:
    python q4_recommendations.py                   # 1000 iterations
    python q4_recommendations.py --iterations 100
    python q4_recommendations.py --dry-run
    python q4_recommendations.py --pool-size 200
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
TOP_N    = 10

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_product_id_pool(session, pool_size: int) -> list:
    """
    Fetch product_ids that have at least one also_bought entry.
    SELECT product_id FROM also_bought LIMIT n — no ALLOW FILTERING needed.
    """
    rows = list(session.execute(
        f"SELECT product_id FROM also_bought LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No rows in also_bought — run cassandra_optimised_loader.py first.")
    ids = list({r.product_id for r in rows})
    random.shuffle(ids)
    print(f"  Product ID pool: {len(ids):,} products with co-purchases loaded.")
    return ids

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, product_ids: list):
    """
    Single partition read with LIMIT 10.
    Clustering is co_purchase_count DESC so top recommendations come first.
    No Python sorting, no aggregation, no secondary lookups.
    """
    def _run():
        product_id = random.choice(product_ids)
        rows = list(session.execute(
            "SELECT co_product_id, co_product_name, co_product_type, "
            "co_product_price_usd, co_product_is_active, "
            "co_purchase_count, confidence "
            "FROM also_bought WHERE product_id = %s LIMIT %s",
            (product_id, TOP_N),
        ))
        return rows
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, product_ids: list):
    pid = product_ids[0]
    print(f"\n  DRY RUN — Q4 optimised recommendations for product {pid}\n")
    rows = make_query_fn(session, [pid])()
    if not rows:
        print("  ⚠  No recommendations found for this product.")
        return
    print(f"  {len(rows)} recommendation(s) (pre-sorted by co_purchase_count DESC):\n")
    print(f"  {'Name':<40} {'Type':<15} {'Price':>8} {'Count':>8} {'Conf':>8}")
    print(f"  {'─'*40} {'─'*15} {'─'*8} {'─'*8} {'─'*8}")
    for r in rows:
        print(
            f"  {str(r.co_product_name or '')[:40]:<40} "
            f"{str(r.co_product_type or ''):<15} "
            f"{str(r.co_product_price_usd or ''):>8} "
            f"{r.co_purchase_count:>8} "
            f"{float(r.confidence or 0):.4f}"
        )
    print(f"\n  Method: 1 partition read, LIMIT {TOP_N} — no Python aggregation.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q4 recommendations benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=500, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q4.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q4 Recommendations Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_optimised (also_bought)")
    print("  Method : Single partition read LIMIT 10, pre-sorted DESC")
    print("           Co-purchase counts pre-aggregated at load time")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pool = fetch_product_id_pool(session, args.pool_size)
        if args.dry_run:
            dry_run(session, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(session, pool),
            db="cassandra_optimised",
            query_id="Q4",
            label=(
                f"Top-{TOP_N} co-purchase recommendations. "
                "Table: also_bought PK ((product_id), co_purchase_count DESC, co_product_id). "
                "Single partition read LIMIT 10 — results pre-sorted by clustering key. "
                "Co-purchase counts and product details pre-computed at load time. "
                "No ALLOW FILTERING, no Python aggregation, no secondary lookups. "
                f"Pool of {len(pool):,} product IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()