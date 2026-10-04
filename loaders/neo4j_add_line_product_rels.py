"""
loaders/neo4j_add_line_product_rels.py — One-off patch
=======================================================
Adds the missing (InvoiceLine)-[:LINE_FOR_PRODUCT]->(Product)
relationships to both naive and optimised Neo4j containers.

This relationship is required by Q2 to fetch product details
for marketplace invoice lines. Subscription lines have no
product_id and are skipped (OPTIONAL MATCH in Q2 handles them).

Run once against both containers after the main loaders have finished.

Usage:
    python loaders/neo4j_add_line_product_rels.py
"""

import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.neo4j.neo4j_conn import get_driver
from neo4j_naive_loader import ok, err, info, warn, run_batches, read_csv

load_dotenv()

CYPHER = """
UNWIND $rows AS row
MATCH (il:InvoiceLine {id: row.id})
MATCH (p:Product      {id: row.product_id})
MERGE (il)-[:LINE_FOR_PRODUCT]->(p)
"""


def add_rels(driver, port: int):
    # Merge both line CSVs — only rows with a non-empty product_id
    market_rows = read_csv("marketplace_invoice_lines.csv")
    sub_rows    = read_csv("subscription_invoice_lines.csv")
    rows = [
        r for r in market_rows + sub_rows
        if r.get("product_id", "").strip()
    ]
    info(f"  {len(rows):,} invoice lines have a product_id")
    t0 = time.perf_counter()
    run_batches(driver, CYPHER, rows)
    elapsed = time.perf_counter() - t0
    ok(f"  (InvoiceLine)-[:LINE_FOR_PRODUCT]->(Product)  "
       f"{len(rows):,} rels  {elapsed:.1f}s  (port {port})")


def main():
    for label, env_var, default_port in [
        ("naive",     "NEO4J_NAIVE_PORT",     7687),
        ("optimised", "NEO4J_OPTIMISED_PORT", 7688),
    ]:
        port = int(os.getenv(env_var, default_port))
        print(f"\n{'=' * 55}")
        print(f"  Adding LINE_FOR_PRODUCT rels — Neo4j {label} (port {port})")
        print(f"{'=' * 55}")
        driver = get_driver(port=port)
        try:
            driver.verify_connectivity()
            ok(f"Connected to Neo4j {label}")
            add_rels(driver, port)
        except Exception as e:
            err(f"Failed on {label}: {e}")
            sys.exit(1)
        finally:
            driver.close()

    print(f"\n  All done.\n")


if __name__ == "__main__":
    main()