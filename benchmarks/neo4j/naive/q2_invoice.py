"""
benchmarks/neo4j/naive/q2_invoice.py — Neo4j Naive: Q2
=======================================================
Q2: Fetch a complete invoice with customer info, all line items,
    and full product details.

Naive schema: two Cypher queries — one for the invoice + customer +
lines, one for product details on lines that have a product_id.
This mirrors the 4-table JOIN in PostgreSQL:
    invoices → users → invoice_lines → products

Two queries are needed because Neo4j cannot express a conditional
LEFT JOIN in a single MATCH without OPTIONAL MATCH adding complexity.
The split is clean: fetch all lines first, then batch-fetch products
only for lines that reference one.

Random invoice ID pool pre-fetched at startup — same anti-cache-bias
pattern as the PostgreSQL baseline.

Usage:
    python q2_invoice.py
    python q2_invoice.py --iterations 100
    python q2_invoice.py --dry-run
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

# ── Cypher queries ────────────────────────────────────────────────────────────

# Fetch invoice header + customer + all line items in one traversal.
# OPTIONAL MATCH on Product handles subscription lines (no product_id).
Q2_CYPHER = """
MATCH (u:User)-[:HAS_INVOICE]->(i:Invoice {id: $invoice_id})
MATCH (i)-[:HAS_LINE]->(il:InvoiceLine)
OPTIONAL MATCH (il)-[:LINE_FOR_PRODUCT]->(p:Product)
RETURN
    i.id                    AS invoice_id,
    i.invoice_type          AS invoice_type,
    i.status                AS invoice_status,
    i.subtotal_usd          AS subtotal_usd,
    i.tax_usd               AS tax_usd,
    i.discount_usd          AS discount_usd,
    i.total_usd             AS total_usd,
    i.due_at                AS due_at,
    i.paid_at               AS paid_at,
    i.created_at            AS invoice_created_at,
    u.id                    AS customer_id,
    u.full_name             AS customer_name,
    u.email                 AS customer_email,
    u.country_code          AS customer_country,
    il.id                   AS line_id,
    il.description          AS line_description,
    il.quantity             AS quantity,
    il.unit_price_usd       AS unit_price_usd,
    il.line_total_usd       AS line_total_usd,
    p.id                    AS product_id,
    p.name                  AS product_name,
    p.product_type          AS product_type,
    p.price_usd             AS product_current_price_usd,
    p.attributes            AS product_attributes
ORDER BY il.created_at
"""

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(driver, pool_size: int) -> list[str]:
    """
    Pre-fetch a random sample of real invoice IDs from Neo4j.
    Equivalent to the PostgreSQL ORDER BY RANDOM() LIMIT pool_size.
    apoc.coll.randomItems requires APOC — which is installed in the container.
    """
    cypher = """
    MATCH (i:Invoice)
    WITH i, rand() AS r
    ORDER BY r
    LIMIT $pool_size
    RETURN i.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, pool_size=pool_size)
        ids = [row["id"] for row in result]

    if not ids:
        raise RuntimeError(
            "No Invoice nodes found. Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} invoice IDs for random sampling.")
    return ids


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, invoice_ids: list[str]):
    def _run():
        invoice_id = random.choice(invoice_ids)
        with driver.session() as session:
            result = session.run(Q2_CYPHER, invoice_id=invoice_id)
            result.data()   # fully materialise
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — Neo4j Naive Q2 result for invoice {invoice_id}:\n")
    with driver.session() as session:
        result = session.run(Q2_CYPHER, invoice_id=invoice_id)
        rows = result.data()

    if not rows:
        print("  ⚠  No rows returned — check the invoice ID exists.")
        return

    r = rows[0]
    print(f"  Invoice ID  : {r['invoice_id']}")
    print(f"  Type        : {r['invoice_type']}")
    print(f"  Status      : {r['invoice_status']}")
    print(f"  Customer    : {r['customer_name']} <{r['customer_email']}> ({r['customer_country']})")
    print(f"  Total USD   : {r['total_usd']}")
    print(f"\n  Line items ({len(rows)}):")
    print(f"  {'Description':<40} {'Qty':>4} {'Unit':>10} {'Total':>10} {'Product type':<16}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*10} {'─'*16}")
    for row in rows:
        print(
            f"  {str(row['line_description']):<40} "
            f"{str(row['quantity']):>4} "
            f"{str(row['unit_price_usd']):>10} "
            f"{str(row['line_total_usd']):>10} "
            f"{str(row['product_type'] or 'N/A'):<16}"
        )
    print(f"\n  ... {len(rows)} line(s) returned")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Naive Q2 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size",  type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_naive_Q2.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Naive — Q2 Invoice Fetch Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))

    try:
        invoice_ids = fetch_invoice_id_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, invoice_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, invoice_ids),
            db="neo4j_naive",
            query_id="Q2",
            label=(
                "Complete invoice fetch (customer + line items + product details). "
                "Single Cypher traversal: (User)-[:HAS_INVOICE]->(Invoice)"
                "-[:HAS_LINE]->(InvoiceLine) + OPTIONAL MATCH to Product. "
                f"Random invoice sampled from pool of {len(invoice_ids)} IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()