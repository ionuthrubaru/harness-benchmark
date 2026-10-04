"""
benchmarks/postgres/q2_invoice.py — PostgreSQL Baseline: Q2
============================================================
Q2: Fetch a complete invoice with customer info, all line items,
    and full product details in a single query.

    This is the PostgreSQL baseline for the MongoDB comparison (Q2).
    MongoDB will serve the same data from a single embedded document
    with zero JOINs. Here we require a 4-table JOIN:
        invoices → users (customer)
                 → invoice_lines → products

Killer feature demonstrated (MongoDB side):
    Embedded document model — the entire invoice, customer snapshot,
    line items and product details are stored as one document and
    retrieved in a single read with no JOINs.

PostgreSQL baseline design notes:
    - A random invoice ID is chosen each iteration from a pre-fetched
      pool of real IDs. This prevents the buffer cache from serving
      the same page repeatedly and producing artificially low latency.
    - Results are fully materialised (fetchall) so network + transfer
      time is included — this matches what MongoDB will measure.
    - The query uses a single JOIN chain rather than separate queries,
      giving PostgreSQL the best realistic chance against MongoDB's
      single-document fetch.

Usage:
    python q2_invoice.py                   # 1000 iterations, save results
    python q2_invoice.py --iterations 100  # quick smoke test
    python q2_invoice.py --explain         # print EXPLAIN ANALYZE, no benchmark
    python q2_invoice.py --dry-run         # run once, print result sample
    python q2_invoice.py --pool-size 500   # pre-fetch 500 invoice IDs (default: 1000)
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
# Design notes
# ────────────
# The JOIN chain mirrors the document structure MongoDB will embed:
#
#   invoices (root document)
#     → users            (customer snapshot: name, email, country)
#     → invoice_lines    (embedded array of line items)
#     → products         (product details embedded inside each line item)
#
# product_id is nullable on invoice_lines (subscription renewal lines have
# no product), so we use LEFT JOIN for products to avoid dropping those rows.
#
# The result is one row per invoice line — the application (or MongoDB) would
# normally group these into a nested structure, but for benchmarking purposes
# we measure the raw data retrieval cost, which is what matters.

Q2_SQL = """
SELECT
    -- Invoice fields
    i.id                    AS invoice_id,
    i.invoice_type,
    i.status                AS invoice_status,
    i.subtotal_usd,
    i.tax_usd,
    i.discount_usd,
    i.total_usd,
    i.due_at,
    i.paid_at,
    i.created_at            AS invoice_created_at,

    -- Customer fields (denormalised snapshot — mirrors MongoDB embedding)
    u.id                    AS customer_id,
    u.full_name             AS customer_name,
    u.email                 AS customer_email,
    u.country_code          AS customer_country,

    -- Invoice line fields
    il.id                   AS line_id,
    il.description          AS line_description,
    il.quantity,
    il.unit_price_usd,
    il.line_total_usd,

    -- Product fields (NULL for subscription renewal lines)
    p.id                    AS product_id,
    p.name                  AS product_name,
    p.product_type,
    p.price_usd             AS product_current_price_usd,
    p.attributes            AS product_attributes

FROM invoices i
JOIN users         u  ON u.id  = i.user_id
JOIN invoice_lines il ON il.invoice_id = i.id
LEFT JOIN products p  ON p.id  = il.product_id

WHERE i.id = %s
ORDER BY il.created_at;
"""

# ── EXPLAIN wrapper ───────────────────────────────────────────────────────────

Q2_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q2_SQL

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(conn, pool_size: int) -> list[str]:
    """
    Pre-fetch a random sample of real invoice IDs.
    Using a pool rather than the same ID every iteration prevents
    PostgreSQL's shared buffer cache from serving identical pages on
    every run — which would produce unrealistically low latencies and
    make the PostgreSQL baseline look better than it really is at scale.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM invoices
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            (pool_size,),
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            "No invoices found in the database. "
            "Run the data loader before benchmarking."
        )

    ids = [str(row[0]) for row in rows]
    print(f"  Loaded pool of {len(ids)} invoice IDs for random sampling.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, invoice_ids: list[str]):
    """
    Return a zero-argument callable that fetches a randomly chosen invoice
    on every call. Random selection from the pre-fetched pool avoids the
    cache-warming bias described above.
    """
    def _run():
        invoice_id = random.choice(invoice_ids)
        with conn.cursor() as cur:
            cur.execute(Q2_SQL, (invoice_id,))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, invoice_ids: list[str]):
    """Fetch one invoice and print the result rows for sanity checking."""
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2 result for invoice {invoice_id}:\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q2_SQL, (invoice_id,))
        rows = cur.fetchall()

    if not rows:
        print("  ⚠  No rows returned — check the invoice ID exists.")
        return

    # Print invoice-level fields once
    r = rows[0]
    print(f"  Invoice ID    : {r['invoice_id']}")
    print(f"  Type          : {r['invoice_type']}")
    print(f"  Status        : {r['invoice_status']}")
    print(f"  Customer      : {r['customer_name']} <{r['customer_email']}> ({r['customer_country']})")
    print(f"  Total USD     : {r['total_usd']}")
    print(f"  Created at    : {r['invoice_created_at']}")
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


def explain(conn, invoice_ids: list[str]):
    """Print EXPLAIN ANALYZE for one invoice fetch."""
    invoice_id = invoice_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q2 (invoice {invoice_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q2_EXPLAIN_SQL, (invoice_id,))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Q2 invoice benchmark")
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of invoice IDs to pre-fetch for random sampling (default: 1000)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch one invoice and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q2_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q2_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q2 Invoice Fetch Benchmark")
    print("=" * 50)

    conn = get_connection()

    try:
        invoice_ids = fetch_invoice_id_pool(conn, args.pool_size)

        if args.explain:
            explain(conn, invoice_ids)
            return

        if args.dry_run:
            dry_run(conn, invoice_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, invoice_ids),
            db="postgres",
            query_id="Q2",
            label=(
                "Complete invoice fetch (customer + line items + product details) "
                "via 4-table JOIN: invoices → users → invoice_lines → products. "
                "Random invoice sampled from pool of "
                f"{len(invoice_ids)} IDs per iteration to avoid buffer cache bias."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()