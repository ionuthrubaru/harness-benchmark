"""
benchmarks/timescaledb/naive/q2_invoice.py — TimescaleDB Naive: Q2
===================================================================
Q2: Fetch a complete invoice with customer info, all line items,
    and full product details in a single query.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
The same 4-table JOIN chain runs unchanged on TimescaleDB:
    invoices → users → invoice_lines → products

TimescaleDB adds nothing for this query — it is a point lookup by
invoice ID, not a time-range scan. The invoices hypertable is queried
by a non-partition column (id), so no chunk pruning occurs. All chunks
are probed via the index on invoices.id.

This is the expected result: Q2 is MongoDB's killer query (embedded
document model, zero JOINs). TimescaleDB's time-series features are
irrelevant here. The latency should be comparable to PostgreSQL,
demonstrating that TimescaleDB does not penalise non-time-series queries.

Schema note — dropped FK
─────────────────────────
The FK from invoice_lines.invoice_id → invoices.id was dropped in the
TimescaleDB naive schema because TimescaleDB requires all unique constraints
on a hypertable to include the partition column (created_at). The JOIN
`il.invoice_id = i.id` still works correctly via the index on
invoice_lines(invoice_id).

Usage:
    cd benchmarks/timescaledb/naive
    python q2_invoice.py                   # 1000 iterations
    python q2_invoice.py --iterations 100  # quick smoke test
    python q2_invoice.py --explain         # EXPLAIN ANALYZE
    python q2_invoice.py --dry-run         # run once, print result
    python q2_invoice.py --pool-size 500   # smaller pool
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

# ── query — identical to PostgreSQL Q2 ────────────────────────────────────────

Q2_SQL = """
SELECT
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

    u.id                    AS customer_id,
    u.full_name             AS customer_name,
    u.email                 AS customer_email,
    u.country_code          AS customer_country,

    il.id                   AS line_id,
    il.description          AS line_description,
    il.quantity,
    il.unit_price_usd,
    il.line_total_usd,

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

Q2_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q2_SQL

# ── ID pool ────────────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(conn, pool_size: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM invoices ORDER BY RANDOM() LIMIT %s", (pool_size,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No invoices found — run timescaledb_naive_loader.py first.")
    ids = [str(row[0]) for row in rows]
    print(f"  Invoice ID pool: {len(ids):,} entries loaded.")
    return ids

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, invoice_ids: list[str]):
    def _run():
        invoice_id = random.choice(invoice_ids)
        with conn.cursor() as cur:
            cur.execute(Q2_SQL, (invoice_id,))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2 naive for invoice {invoice_id}:\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q2_SQL, (invoice_id,))
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No rows returned.")
        return
    r = rows[0]
    print(f"  Invoice   : {r['invoice_id']}")
    print(f"  Type      : {r['invoice_type']}   Status: {r['invoice_status']}")
    print(f"  Customer  : {r['customer_name']} <{r['customer_email']}>")
    print(f"  Total USD : {r['total_usd']}")
    print(f"\n  Lines ({len(rows)}):")
    print(f"  {'Description':<40} {'Qty':>4} {'Total':>10} {'Product type':<15}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*15}")
    for row in rows:
        print(
            f"  {str(row['line_description'])[:40]:<40} "
            f"{str(row['quantity']):>4} "
            f"{str(row['line_total_usd']):>10} "
            f"{str(row['product_type'] or 'N/A'):<15}"
        )


def explain(conn, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  EXPLAIN ANALYZE — Q2 naive (invoice {invoice_id}):\n")
    with conn.cursor() as cur:
        cur.execute(Q2_EXPLAIN_SQL, (invoice_id,))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q2 invoice benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q2.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q2 Invoice Fetch Benchmark")
    print("=" * 60)
    print("  Schema : naive (invoices hypertable, 7-day chunks)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : PK lookup — no chunk pruning (id is not partition col)")

    conn = get_connection()
    try:
        pool = fetch_invoice_id_pool(conn, args.pool_size)
        if args.explain:
            explain(conn, pool)
            return
        if args.dry_run:
            dry_run(conn, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(conn, pool),
            db="timescaledb_naive",
            query_id="Q2",
            label=(
                "Full invoice fetch via 4-table JOIN: "
                "invoices → users → invoice_lines → products. "
                "SQL identical to PostgreSQL baseline. "
                "No chunk pruning — query by invoice id (not partition column). "
                "invoice_lines.invoice_id FK dropped for hypertable compatibility "
                "(JOIN works via index). "
                f"Pool of {len(pool):,} invoice IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()