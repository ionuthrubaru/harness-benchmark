"""
benchmarks/cassandra/optimised/q2_invoice.py — Cassandra Optimised: Q2
=======================================================================
Q2: Fetch a complete invoice with customer info, all line items, and
    full product details.

Optimised schema — single partition read
─────────────────────────────────────────
Table: invoices_full
PK:    ((invoice_id), line_id)

All four source tables (invoices, users, invoice_lines, products) were
joined at load time. One Cassandra row exists per invoice line, with
invoice-level fields, customer snapshot, and product details all
embedded. Q2 reads one partition (all lines for one invoice_id) and
reconstructs the document in Python.

Schema effect vs naive:
  Naive  : PK lookup (invoice) + ALLOW FILTERING scan (lines by invoice_id)
           + PK lookup (user) + N PK lookups (products) → 3+N round trips
  Optimised : single partition read of invoices_full → 1 round trip
              (plus Python assembly of the result dict)

This is the Cassandra analogue of MongoDB's embedded document model:
both return a complete invoice in one storage operation. The comparison
is deliberately interesting — Cassandra does it via a multi-row
partition (one row per line) while MongoDB does it as a single BSON
document.

Usage:
    python q2_invoice.py                   # 1000 iterations
    python q2_invoice.py --iterations 100
    python q2_invoice.py --dry-run
    python q2_invoice.py --pool-size 500
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

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(session, pool_size: int) -> list:
    rows = list(session.execute(
        f"SELECT invoice_id FROM invoices_full LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No rows in invoices_full — run cassandra_optimised_loader.py first.")
    ids = list({r.invoice_id for r in rows})   # deduplicate (multiple lines per invoice)
    random.shuffle(ids)
    print(f"  Invoice ID pool: {len(ids):,} unique invoices loaded.")
    return ids

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, invoice_ids: list):
    """
    Single partition read of invoices_full for the chosen invoice_id.
    All lines, customer data, and product details come back in one scan.
    Python assembles the result dict — no additional queries issued.
    """
    def _run():
        inv_id = random.choice(invoice_ids)
        rows = list(session.execute(
            "SELECT invoice_id, line_id, invoice_type, invoice_status, "
            "subtotal_usd, tax_usd, discount_usd, total_usd, "
            "subscription_id, billing_period_start, billing_period_end, "
            "paid_at, due_at, invoice_created_at, "
            "customer_id, customer_full_name, customer_email, customer_country_code, "
            "line_description, line_quantity, line_unit_price_usd, line_total_usd, "
            "product_id, product_name, product_type, product_price_usd "
            "FROM invoices_full WHERE invoice_id = %s",
            (inv_id,),
        ))
        if not rows:
            return None
        first = rows[0]
        return {
            "invoice": {
                "id":                   first.invoice_id,
                "type":                 first.invoice_type,
                "status":               first.invoice_status,
                "subtotal_usd":         first.subtotal_usd,
                "tax_usd":              first.tax_usd,
                "discount_usd":         first.discount_usd,
                "total_usd":            first.total_usd,
                "created_at":           first.invoice_created_at,
            },
            "customer": {
                "id":           first.customer_id,
                "full_name":    first.customer_full_name,
                "email":        first.customer_email,
                "country_code": first.customer_country_code,
            },
            "lines": [
                {
                    "line_id":          r.line_id,
                    "description":      r.line_description,
                    "quantity":         r.line_quantity,
                    "unit_price_usd":   r.line_unit_price_usd,
                    "line_total_usd":   r.line_total_usd,
                    "product_id":       r.product_id,
                    "product_name":     r.product_name,
                    "product_type":     r.product_type,
                    "product_price_usd": r.product_price_usd,
                }
                for r in rows
            ],
        }
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, invoice_ids: list):
    inv_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2 optimised for invoice {inv_id}\n")
    result = make_query_fn(session, [inv_id])()
    if not result:
        print("  ⚠  Invoice not found.")
        return
    inv  = result["invoice"]
    cust = result["customer"]
    lines = result["lines"]
    print(f"  Invoice  : {inv['id']}")
    print(f"  Type     : {inv['type']}   Status: {inv['status']}")
    print(f"  Total    : ${inv['total_usd']}")
    print(f"  Customer : {cust['full_name']} ({cust['email']})")
    print(f"  Lines    : {len(lines)}  (single partition read — 1 round trip)")
    for i, l in enumerate(lines, 1):
        print(f"    {i}. {str(l['description'])[:50]:<50} ${l['line_total_usd']}"
              + (f"  [{l['product_name'][:30]}]" if l['product_name'] else ""))

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q2 invoice benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q2.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q2 Invoice Fetch Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_optimised (invoices_full)")
    print("  Method : Single partition read — invoice + customer + lines")
    print("           + products all denormalised into one partition")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pool = fetch_invoice_id_pool(session, args.pool_size)
        if args.dry_run:
            dry_run(session, pool)
            return
        run_benchmark(
            query_fn=make_query_fn(session, pool),
            db="cassandra_optimised",
            query_id="Q2",
            label=(
                "Full invoice fetch. Table: invoices_full PK ((invoice_id), line_id). "
                "One partition read returns all lines with embedded invoice, customer, "
                "and product fields. Python assembles result dict. "
                "1 round trip vs naive 3+N. "
                f"Pool of {len(pool):,} invoice IDs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()