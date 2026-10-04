"""
benchmarks/cassandra/naive/q2_invoice.py — Cassandra Naive: Q2
===============================================================
Q2: Fetch a complete invoice with customer info, all line items,
    and full product details.

Naive schema access pattern — 3 query types, N+2 round trips
──────────────────────────────────────────────────────────────
PostgreSQL Q2 resolves this in one JOIN chain (4 tables). MongoDB
optimised resolves it in a single document read. In the Cassandra
naive schema:

  1. Invoice by PK — fast single-partition read (id is the PK).

  2. Invoice lines by invoice_id — ALLOW FILTERING required.
     invoice_lines has id as its sole partition key; invoice_id is a
     plain column. Cassandra must scan every partition to find rows
     where invoice_id matches. This is the primary cost driver.

  3. User by PK — fast single-partition read.

  4. Products by PK — one fast read per line item with a non-null
     product_id. These are O(1) partition lookups, not scans, but
     the N individual round trips add latency for invoices with many
     lines.

The total round trips per iteration = 3 + (lines with product_id).
For a typical invoice with 1–5 line items this is 4–8 queries where
PostgreSQL uses 1 and MongoDB optimised uses 1.

Pool design
────────────
Invoice IDs are pre-fetched at startup so each iteration picks a
random one. This mirrors the PostgreSQL Q2 design and prevents the
OS page cache from serving the same partition repeatedly, producing
artificially stable latency. The pool also ensures every sampled
invoice actually exists in the keyspace.

Usage:
    python q2_invoice.py                   # 1000 iterations
    python q2_invoice.py --iterations 100  # quick smoke test
    python q2_invoice.py --dry-run         # run once, print result sample
    python q2_invoice.py --pool-size 500   # smaller pool
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

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")

# ── pool helpers ──────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(session, pool_size: int) -> list:
    """
    Fetch a pool of invoice IDs from the invoices table.
    SELECT id FROM invoices has no WHERE clause so no ALLOW FILTERING is needed —
    Cassandra returns the first pool_size rows it encounters across token ranges.
    The ids are UUID objects (Cassandra driver materialises them automatically).
    """
    rows = list(session.execute(
        f"SELECT id FROM invoices LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No invoices found — run cassandra_naive_loader.py first.")
    ids = [r.id for r in rows]
    random.shuffle(ids)
    print(f"  Invoice ID pool: {len(ids):,} entries loaded.")
    return ids

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, invoice_ids: list):
    """
    Each call:
      1. Fetch invoice by PK (fast)
      2. Fetch lines by invoice_id — ALLOW FILTERING (expensive)
      3. Fetch user by PK (fast)
      4. Fetch each product by PK (fast, N round trips)
    Returns a dict representing the assembled invoice document.
    """
    def _run():
        inv_id = random.choice(invoice_ids)

        # ── 1. Invoice by PK ──────────────────────────────────────────────────
        inv_row = session.execute(
            "SELECT id, user_id, invoice_type, status, subtotal_usd, tax_usd, "
            "discount_usd, total_usd, subscription_id, billing_period_start, "
            "billing_period_end, paid_at, due_at, created_at "
            "FROM invoices WHERE id = %s",
            (inv_id,),
        ).one()

        if inv_row is None:
            return None

        # ── 2. Invoice lines by invoice_id (ALLOW FILTERING) ──────────────────
        lines = list(session.execute(
            "SELECT id, invoice_id, product_id, description, quantity, "
            "unit_price_usd, line_total_usd, created_at "
            "FROM invoice_lines WHERE invoice_id = %s ALLOW FILTERING",
            (inv_id,),
        ))

        # ── 3. User by PK ─────────────────────────────────────────────────────
        user_row = session.execute(
            "SELECT id, full_name, email, country_code FROM users WHERE id = %s",
            (inv_row.user_id,),
        ).one()

        # ── 4. Products by PK (one per line with product_id) ──────────────────
        product_cache = {}
        for line in lines:
            if line.product_id and str(line.product_id) not in product_cache:
                prod_row = session.execute(
                    "SELECT id, name, product_type, price_usd, attributes "
                    "FROM products WHERE id = %s",
                    (line.product_id,),
                ).one()
                if prod_row:
                    product_cache[str(line.product_id)] = prod_row

        # ── Assemble result dict ───────────────────────────────────────────────
        return {
            "invoice":  inv_row,
            "customer": user_row,
            "lines": [
                {
                    "line":    l,
                    "product": product_cache.get(str(l.product_id)),
                }
                for l in lines
            ],
        }

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, invoice_ids: list):
    inv_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2 naive for invoice {inv_id}\n")

    fn = make_query_fn(session, [inv_id])
    result = fn()
    if not result or not result["invoice"]:
        print("  ⚠  Invoice not found.")
        return

    inv = result["invoice"]
    cust = result["customer"]
    lines = result["lines"]
    print(f"  Invoice      : {inv.id}")
    print(f"  Type         : {inv.invoice_type}   Status: {inv.status}")
    print(f"  Total        : ${inv.total_usd}")
    print(f"  Customer     : {cust.full_name if cust else 'N/A'} ({cust.email if cust else 'N/A'})")
    print(f"  Lines        : {len(lines)}")
    for i, item in enumerate(lines, 1):
        l = item["line"]
        p = item["product"]
        print(f"    {i}. {l.description[:50]:<50} ${l.line_total_usd}"
              + (f"  [{p.name[:30]}]" if p else ""))

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q2 invoice benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q2.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q2 Invoice Fetch Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_naive")
    print("  Method : PK lookup (invoice) + ALLOW FILTERING (lines)")
    print("           + PK lookup (user) + N PK lookups (products)")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        pool = fetch_invoice_id_pool(session, args.pool_size)

        if args.dry_run:
            dry_run(session, pool)
            return

        run_benchmark(
            query_fn=make_query_fn(session, pool),
            db="cassandra_naive",
            query_id="Q2",
            label=(
                "Full invoice fetch: PK lookup for invoice + ALLOW FILTERING scan "
                "for invoice_lines by invoice_id + PK lookup for user + N PK lookups "
                "for products (one per line item). Round trips = 3 + N line items. "
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