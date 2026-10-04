"""
benchmarks/mongodb/optimised/q2_invoice.py — MongoDB Optimised: Q2
====================================================================
Q2: Fetch a complete invoice with customer info, all line items, and
    product details in a single read.

Optimised schema changes vs naive:
  - invoice document embeds a `lines` array (each line has product snapshot)
  - invoice document embeds a `customer` subdocument (full_name, email,
    country_code from users at write time)
  - invoice_lines collection is ELIMINATED — all data is in the invoice doc
  - No second query, no Python-side JOIN

This is the flagship optimised query for MongoDB. The schema effect is
large: naive required 2 network round-trips (invoice + lines) plus a
Python JOIN; optimised requires exactly one `find_one`.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive — this query shows the largest
  schema effect in the entire benchmark suite.

Usage:
    python q2_invoice.py
    python q2_invoice.py --iterations 100
    python q2_invoice.py --dry-run
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── sample invoice IDs (loaded once, re-used across iterations) ───────────────

def load_invoice_ids(db, sample_size: int = 500) -> list[str]:
    ids = [d["_id"] for d in db["invoices"].find({}, {"_id": 1}).limit(sample_size * 4)]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    return ids

# ── core Q2 logic (timed portion) ─────────────────────────────────────────────

def run_q2(db, invoice_ids: list[str]) -> dict | None:
    """
    Single find_one: returns the full invoice document with:
      - doc["customer"]       → embedded user snapshot {full_name, email, country_code}
      - doc["lines"]          → embedded array of line items, each containing:
          line["product_id"], line["description"], line["quantity"],
          line["unit_price_usd"], line["line_total_usd"],
          line["product"]     → embedded product snapshot {name, product_type, price_usd}

    Zero JOINs. One network round-trip.
    """
    invoice_id = random.choice(invoice_ids)
    return db["invoices"].find_one({"_id": invoice_id})

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, invoice_ids):
    def _run():
        run_q2(db, invoice_ids)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, invoice_ids):
    print("\n  DRY RUN — MongoDB Optimised Q2 result sample:\n")
    doc = run_q2(db, invoice_ids)
    if not doc:
        print("  ⚠  No invoice returned — is the optimised DB populated?")
        return

    customer = doc.get("customer", {})
    lines    = doc.get("lines", [])

    print(f"  Invoice ID  : {doc.get('_id')}")
    print(f"  Type        : {doc.get('invoice_type')}")
    print(f"  Status      : {doc.get('status')}")
    print(f"  Total USD   : {doc.get('total_usd')}")
    print(f"  Customer    : {customer.get('full_name')} ({customer.get('email')})")
    print(f"  Country     : {customer.get('country_code')}")
    print(f"  Line count  : {len(lines)}")
    if lines:
        print(f"\n  First line  :")
        l = lines[0]
        print(f"    description    : {l.get('description')}")
        print(f"    quantity       : {l.get('quantity')}")
        print(f"    unit_price_usd : {l.get('unit_price_usd')}")
        product = l.get("product", {})
        print(f"    product.name   : {product.get('name')}")
        print(f"    product.type   : {product.get('product_type')}")
    print(f"\n  Queries issued: 1 (find_one on invoices — lines embedded)")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q2 — invoice fetch benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q2.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q2 Invoice Fetch Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    print("  Sampling invoice IDs...")
    invoice_ids = load_invoice_ids(db)
    print(f"  Loaded {len(invoice_ids):,} invoice IDs for random sampling.\n")

    if args.dry_run:
        dry_run(db, invoice_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, invoice_ids),
        db="mongodb_optimised",
        query_id="Q2",
        label=(
            "Full invoice fetch: single find_one returning embedded customer "
            "snapshot and embedded lines array with product snapshots. "
            "Optimised: invoice_lines collection eliminated — zero JOINs, "
            "one network round-trip. Largest schema effect in the benchmark suite."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()