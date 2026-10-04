"""
benchmarks/mongodb/naive/q2_invoice.py — MongoDB Naive: Q2
===========================================================
Q2: Fetch a complete invoice with customer info, all line items,
    and full product details.

Naive implementation notes:
    Collections are flat mirrors of the PostgreSQL schema — no embedding.
    Replicating the PostgreSQL 4-table JOIN requires four separate
    MongoDB queries per iteration:

      1. find_one("invoices",      {_id: invoice_id})
      2. find_one("users",         {_id: user_id})
      3. find("invoice_lines",     {invoice_id: invoice_id})
      4. find("products",          {_id: {$in: product_ids}})

    This is the naive equivalent of:
        invoices → users → invoice_lines → products

    The four-query round-trip is intentionally expensive — it
    demonstrates the cost of a relational access pattern against
    a flat document store with no embedding or denormalisation.
    The optimised version will collapse this into a single find_one
    on an embedded document.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive result.
    This file measures the naive (engine-only) side.

Usage:
    cd benchmarks/mongodb/naive
    python q2_invoice.py                   # 1000 iterations, save results
    python q2_invoice.py --iterations 100  # quick smoke test
    python q2_invoice.py --dry-run         # run once, print result sample
    python q2_invoice.py --pool-size 500   # use 500 invoice IDs (default: 1000)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(db, pool_size: int) -> list[str]:
    """
    Pre-fetch a random sample of invoice IDs from MongoDB.
    Random sampling avoids serving the same document from MongoDB's
    in-memory cache on every iteration — same rationale as the
    PostgreSQL baseline.
    """
    pipeline = [
        {"$sample": {"size": pool_size}},
        {"$project": {"_id": 1}},
    ]
    ids = [doc["_id"] for doc in db["invoices"].aggregate(pipeline)]
    if not ids:
        raise RuntimeError(
            "No invoices found in MongoDB. "
            "Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} invoice IDs for random sampling.")
    return ids

# ── core Q2 logic ─────────────────────────────────────────────────────────────

def run_q2(db, invoice_id: str) -> dict | None:
    """
    Fetch a complete invoice using four separate MongoDB queries.

    Query 1 — invoice root
    Query 2 — customer (user) document
    Query 3 — all invoice lines for this invoice
    Query 4 — all products referenced by those lines

    Returns a dict mirroring the PostgreSQL Q2 output structure,
    or None if the invoice does not exist.
    """
    # ── query 1: invoice ──────────────────────────────────────────────────
    invoice = db["invoices"].find_one({"_id": invoice_id})
    if not invoice:
        return None

    # ── query 2: customer ─────────────────────────────────────────────────
    user = db["users"].find_one(
        {"_id": invoice["user_id"]},
        {"_id": 1, "full_name": 1, "email": 1, "country_code": 1},
    )

    # ── query 3: invoice lines ────────────────────────────────────────────
    lines = list(db["invoice_lines"].find(
        {"invoice_id": invoice_id},
        {"_id": 1, "product_id": 1, "description": 1,
         "quantity": 1, "unit_price_usd": 1, "line_total_usd": 1,
         "created_at": 1},
    ))

    # ── query 4: products (batch fetch for all line product_ids) ──────────
    product_ids = [l["product_id"] for l in lines if l.get("product_id")]
    products_by_id = {}
    if product_ids:
        for prod in db["products"].find(
            {"_id": {"$in": product_ids}},
            {"_id": 1, "name": 1, "product_type": 1,
             "price_usd": 1, "attributes": 1},
        ):
            products_by_id[prod["_id"]] = prod

    # ── assemble result ───────────────────────────────────────────────────
    assembled_lines = []
    for line in lines:
        prod = products_by_id.get(line.get("product_id", ""), {})
        assembled_lines.append({
            "line_id":                  line["_id"],
            "line_description":         line.get("description"),
            "quantity":                 line.get("quantity"),
            "unit_price_usd":           line.get("unit_price_usd"),
            "line_total_usd":           line.get("line_total_usd"),
            "product_id":               prod.get("_id"),
            "product_name":             prod.get("name"),
            "product_type":             prod.get("product_type"),
            "product_current_price_usd": prod.get("price_usd"),
            "product_attributes":       prod.get("attributes"),
        })

    return {
        "invoice_id":       invoice["_id"],
        "invoice_type":     invoice.get("invoice_type"),
        "invoice_status":   invoice.get("status"),
        "subtotal_usd":     invoice.get("subtotal_usd"),
        "tax_usd":          invoice.get("tax_usd"),
        "discount_usd":     invoice.get("discount_usd"),
        "total_usd":        invoice.get("total_usd"),
        "due_at":           invoice.get("due_at"),
        "paid_at":          invoice.get("paid_at"),
        "invoice_created_at": invoice.get("created_at"),
        "customer_id":      user.get("_id") if user else None,
        "customer_name":    user.get("full_name") if user else None,
        "customer_email":   user.get("email") if user else None,
        "customer_country": user.get("country_code") if user else None,
        "lines":            assembled_lines,
    }

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, invoice_ids: list[str]):
    def _run():
        invoice_id = random.choice(invoice_ids)
        run_q2(db, invoice_id)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — MongoDB Naive Q2 result for invoice {invoice_id}:\n")
    result = run_q2(db, invoice_id)
    if not result:
        print("  ⚠  No invoice found — check the ID exists.")
        return
    print(f"  Invoice ID    : {result['invoice_id']}")
    print(f"  Type          : {result['invoice_type']}")
    print(f"  Status        : {result['invoice_status']}")
    print(f"  Customer      : {result['customer_name']} "
          f"<{result['customer_email']}> ({result['customer_country']})")
    print(f"  Total USD     : {result['total_usd']}")
    print(f"  Created at    : {result['invoice_created_at']}")
    lines = result["lines"]
    print(f"\n  Line items ({len(lines)}):")
    print(f"  {'Description':<40} {'Qty':>4} {'Unit':>10} {'Total':>10} {'Product type':<16}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*10} {'─'*16}")
    for line in lines:
        print(
            f"  {str(line['line_description']):<40} "
            f"{str(line['quantity']):>4} "
            f"{str(line['unit_price_usd']):>10} "
            f"{str(line['line_total_usd']):>10} "
            f"{str(line['product_type'] or 'N/A'):<16}"
        )

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q2 — invoice fetch benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of invoice IDs to pre-fetch for random sampling (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch one invoice and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q2.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q2.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q2 Invoice Fetch Benchmark")
    print("=" * 55)

    db = get_db()
    invoice_ids = fetch_invoice_id_pool(db, args.pool_size)

    if args.dry_run:
        dry_run(db, invoice_ids)
        return

    run_benchmark(
        query_fn=make_query_fn(db, invoice_ids),
        db="mongodb_naive",
        query_id="Q2",
        label=(
            "Complete invoice fetch via four separate MongoDB queries: "
            "invoices → users → invoice_lines → products. "
            "No embedding — flat collections mirroring PostgreSQL schema. "
            f"Random invoice sampled from pool of {len(invoice_ids)} IDs "
            "per iteration to avoid cache bias."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()