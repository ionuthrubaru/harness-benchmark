"""
loaders/mongodb_optimised_loader.py — Phase 3: MongoDB Optimised Schema
========================================================================
Loads all 12 StreamCart tables into MongoDB using an idiomatic document
model designed around the actual query access patterns (Q1–Q8).

This loader produces the OPTIMISED schema used in Phase 3/4 benchmarks.
Compare against mongodb_naive_loader.py which ports the PostgreSQL schema
literally (one flat collection per table, no embedding, no denormalisation).

The academic purpose of comparing naive vs optimised:
    naive result    − PostgreSQL baseline = ENGINE effect
    optimised result − naive result       = SCHEMA effect

This file measures the schema effect for MongoDB specifically.

════════════════════════════════════════════════════════════════════════
OPTIMISED SCHEMA DESIGN — ALL 12 COLLECTIONS
════════════════════════════════════════════════════════════════════════

┌─────────────────────────────┬──────────────────────────────────────────────────────────────────┐
│ Collection                  │ Optimisation & Motivation                                        │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ invoices                    │ Embed invoice_lines array directly inside each invoice document.  │
│                             │ Also embed a customer snapshot (full_name, email, country_code,  │
│                             │ city) from users. Each line item includes a product snapshot      │
│                             │ (name, product_type, price_usd) from products.                   │
│                             │ MOTIVATION: Q2 becomes a single find_one() — zero additional     │
│                             │ queries needed for lines or customer info. In the naive schema    │
│                             │ Q2 requires 4 round trips (invoice + user + lines + products).   │
│                             │ Compound index on (user_id, created_at DESC) serves Q1 and Q7.   │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ invoice_lines               │ ELIMINATED as a standalone collection. Lines are embedded        │
│                             │ inside their parent invoice document (see above).                │
│                             │ MOTIVATION: invoice lines have no independent access pattern —   │
│                             │ they are always fetched as part of their invoice. Embedding       │
│                             │ removes the need for a separate collection, a foreign key index,  │
│                             │ and a second query. MongoDB's document model is designed for      │
│                             │ exactly this 1:few parent-child relationship.                     │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ sessions                    │ cart stored as a native BSON array of subdocuments instead of a  │
│                             │ JSON string. All session scalar fields stored as top-level BSON  │
│                             │ fields (no nested JSON strings anywhere).                        │
│                             │ MOTIVATION: the naive schema stores cart as a raw JSON string    │
│                             │ because the CSV loader returns all values as text. Deserialising  │
│                             │ that string in Python on every Q3 read adds latency and prevents │
│                             │ MongoDB from indexing into cart item fields. A native array       │
│                             │ returns a Python list directly with no extra work.               │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ events                      │ metadata stored as a native BSON subdocument instead of a JSON  │
│                             │ string. Compound index on (user_id, occurred_at DESC) replaces   │
│                             │ the naive separate indexes.                                      │
│                             │ MOTIVATION: same deserialisation argument as cart in sessions.   │
│                             │ The compound index matches the exact Q6 access pattern           │
│                             │ (user_id equality + occurred_at range) and allows MongoDB to     │
│                             │ satisfy the query and sort entirely from the index.              │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ orders                      │ Embed order_items array directly inside each order document.     │
│                             │ Each embedded item includes a product snapshot (name,            │
│                             │ product_type, price_usd).                                        │
│                             │ MOTIVATION: order items have no independent access pattern —     │
│                             │ they are always fetched as part of their order. Q4 co-purchase   │
│                             │ traversal benefits from being able to read all items in an       │
│                             │ order in a single document fetch rather than a separate          │
│                             │ collection scan. Compound index (user_id, created_at DESC).      │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ order_items                 │ ELIMINATED as a standalone collection. Items are embedded        │
│                             │ inside their parent order document (see above).                  │
│                             │ MOTIVATION: same argument as invoice_lines. Order items are      │
│                             │ never accessed independently of their order in Q1–Q8.            │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ products                    │ Text index on (name, description) for Q5 $text search —         │
│                             │ same as naive. attributes stored as native BSON subdocument     │
│                             │ (parsed from JSON string in CSV).                               │
│                             │ Compound index (seller_id, created_at DESC) for seller queries. │
│                             │ MOTIVATION: attributes in the naive schema is stored as a raw   │
│                             │ JSON string. Parsing it to a native subdocument allows MongoDB  │
│                             │ to index into attribute fields and return structured data        │
│                             │ without application-side deserialisation.                       │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ users                       │ preferences stored as native BSON subdocument (parsed from      │
│                             │ JSON string). Indexes on (email), (country_code, created_at).   │
│                             │ MOTIVATION: preferences in the naive schema is a raw JSON        │
│                             │ string. Native subdocument enables field-level access without   │
│                             │ deserialisation. The compound country+date index supports        │
│                             │ geo-filtered user queries.                                       │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ seller_profiles             │ No structural change — seller_profiles is a 1:1 extension of    │
│                             │ users with no child relationships to embed. Flat document is     │
│                             │ already idiomatic. Index on (is_verified, country_code).        │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ subscriptions               │ Compound index (user_id, started_at DESC) replaces the naive    │
│                             │ separate indexes.                                                │
│                             │ MOTIVATION: Q1 and Q7 both perform a per-user tier attribution  │
│                             │ by finding the most recent subscription before a given invoice  │
│                             │ date. This is a user_id equality + started_at range scan —      │
│                             │ exactly what (user_id, started_at DESC) covers. In the naive    │
│                             │ schema, subscriptions are loaded into a Python dict keyed by    │
│                             │ user_id but the index on the collection itself is not compound. │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ subscription_tiers          │ No change. 3 static documents, always resident in WiredTiger    │
│                             │ cache. No optimisation possible or necessary.                   │
├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ subscription_tier_pricing   │ No change. 5 static documents. Index on (tier_id, valid_from). │
└─────────────────────────────┴──────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
EMBEDDING DEPTH DECISIONS
════════════════════════════════════════════════════════════════════════

Invoice line product snapshot — name, product_type, price_usd ONLY.
    attributes is excluded because it can reach several hundred bytes per
    product and invoices have 1–10 lines each. Across 700k+ invoice
    documents the storage cost is significant. attributes is only needed
    for product detail pages, not invoice rendering — the common read path
    that Q2 measures. This keeps documents lean while eliminating the
    JOIN for the meaningful access pattern.

Customer snapshot in invoice — full_name, email, country_code, city.
    These four fields are sufficient to render invoice headers and match
    the PostgreSQL Q2 baseline output. Excluding preferences and
    last_login_at which are irrelevant to invoice display.

Order item product snapshot — name, product_type, price_usd ONLY.
    Same rationale as invoice line snapshot — attributes excluded for
    the same size/relevance argument.

════════════════════════════════════════════════════════════════════════
NO MULTIKEY INDEX ON EMBEDDED ARRAYS
════════════════════════════════════════════════════════════════════════

MongoDB supports multikey indexes on embedded array fields
(e.g. {"lines.product_id": 1}) but none are created here.
The access pattern for invoice lines and order items is always
"give me document X and all its children" — a primary key fetch.
No query in Q1–Q8 filters invoices by a product appearing in their
lines, so a multikey index would add write overhead with no read benefit.

════════════════════════════════════════════════════════════════════════
NATIVE BSON vs JSON STRING — THE CORE NAIVE→OPTIMISED DIFFERENCE
════════════════════════════════════════════════════════════════════════

The naive loader stores all values as strings because csv.DictReader
returns every field as text. This means JSONB columns (cart, metadata,
attributes, preferences, features) arrive as raw JSON strings like:
    '{"theme": "dark", "language": "en"}'

The optimised loader calls json.loads() on these fields before inserting,
so MongoDB stores them as native BSON subdocuments or arrays:
    {"theme": "dark", "language": "en"}

Benefits:
  1. No application-side json.loads() on every read
  2. MongoDB can index into subdocument fields
  3. Aggregation pipelines can reference subdocument fields directly
  4. $text search over embedded string fields works correctly
  5. Document size is slightly smaller (no quote escaping overhead)

This change alone accounts for the schema effect on Q3 (cart) and Q6
(metadata). The embedding changes account for the schema effect on Q2.

Usage:
    python loaders/mongodb_optimised_loader.py
    python loaders/mongodb_optimised_loader.py --drop     # drop + reload
    python loaders/mongodb_optimised_loader.py --verify   # check counts only
"""

import csv
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv()


# ── connection ────────────────────────────────────────────────────────────────

def get_client():
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    if not user or not password:
        raise RuntimeError("MONGO_USER and MONGO_PASSWORD must be set in .env")
    return MongoClient(
        f"mongodb://{user}:{password}@localhost:27017/",
        serverSelectionTimeoutMS=30_000,
        socketTimeoutMS=60_000,
    )


def get_db():
    db_name = os.getenv("MONGO_DB_OPTIMISED")
    if not db_name:
        raise RuntimeError("MONGO_DB_OPTIMISED not set in .env")
    return get_client()[db_name]


# ── helpers ───────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BATCH_SIZE = 250

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")


def fail(msg): print(f"  {RED}✘ {msg}{RESET}")


def info(msg): print(f"  {BLUE}> {msg}{RESET}")


def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")


def csv_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def load_csv(filename: str) -> list[dict]:
    path = csv_path(filename)
    if not os.path.exists(path):
        warn(f"CSV not found: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_json_field(value: str, default):
    """
    Parse a JSON string field into a native Python object.
    Falls back to default if the value is empty or unparseable.
    """
    if not value or value.strip() in ("", "null", "NULL"):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def insert_batches(collection, docs: list[dict], label: str):
    if not docs:
        warn(f"  No documents to insert for {label}")
        return
    total = len(docs)
    inserted = 0
    t0 = time.perf_counter()
    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        collection.insert_many(batch, ordered=False)
        inserted += len(batch)
        if inserted % 10_000 == 0 or inserted == total:
            elapsed = time.perf_counter() - t0
            print(f"    {inserted:>8,} / {total:,}  ({elapsed:.1f}s)", end="\r")
    elapsed = time.perf_counter() - t0
    print(f"    {inserted:>8,} / {total:,}  ({elapsed:.1f}s)  ✔            ")
    ok(f"{label}: {inserted:,} documents in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION LOADERS
# ══════════════════════════════════════════════════════════════════════════════

# ── users ─────────────────────────────────────────────────────────────────────

def load_users(db):
    info("Loading users...")
    rows = load_csv("users.csv")
    docs = []
    for r in rows:
        docs.append({
            "_id": r["id"],
            "email": r["email"],
            "full_name": r["full_name"],
            "country_code": r["country_code"],
            "city": r["city"] or None,
            "created_at": r["created_at"],
            "last_login_at": r["last_login_at"] or None,
            "is_active": r["is_active"].lower() == 'true' if isinstance(r["is_active"], str) else r["is_active"],
            # OPTIMISATION: parse JSON string → native BSON subdocument
            "preferences": parse_json_field(r.get("preferences"), {}),
        })
    insert_batches(db["users"], docs, "users")


def create_users_indexes(db):
    db["users"].create_index([("email", ASCENDING)], unique=True)
    db["users"].create_index([("country_code", ASCENDING), ("created_at", ASCENDING)])
    db["users"].create_index([("created_at", ASCENDING)])


# ── seller_profiles ───────────────────────────────────────────────────────────
# FIXED: Removed 'total_sales' column as it doesn't exist in the CSV

def load_seller_profiles(db):
    info("Loading seller_profiles...")
    rows = load_csv("seller_profiles.csv")
    docs = []
    for r in rows:
        docs.append({
            "_id": r["user_id"],
            "display_name": r["display_name"],
            "legal_name": r.get("legal_name") or None,
            "tax_id": r.get("tax_id") or None,
            "payout_email": r.get("payout_email") or None,
            "country_code": r.get("country_code") or None,
            "is_verified": r["is_verified"].lower() == 'true' if isinstance(r["is_verified"], str) else r[
                "is_verified"],
            "bio": r.get("bio") or None,
            # 'total_sales' column removed - doesn't exist in CSV
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    insert_batches(db["seller_profiles"], docs, "seller_profiles")


def create_seller_profiles_indexes(db):
    db["seller_profiles"].create_index(
        [("is_verified", ASCENDING), ("country_code", ASCENDING)]
    )


# ── subscription_tiers ────────────────────────────────────────────────────────

TIER_DOCS = [
    {
        "_id": "1",
        "name": "Free",
        "description": "Basic software access, up to 5 marketplace purchases/year",
        "features": {
            "seats": 1, "api_access": False, "priority_support": False,
            "marketplace_purchases_per_year": 5,
            "apps": {"CanvasEditor": "free", "VideoSuite": None},
        },
    },
    {
        "_id": "2",
        "name": "Pro",
        "description": "Full software, unlimited purchases, early access",
        "features": {
            "seats": 1, "api_access": False, "priority_support": False,
            "marketplace_purchases_per_year": -1,
            "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"},
        },
    },
    {
        "_id": "3",
        "name": "Business",
        "description": "Everything in Pro plus team seats, API access, priority support",
        "features": {
            "seats": 10, "api_access": True, "priority_support": True,
            "marketplace_purchases_per_year": -1,
            "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"},
        },
    },
]


def load_subscription_tiers(db):
    info("Loading subscription_tiers (hardcoded — no CSV)...")
    db["subscription_tiers"].insert_many(TIER_DOCS)
    ok("subscription_tiers: 3 documents")


# ── subscription_tier_pricing ─────────────────────────────────────────────────

PRICING_DOCS = [
    {"_id": "1_2023-01-01", "tier_id": "1", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": None, "monthly_price_usd": 0.00, "is_active": True},
    {"_id": "2_2023-01-01", "tier_id": "2", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 14.99, "is_active": False},
    {"_id": "2_2024-06-01", "tier_id": "2", "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None, "monthly_price_usd": 19.99, "is_active": True},
    {"_id": "3_2023-01-01", "tier_id": "3", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 39.99, "is_active": False},
    {"_id": "3_2024-06-01", "tier_id": "3", "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None, "monthly_price_usd": 49.99, "is_active": True},
]


def load_subscription_tier_pricing(db):
    info("Loading subscription_tier_pricing (hardcoded — no CSV)...")
    db["subscription_tier_pricing"].insert_many(PRICING_DOCS)
    ok("subscription_tier_pricing: 5 documents")


def create_subscription_tier_pricing_indexes(db):
    db["subscription_tier_pricing"].create_index(
        [("tier_id", ASCENDING), ("valid_from", ASCENDING)]
    )


# ── subscriptions ─────────────────────────────────────────────────────────────

def load_subscriptions(db):
    info("Loading subscriptions...")
    rows = load_csv("subscriptions.csv")
    docs = [{
        "_id": r["id"],
        "user_id": r["user_id"],
        "tier_id": r["tier_id"],
        "status": r["status"],
        "started_at": r["started_at"],
        "current_period_start": r["current_period_start"],
        "current_period_end": r["current_period_end"],
        "cancelled_at": r.get("cancelled_at") or None,
        "cancel_reason": r.get("cancel_reason") or None,
        "billing_cycle": r["billing_cycle"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    } for r in rows]
    insert_batches(db["subscriptions"], docs, "subscriptions")


def create_subscriptions_indexes(db):
    db["subscriptions"].create_index(
        [("user_id", ASCENDING), ("started_at", DESCENDING)]
    )
    db["subscriptions"].create_index([("tier_id", ASCENDING)])
    db["subscriptions"].create_index([("status", ASCENDING)])


# ── products ──────────────────────────────────────────────────────────────────

def load_products(db):
    info("Loading products...")
    rows = load_csv("products.csv")
    docs = []
    for r in rows:
        docs.append({
            "_id": r["id"],
            "name": r["name"],
            "slug": r["slug"],
            "product_type": r["product_type"],
            "description": r["description"],
            "price_usd": float(r["price_usd"]) if r.get("price_usd") else None,
            "currency": r["currency"],
            "is_active": r["is_active"].lower() == 'true' if isinstance(r["is_active"], str) else r["is_active"],
            "seller_id": r["seller_id"],
            # OPTIMISATION: parse JSON string → native BSON subdocument
            "attributes": parse_json_field(r.get("attributes"), {}),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    insert_batches(db["products"], docs, "products")


def create_products_indexes(db):
    db["products"].create_index([("slug", ASCENDING)], unique=True)
    db["products"].create_index([("product_type", ASCENDING)])
    db["products"].create_index([("is_active", ASCENDING)])
    db["products"].create_index(
        [("seller_id", ASCENDING), ("created_at", DESCENDING)]
    )
    db["products"].create_index(
        [("name", "text"), ("description", "text")],
        name="products_text_search",
    )


# ── invoices (with embedded lines + customer snapshot) ────────────────────────

def build_invoice_docs(db, rows_invoices, rows_lines):
    """
    Build invoice documents with embedded lines and customer snapshot.
    """
    info("  Building user lookup for customer snapshots...")
    users_lookup = {
        u["_id"]: u
        for u in db["users"].find(
            {},
            {"_id": 1, "full_name": 1, "email": 1,
             "country_code": 1, "city": 1},
        )
    }

    info("  Building product lookup for line item snapshots...")
    products_lookup = {
        p["_id"]: p
        for p in db["products"].find(
            {},
            {"_id": 1, "name": 1, "product_type": 1, "price_usd": 1},
        )
    }

    info("  Building invoice_lines index by invoice_id...")
    lines_by_invoice: dict[str, list] = {}
    for r in rows_lines:
        inv_id = r["invoice_id"]
        if inv_id not in lines_by_invoice:
            lines_by_invoice[inv_id] = []
        prod = products_lookup.get(r.get("product_id", ""), {})

        # Handle case where product_id might be empty string or None
        product_id = r.get("product_id") or None

        line_doc = {
            "_id": r["id"],
            "product_id": product_id,
            "description": r["description"],
            "quantity": int(r["quantity"]) if r.get("quantity") else 1,
            "unit_price_usd": float(r["unit_price_usd"]) if r.get("unit_price_usd") else None,
            "line_total_usd": float(r["line_total_usd"]) if r.get("line_total_usd") else None,
            "created_at": r["created_at"],
        }

        # Add product snapshot only if product exists
        if prod:
            line_doc.update({
                "product_name": prod.get("name"),
                "product_type": prod.get("product_type"),
                "product_price_usd": prod.get("price_usd"),
            })

        lines_by_invoice[inv_id].append(line_doc)

    info("  Assembling invoice documents...")
    docs = []
    for r in rows_invoices:
        inv_id = r["id"]
        user = users_lookup.get(r.get("user_id", ""), {})
        docs.append({
            "_id": inv_id,
            "user_id": r["user_id"],
            "customer": {
                "full_name": user.get("full_name"),
                "email": user.get("email"),
                "country_code": user.get("country_code"),
                "city": user.get("city"),
            },
            "invoice_type": r["invoice_type"],
            "status": r["status"],
            "subtotal_usd": float(r["subtotal_usd"]) if r.get("subtotal_usd") else None,
            "tax_usd": float(r["tax_usd"]) if r.get("tax_usd") else None,
            "discount_usd": float(r["discount_usd"]) if r.get("discount_usd") else None,
            "total_usd": float(r["total_usd"]) if r.get("total_usd") else None,
            "subscription_id": r.get("subscription_id") or None,
            "billing_period_start": r.get("billing_period_start") or None,
            "billing_period_end": r.get("billing_period_end") or None,
            "paid_at": r.get("paid_at") or None,
            "due_at": r.get("due_at") or None,
            "created_at": r["created_at"],
            "lines": lines_by_invoice.get(inv_id, []),
        })
    return docs


def load_invoices(db):
    info("Loading invoices (with embedded lines + customer snapshot)...")
    rows_sub = load_csv("subscription_invoices.csv")
    rows_mkt = load_csv("marketplace_invoices.csv")
    rows_all = rows_sub + rows_mkt
    info(f"  {len(rows_sub):,} subscription + {len(rows_mkt):,} marketplace "
         f"= {len(rows_all):,} total invoices")

    lines_sub = load_csv("subscription_invoice_lines.csv")
    lines_mkt = load_csv("marketplace_invoice_lines.csv")
    rows_lines = lines_sub + lines_mkt
    info(f"  {len(rows_lines):,} invoice lines to embed")

    docs = build_invoice_docs(db, rows_all, rows_lines)
    insert_batches(db["invoices"], docs, "invoices (with embedded lines)")


def create_invoices_indexes(db):
    db["invoices"].create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)]
    )
    db["invoices"].create_index([("status", ASCENDING)])
    db["invoices"].create_index([("invoice_type", ASCENDING)])
    db["invoices"].create_index([("subscription_id", ASCENDING)])
    db["invoices"].create_index([("created_at", DESCENDING)])


# ── orders (with embedded order_items + product snapshot) ─────────────────────

def build_order_docs(db, rows_orders, rows_items):
    info("  Building product lookup for order item snapshots...")
    products_lookup = {
        p["_id"]: p
        for p in db["products"].find(
            {},
            {"_id": 1, "name": 1, "product_type": 1, "price_usd": 1},
        )
    }

    info("  Building order_items index by order_id...")
    items_by_order: dict[str, list] = {}
    for r in rows_items:
        oid = r["order_id"]
        if oid not in items_by_order:
            items_by_order[oid] = []
        prod = products_lookup.get(r.get("product_id", ""), {})

        item_doc = {
            "_id": r["id"],
            "product_id": r["product_id"],
            "quantity": int(r["quantity"]) if r.get("quantity") else 1,
            "unit_price_usd": float(r["unit_price_usd"]) if r.get("unit_price_usd") else None,
            "line_total_usd": float(r["line_total_usd"]) if r.get("line_total_usd") else None,
            "fulfilment_status": r["fulfilment_status"],
            "created_at": r["created_at"],
        }

        # Add product snapshot
        if prod:
            item_doc.update({
                "product_name": prod.get("name"),
                "product_type": prod.get("product_type"),
                "product_price_usd": prod.get("price_usd"),
            })

        items_by_order[oid].append(item_doc)

    info("  Assembling order documents...")
    docs = []
    for r in rows_orders:
        docs.append({
            "_id": r["id"],
            "user_id": r["user_id"],
            "invoice_id": r["invoice_id"],
            "status": r["status"],
            "shipping_name": r.get("shipping_name") or None,
            "shipping_address": r.get("shipping_address") or None,
            "shipping_city": r.get("shipping_city") or None,
            "shipping_country": r.get("shipping_country") or None,
            "shipping_postal": r.get("shipping_postal") or None,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "items": items_by_order.get(r["id"], []),
        })
    return docs


def load_orders(db):
    info("Loading orders (with embedded order_items + product snapshot)...")
    rows_orders = load_csv("orders.csv")
    rows_items = load_csv("order_items.csv")
    info(f"  {len(rows_orders):,} orders, {len(rows_items):,} order items to embed")
    docs = build_order_docs(db, rows_orders, rows_items)
    insert_batches(db["orders"], docs, "orders (with embedded items)")


def create_orders_indexes(db):
    db["orders"].create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)]
    )
    db["orders"].create_index([("invoice_id", ASCENDING)])
    db["orders"].create_index([("status", ASCENDING)])
    db["orders"].create_index([("items.product_id", ASCENDING)])


# ── sessions ──────────────────────────────────────────────────────────────────

def load_sessions(db):
    info("Loading sessions (native cart array)...")
    rows = load_csv("sessions.csv")
    docs = []
    for r in rows:
        # Parse cart - handle both string and already-parsed cases
        cart = r.get("cart", "[]")
        if isinstance(cart, str):
            cart = parse_json_field(cart, [])

        docs.append({
            "_id": r["id"],
            "user_id": r["user_id"],
            "cart": cart,
            "ip_address": r.get("ip_address") or None,
            "user_agent": r.get("user_agent") or None,
            "created_at": r["created_at"],
            "last_active_at": r["last_active_at"],
            "expires_at": r["expires_at"],
        })
    insert_batches(db["sessions"], docs, "sessions")


def create_sessions_indexes(db):
    db["sessions"].create_index(
        [("user_id", ASCENDING), ("last_active_at", DESCENDING)]
    )
    db["sessions"].create_index([("expires_at", ASCENDING)])


# ── events ────────────────────────────────────────────────────────────────────

def load_events(db):
    info("Loading events (native metadata subdocument)...")
    rows = load_csv("events.csv")

    # Also load events_q8.csv if it exists and combine
    rows_q8 = load_csv("events_q8.csv")
    if rows_q8:
        info(f"  Found {len(rows_q8)} additional events from events_q8.csv")
        rows.extend(rows_q8)

    docs = []
    for r in rows:
        # Parse metadata - handle both string and already-parsed cases
        metadata = r.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = parse_json_field(metadata, {})

        docs.append({
            "_id": r["id"],
            "user_id": r["user_id"],
            "event_type": r["event_type"],
            "product_id": r.get("product_id") or None,
            "session_id": r.get("session_id") or None,
            "metadata": metadata,
            "occurred_at": r["occurred_at"],
        })
    insert_batches(db["events"], docs, "events")


def create_events_indexes(db):
    db["events"].create_index(
        [("user_id", ASCENDING), ("occurred_at", DESCENDING)]
    )
    db["events"].create_index([("event_type", ASCENDING)])
    db["events"].create_index([("product_id", ASCENDING)])
    db["events"].create_index([("occurred_at", DESCENDING)])


# ══════════════════════════════════════════════════════════════════════════════
# INDEX CREATION — all collections
# ══════════════════════════════════════════════════════════════════════════════

def create_all_indexes(db):
    info("Creating indexes...")
    create_users_indexes(db)
    create_seller_profiles_indexes(db)
    create_subscription_tier_pricing_indexes(db)
    create_subscriptions_indexes(db)
    create_products_indexes(db)
    create_invoices_indexes(db)
    create_orders_indexes(db)
    create_sessions_indexes(db)
    create_events_indexes(db)
    ok("All indexes created.")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_COLLECTIONS = [
    "users", "seller_profiles", "subscription_tiers",
    "subscription_tier_pricing", "subscriptions", "products",
    "invoices", "orders", "sessions", "events",
]

ELIMINATED_COLLECTIONS = ["invoice_lines", "order_items"]


def verify(db):
    print(f"\n{'═' * 55}")
    print("  Verification — Optimised Collection Counts")
    print(f"{'═' * 55}")
    all_ok = True
    for name in EXPECTED_COLLECTIONS:
        count = db[name].count_documents({})
        if count > 0:
            ok(f"{name:<30} {count:>10,} documents")
        else:
            fail(f"{name:<30} EMPTY")
            all_ok = False

    print(f"\n  Collections eliminated by embedding:")
    for name in ELIMINATED_COLLECTIONS:
        exists = name in db.list_collection_names()
        if not exists:
            ok(f"{name:<30} correctly absent (embedded)")
        else:
            count = db[name].count_documents({})
            warn(f"{name:<30} still exists ({count:,} docs) — "
                 f"should be embedded in parent")

    # Spot-check: verify a random invoice has embedded lines
    sample = db["invoices"].find_one({"lines": {"$exists": True, "$ne": []}})
    if sample:
        ok(f"Invoice embedding verified — sample has {len(sample['lines'])} line(s)")
    else:
        fail("No invoices found with embedded lines — check loader")

    # Spot-check: verify a random session has native cart array
    sample_session = db["sessions"].find_one({"cart": {"$type": "array"}})
    if sample_session:
        ok("Session cart verified — stored as native BSON array")
    else:
        warn("No sessions found with native cart array — may still be JSON strings")

    # Spot-check: verify a random order has embedded items
    sample_order = db["orders"].find_one({"items": {"$exists": True}})
    if sample_order:
        ok(f"Order embedding verified — sample has {len(sample_order.get('items', []))} item(s)")
    else:
        fail("No orders found with embedded items — check loader")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB optimised schema loader"
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop all optimised collections before loading (safe re-run)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify collection counts — do not load data",
    )
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  MongoDB Optimised Schema Loader")
    print("═" * 55)
    print("  Schema changes vs naive:")
    print("  • invoices:     embedded lines + customer snapshot")
    print("  • invoice_lines: ELIMINATED (embedded in invoices)")
    print("  • orders:       embedded order_items + product snapshots")
    print("  • order_items:  ELIMINATED (embedded in orders)")
    print("  • sessions:     cart as native BSON array")
    print("  • events:       metadata as native BSON subdocument")
    print("  • products:     attributes as native BSON subdocument")
    print("  • users:        preferences as native BSON subdocument")
    print("  • subscriptions: compound index (user_id, started_at DESC)")
    print()

    db = get_db()

    if args.verify:
        verify(db)
        return

    if args.drop:
        warn("Dropping all optimised collections...")
        for name in EXPECTED_COLLECTIONS + ELIMINATED_COLLECTIONS:
            db.drop_collection(name)
        ok("Collections dropped.")

    total_start = time.perf_counter()

    # Load in dependency order
    load_users(db)
    load_seller_profiles(db)
    load_subscription_tiers(db)
    load_subscription_tier_pricing(db)
    load_subscriptions(db)
    load_products(db)
    load_invoices(db)  # depends on users + products
    load_orders(db)  # depends on products
    load_sessions(db)
    load_events(db)

    create_all_indexes(db)

    total_elapsed = time.perf_counter() - total_start

    verify(db)

    print(f"\n{'═' * 55}")
    ok(f"Optimised load complete in {total_elapsed:.1f}s "
       f"({total_elapsed / 60:.1f} min)")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()