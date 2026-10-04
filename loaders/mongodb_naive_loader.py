"""
loaders/mongodb_naive_loader.py — MongoDB Naive Loader
=======================================================
Loads all 12 tables into MongoDB as flat collections that mirror
the PostgreSQL schema as literally as possible.

Naive constraints (intentional — do not change):
  - No embedding, no denormalisation
  - Foreign keys remain as plain string ID fields
  - cart in sessions is stored as a JSON string (not a native array)
  - No compound or specialised indexes beyond FK mirrors

Data sources:
  - 10 tables loaded from CSV files in data/
  - subscription_tiers and subscription_tier_pricing are hardcoded
    inline — they were seeded directly via schema.sql and have no
    corresponding CSV files

The PostgreSQL invoices table is split across two CSV files
(marketplace_invoices.csv, subscription_invoices.csv) purely as
a generator artifact — both are loaded into one invoices collection,
which directly mirrors the single PostgreSQL table.
Same applies to invoice_lines.

Usage:
    python loaders/mongodb_naive_loader.py
    python loaders/mongodb_naive_loader.py --drop    # drop collections first

    db.products.createIndex({ name: "text", description: "text" })
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
BATCH_SIZE = 1_000   # documents per insert_many call

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}"); sys.exit(1)
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── seed data (sourced from schema.sql — no CSV files exist for these) ────────

SUBSCRIPTION_TIERS = [
    {
        "_id": "1",
        "name": "Free",
        "description": "Basic software access, up to 5 marketplace purchases/year",
        "features": '{"seats": 1, "api_access": false, "priority_support": false, "marketplace_purchases_per_year": 5, "apps": {"CanvasEditor": "free", "VideoSuite": null}}',
    },
    {
        "_id": "2",
        "name": "Pro",
        "description": "Full software, unlimited purchases, early access",
        "features": '{"seats": 1, "api_access": false, "priority_support": false, "marketplace_purchases_per_year": -1, "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"}}',
    },
    {
        "_id": "3",
        "name": "Business",
        "description": "Everything in Pro plus team seats, API access, priority support",
        "features": '{"seats": 10, "api_access": true, "priority_support": true, "marketplace_purchases_per_year": -1, "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"}}',
    },
]

SUBSCRIPTION_TIER_PRICING = [
    # tier_id is the FK field — stored as a plain string (naive)
    # PostgreSQL uses a composite PK (tier_id, valid_from) — MongoDB requires
    # a single _id, so we construct a stable synthetic one from both PK parts
    {"_id": "1_2023-01-01T00:00:00+00:00", "tier_id": "1", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": "",                         "monthly_price_usd": "0.00",  "is_active": "True"},
    {"_id": "2_2023-01-01T00:00:00+00:00", "tier_id": "2", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": "14.99", "is_active": "False"},
    {"_id": "2_2024-06-01T00:00:00+00:00", "tier_id": "2", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": "",                         "monthly_price_usd": "19.99", "is_active": "True"},
    {"_id": "3_2023-01-01T00:00:00+00:00", "tier_id": "3", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": "39.99", "is_active": "False"},
    {"_id": "3_2024-06-01T00:00:00+00:00", "tier_id": "3", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": "",                         "monthly_price_usd": "49.99", "is_active": "True"},
]

# ── connection ────────────────────────────────────────────────────────────────

def get_mongo_db():
    user     = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    db_name  = os.getenv("MONGO_DB_NAIVE", "dissertation")
    client   = MongoClient(
        f"mongodb://{user}:{password}@localhost:27017/",
        serverSelectionTimeoutMS=30_000,
        socketTimeoutMS=60_000,
    )
    return client[db_name]

# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_csv(filename: str) -> list[dict]:
    """Read a CSV from DATA_DIR and return rows as plain string dicts."""
    path = DATA_DIR / filename
    if not path.exists():
        fail(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare_doc(row: dict) -> dict:
    """
    Convert a CSV row dict into a MongoDB document.
    - Renames 'id' → '_id' so MongoDB uses the existing UUID as the primary key.
      This mirrors the relational PK and makes FK lookups consistent.
    - All other values remain as strings (naive — no type casting).
    """
    doc = dict(row)
    if "id" in doc:
        doc["_id"] = doc.pop("id")
    return doc

# ── collection loader ─────────────────────────────────────────────────────────

def load_collection(db, collection_name: str, docs: list[dict], drop: bool) -> int:
    """
    Insert a list of documents into a MongoDB collection.
    Returns the number of documents inserted.
    """
    col = db[collection_name]

    if drop:
        col.drop()
        info(f"Dropped collection: {collection_name}")

    if not docs:
        warn(f"{collection_name}: no documents to insert, skipping")
        return 0

    total    = len(docs)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        col.insert_many(batch, ordered=False)
        inserted += len(batch)
        print(f"\r    {collection_name}: {inserted}/{total}", end="", flush=True)

    print()
    ok(f"{collection_name}: {inserted} documents inserted")
    return inserted


def docs_from_csv(csv_files: list[str]) -> list[dict]:
    rows = []
    for filename in csv_files:
        rows.extend(read_csv(filename))
    return [prepare_doc(r) for r in rows]

# ── index creation ────────────────────────────────────────────────────────────

def create_indexes(db):
    """
    Create indexes that mirror the PostgreSQL FK and query indexes.
    Single-field indexes only — no compound or specialised indexes,
    which would cross into optimised territory.
    The one exception is events (user_id, occurred_at DESC) which mirrors
    the composite index already present in the PostgreSQL schema.
    """
    info("Creating indexes...")

    indexes = [
        # users
        ("users",                     [("email",               ASCENDING)],  {"unique": True}),
        ("users",                     [("country_code",        ASCENDING)],  {}),
        ("users",                     [("created_at",          ASCENDING)],  {}),
        # seller_profiles
        ("seller_profiles",           [("is_verified",         ASCENDING)],  {}),
        ("seller_profiles",           [("country_code",        ASCENDING)],  {}),
        # subscription_tier_pricing
        ("subscription_tier_pricing", [("tier_id",             ASCENDING)],  {}),
        ("subscription_tier_pricing", [("valid_from",          ASCENDING)],  {}),
        # subscriptions
        ("subscriptions",             [("user_id",             ASCENDING)],  {}),
        ("subscriptions",             [("tier_id",             ASCENDING)],  {}),
        ("subscriptions",             [("status",              ASCENDING)],  {}),
        ("subscriptions",             [("current_period_end",  ASCENDING)],  {}),
        # products
        ("products",                  [("product_type",        ASCENDING)],  {}),
        ("products",                  [("seller_id",           ASCENDING)],  {}),
        ("products",                  [("price_usd",           ASCENDING)],  {}),
        ("products",                  [("is_active",           ASCENDING)],  {}),
        # invoices
        ("invoices",                  [("user_id",             ASCENDING)],  {}),
        ("invoices",                  [("subscription_id",     ASCENDING)],  {}),
        ("invoices",                  [("status",              ASCENDING)],  {}),
        ("invoices",                  [("created_at",          ASCENDING)],  {}),
        ("invoices",                  [("invoice_type",        ASCENDING)],  {}),
        # invoice_lines
        ("invoice_lines",             [("invoice_id",          ASCENDING)],  {}),
        ("invoice_lines",             [("product_id",          ASCENDING)],  {}),
        # orders
        ("orders",                    [("user_id",             ASCENDING)],  {}),
        ("orders",                    [("invoice_id",          ASCENDING)],  {}),
        ("orders",                    [("status",              ASCENDING)],  {}),
        ("orders",                    [("created_at",          ASCENDING)],  {}),
        # order_items
        ("order_items",               [("order_id",            ASCENDING)],  {}),
        ("order_items",               [("product_id",          ASCENDING)],  {}),
        # sessions
        ("sessions",                  [("user_id",             ASCENDING)],  {}),
        ("sessions",                  [("expires_at",          ASCENDING)],  {}),
        ("sessions",                  [("last_active_at",      ASCENDING)],  {}),
        # events — composite mirrors PostgreSQL idx_events_user_time exactly
        ("events",                    [("user_id",    ASCENDING),
                                       ("occurred_at", DESCENDING)],          {}),
        ("events",                    [("event_type",  ASCENDING)],           {}),
        ("events",                    [("product_id",  ASCENDING)],           {}),
        ("events",                    [("occurred_at", DESCENDING)],          {}),
    ]

    for collection_name, key_spec, kwargs in indexes:
        db[collection_name].create_index(key_spec, **kwargs)

    ok("All indexes created")

# ── collection manifest ───────────────────────────────────────────────────────

# Each entry is (collection_name, source) where source is either:
#   - a list of CSV filenames  → loaded from data/
#   - None                     → loaded from inline seed data (SEED_DATA below)
COLLECTIONS = [
    ("users",                     ["users.csv"]),
    ("seller_profiles",           ["seller_profiles.csv"]),
    ("subscription_tiers",        None),
    ("subscription_tier_pricing", None),
    ("subscriptions",             ["subscriptions.csv"]),
    ("products",                  ["products.csv"]),
    ("invoices",                  ["subscription_invoices.csv",
                                   "marketplace_invoices.csv"]),
    ("invoice_lines",             ["subscription_invoice_lines.csv",
                                   "marketplace_invoice_lines.csv"]),
    ("orders",                    ["orders.csv"]),
    ("order_items",               ["order_items.csv"]),
    ("sessions",                  ["sessions.csv"]),
    ("events",                    ["events.csv"]),
]

SEED_DATA = {
    "subscription_tiers":        SUBSCRIPTION_TIERS,
    "subscription_tier_pricing": SUBSCRIPTION_TIER_PRICING,
}

# ── main ──────────────────────────────────────────────────────────────────────

def main(drop: bool):
    print("\n" + "═" * 55)
    print("  MongoDB Naive Loader")
    print("═" * 55)

    db = get_mongo_db()
    info(f"Connected to MongoDB — database: {db.name}")

    total_docs = 0
    wall_start = time.perf_counter()

    for collection_name, source in COLLECTIONS:
        if source is None:
            docs = SEED_DATA[collection_name]
        else:
            docs = docs_from_csv(source)
        n = load_collection(db, collection_name, docs, drop=drop)
        total_docs += n

    create_indexes(db)

    wall_elapsed = time.perf_counter() - wall_start
    print("\n" + "─" * 55)
    ok(f"Load complete — {total_docs:,} total documents in {wall_elapsed:.1f}s")
    print("─" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MongoDB naive loader")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all collections before loading (safe re-run)",
    )
    args = parser.parse_args()
    main(drop=args.drop)
