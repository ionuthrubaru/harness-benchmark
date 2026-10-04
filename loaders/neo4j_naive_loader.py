"""
loaders/neo4j_naive_loader.py — Phase 3: Neo4j Naive Load
==========================================================
Loads all 12 tables into Neo4j naive container (bolt://localhost:7687)
as flat nodes with properties mirroring the PostgreSQL schema.

Naive schema principle:
  Every table row becomes a node with all columns as flat properties.
  Relationships mirror FK structure exactly — no precomputed weights,
  no embeddings, no graph-native optimisations.
  JSONB columns (preferences, attributes, metadata, features, cart)
  are stored as JSON strings — Neo4j properties do not support nested maps.

Constraints and indexes:
  Uniqueness constraints are created on every primary key property.
  Additional property indexes are created on FK-equivalent properties
  (user_id, invoice_id, product_id, etc.) so that relationship creation
  and FK-style lookups are index-backed. This makes the naive schema a
  fair relational port, not a deliberately hobbled one.

Relationships created (mirrors FK structure):
  (SellerProfile)-[:PROFILE_OF]->(User)
  (Subscription)-[:ON_TIER]->(SubscriptionTier)
  (User)-[:HAS_SUBSCRIPTION]->(Subscription)
  (User)-[:HAS_INVOICE]->(Invoice)
  (Invoice)-[:FOR_SUBSCRIPTION]->(Subscription)   -- subscription invoices only
  (Invoice)-[:HAS_LINE]->(InvoiceLine)
  (User)-[:PLACED]->(Order)
  (Order)-[:CONTAINS]->(OrderItem)
  (OrderItem)-[:FOR_PRODUCT]->(Product)
  (User)-[:HAS_SESSION]->(Session)
  (User)-[:TRIGGERED]->(Event)
  (Event)-[:RELATES_TO]->(Product)                -- only where product_id is set

Usage:
    python loaders/neo4j_naive_loader.py
    python loaders/neo4j_naive_loader.py --dry-run   (count rows, no writes)
    python loaders/neo4j_naive_loader.py --wipe      (wipe DB before loading)
"""

import argparse
import csv
import json
import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
BATCH_SIZE  = 500

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")
def err(msg):  print(f"  {RED}✘ {msg}{RESET}")


# ── helpers ───────────────────────────────────────────────────────────────────

def csv_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def read_csv(filename: str) -> list[dict]:
    path = csv_path(filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_batches(driver, cypher: str, rows: list[dict], batch_size: int = BATCH_SIZE):
    """
    Execute a Cypher statement in batches using UNWIND.
    Each batch is a separate transaction. The Cypher must accept
    a parameter named $rows and iterate with UNWIND $rows AS row.
    """
    total = len(rows)
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        with driver.session() as session:
            session.execute_write(lambda tx, b=batch: tx.run(cypher, rows=b))
    return total


def run_query(driver, cypher: str, **params):
    """Execute a single Cypher statement (no UNWIND batching needed)."""
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(cypher, **params))


# ── hardcoded seed data ───────────────────────────────────────────────────────

SUBSCRIPTION_TIERS = [
    {"id": "1", "name": "Free",
     "description": "Basic software access, up to 5 marketplace purchases/year",
     "features": json.dumps({
         "seats": 1, "api_access": False, "priority_support": False,
         "marketplace_purchases_per_year": 5,
         "apps": {"CanvasEditor": "free", "VideoSuite": None},
     })},
    {"id": "2", "name": "Pro",
     "description": "Full software, unlimited purchases, early access",
     "features": json.dumps({
         "seats": 1, "api_access": False, "priority_support": False,
         "marketplace_purchases_per_year": -1,
         "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"},
     })},
    {"id": "3", "name": "Business",
     "description": "Everything in Pro plus team seats, API access, priority support",
     "features": json.dumps({
         "seats": 10, "api_access": True, "priority_support": True,
         "marketplace_purchases_per_year": -1,
         "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"},
     })},
]

SUBSCRIPTION_TIER_PRICING = [
    {"tier_id": "1", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": "0.00", "is_active": "true"},
    {"tier_id": "2", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00",
     "monthly_price_usd": "14.99", "is_active": "false"},
    {"tier_id": "2", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": "19.99", "is_active": "true"},
    {"tier_id": "3", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00",
     "monthly_price_usd": "39.99", "is_active": "false"},
    {"tier_id": "3", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": "49.99", "is_active": "true"},
]


# ── Step 1: constraints and indexes ──────────────────────────────────────────

def create_constraints_and_indexes(driver):
    """
    Create uniqueness constraints on all primary keys and property indexes
    on all FK-equivalent properties. Constraints implicitly create indexes.
    All use IF NOT EXISTS so re-runs are safe.
    """
    constraints = [
        # Primary key uniqueness constraints
        "CREATE CONSTRAINT user_pk IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT seller_pk IF NOT EXISTS FOR (n:SellerProfile) REQUIRE n.user_id IS UNIQUE",
        "CREATE CONSTRAINT tier_pk IF NOT EXISTS FOR (n:SubscriptionTier) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT subscription_pk IF NOT EXISTS FOR (n:Subscription) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT product_pk IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT invoice_pk IF NOT EXISTS FOR (n:Invoice) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT invoice_line_pk IF NOT EXISTS FOR (n:InvoiceLine) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT order_pk IF NOT EXISTS FOR (n:Order) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT order_item_pk IF NOT EXISTS FOR (n:OrderItem) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT session_pk IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT event_pk IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
    ]

    indexes = [
        # FK-equivalent property indexes for relationship creation and lookups
        "CREATE INDEX sub_user_id IF NOT EXISTS FOR (n:Subscription) ON (n.user_id)",
        "CREATE INDEX sub_tier_id IF NOT EXISTS FOR (n:Subscription) ON (n.tier_id)",
        "CREATE INDEX invoice_user_id IF NOT EXISTS FOR (n:Invoice) ON (n.user_id)",
        "CREATE INDEX invoice_sub_id IF NOT EXISTS FOR (n:Invoice) ON (n.subscription_id)",
        "CREATE INDEX invoice_type IF NOT EXISTS FOR (n:Invoice) ON (n.invoice_type)",
        "CREATE INDEX invoice_created_at IF NOT EXISTS FOR (n:Invoice) ON (n.created_at)",
        "CREATE INDEX invoice_status IF NOT EXISTS FOR (n:Invoice) ON (n.status)",
        "CREATE INDEX invoice_line_invoice_id IF NOT EXISTS FOR (n:InvoiceLine) ON (n.invoice_id)",
        "CREATE INDEX order_user_id IF NOT EXISTS FOR (n:Order) ON (n.user_id)",
        "CREATE INDEX order_item_order_id IF NOT EXISTS FOR (n:OrderItem) ON (n.order_id)",
        "CREATE INDEX order_item_product_id IF NOT EXISTS FOR (n:OrderItem) ON (n.product_id)",
        "CREATE INDEX session_user_id IF NOT EXISTS FOR (n:Session) ON (n.user_id)",
        "CREATE INDEX event_user_id IF NOT EXISTS FOR (n:Event) ON (n.user_id)",
        "CREATE INDEX event_occurred_at IF NOT EXISTS FOR (n:Event) ON (n.occurred_at)",
        "CREATE INDEX event_product_id IF NOT EXISTS FOR (n:Event) ON (n.product_id)",
        "CREATE INDEX tier_pricing_tier_id IF NOT EXISTS FOR (n:SubscriptionTierPricing) ON (n.tier_id)",
        "CREATE INDEX product_type IF NOT EXISTS FOR (n:Product) ON (n.product_type)",
    ]

    with driver.session() as session:
        for cypher in constraints + indexes:
            session.run(cypher)

    ok(f"Constraints and indexes created ({len(constraints)} constraints, "
       f"{len(indexes)} indexes)")


# ── Step 2: node loaders ──────────────────────────────────────────────────────

def load_subscription_tiers(driver, dry_run: bool) -> int:
    if dry_run:
        return len(SUBSCRIPTION_TIERS)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:SubscriptionTier {id: row.id})
    SET n += row
    """
    return run_batches(driver, cypher, SUBSCRIPTION_TIERS)


def load_subscription_tier_pricing(driver, dry_run: bool) -> int:
    if dry_run:
        return len(SUBSCRIPTION_TIER_PRICING)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:SubscriptionTierPricing {tier_id: row.tier_id, valid_from: row.valid_from})
    SET n += row
    """
    return run_batches(driver, cypher, SUBSCRIPTION_TIER_PRICING)


def load_users(driver, dry_run: bool) -> int:
    rows = read_csv("users.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:User {id: row.id})
    SET n.email          = row.email,
        n.full_name      = row.full_name,
        n.country_code   = row.country_code,
        n.city           = row.city,
        n.created_at     = row.created_at,
        n.last_login_at  = row.last_login_at,
        n.is_active      = row.is_active,
        n.preferences    = row.preferences
    """
    return run_batches(driver, cypher, rows)


def load_seller_profiles(driver, dry_run: bool) -> int:
    rows = read_csv("seller_profiles.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:SellerProfile {user_id: row.user_id})
    SET n.display_name  = row.display_name,
        n.legal_name    = row.legal_name,
        n.tax_id        = row.tax_id,
        n.payout_email  = row.payout_email,
        n.country_code  = row.country_code,
        n.is_verified   = row.is_verified,
        n.bio           = row.bio,
        n.total_sales   = row.total_sales,
        n.created_at    = row.created_at,
        n.updated_at    = row.updated_at
    """
    return run_batches(driver, cypher, rows)


def load_subscriptions(driver, dry_run: bool) -> int:
    rows = read_csv("subscriptions.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Subscription {id: row.id})
    SET n.user_id               = row.user_id,
        n.tier_id               = row.tier_id,
        n.status                = row.status,
        n.started_at            = row.started_at,
        n.current_period_start  = row.current_period_start,
        n.current_period_end    = row.current_period_end,
        n.cancelled_at          = row.cancelled_at,
        n.cancel_reason         = row.cancel_reason,
        n.billing_cycle         = row.billing_cycle,
        n.created_at            = row.created_at,
        n.updated_at            = row.updated_at
    """
    return run_batches(driver, cypher, rows)


def load_products(driver, dry_run: bool) -> int:
    rows = read_csv("products.csv")
    if dry_run:
        return len(rows)
    # Drop search_vector — PostgreSQL tsvector, meaningless in Neo4j
    for row in rows:
        row.pop("search_vector", None)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Product {id: row.id})
    SET n.name          = row.name,
        n.slug          = row.slug,
        n.product_type  = row.product_type,
        n.description   = row.description,
        n.price_usd     = row.price_usd,
        n.currency      = row.currency,
        n.is_active     = row.is_active,
        n.seller_id     = row.seller_id,
        n.attributes    = row.attributes,
        n.created_at    = row.created_at,
        n.updated_at    = row.updated_at
    """
    return run_batches(driver, cypher, rows)


def load_invoices(driver, dry_run: bool) -> int:
    """Merged from marketplace_invoices.csv + subscription_invoices.csv."""
    market_rows = read_csv("marketplace_invoices.csv")
    sub_rows    = read_csv("subscription_invoices.csv")
    rows        = market_rows + sub_rows
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Invoice {id: row.id})
    SET n.user_id               = row.user_id,
        n.invoice_type          = row.invoice_type,
        n.status                = row.status,
        n.subtotal_usd          = row.subtotal_usd,
        n.tax_usd               = row.tax_usd,
        n.discount_usd          = row.discount_usd,
        n.total_usd             = row.total_usd,
        n.subscription_id       = row.subscription_id,
        n.billing_period_start  = row.billing_period_start,
        n.billing_period_end    = row.billing_period_end,
        n.paid_at               = row.paid_at,
        n.due_at                = row.due_at,
        n.created_at            = row.created_at
    """
    return run_batches(driver, cypher, rows)


def load_invoice_lines(driver, dry_run: bool) -> int:
    """Merged from marketplace_invoice_lines.csv + subscription_invoice_lines.csv."""
    market_rows = read_csv("marketplace_invoice_lines.csv")
    sub_rows    = read_csv("subscription_invoice_lines.csv")
    rows        = market_rows + sub_rows
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:InvoiceLine {id: row.id})
    SET n.invoice_id    = row.invoice_id,
        n.product_id    = row.product_id,
        n.description   = row.description,
        n.quantity      = row.quantity,
        n.unit_price_usd = row.unit_price_usd,
        n.line_total_usd = row.line_total_usd,
        n.created_at    = row.created_at
    """
    return run_batches(driver, cypher, rows)


def load_orders(driver, dry_run: bool) -> int:
    rows = read_csv("orders.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Order {id: row.id})
    SET n.user_id           = row.user_id,
        n.invoice_id        = row.invoice_id,
        n.status            = row.status,
        n.shipping_name     = row.shipping_name,
        n.shipping_address  = row.shipping_address,
        n.shipping_city     = row.shipping_city,
        n.shipping_country  = row.shipping_country,
        n.shipping_postal   = row.shipping_postal,
        n.created_at        = row.created_at,
        n.updated_at        = row.updated_at
    """
    return run_batches(driver, cypher, rows)


def load_order_items(driver, dry_run: bool) -> int:
    rows = read_csv("order_items.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:OrderItem {id: row.id})
    SET n.order_id          = row.order_id,
        n.product_id        = row.product_id,
        n.quantity          = row.quantity,
        n.unit_price_usd    = row.unit_price_usd,
        n.line_total_usd    = row.line_total_usd,
        n.fulfilment_status = row.fulfilment_status,
        n.created_at        = row.created_at
    """
    return run_batches(driver, cypher, rows)


def load_sessions(driver, dry_run: bool) -> int:
    rows = read_csv("sessions.csv")
    if dry_run:
        return len(rows)
    # cart is a JSON string in the CSV — stored as-is (naive: no structural change)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Session {id: row.id})
    SET n.user_id       = row.user_id,
        n.cart          = row.cart,
        n.ip_address    = row.ip_address,
        n.user_agent    = row.user_agent,
        n.created_at    = row.created_at,
        n.last_active_at = row.last_active_at,
        n.expires_at    = row.expires_at
    """
    return run_batches(driver, cypher, rows)


def load_events(driver, dry_run: bool) -> int:
    rows = read_csv("events.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Event {id: row.id})
    SET n.user_id       = row.user_id,
        n.event_type    = row.event_type,
        n.product_id    = row.product_id,
        n.session_id    = row.session_id,
        n.metadata      = row.metadata,
        n.occurred_at   = row.occurred_at
    """
    return run_batches(driver, cypher, rows)


# ── Step 3: relationship loaders ──────────────────────────────────────────────
#
# Each relationship is created in a separate pass after all nodes exist.
# This avoids MERGE ordering issues and keeps each step simple.
# All MATCH calls are index-backed (constraints/indexes created in Step 1).

def create_seller_profile_relationships(driver, dry_run: bool) -> int:
    rows = read_csv("seller_profiles.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MATCH (sp:SellerProfile {user_id: row.user_id})
    MATCH (u:User {id: row.user_id})
    MERGE (sp)-[:PROFILE_OF]->(u)
    """
    return run_batches(driver, cypher, rows)


def create_subscription_relationships(driver, dry_run: bool) -> int:
    rows = read_csv("subscriptions.csv")
    if dry_run:
        return len(rows) * 2  # two relationships per row
    # (User)-[:HAS_SUBSCRIPTION]->(Subscription)
    cypher_user = """
    UNWIND $rows AS row
    MATCH (u:User {id: row.user_id})
    MATCH (s:Subscription {id: row.id})
    MERGE (u)-[:HAS_SUBSCRIPTION]->(s)
    """
    # (Subscription)-[:ON_TIER]->(SubscriptionTier)
    cypher_tier = """
    UNWIND $rows AS row
    MATCH (s:Subscription {id: row.id})
    MATCH (t:SubscriptionTier {id: row.tier_id})
    MERGE (s)-[:ON_TIER]->(t)
    """
    run_batches(driver, cypher_user, rows)
    run_batches(driver, cypher_tier, rows)
    return len(rows) * 2


def create_invoice_relationships(driver, dry_run: bool) -> int:
    market_rows = read_csv("marketplace_invoices.csv")
    sub_rows    = read_csv("subscription_invoices.csv")
    all_rows    = market_rows + sub_rows
    if dry_run:
        return len(all_rows)

    # (User)-[:HAS_INVOICE]->(Invoice) — all invoices
    cypher_user = """
    UNWIND $rows AS row
    MATCH (u:User {id: row.user_id})
    MATCH (i:Invoice {id: row.id})
    MERGE (u)-[:HAS_INVOICE]->(i)
    """
    # (Invoice)-[:FOR_SUBSCRIPTION]->(Subscription) — subscription invoices only
    cypher_sub = """
    UNWIND $rows AS row
    MATCH (i:Invoice {id: row.id})
    MATCH (s:Subscription {id: row.subscription_id})
    MERGE (i)-[:FOR_SUBSCRIPTION]->(s)
    """
    run_batches(driver, cypher_user, all_rows)
    run_batches(driver, cypher_sub,
                [r for r in sub_rows if r.get("subscription_id")])
    return len(all_rows)


def create_invoice_line_relationships(driver, dry_run: bool) -> int:
    market_rows = read_csv("marketplace_invoice_lines.csv")
    sub_rows    = read_csv("subscription_invoice_lines.csv")
    rows        = market_rows + sub_rows
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MATCH (i:Invoice {id: row.invoice_id})
    MATCH (l:InvoiceLine {id: row.id})
    MERGE (i)-[:HAS_LINE]->(l)
    """
    return run_batches(driver, cypher, rows)


def create_order_relationships(driver, dry_run: bool) -> int:
    rows = read_csv("orders.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MATCH (u:User {id: row.user_id})
    MATCH (o:Order {id: row.id})
    MERGE (u)-[:PLACED]->(o)
    """
    return run_batches(driver, cypher, rows)


def create_order_item_relationships(driver, dry_run: bool) -> int:
    rows = read_csv("order_items.csv")
    if dry_run:
        return len(rows) * 2
    # (Order)-[:CONTAINS]->(OrderItem)
    cypher_order = """
    UNWIND $rows AS row
    MATCH (o:Order {id: row.order_id})
    MATCH (oi:OrderItem {id: row.id})
    MERGE (o)-[:CONTAINS]->(oi)
    """
    # (OrderItem)-[:FOR_PRODUCT]->(Product)
    cypher_product = """
    UNWIND $rows AS row
    MATCH (oi:OrderItem {id: row.id})
    MATCH (p:Product {id: row.product_id})
    MERGE (oi)-[:FOR_PRODUCT]->(p)
    """
    run_batches(driver, cypher_order, rows)
    run_batches(driver, cypher_product, rows)
    return len(rows) * 2


def create_session_relationships(driver, dry_run: bool) -> int:
    rows = read_csv("sessions.csv")
    if dry_run:
        return len(rows)
    cypher = """
    UNWIND $rows AS row
    MATCH (u:User {id: row.user_id})
    MATCH (s:Session {id: row.id})
    MERGE (u)-[:HAS_SESSION]->(s)
    """
    return run_batches(driver, cypher, rows)


def create_event_relationships(driver, dry_run: bool) -> int:
    """
    Two relationship types:
      (User)-[:TRIGGERED]->(Event)          — all events
      (Event)-[:RELATES_TO]->(Product)      — only where product_id is non-empty
    Largest step — 6.3M events, processed in batches of 500.
    """
    rows = read_csv("events.csv")
    if dry_run:
        return len(rows)

    cypher_user = """
    UNWIND $rows AS row
    MATCH (u:User {id: row.user_id})
    MATCH (e:Event {id: row.id})
    MERGE (u)-[:TRIGGERED]->(e)
    """
    product_rows = [r for r in rows if r.get("product_id")]
    cypher_product = """
    UNWIND $rows AS row
    MATCH (e:Event {id: row.id})
    MATCH (p:Product {id: row.product_id})
    MERGE (e)-[:RELATES_TO]->(p)
    """
    run_batches(driver, cypher_user, rows)
    run_batches(driver, cypher_product, product_rows)
    return len(rows)


# ── orchestration ─────────────────────────────────────────────────────────────

NODE_STEPS = [
    ("SubscriptionTier nodes",         load_subscription_tiers),
    ("SubscriptionTierPricing nodes",  load_subscription_tier_pricing),
    ("User nodes",                     load_users),
    ("SellerProfile nodes",            load_seller_profiles),
    ("Subscription nodes",             load_subscriptions),
    ("Product nodes",                  load_products),
    ("Invoice nodes",                  load_invoices),
    ("InvoiceLine nodes",              load_invoice_lines),
    ("Order nodes",                    load_orders),
    ("OrderItem nodes",                load_order_items),
    ("Session nodes",                  load_sessions),
    ("Event nodes",                    load_events),
]

REL_STEPS = [
    ("SellerProfile→User rels",        create_seller_profile_relationships),
    ("Subscription rels",              create_subscription_relationships),
    ("Invoice rels",                   create_invoice_relationships),
    ("InvoiceLine rels",               create_invoice_line_relationships),
    ("Order rels",                     create_order_relationships),
    ("OrderItem rels",                 create_order_item_relationships),
    ("Session rels",                   create_session_relationships),
    ("Event rels",                     create_event_relationships),
]


def main():
    parser = argparse.ArgumentParser(description="Neo4j naive loader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count rows only — no writes to Neo4j")
    parser.add_argument("--wipe", action="store_true",
                        help="Delete all nodes and relationships before loading")
    args = parser.parse_args()

    port = int(os.getenv("NEO4J_NAIVE_PORT", 7687))

    print(f"\n{'=' * 55}")
    print(f"  Neo4j Naive Loader  (bolt://localhost:{port})")
    print(f"{'=' * 55}")

    if args.dry_run:
        warn("DRY RUN — no data will be written")

    driver = get_driver(port=port)
    try:
        driver.verify_connectivity()
        ok(f"Connected to Neo4j on port {port}")
    except Exception as e:
        err(f"Cannot connect to Neo4j: {e}")
        sys.exit(1)

    if args.wipe and not args.dry_run:
        warn("Wiping all nodes and relationships ...")
        # Delete in batches to avoid memory pressure on large graphs
        with driver.session() as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS cnt"
                )
                cnt = result.single()["cnt"]
                if cnt == 0:
                    break
        ok("Database wiped")

    wall_start = time.perf_counter()

    if not args.dry_run:
        t0 = time.perf_counter()
        create_constraints_and_indexes(driver)
        # Wait briefly for indexes to come online before loading
        time.sleep(2)

    print(f"\n  {'─' * 45}")
    print("  Loading nodes ...")
    print(f"  {'─' * 45}")

    total_rows = 0
    for name, fn in NODE_STEPS:
        t0 = time.perf_counter()
        try:
            n = fn(driver, dry_run=args.dry_run)
            elapsed = time.perf_counter() - t0
            ok(f"{name:<38} {n:>9,}   {elapsed:.1f}s")
            total_rows += n
        except Exception as e:
            err(f"{name:<38} FAILED — {e}")
            driver.close()
            sys.exit(1)

    print(f"\n  {'─' * 45}")
    print("  Creating relationships ...")
    print(f"  {'─' * 45}")

    for name, fn in REL_STEPS:
        t0 = time.perf_counter()
        try:
            n = fn(driver, dry_run=args.dry_run)
            elapsed = time.perf_counter() - t0
            ok(f"{name:<38} {n:>9,}   {elapsed:.1f}s")
        except Exception as e:
            err(f"{name:<38} FAILED — {e}")
            driver.close()
            sys.exit(1)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n{'─' * 55}")
    action = "counted" if args.dry_run else "loaded"
    ok(f"Total {action}: {total_rows:,} rows in {wall_elapsed:.1f}s")
    print(f"{'=' * 55}\n")

    driver.close()


if __name__ == "__main__":
    main()