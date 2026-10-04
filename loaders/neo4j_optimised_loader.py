"""
loaders/neo4j_optimised_loader.py — Phase 3: Neo4j Optimised Load
==================================================================
Loads all 12 tables into Neo4j optimised container (bolt://localhost:7688).

Reuses all node and relationship loaders from neo4j_naive_loader.py —
the node/relationship structure is identical. Only the following differ:

Schema changes vs naive:
  1. Session.cart stored as a native Neo4j list property (not a JSON string).
     Parsed from the CSV JSON string at load time. Q3 can access individual
     cart items without json.loads() in the benchmark driver.

  2. Full-text index on Product(name, description) — enables native BM25
     full-text search in Q5 via db.index.fulltext.queryNodes().

  3. Composite index on Event(user_id, occurred_at) — turns Q6's range
     scan into a single index-backed lookup instead of two separate indexes.

  4. ALSO_BOUGHT relationships between Products, with properties:
       count       — number of orders in which both products appear together
       confidence  — count / total_orders_containing(source_product)
     Computed in Python from order_items.csv before any Neo4j writes.
     Q4 benchmark: single-hop ALSO_BOUGHT traversal, ordered by count DESC.

All other constraints and indexes from the naive schema are retained —
the optimised schema is strictly a superset of the naive schema.

Usage:
    python loaders/neo4j_optimised_loader.py
    python loaders/neo4j_optimised_loader.py --dry-run
    python loaders/neo4j_optimised_loader.py --wipe
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.neo4j.neo4j_conn import get_driver

# Re-use every node and relationship loader from the naive loader.
# Only load_sessions and the constraint/index step are overridden here.
from neo4j_naive_loader import (
    DATA_DIR, BATCH_SIZE,
    ok, info, warn, err,
    read_csv, run_batches, run_query,
    SUBSCRIPTION_TIERS, SUBSCRIPTION_TIER_PRICING,
    load_subscription_tiers, load_subscription_tier_pricing,
    load_users, load_seller_profiles, load_subscriptions,
    load_products, load_invoices, load_invoice_lines,
    load_orders, load_order_items, load_events,
    create_seller_profile_relationships, create_subscription_relationships,
    create_invoice_relationships, create_invoice_line_relationships,
    create_order_relationships, create_order_item_relationships,
    create_session_relationships, create_event_relationships,
)

load_dotenv()


# ── Step 1: constraints and indexes (superset of naive) ───────────────────────

def create_constraints_and_indexes(driver):
    """
    All naive constraints and indexes, plus:
      - Full-text index on Product(name, description)  → Q5
      - Composite index on Event(user_id, occurred_at) → Q6 (replaces two
        separate naive indexes with one compound index)
    """
    constraints = [
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
        # All naive FK-equivalent indexes retained
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
        "CREATE INDEX tier_pricing_tier_id IF NOT EXISTS FOR (n:SubscriptionTierPricing) ON (n.tier_id)",
        "CREATE INDEX product_type IF NOT EXISTS FOR (n:Product) ON (n.product_type)",
        # Optimised: composite index replaces two naive single-property indexes on Event
        "CREATE INDEX event_user_time IF NOT EXISTS FOR (n:Event) ON (n.user_id, n.occurred_at)",
    ]

    full_text = [
        # Full-text index on Product name + description for Q5.
        # Enables db.index.fulltext.queryNodes() with BM25 scoring.
        # name is weighted higher via the Cypher query at benchmark time.
        """
        CREATE FULLTEXT INDEX product_fulltext IF NOT EXISTS
        FOR (n:Product) ON EACH [n.name, n.description]
        OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}
        """,
    ]

    with driver.session() as session:
        for cypher in constraints + indexes + full_text:
            session.run(cypher)

    ok(f"Constraints and indexes created ({len(constraints)} constraints, "
       f"{len(indexes)} indexes, {len(full_text)} full-text index)")


# ── Overridden session loader ─────────────────────────────────────────────────

def load_sessions(driver, dry_run: bool) -> int:
    """
    Optimised override: cart stored as a native Neo4j list property.
    The CSV contains a JSON array string — parsed at load time so the
    benchmark can access cart items directly without json.loads().

    Each cart item is stored as a JSON string element within the list
    (Neo4j lists can hold strings but not nested maps), preserving the
    item structure while eliminating the outer JSON deserialisation.
    """
    rows = read_csv("sessions.csv")
    if dry_run:
        return len(rows)

    parsed = []
    for row in rows:
        cart_raw = row.pop("cart", "[]") or "[]"
        try:
            cart_items = json.loads(cart_raw)
        except (json.JSONDecodeError, TypeError):
            cart_items = []
        # Store each item as a JSON string within the list
        row["cart"] = [
            item if isinstance(item, str) else json.dumps(item)
            for item in cart_items
        ]
        parsed.append(row)

    cypher = """
    UNWIND $rows AS row
    MERGE (n:Session {id: row.id})
    SET n.user_id        = row.user_id,
        n.cart           = row.cart,
        n.ip_address     = row.ip_address,
        n.user_agent     = row.user_agent,
        n.created_at     = row.created_at,
        n.last_active_at = row.last_active_at,
        n.expires_at     = row.expires_at
    """
    return run_batches(driver, cypher, parsed)


# ── ALSO_BOUGHT computation and loading ───────────────────────────────────────

def build_copurchase_data() -> list[dict]:
    """
    Read order_items.csv in memory. Group product_ids by order_id.
    For each order with 2+ products, compute every pair (A, B):
      - Increment copurchase_count[A][B] and [B][A]
      - Track total orders containing each product for confidence calculation

    Returns a flat list of dicts ready for UNWIND batching:
      [{"from_id": A, "to_id": B, "count": n, "confidence": f}, ...]
    """
    info("Computing ALSO_BOUGHT data from order_items.csv ...")
    rows = read_csv("order_items.csv")

    # Group products by order
    order_products: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        order_products[row["order_id"]].append(row["product_id"])

    # Count co-occurrences and total order appearances
    copurchase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_orders: dict[str, int] = defaultdict(int)

    for products in order_products.values():
        unique_products = list(set(products))  # deduplicate within same order
        for pid in unique_products:
            total_orders[pid] += 1
        if len(unique_products) < 2:
            continue
        for pid_a, pid_b in combinations(unique_products, 2):
            copurchase[pid_a][pid_b] += 1
            copurchase[pid_b][pid_a] += 1

    # Flatten to list of relationship dicts
    rels = []
    for pid_a, others in copurchase.items():
        total_a = total_orders[pid_a]
        for pid_b, count in others.items():
            rels.append({
                "from_id":    pid_a,
                "to_id":      pid_b,
                "count":      count,
                "confidence": round(count / total_a, 6) if total_a > 0 else 0.0,
            })

    ok(f"ALSO_BOUGHT: {len(rels):,} directed relationships "
       f"across {len(copurchase):,} products")
    return rels


def load_also_bought(driver, rels: list[dict], dry_run: bool) -> int:
    """
    Write ALSO_BOUGHT relationships to Neo4j.
    Properties: count (int), confidence (float).
    Q4 uses: MATCH (p:Product {id:$id})-[r:ALSO_BOUGHT]->(rec)
             RETURN rec ORDER BY r.count DESC LIMIT 10
    """
    if dry_run:
        return len(rels)

    cypher = """
    UNWIND $rows AS row
    MATCH (a:Product {id: row.from_id})
    MATCH (b:Product {id: row.to_id})
    MERGE (a)-[r:ALSO_BOUGHT]->(b)
    SET r.count      = row.count,
        r.confidence = row.confidence
    """
    return run_batches(driver, cypher, rels)


# ── orchestration ─────────────────────────────────────────────────────────────

NODE_STEPS = [
    ("SubscriptionTier nodes",        load_subscription_tiers),
    ("SubscriptionTierPricing nodes", load_subscription_tier_pricing),
    ("User nodes",                    load_users),
    ("SellerProfile nodes",           load_seller_profiles),
    ("Subscription nodes",            load_subscriptions),
    ("Product nodes",                 load_products),
    ("Invoice nodes",                 load_invoices),
    ("InvoiceLine nodes",             load_invoice_lines),
    ("Order nodes",                   load_orders),
    ("OrderItem nodes",               load_order_items),
    ("Session nodes",                 load_sessions),       # overridden
    ("Event nodes",                   load_events),
]

REL_STEPS = [
    ("SellerProfile→User rels",       create_seller_profile_relationships),
    ("Subscription rels",             create_subscription_relationships),
    ("Invoice rels",                  create_invoice_relationships),
    ("InvoiceLine rels",              create_invoice_line_relationships),
    ("Order rels",                    create_order_relationships),
    ("OrderItem rels",                create_order_item_relationships),
    ("Session rels",                  create_session_relationships),
    ("Event rels",                    create_event_relationships),
]


def main():
    parser = argparse.ArgumentParser(description="Neo4j optimised loader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count rows only — no writes to Neo4j")
    parser.add_argument("--wipe", action="store_true",
                        help="Delete all nodes and relationships before loading")
    args = parser.parse_args()

    port = int(os.getenv("NEO4J_OPTIMISED_PORT", 7688))

    print(f"\n{'=' * 55}")
    print(f"  Neo4j Optimised Loader  (bolt://localhost:{port})")
    print(f"{'=' * 55}")

    if args.dry_run:
        warn("DRY RUN — no data will be written")

    # Precompute ALSO_BOUGHT data from CSVs before connecting to Neo4j
    also_bought_rels = build_copurchase_data()

    driver = get_driver(port=port)
    try:
        driver.verify_connectivity()
        ok(f"Connected to Neo4j on port {port}")
    except Exception as e:
        err(f"Cannot connect to Neo4j: {e}")
        sys.exit(1)

    if args.wipe and not args.dry_run:
        warn("Wiping all nodes and relationships ...")
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

    # Post-load: ALSO_BOUGHT relationships
    print(f"\n  {'─' * 45}")
    print("  Post-load: ALSO_BOUGHT relationships ...")
    print(f"  {'─' * 45}")
    t0 = time.perf_counter()
    try:
        n = load_also_bought(driver, also_bought_rels, dry_run=args.dry_run)
        elapsed = time.perf_counter() - t0
        ok(f"{'ALSO_BOUGHT rels':<38} {n:>9,}   {elapsed:.1f}s")
    except Exception as e:
        err(f"ALSO_BOUGHT rels FAILED — {e}")
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