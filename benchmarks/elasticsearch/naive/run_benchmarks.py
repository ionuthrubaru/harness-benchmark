"""
benchmarks/elasticsearch/naive/run_benchmarks.py — Elasticsearch Naive Q1–Q7
=============================================================================
Runs all seven read benchmarks against the naive Elasticsearch schema in one
file. Structured identically to the TimescaleDB benchmark pattern for
consistency — each query has its own request body / factory function,
dry-run printer, and labelled harness call.

Naive schema: 12 flat indices mirroring the PostgreSQL table structure.
No custom analysers, no field boosting, no embedded documents.
Index prefix: naive_

Query design notes (naive)
──────────────────────────
  Q1: revenue by month x subscription tier x price in effect (full PostgreSQL
      equivalent). tier_id is not stored on naive invoice documents, so each
      call does the client-side join (timed): PIT scans of naive_subscriptions
      and of the paid invoices in the window, Python tier attribution and
      temporal price lookup, aggregation. See es_revenue.py.

  Q2: 4 round trips mirroring the PostgreSQL 4-table JOIN:
        get(naive_invoices)  → get(naive_users)
        search(naive_invoice_lines) → mget(naive_products)
      Engine effect: network cost of normalised document model.

  Q3: search(naive_sessions, user_id) — one round trip.
      Cart is embedded in the session doc (same in naive + optimised).
      50 concurrent threads share a single thread-safe ES client.

  Q4: 3-step Python orchestration across 3 indices:
        (1) terms agg on naive_order_items → order_ids containing product X
        (2) search naive_orders            → filter to valid statuses
        (3) terms agg on naive_order_items → co-purchased product counts
      Engine effect: 3 round trips vs Neo4j's 1-hop index-free traversal.

  Q5: multi_match on name / description / product_type, default BM25.
      No field boosting, no custom analyser. Equivalent to PostgreSQL
      ts_rank_cd. The schema effect (custom analyser + boosts) is in
      optimised only.

  Q6: bool/filter on user_id + range on occurred_at.
      Full inverted-index scan — no partition locality (Cassandra's edge).
      Window centred on a real anchor event → guaranteed non-empty.

  Q7: daily revenue per tier + gap-fill + rolling 7-day avg (full PostgreSQL
      equivalent). Same client-side tier attribution as Q1, then Python
      daily aggregation, gap-filling and rolling average — all timed.

Field naming notes (naive dynamic mapping)
───────────────────────────────────────────
  UUID / string enum fields → loaded as pure keyword (no .keyword sub-field).
  Use bare field name for term/terms queries — no sub-field needed.
  Date fields (occurred_at, created_at) → detected as date by dynamic mapping.
  Boolean fields (is_active) → stored as boolean if loader passed Python bool.
  If Q5 dry-run returns 0 results, check: GET /naive_products/_mapping/field/is_active

Usage:
    cd benchmarks/elasticsearch/naive
    python run_benchmarks.py                     # all Q1–Q7, 1000 iterations
    python run_benchmarks.py --only Q5           # single query
    python run_benchmarks.py --only Q1 Q5 Q7     # subset
    python run_benchmarks.py --iterations 100    # quick smoke test
    python run_benchmarks.py --dry-run           # run each query once, print results
    python run_benchmarks.py --dry-run --only Q2 # dry-run single query
"""

import argparse
import os
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from benchmarks.harness import run_benchmark
from benchmarks.elasticsearch.es_revenue import (
    Q1_AS_OF, naive_q1, naive_q7, print_q1, print_q7,
)

load_dotenv()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "benchmarks", "elasticsearch", "naive", "results")
WINDOW_DAYS_Q6 = 30
WINDOW_DAYS_Q7 = 183

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour", "photoshop brushes", "video editing",
    "certificate course", "logo design", "colour palette", "font pack",
    "texture pack", "motion graphics", "social media", "icon set",
    "web design", "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")


# ── ES client ──────────────────────────────────────────────────────────────────

def get_client() -> Elasticsearch:
    """
    Single shared Elasticsearch 8.x client.
    Thread-safe: urllib3 connection pool handles Q3's 50-thread concurrency.
    No auth — xpack.security.enabled=false in docker-compose.yml.
    """
    return Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


# ── pool fetch ─────────────────────────────────────────────────────────────────

def fetch_invoice_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    """Random sample of invoice IDs from naive_invoices."""
    resp = client.search(
        index="naive_invoices",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        size=pool_size,
        _source=False,
    )
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    if not ids:
        raise RuntimeError("naive_invoices is empty — run the loader first.")
    info(f"Q2 pool: {len(ids):,} invoice IDs")
    return ids


def fetch_user_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    """Distinct user IDs that have at least one session."""
    resp = client.search(
        index="naive_sessions",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=["user_id"],
        size=pool_size,
    )
    ids = list({h["_source"]["user_id"] for h in resp["hits"]["hits"]})
    if not ids:
        raise RuntimeError("naive_sessions is empty — run the loader first.")
    info(f"Q3 pool: {len(ids):,} user IDs with sessions")
    return ids


def fetch_product_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    """
    Product IDs with purchase history, ranked by co-purchase frequency.
    Uses a terms aggregation on naive_order_items — same intent as the
    PostgreSQL pool fetch (products that appear in confirmed orders).
    """
    resp = client.search(
        index="naive_order_items",
        query={"match_all": {}},
        aggs={"products": {"terms": {"field": "product_id", "size": pool_size}}},
        size=0,
    )
    ids = [b["key"] for b in resp["aggregations"]["products"]["buckets"]]
    if not ids:
        raise RuntimeError("naive_order_items is empty — run the loader first.")
    info(f"Q4 pool: {len(ids):,} product IDs with purchase history")
    return ids


def fetch_anchor_pool(client: Elasticsearch, pool_size: int = 1000) -> list[tuple]:
    """
    Random (user_id, occurred_at) pairs from naive_events.
    30-day window centred on occurred_at → guaranteed non-empty result
    per iteration. Mirrors the anchor-based methodology in all other databases.
    """
    resp = client.search(
        index="naive_events",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=["user_id", "occurred_at"],
        size=pool_size,
    )
    pairs = [
        (h["_source"]["user_id"], h["_source"]["occurred_at"])
        for h in resp["hits"]["hits"]
        if h["_source"].get("user_id") and h["_source"].get("occurred_at")
    ]
    if not pairs:
        raise RuntimeError("naive_events is empty — run the loader first.")
    info(f"Q6 pool: {len(pairs):,} (user_id, anchor) pairs")
    return pairs


def load_date_range(client: Elasticsearch) -> tuple[date, date]:
    """
    Min / max occurred_at from naive_events — used to anchor Q7's random
    6-month window within actual data, consistent with all other databases.
    """
    resp = client.search(
        index="naive_events",
        aggs={
            "min_ts": {"min": {"field": "occurred_at"}},
            "max_ts": {"max": {"field": "occurred_at"}},
        },
        size=0,
    )
    def _ms_to_date(ms: float) -> date:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).date()
    min_d = _ms_to_date(resp["aggregations"]["min_ts"]["value"])
    max_d = _ms_to_date(resp["aggregations"]["max_ts"]["value"])
    return min_d, max_d


def _parse_anchor(anchor) -> datetime:
    """
    Convert ES occurred_at value (ISO string or millisecond epoch) to datetime.
    Handles both the string format ES returns in _source and numeric epoch.
    """
    if isinstance(anchor, (int, float)):
        return datetime.fromtimestamp(anchor / 1000.0, tz=timezone.utc)
    s = str(anchor).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s[:10]).replace(tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — Monthly revenue by subscription tier, last 12 months (temporal pricing)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Full PostgreSQL-equivalent output (month, tier, price in effect, count,
# revenue). tier_id is not on naive invoice documents, so each call performs
# the client-side join an application would need on this schema (all timed):
# tiers + pricing reads, a PIT scan of naive_subscriptions, a PIT scan of the
# paid invoices of the window, Python tier attribution (subscription_id, or
# the user's most recent subscription started_at <= created_at), temporal
# price lookup and aggregation. See benchmarks/elasticsearch/es_revenue.py.
# Window anchored to the PostgreSQL Q1 baseline run time (Q1_AS_OF).

def make_q1_fn(client: Elasticsearch):
    def _run():
        return naive_q1(client)
    return _run


def dry_run_q1(client: Elasticsearch):
    print(f"\n  DRY RUN — Q1: monthly revenue by tier, 12 months to {Q1_AS_OF:%Y-%m-%d %H:%M} UTC\n")
    t0 = time.perf_counter()
    rows = naive_q1(client)
    print_q1(rows)
    info(f"One call took {(time.perf_counter() - t0) * 1000:,.0f} ms")


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — Full invoice fetch (4 round trips)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mirrors PostgreSQL's 4-table JOIN as 4 sequential ES requests:
#   1. get(naive_invoices, id)            → invoice header + financials
#   2. get(naive_users, user_id)          → customer snapshot
#   3. search(naive_invoice_lines, ...)   → all line items for this invoice
#   4. mget(naive_products, product_ids)  → product details per line
#
# Engine effect: each request is a separate network round trip. PostgreSQL
# executes the equivalent in a single query plan with hash joins entirely
# in shared memory. The latency gap measures the cost of normalisation in
# a document store without native JOIN support.

def make_q2_fn(client: Elasticsearch, invoice_ids: list[str]):
    def _run():
        invoice_id = random.choice(invoice_ids)

        # 1. Invoice header
        inv     = client.get(index="naive_invoices", id=invoice_id)
        user_id = inv["_source"].get("user_id")

        # 2. Customer
        if user_id:
            client.get(index="naive_users", id=user_id)

        # 3. Line items
        lines_resp = client.search(
            index="naive_invoice_lines",
            query={"term": {"invoice_id": invoice_id}},
            size=100,
        )

        # 4. Product details (batch via mget — one request regardless of line count)
        product_ids = [
            h["_source"]["product_id"]
            for h in lines_resp["hits"]["hits"]
            if h["_source"].get("product_id")
        ]
        if product_ids:
            client.mget(index="naive_products", ids=product_ids)

    return _run


def dry_run_q2(client: Elasticsearch, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2: full invoice fetch for {invoice_id}\n")

    inv = client.get(index="naive_invoices", id=invoice_id)
    src = inv["_source"]
    user_id = src.get("user_id")

    user = {}
    if user_id:
        try:
            user = client.get(index="naive_users", id=user_id)["_source"]
        except Exception:
            pass

    lines_resp = client.search(
        index="naive_invoice_lines",
        query={"term": {"invoice_id": invoice_id}},
        size=100,
    )
    lines = [h["_source"] for h in lines_resp["hits"]["hits"]]

    product_ids = [l["product_id"] for l in lines if l.get("product_id")]
    products = {}
    if product_ids:
        mg = client.mget(index="naive_products", ids=product_ids)
        products = {d["_id"]: d["_source"] for d in mg["docs"] if d.get("found")}

    print(f"  Invoice ID : {invoice_id}")
    print(f"  Type       : {src.get('invoice_type')}  |  Status: {src.get('status')}")
    print(f"  Total      : ${src.get('total_usd', 0):.2f}")
    print(f"  Customer   : {user.get('full_name', 'N/A')} <{user.get('email', 'N/A')}> ({user.get('country_code', '?')})")
    print(f"\n  Lines ({len(lines)}):")
    print(f"  {'Description':<40} {'Qty':>4} {'Unit':>10} {'Total':>10}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*10}")
    for l in lines:
        print(
            f"  {str(l.get('description', ''))[:40]:<40} "
            f"{l.get('quantity', 1):>4} "
            f"{float(l.get('unit_price_usd', 0)):>10.2f} "
            f"{float(l.get('line_total_usd', 0)):>10.2f}"
        )
    print(f"\n  Note: 4 round trips (get invoice → get user → search lines → mget products)")


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — Active session + cart (50 concurrent threads)
# ═══════════════════════════════════════════════════════════════════════════════
#
# search(naive_sessions) filtered by user_id, sorted by last_active_at,
# size=1 — mirrors the PostgreSQL pattern (ORDER BY last_active_at DESC LIMIT 1).
#
# Cart is embedded in the session document in both naive and optimised schemas.
# The naive vs optimised distinction for Q3 is storage type only:
#   naive:     cart stored as a serialised JSON string
#   optimised: cart stored as a native array of objects (avoids JSON parse overhead)
#
# 50 concurrent threads share one ES client. The elasticsearch-py client is
# thread-safe (urllib3 connection pooling). No thread-local client needed.

def make_q3_fn(client: Elasticsearch, user_ids: list[str]):
    def _run():
        user_id = random.choice(user_ids)
        client.search(
            index="naive_sessions",
            query={"term": {"user_id": user_id}},
            sort=[{"last_active_at": {"order": "desc"}}],
            size=1,
        )
    return _run


def dry_run_q3(client: Elasticsearch, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Q3: active session for user {user_id}\n")
    resp = client.search(
        index="naive_sessions",
        query={"term": {"user_id": user_id}},
        sort=[{"last_active_at": {"order": "desc"}}],
        size=1,
    )
    hits = resp["hits"]["hits"]
    if not hits:
        warn(f"No session found for user {user_id}.")
        return
    src = hits[0]["_source"]
    print(f"  Session ID  : {hits[0]['_id']}")
    print(f"  User ID     : {src.get('user_id')}")
    print(f"  IP          : {src.get('ip_address')}")
    print(f"  Last active : {src.get('last_active_at')}")
    print(f"  Expires     : {src.get('expires_at')}")
    cart = src.get("cart", [])
    if isinstance(cart, str):
        import json
        try:
            cart = json.loads(cart)
        except Exception:
            cart = []
    if not cart:
        print(f"  Cart        : (empty)")
    else:
        print(f"\n  Cart ({len(cart)} item(s)):")
        for item in (cart if isinstance(cart, list) else []):
            print(
                f"    • {item.get('product_name', '?')} "
                f"× {item.get('quantity', '?')} "
                f"@ ${item.get('price_usd', '?')}"
            )
    warn("Naive: cart stored as JSON string — deserialisation cost on every read.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — Top-10 co-purchase recommendations (3-step Python orchestration)
# ═══════════════════════════════════════════════════════════════════════════════
#
# No graph traversal — pure Python coordination across three index lookups:
#
#   Step 1: terms agg on naive_order_items
#           → all order_ids that contain product X
#
#   Step 2: search naive_orders with ids query + status filter
#           → retain only confirmed / shipped / delivered orders
#
#   Step 3: terms agg on naive_order_items
#           → product_ids co-purchased in valid orders (excluding X)
#           → top 10 by doc_count = co-purchase frequency
#
# Engine effect: 3 ES round trips for a result Neo4j delivers in 1 hop using
# index-free adjacency on pre-computed ALSO_BOUGHT relationships.

def make_q4_fn(client: Elasticsearch, product_ids: list[str]):
    def _run():
        product_id = random.choice(product_ids)

        # Step 1: order_ids containing product X
        step1 = client.search(
            index="naive_order_items",
            query={"term": {"product_id": product_id}},
            aggs={"order_ids": {"terms": {"field": "order_id", "size": 5000}}},
            size=0,
        )
        order_ids = [b["key"] for b in step1["aggregations"]["order_ids"]["buckets"]]
        if not order_ids:
            return

        # Step 2: filter to valid order statuses
        step2 = client.search(
            index="naive_orders",
            query={
                "bool": {
                    "filter": [
                        {"ids":   {"values": order_ids}},
                        {"terms": {"status": ["confirmed", "shipped", "delivered"]}},
                    ]
                }
            },
            _source=False,
            size=min(len(order_ids), 5000),
        )
        valid_ids = [h["_id"] for h in step2["hits"]["hits"]]
        if not valid_ids:
            return

        # Step 3: co-purchased products in valid orders, top 10
        client.search(
            index="naive_order_items",
            query={
                "bool": {
                    "filter":    [{"terms": {"order_id": valid_ids}}],
                    "must_not":  [{"term":  {"product_id": product_id}}],
                }
            },
            aggs={"co_products": {"terms": {"field": "product_id", "size": 10}}},
            size=0,
        )

    return _run


def dry_run_q4(client: Elasticsearch, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  DRY RUN — Q4: co-purchase recommendations for product {product_id}\n")

    try:
        p = client.get(index="naive_products", id=product_id)
        src = p["_source"]
        print(f"  Source: {src.get('name')} ({src.get('product_type')})\n")
    except Exception:
        pass

    step1 = client.search(
        index="naive_order_items",
        query={"term": {"product_id": product_id}},
        aggs={"order_ids": {"terms": {"field": "order_id", "size": 5000}}},
        size=0,
    )
    order_ids = [b["key"] for b in step1["aggregations"]["order_ids"]["buckets"]]
    print(f"  Step 1: {len(order_ids):,} order_ids contain product X")
    if not order_ids:
        warn("No orders found for this product."); return

    step2 = client.search(
        index="naive_orders",
        query={
            "bool": {
                "filter": [
                    {"ids":   {"values": order_ids}},
                    {"terms": {"status": ["confirmed", "shipped", "delivered"]}},
                ]
            }
        },
        _source=False,
        size=min(len(order_ids), 5000),
    )
    valid_ids = [h["_id"] for h in step2["hits"]["hits"]]
    print(f"  Step 2: {len(valid_ids):,} valid orders after status filter")
    if not valid_ids:
        warn("No confirmed/shipped/delivered orders found."); return

    step3 = client.search(
        index="naive_order_items",
        query={
            "bool": {
                "filter":   [{"terms": {"order_id": valid_ids}}],
                "must_not": [{"term":  {"product_id": product_id}}],
            }
        },
        aggs={"co_products": {"terms": {"field": "product_id", "size": 10}}},
        size=0,
    )
    buckets = step3["aggregations"]["co_products"]["buckets"]
    if not buckets:
        warn("No co-purchased products found."); return

    pids = [b["key"] for b in buckets]
    mg   = client.mget(index="naive_products", ids=pids)
    names = {d["_id"]: d["_source"] for d in mg["docs"] if d.get("found")}

    print(f"\n  Top {len(buckets)} recommendations:")
    print(f"  {'#':<3} {'Product name':<35} {'Type':<16} {'Price':>8} {'Co-buys':>8}")
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, b in enumerate(buckets, 1):
        p = names.get(b["key"], {})
        print(
            f"  {i:<3} {str(p.get('name', '?'))[:35]:<35} "
            f"{str(p.get('product_type', '?')):<16} "
            f"{float(p.get('price_usd', 0)):>8.2f} "
            f"{b['doc_count']:>8}"
        )
    print(f"\n  Note: 3 ES round trips vs Neo4j's 1-hop ALSO_BOUGHT traversal")


# ═══════════════════════════════════════════════════════════════════════════════
# Q5 — Full-text product search (default BM25, no boosts or custom analyser)
# ═══════════════════════════════════════════════════════════════════════════════
#
# multi_match across name / description / product_type with default BM25.
# is_active filter: stored as Python bool True by the loader (CSV "True" → bool).
# If this returns 0 results, inspect: GET /naive_products/_mapping/field/is_active
# and adjust the filter accordingly (e.g. {"term": {"is_active": "True"}} for string).
#
# Same search term pool as PostgreSQL Q5, random per iteration.
# Naive vs optimised schema effect: custom English analyser + name^3 / description^1.5
# boosts are absent here; optimised adds those and measures the ranking improvement.

def make_q5_fn(client: Elasticsearch):
    def _run():
        term = random.choice(SEARCH_TERMS)
        client.search(
            index="naive_products",
            query={
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query":  term,
                            "fields": ["name", "description", "product_type"],
                        }
                    }],
                    "filter": [{"term": {"is_active": True}}],
                }
            },
            size=20,
        )
    return _run


def dry_run_q5(client: Elasticsearch):
    term = random.choice(SEARCH_TERMS)
    print(f"\n  DRY RUN — Q5: multi_match search for '{term}'\n")
    resp = client.search(
        index="naive_products",
        query={
            "bool": {
                "must": [{
                    "multi_match": {
                        "query":  term,
                        "fields": ["name", "description", "product_type"],
                    }
                }],
                "filter": [{"term": {"is_active": True}}],
            }
        },
        size=20,
    )
    hits  = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    if not hits:
        warn(
            f"No results for '{term}'. Check is_active field mapping:\n"
            "  GET /naive_products/_mapping/field/is_active"
        )
        return
    print(f"  {total:,} total results (showing up to 20):\n")
    print(f"  {'#':<3} {'Product name':<35} {'Type':<16} {'Price':>8} {'Score':>8}")
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, h in enumerate(hits, 1):
        s = h["_source"]
        print(
            f"  {i:<3} {str(s.get('name', ''))[:35]:<35} "
            f"{str(s.get('product_type', '')):<16} "
            f"{float(s.get('price_usd', 0)):>8.2f} "
            f"{h['_score']:>8.4f}"
        )
    warn("No field boosts or custom analyser — optimised adds name^3, description^1.5, English stemmer.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q6 — User activity events in a 30-day window
# ═══════════════════════════════════════════════════════════════════════════════
#
# bool/filter: term on user_id + range on occurred_at.
# sort: occurred_at desc (mirrors PostgreSQL ORDER BY occurred_at DESC).
# size=1000: enough for a typical 30-day event window per user.
# track_total_hits=False: avoids the exact-count overhead when only results matter.
#
# Window centred ±15 days on anchor event → guaranteed non-empty result.
# Cassandra's edge: partition scan is O(events in partition); ES inverted index
# must resolve postings lists for the range predicate across all shards.

def make_q6_fn(client: Elasticsearch, pairs: list[tuple]):
    def _run():
        user_id, anchor = random.choice(pairs)
        anchor_dt = _parse_anchor(anchor)
        start = anchor_dt - timedelta(days=15)
        end   = anchor_dt + timedelta(days=15)
        client.search(
            index="naive_events",
            query={
                "bool": {
                    "filter": [
                        {"term":  {"user_id": user_id}},
                        {"range": {"occurred_at": {
                            "gte": start.isoformat(),
                            "lt":  end.isoformat(),
                        }}},
                    ]
                }
            },
            sort=[{"occurred_at": {"order": "desc"}}],
            size=1000,
            track_total_hits=False,
        )
    return _run


def dry_run_q6(client: Elasticsearch, pairs: list[tuple]):
    user_id, anchor = pairs[0]
    anchor_dt = _parse_anchor(anchor)
    start = anchor_dt - timedelta(days=15)
    end   = anchor_dt + timedelta(days=15)
    print(f"\n  DRY RUN — Q6: events for user {user_id}")
    print(f"  Window: {start.date()} → {end.date()} ({WINDOW_DAYS_Q6} days)\n")
    resp = client.search(
        index="naive_events",
        query={
            "bool": {
                "filter": [
                    {"term":  {"user_id": user_id}},
                    {"range": {"occurred_at": {
                        "gte": start.isoformat(),
                        "lt":  end.isoformat(),
                    }}},
                ]
            }
        },
        sort=[{"occurred_at": {"order": "desc"}}],
        size=1000,
        track_total_hits=True,
    )
    hits  = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    if not hits:
        warn("No events found for this user/window."); return
    print(f"  {total:,} event(s) in window (showing up to 20):\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<32}")
    print(f"  {'─'*4} {'─'*25} {'─'*32}")
    for i, h in enumerate(hits[:20], 1):
        s = h["_source"]
        print(f"  {i:<4} {str(s.get('event_type', '')):<25} {str(s.get('occurred_at', '')):<32}")
    if total > 20:
        print(f"  ... and {total - 20} more")


# ═══════════════════════════════════════════════════════════════════════════════
# Q7 — Daily revenue per tier + gap-fill + 7-day rolling average (6-month window)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Full PostgreSQL-equivalent output (day x tier, daily revenue, rolling 7-day
# avg). Same client-side tier attribution as Q1 naive, then Python daily
# aggregation per tier, gap-filling (all 3 tiers x every day) and the rolling
# average over the gap-filled series — all inside the timed call.

def _q7_start(data_min: date, data_max: date) -> date:
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS_Q7)
    return data_min + timedelta(days=random.randint(0, max_start))


def make_q7_fn(client: Elasticsearch, data_min: date, data_max: date):
    def _run():
        return naive_q7(client, _q7_start(data_min, data_max), WINDOW_DAYS_Q7)
    return _run


def dry_run_q7(client: Elasticsearch, data_min: date, data_max: date):
    start = _q7_start(data_min, data_max)
    print(f"\n  DRY RUN — Q7: rolling 7-day revenue per tier, gap-filled")
    print(f"  Window: {start} + {WINDOW_DAYS_Q7} days\n")
    t0 = time.perf_counter()
    rows = naive_q7(client, start, WINDOW_DAYS_Q7)
    print_q7(rows, WINDOW_DAYS_Q7)
    info(f"One call took {(time.perf_counter() - t0) * 1000:,.0f} ms")


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def run_query(
    query_id:    str,
    client:      Elasticsearch,
    iterations:  int,
    dry:         bool,
    pools:       dict,
    results_dir: str,
):
    output_path = os.path.join(results_dir, f"elasticsearch_naive_{query_id}.json")

    print(f"\n{'═'*60}")
    print(f"  {query_id} — Elasticsearch Naive")
    print(f"{'═'*60}")

    try:
        if query_id == "Q1":
            if dry:
                dry_run_q1(client); return
            run_benchmark(
                query_fn=make_q1_fn(client),
                db="elasticsearch_naive", query_id="Q1",
                label=(
                    "Monthly revenue by subscription tier, 12 months to Q1_AS_OF "
                    f"({Q1_AS_OF.isoformat()}, = PostgreSQL Q1 baseline run time), "
                    "with price in effect (temporal pricing). Naive: tier_id not on "
                    "invoices → per call (timed): tiers+pricing reads, PIT scan of "
                    "naive_subscriptions, PIT scan of paid invoices in window, Python "
                    "tier attribution (subscription_id / most recent subscription "
                    "started_at <= created_at), temporal price, aggregation. "
                    "Full PostgreSQL-equivalent output."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q2":
            pool = pools.get("invoices") or fetch_invoice_pool(client)
            pools["invoices"] = pool
            if dry:
                dry_run_q2(client, pool); return
            run_benchmark(
                query_fn=make_q2_fn(client, pool),
                db="elasticsearch_naive", query_id="Q2",
                label=(
                    "Full invoice fetch: get(invoice) → get(user) → "
                    "search(invoice_lines) → mget(products). "
                    "4 round trips vs PostgreSQL's single 4-table JOIN. "
                    f"Random invoice from pool of {len(pool):,} IDs per iteration."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q3":
            pool = pools.get("users") or fetch_user_pool(client)
            pools["users"] = pool
            if dry:
                dry_run_q3(client, pool); return
            run_benchmark(
                query_fn=make_q3_fn(client, pool),
                db="elasticsearch_naive", query_id="Q3",
                label=(
                    "Active session + cart retrieval by user_id (50 concurrent threads). "
                    "search(naive_sessions, user_id, sort=last_active_at, size=1). "
                    "Naive: cart stored as JSON string — deserialisation on every read. "
                    f"Pool of {len(pool):,} user IDs."
                ),
                iterations=iterations, concurrency=50, output_path=output_path,
            )

        elif query_id == "Q4":
            pool = pools.get("products") or fetch_product_pool(client)
            pools["products"] = pool
            if dry:
                dry_run_q4(client, pool); return
            run_benchmark(
                query_fn=make_q4_fn(client, pool),
                db="elasticsearch_naive", query_id="Q4",
                label=(
                    "Top-10 co-purchase recommendations via 3-step Python orchestration: "
                    "terms agg(order_items) → ids+status filter(orders) → "
                    "terms agg(order_items, exclude product X). "
                    "No graph traversal — 3 ES round trips vs Neo4j's 1-hop adjacency. "
                    f"Pool of {len(pool):,} product IDs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q5":
            if dry:
                dry_run_q5(client); return
            run_benchmark(
                query_fn=make_q5_fn(client),
                db="elasticsearch_naive", query_id="Q5",
                label=(
                    "Full-text search: multi_match on name/description/product_type, "
                    "default BM25, no field boosts, no custom analyser. "
                    "is_active=True filter. 20 results. "
                    f"{len(SEARCH_TERMS)} search terms, random per iteration."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q6":
            pool = pools.get("anchors") or fetch_anchor_pool(client)
            pools["anchors"] = pool
            if dry:
                dry_run_q6(client, pool); return
            run_benchmark(
                query_fn=make_q6_fn(client, pool),
                db="elasticsearch_naive", query_id="Q6",
                label=(
                    f"All events for a user in a {WINDOW_DAYS_Q6}-day window. "
                    "bool/filter: term(user_id) + range(occurred_at). "
                    "sort: occurred_at desc. size=1000. track_total_hits=False. "
                    "Window centred on anchor event — guaranteed non-empty. "
                    f"Pool of {len(pool):,} (user_id, anchor) pairs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q7":
            data_min, data_max = pools.get("date_range") or load_date_range(client)
            pools["date_range"] = (data_min, data_max)
            if dry:
                dry_run_q7(client, data_min, data_max); return
            run_benchmark(
                query_fn=make_q7_fn(client, data_min, data_max),
                db="elasticsearch_naive", query_id="Q7",
                label=(
                    f"7-day rolling avg of daily revenue per tier over {WINDOW_DAYS_Q7} days, "
                    "gap-filled. Naive: per call (timed) PIT scans of naive_subscriptions "
                    "and paid invoices in window, Python tier attribution, daily "
                    "aggregation per tier, gap-fill (3 tiers x every day), rolling avg "
                    "(current + 6 preceding days). Full PostgreSQL-equivalent output. "
                    f"Data range: {data_min} → {data_max}."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        else:
            warn(f"Unknown query ID: {query_id}")
            return

        ok(f"{query_id} complete → {output_path}")

    except Exception as e:
        fail(f"{query_id} failed: {e}")
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Elasticsearch naive Q1–Q7 benchmarks"
    )
    parser.add_argument("--iterations",  type=int,  default=1000)
    parser.add_argument("--dry-run",     action="store_true", dest="dry_run")
    parser.add_argument("--only",        nargs="+", metavar="Q")
    parser.add_argument("--results-dir", type=str,  default=RESULTS_DIR, dest="results_dir")
    args = parser.parse_args()

    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]
    iterations = args.iterations

    print("\n" + "═" * 60)
    print("  Elasticsearch Naive — Q1–Q7 Benchmarks")
    print("═" * 60)
    print(f"  Queries    : {queries}")
    print(f"  Iterations : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir: {args.results_dir}")
    print()
    print("  Schema: 12 flat indices, dynamic mapping, no custom analysers.")
    print("  Index prefix: naive_")
    print("  Q1: client-side tier attribution + temporal price (PIT scans, timed)")
    print("  Q2: 4 round trips  (vs 1 PostgreSQL JOIN)")
    print("  Q3: search by user_id, 50 concurrent threads")
    print("  Q4: 3-step Python orchestration across order_items / orders")
    print("  Q5: default BM25, no boosts, no custom analyser")
    print("  Q6: range + term filter on naive_events (6.3M docs)")
    print("  Q7: client-side tier attribution, Python gap-fill + rolling avg (timed)")

    os.makedirs(args.results_dir, exist_ok=True)

    client = get_client()
    try:
        info_resp = client.info()
        info(f"Connected — ES {info_resp['version']['number']}")
    except Exception as e:
        fail(f"Cannot connect to Elasticsearch: {e}")
        sys.exit(1)

    pools       = {}
    total_start = time.perf_counter()

    for qid in queries:
        run_query(
            query_id=qid,
            client=client,
            iterations=iterations,
            dry=args.dry_run,
            pools=pools,
            results_dir=args.results_dir,
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  {GREEN}Done.{RESET}\n")


if __name__ == "__main__":
    main()