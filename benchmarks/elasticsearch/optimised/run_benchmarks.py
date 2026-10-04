"""
benchmarks/elasticsearch/optimised/run_benchmarks.py — Elasticsearch Optimised Q1–Q7
======================================================================================
Runs all seven read benchmarks against the optimised Elasticsearch schema.
Structured identically to the naive benchmark for consistency.

Optimised schema features used per query
──────────────────────────────────────────
  Q1: tier_id / tier_name / price_in_effect_usd denormalised onto invoices at
      load (es_tier_attribution.py) → single date_histogram → terms → terms → sum.

  Q2: invoice_lines embedded as nested objects inside invoice documents →
      single get() retrieves the full invoice + all lines + product snapshots.
      Naive requires 4 round trips.

  Q3: cart stored as native array of objects (not JSON string) in session docs →
      single get() by session_id. No deserialisation cost on the read path.
      Naive searches by user_id and receives a string cart.

  Q4: Same 3-step Python orchestration as naive — ES has no graph traversal.
      Included for completeness and schema-effect isolation (none for Q4).

  Q5: multi_match with best_fields + name^3 / description^1.5 boosts +
      custom English analyser (stemming + stop words + synonyms). The schema
      effect is the ranking quality improvement over naive's unweighted BM25.

  Q6: Same bool/filter query. Schema effect: optimised_events mapping has a
      composite (user_id, occurred_at) sort optimisation. SQL/query unchanged.

  Q7: filters(tier) → date_histogram(day, min_doc_count=0, extended_bounds)
      → sum + moving_fn: per-tier daily revenue, gap-filling and the rolling
      7-day average all server-side in one search.

Schema effect vs naive for each query
───────────────────────────────────────
  Q1: load-time tier denormalisation → 1 server-side aggregation (vs client join)
  Q2: nested lines embedding → 1 round trip (vs 4 naive)
  Q3: native cart array + session_id lookup → 1 GET (vs search + string parse)
  Q4: none (no graph in ES — same orchestration as naive)
  Q5: field boosts + custom analyser → better BM25 ranking quality
  Q6: none functionally (query unchanged — storage layout effect only)
  Q7: tier denormalisation + extended_bounds + moving_fn → 1 server-side search

Field mapping notes (optimised schema)
───────────────────────────────────────
  Same as naive: UUIDs and enums are pure keyword (no .keyword sub-field).
  Use bare field names in term/terms queries.
  Nested lines in optimised_invoices → use nested query context if filtering
  inside lines; for full-document retrieval a plain get() suffices.

Usage:
    cd benchmarks/elasticsearch/optimised
    python run_benchmarks.py                     # all Q1–Q7, 1000 iterations
    python run_benchmarks.py --only Q2           # single query
    python run_benchmarks.py --only Q1 Q5 Q7     # subset
    python run_benchmarks.py --iterations 100    # quick smoke test
    python run_benchmarks.py --dry-run           # run each query once, print results
    python run_benchmarks.py --dry-run --only Q7 # dry-run single query
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
    Q1_AS_OF, optimised_q1, optimised_q7, optimised_tier_names, print_q1, print_q7,
)

load_dotenv()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "benchmarks", "elasticsearch", "optimised", "results")
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
    return Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


# ── pool fetch ─────────────────────────────────────────────────────────────────

def fetch_invoice_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    resp = client.search(
        index="optimised_invoices",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=False,
        size=pool_size,
    )
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    if not ids:
        raise RuntimeError("optimised_invoices is empty — run the loader first.")
    info(f"Q2 pool: {len(ids):,} invoice IDs")
    return ids


def fetch_session_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    """
    Q3 optimised fetches by session_id directly (single GET) rather than
    searching by user_id. Pool contains session document IDs (_id = session token).
    """
    resp = client.search(
        index="optimised_sessions",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=False,
        size=pool_size,
    )
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    if not ids:
        raise RuntimeError("optimised_sessions is empty — run the loader first.")
    info(f"Q3 pool: {len(ids):,} session IDs")
    return ids


def fetch_product_pool(client: Elasticsearch, pool_size: int = 1000) -> list[str]:
    resp = client.search(
        index="optimised_order_items",
        query={"match_all": {}},
        aggs={"products": {"terms": {"field": "product_id", "size": pool_size}}},
        size=0,
    )
    ids = [b["key"] for b in resp["aggregations"]["products"]["buckets"]]
    if not ids:
        raise RuntimeError("optimised_order_items is empty — run the loader first.")
    info(f"Q4 pool: {len(ids):,} product IDs with purchase history")
    return ids


def fetch_anchor_pool(client: Elasticsearch, pool_size: int = 1000) -> list[tuple]:
    resp = client.search(
        index="optimised_events",
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
        raise RuntimeError("optimised_events is empty — run the loader first.")
    info(f"Q6 pool: {len(pairs):,} (user_id, anchor) pairs")
    return pairs


def load_date_range(client: Elasticsearch) -> tuple[date, date]:
    resp = client.search(
        index="optimised_events",
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
# Schema effect: tier_id, tier_name and price_in_effect_usd are denormalised
# onto every invoice at load time (loaders/es_tier_attribution.py — same
# pre-resolution as the Cassandra optimised schema). Q1 is one search:
#   date_histogram(month) → terms(tier_name) → terms(price_in_effect_usd) → sum
# Full PostgreSQL-equivalent output, computed server-side.
# Window anchored to the PostgreSQL Q1 baseline run time (Q1_AS_OF).

def make_q1_fn(client: Elasticsearch):
    def _run():
        return optimised_q1(client)
    return _run


def dry_run_q1(client: Elasticsearch):
    print(f"\n  DRY RUN — Q1: monthly revenue by tier, 12 months to {Q1_AS_OF:%Y-%m-%d %H:%M} UTC\n")
    n = client.count(index="optimised_invoices",
                     query={"exists": {"field": "tier_id"}})["count"]
    if n == 0:
        warn("optimised_invoices has no tier_id — run: python loaders/es_tier_attribution.py")
        return
    print_q1(optimised_q1(client))


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — Full invoice fetch (single GET — lines embedded as nested objects)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Schema effect: invoice_lines are embedded as a nested array inside each
# optimised_invoices document. A single get() retrieves the invoice header,
# all line items, and product detail snapshots in one round trip.
# Naive requires 4 round trips (get invoice → get user → search lines → mget products).

def make_q2_fn(client: Elasticsearch, invoice_ids: list[str]):
    def _run():
        client.get(index="optimised_invoices", id=random.choice(invoice_ids))
    return _run


def dry_run_q2(client: Elasticsearch, invoice_ids: list[str]):
    invoice_id = invoice_ids[0]
    print(f"\n  DRY RUN — Q2: single GET for invoice {invoice_id}\n")
    doc = client.get(index="optimised_invoices", id=invoice_id)
    src = doc["_source"]
    lines = src.get("lines", [])
    print(f"  Invoice ID : {invoice_id}")
    print(f"  Type       : {src.get('invoice_type')}  |  Status: {src.get('status')}")
    print(f"  Total      : ${src.get('total_usd', 0):.2f}")
    print(f"  Tier ID    : {src.get('tier_id', 'N/A')}")
    print(f"\n  Embedded lines ({len(lines)}):")
    print(f"  {'Description':<40} {'Qty':>4} {'Unit':>10} {'Total':>10}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*10}")
    for l in lines[:10]:
        print(
            f"  {str(l.get('description', ''))[:40]:<40} "
            f"{l.get('quantity', 1):>4} "
            f"{float(l.get('unit_price_usd', 0)):>10.2f} "
            f"{float(l.get('line_total_usd', 0)):>10.2f}"
        )
    if len(lines) > 10:
        print(f"  ... and {len(lines) - 10} more lines")
    print()
    info("Schema effect: 1 GET (vs 4 round trips in naive).")


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — Active session + cart (single GET by session_id, 50 concurrent threads)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Schema effect: cart is a native array of objects (not a JSON string).
# No deserialisation needed on the read path — ES returns the array directly.
# Lookup is by session_id (_id) — a single GET vs naive's search-by-user_id.
# 50 concurrent threads share the thread-safe ES client.

def make_q3_fn(client: Elasticsearch, session_ids: list[str]):
    def _run():
        client.get(index="optimised_sessions", id=random.choice(session_ids))
    return _run


def dry_run_q3(client: Elasticsearch, session_ids: list[str]):
    session_id = session_ids[0]
    print(f"\n  DRY RUN — Q3: single GET for session {session_id[:20]}...\n")
    doc = client.get(index="optimised_sessions", id=session_id)
    src = doc["_source"]
    print(f"  Session ID  : {session_id[:40]}")
    print(f"  User ID     : {src.get('user_id')}")
    print(f"  IP          : {src.get('ip_address')}")
    print(f"  Last active : {src.get('last_active_at')}")
    print(f"  Expires     : {src.get('expires_at')}")
    cart = src.get("cart", [])
    if not cart:
        print(f"  Cart        : (empty)")
    else:
        print(f"\n  Cart ({len(cart)} item(s)) — native array, no JSON parse:")
        for item in (cart if isinstance(cart, list) else [])[:5]:
            print(
                f"    • {item.get('product_name', '?')} "
                f"× {item.get('quantity', '?')} "
                f"@ ${item.get('price_usd', '?')}"
            )
    print()
    info("Schema effect: native array cart + GET by session_id (vs search + string parse in naive).")


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — Top-10 co-purchase recommendations (identical to naive)
# ═══════════════════════════════════════════════════════════════════════════════
#
# No schema effect — ES has no graph traversal. Same 3-step Python orchestration
# as naive, using optimised_ indices. Included for completeness and to confirm
# that the engine effect (3 round trips vs Neo4j's 1-hop) is consistent across
# naive and optimised schemas.

def make_q4_fn(client: Elasticsearch, product_ids: list[str]):
    def _run():
        product_id = random.choice(product_ids)

        step1 = client.search(
            index="optimised_order_items",
            query={"term": {"product_id": product_id}},
            aggs={"order_ids": {"terms": {"field": "order_id", "size": 5000}}},
            size=0,
        )
        order_ids = [b["key"] for b in step1["aggregations"]["order_ids"]["buckets"]]
        if not order_ids:
            return

        step2 = client.search(
            index="optimised_orders",
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

        client.search(
            index="optimised_order_items",
            query={
                "bool": {
                    "filter":   [{"terms": {"order_id": valid_ids}}],
                    "must_not": [{"term":  {"product_id": product_id}}],
                }
            },
            aggs={"co_products": {"terms": {"field": "product_id", "size": 10}}},
            size=0,
        )
    return _run


def dry_run_q4(client: Elasticsearch, product_ids: list[str]):
    product_id = product_ids[0]
    print(f"\n  DRY RUN — Q4: co-purchase recommendations for {product_id}\n")

    try:
        p = client.get(index="optimised_products", id=product_id)
        src = p["_source"]
        print(f"  Source: {src.get('name')} ({src.get('product_type')})\n")
    except Exception:
        pass

    step1 = client.search(
        index="optimised_order_items",
        query={"term": {"product_id": product_id}},
        aggs={"order_ids": {"terms": {"field": "order_id", "size": 5000}}},
        size=0,
    )
    order_ids = [b["key"] for b in step1["aggregations"]["order_ids"]["buckets"]]
    print(f"  Step 1: {len(order_ids):,} order_ids contain product X")
    if not order_ids:
        warn("No orders found."); return

    step2 = client.search(
        index="optimised_orders",
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
        warn("No confirmed/shipped/delivered orders."); return

    step3 = client.search(
        index="optimised_order_items",
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
        warn("No co-purchased products."); return

    pids = [b["key"] for b in buckets]
    mg   = client.mget(index="optimised_products", ids=pids)
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
    warn("No schema effect — same 3-step orchestration as naive (ES has no graph traversal).")


# ═══════════════════════════════════════════════════════════════════════════════
# Q5 — Full-text search (best_fields + field boosts + custom analyser)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Schema effect: optimised_products index has a custom English analyser
# (stemming + stop words + synonym expansion) and field-level boost weights:
#   name^3 — highest signal for product discovery
#   description^1.5 — secondary signal
#   product_type^1  — lowest weight (disambiguates merch/course/digital_asset)
#
# best_fields selects the best-matching field per document rather than
# combining all fields (cross_fields). This is the correct mode when fields
# are independent signals — a product matching strongly on name should rank
# higher than one matching weakly across all three fields.
#
# The custom analyser is defined at index creation time in the optimised loader.
# The field name to search is `name.english` / `description.english` if the
# loader created sub-fields with the custom analyser, or the base fields
# directly if the analyser was applied at the field level. Adjust if needed.

def make_q5_fn(client: Elasticsearch):
    def _run():
        term = random.choice(SEARCH_TERMS)
        client.search(
            index="optimised_products",
            query={
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query":  term,
                            "fields": ["name^3", "description^1.5", "product_type"],
                            "type":   "best_fields",
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
    print(f"\n  DRY RUN — Q5: best_fields multi_match for '{term}' (name^3, desc^1.5)\n")
    resp = client.search(
        index="optimised_products",
        query={
            "bool": {
                "must": [{
                    "multi_match": {
                        "query":  term,
                        "fields": ["name^3", "description^1.5", "product_type"],
                        "type":   "best_fields",
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
        warn(f"No results for '{term}'. Check custom analyser is applied to optimised_products.")
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
    info("Schema effect: name^3, description^1.5 boosts + custom English analyser applied.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q6 — User activity events in a 30-day window (identical query, optimised index)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Query is unchanged from naive. Schema effect is at the storage level:
# optimised_events has a composite (user_id, occurred_at) sort optimisation
# defined at index creation time, which improves locality for this access
# pattern (all events for a user are sorted together on disk).

def make_q6_fn(client: Elasticsearch, pairs: list[tuple]):
    def _run():
        user_id, anchor = random.choice(pairs)
        anchor_dt = _parse_anchor(anchor)
        start = anchor_dt - timedelta(days=15)
        end   = anchor_dt + timedelta(days=15)
        client.search(
            index="optimised_events",
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
        index="optimised_events",
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
        warn("No events found."); return
    print(f"  {total:,} event(s) in window (showing up to 20):\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<32}")
    print(f"  {'─'*4} {'─'*25} {'─'*32}")
    for i, h in enumerate(hits[:20], 1):
        s = h["_source"]
        print(f"  {i:<4} {str(s.get('event_type', '')):<25} {str(s.get('occurred_at', '')):<32}")
    info("Schema effect: composite sort optimisation (user_id, occurred_at) on index.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q7 — Daily revenue per tier + gap-fill + 7-day rolling average (6-month window)
# ═══════════════════════════════════════════════════════════════════════════════
#
# One search, all server-side:
#   filters(tier_name)            — every tier present, even with no revenue
#   → date_histogram(day, min_doc_count=0, extended_bounds)   — gap-filling
#   → sum(total_usd) + moving_fn(window=7, shift=1, unweightedAvg)
#     (shift=1 → current day + 6 preceding = ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
# Full PostgreSQL-equivalent output.

def _q7_start(data_min: date, data_max: date) -> date:
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS_Q7)
    return data_min + timedelta(days=random.randint(0, max_start))


def make_q7_fn(client: Elasticsearch, data_min: date, data_max: date, tier_names: list[str]):
    def _run():
        return optimised_q7(client, _q7_start(data_min, data_max), WINDOW_DAYS_Q7, tier_names)
    return _run


def dry_run_q7(client: Elasticsearch, data_min: date, data_max: date, tier_names: list[str]):
    start = _q7_start(data_min, data_max)
    print(f"\n  DRY RUN — Q7: rolling 7-day revenue per tier, gap-filled (server-side)")
    print(f"  Window: {start} + {WINDOW_DAYS_Q7} days\n")
    print_q7(optimised_q7(client, start, WINDOW_DAYS_Q7, tier_names), WINDOW_DAYS_Q7)


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
    output_path = os.path.join(results_dir, f"elasticsearch_optimised_{query_id}.json")

    print(f"\n{'═'*60}")
    print(f"  {query_id} — Elasticsearch Optimised")
    print(f"{'═'*60}")

    try:
        if query_id == "Q1":
            if dry:
                dry_run_q1(client); return
            run_benchmark(
                query_fn=make_q1_fn(client),
                db="elasticsearch_optimised", query_id="Q1",
                label=(
                    "Monthly revenue by subscription tier, 12 months to Q1_AS_OF "
                    f"({Q1_AS_OF.isoformat()}, = PostgreSQL Q1 baseline run time), "
                    "with price in effect. Optimised: tier_id/tier_name/"
                    "price_in_effect_usd denormalised at load (es_tier_attribution.py). "
                    "Single search: date_histogram(month) → terms(tier_name) → "
                    "terms(price) → sum. Full PostgreSQL-equivalent output, server-side."
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
                db="elasticsearch_optimised", query_id="Q2",
                label=(
                    "Full invoice fetch: single get(optimised_invoices). "
                    "Schema effect: invoice_lines embedded as nested objects → "
                    "1 round trip (vs 4 in naive). "
                    f"Pool of {len(pool):,} invoice IDs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q3":
            pool = pools.get("sessions") or fetch_session_pool(client)
            pools["sessions"] = pool
            if dry:
                dry_run_q3(client, pool); return
            run_benchmark(
                query_fn=make_q3_fn(client, pool),
                db="elasticsearch_optimised", query_id="Q3",
                label=(
                    "Session + cart retrieval under 50 concurrent threads. "
                    "Single get(optimised_sessions, session_id). "
                    "Schema effect: cart as native array + GET by session_id "
                    "(vs search-by-user_id + JSON string parse in naive). "
                    f"Pool of {len(pool):,} session IDs."
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
                db="elasticsearch_optimised", query_id="Q4",
                label=(
                    "Top-10 co-purchase recommendations via 3-step Python orchestration. "
                    "Identical to naive — no schema effect (ES has no graph traversal). "
                    f"Pool of {len(pool):,} product IDs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q5":
            if dry:
                dry_run_q5(client); return
            run_benchmark(
                query_fn=make_q5_fn(client),
                db="elasticsearch_optimised", query_id="Q5",
                label=(
                    "Full-text search: multi_match best_fields, "
                    "name^3 / description^1.5 / product_type^1 boosts. "
                    "Custom English analyser (stemming + stop words + synonyms). "
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
                db="elasticsearch_optimised", query_id="Q6",
                label=(
                    f"All events for a user in a {WINDOW_DAYS_Q6}-day window. "
                    "bool/filter: term(user_id) + range(occurred_at) on optimised_events. "
                    "Schema effect: composite (user_id, occurred_at) sort optimisation. "
                    f"Pool of {len(pool):,} (user_id, anchor) pairs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q7":
            data_min, data_max = pools.get("date_range") or load_date_range(client)
            pools["date_range"] = (data_min, data_max)
            tier_names = pools.get("tier_names") or optimised_tier_names(client)
            pools["tier_names"] = tier_names
            if dry:
                dry_run_q7(client, data_min, data_max, tier_names); return
            run_benchmark(
                query_fn=make_q7_fn(client, data_min, data_max, tier_names),
                db="elasticsearch_optimised", query_id="Q7",
                label=(
                    f"7-day rolling avg of daily revenue per tier over {WINDOW_DAYS_Q7} days, "
                    "gap-filled. Single search, server-side: filters(tier_name) → "
                    "date_histogram(day, min_doc_count=0, extended_bounds) → sum + "
                    "moving_fn(window=7, shift=1, unweightedAvg). Tier denormalised at "
                    "load. Full PostgreSQL-equivalent output. "
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
        description="Elasticsearch optimised Q1–Q7 benchmarks"
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
    print("  Elasticsearch Optimised — Q1–Q7 Benchmarks")
    print("═" * 60)
    print(f"  Queries    : {queries}")
    print(f"  Iterations : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir: {args.results_dir}")
    print()
    print("  Schema effects vs naive:")
    print("  • Q1: tier denormalised at load → 1 server-side aggregation")
    print("  • Q2: lines embedded → 1 GET (vs 4 round trips)")
    print("  • Q3: native cart array + GET by session_id (vs search + JSON parse)")
    print("  • Q4: none (ES has no graph traversal)")
    print("  • Q5: name^3/desc^1.5 boosts + custom English analyser")
    print("  • Q6: none functionally (composite sort at storage level)")
    print("  • Q7: filters(tier) + extended_bounds + moving_fn → 1 server-side search")

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