"""
benchmarks/elasticsearch/naive/run_scalability.py — Elasticsearch Naive Scalability
=====================================================================================
Re-runs Q1–Q7 at 10% and 50% data scale to establish the Elasticsearch naive
scalability curve for Chart 3.

Scale methodology — identical to all other databases
──────────────────────────────────────────────────────
  10% scale : queries restricted to the first 10% of the dataset date range
  50% scale : queries restricted to the first 50% of the date range
  100% scale: full dataset — already captured by run_benchmarks.py

Cutoffs are computed from the actual min/max of occurred_at in naive_events,
consistent with PostgreSQL, MongoDB, Neo4j, Cassandra, and TimescaleDB.

Q3 uses row-based scaling — sessions were generated in 2025, so a date cutoff
on created_at returns near-zero rows. We sample scale_pct% of distinct session
user_ids instead (same exception as all other databases).

Q8 excluded — write throughput scaling measured separately.

Field mapping notes (naive schema)
────────────────────────────────────
  UUID / enum fields  → pure keyword (no .keyword sub-field)
  date fields         → native ES date type (ISO 8601 strings accepted)
  Use bare field names in term/terms/range queries — no .keyword suffix needed.

Usage:
    cd benchmarks/elasticsearch/naive
    python run_scalability.py                    # both scales, 1000 iterations
    python run_scalability.py --scale 10
    python run_scalability.py --scale 50
    python run_scalability.py --iterations 100
    python run_scalability.py --only Q1 Q5 Q7
    python run_scalability.py --dry-run
"""

import argparse
import json
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
from benchmarks.elasticsearch.es_revenue import naive_q7

load_dotenv()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "benchmarks", "elasticsearch", "naive", "results", "scale")
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
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


# ── cutoff computation ─────────────────────────────────────────────────────────

def compute_cutoffs(client: Elasticsearch) -> dict:
    """
    Derive 10% and 50% date cutoffs from the actual min/max of occurred_at
    in naive_events. Consistent with the PostgreSQL scalability script which
    reads from the same underlying events table.
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
    range_days = (max_d - min_d).days

    return {
        "min_date":     min_d,
        "max_date":     max_d,
        "range_days":   range_days,
        "cutoff_10pct": min_d + timedelta(days=int(range_days * 0.10)),
        "cutoff_50pct": min_d + timedelta(days=int(range_days * 0.50)),
    }


# ── pool helpers ───────────────────────────────────────────────────────────────

def fetch_invoice_pool(client: Elasticsearch, cutoff: date, pool_size: int = 1000) -> list[str]:
    """Invoice IDs with created_at < cutoff."""
    resp = client.search(
        index="naive_invoices",
        query={"function_score": {
            "query": {"range": {"created_at": {"lt": cutoff.isoformat()}}},
            "random_score": {},
        }},
        _source=False,
        size=pool_size,
    )
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    if not ids:
        raise RuntimeError(f"No invoices before {cutoff}")
    return ids


def fetch_user_pool_sessions(client: Elasticsearch, scale_pct: int, pool_size: int = 1000) -> list[str]:
    """
    Row-based scaling for Q3: sample scale_pct% of distinct session user_ids.
    Date cutoff is not applicable — sessions were generated in 2025.
    """
    # Get total distinct user count first
    count_resp = client.search(
        index="naive_sessions",
        aggs={"distinct_users": {"cardinality": {"field": "user_id"}}},
        size=0,
    )
    total = count_resp["aggregations"]["distinct_users"]["value"]
    limit = max(1, int(total * scale_pct / 100))

    resp = client.search(
        index="naive_sessions",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=["user_id"],
        size=min(limit, 10000),   # ES max size cap
    )
    ids = list({h["_source"]["user_id"] for h in resp["hits"]["hits"]})
    if not ids:
        raise RuntimeError("naive_sessions is empty")
    # Fill to pool_size via random.choices (with replacement if needed)
    return random.choices(ids, k=min(pool_size, len(ids)))


def fetch_product_pool(client: Elasticsearch, cutoff: date, pool_size: int = 1000) -> list[str]:
    """
    Product IDs that appear in confirmed/shipped/delivered orders
    where order created_at < cutoff. Mirrors PostgreSQL pool logic.
    Two-step: (1) collect valid order IDs from naive_orders, (2) agg product IDs.
    """
    # Step 1: order IDs with valid status and created_at < cutoff
    orders_resp = client.search(
        index="naive_orders",
        query={
            "bool": {
                "filter": [
                    {"terms": {"status": ["confirmed", "shipped", "delivered"]}},
                    {"range": {"created_at": {"lt": cutoff.isoformat()}}},
                ]
            }
        },
        aggs={"order_ids": {"terms": {"field": "id", "size": 50000}}},
        _source=False,
        size=0,
    )
    # Use _id directly — the document ID is the order UUID
    orders_resp2 = client.search(
        index="naive_orders",
        query={
            "bool": {
                "filter": [
                    {"terms": {"status": ["confirmed", "shipped", "delivered"]}},
                    {"range": {"created_at": {"lt": cutoff.isoformat()}}},
                ]
            }
        },
        _source=False,
        size=10000,
    )
    valid_order_ids = [h["_id"] for h in orders_resp2["hits"]["hits"]]
    if not valid_order_ids:
        raise RuntimeError(f"No valid orders before {cutoff}")

    # Step 2: distinct product IDs in those orders
    items_resp = client.search(
        index="naive_order_items",
        query={"terms": {"order_id": valid_order_ids}},
        aggs={"products": {"terms": {"field": "product_id", "size": pool_size}}},
        size=0,
    )
    ids = [b["key"] for b in items_resp["aggregations"]["products"]["buckets"]]
    if not ids:
        raise RuntimeError(f"No order items for orders before {cutoff}")
    return ids


def fetch_anchor_pool(client: Elasticsearch, cutoff: date, pool_size: int = 1000) -> list[tuple]:
    """
    Random (user_id, occurred_at) pairs from naive_events where occurred_at < cutoff.
    Window centred on anchor → guaranteed non-empty result each iteration.
    """
    resp = client.search(
        index="naive_events",
        query={
            "function_score": {
                "query": {"range": {"occurred_at": {"lt": cutoff.isoformat()}}},
                "random_score": {},
            }
        },
        _source=["user_id", "occurred_at"],
        size=pool_size,
    )
    pairs = [
        (h["_source"]["user_id"], h["_source"]["occurred_at"])
        for h in resp["hits"]["hits"]
        if h["_source"].get("user_id") and h["_source"].get("occurred_at")
    ]
    if not pairs:
        raise RuntimeError(f"No events before {cutoff}")
    return pairs


def _parse_anchor(anchor) -> datetime:
    """Convert ES occurred_at value (ISO string or ms epoch) to datetime."""
    if isinstance(anchor, (int, float)):
        return datetime.fromtimestamp(anchor / 1000.0, tz=timezone.utc)
    s = str(anchor).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s[:10]).replace(tzinfo=timezone.utc)


# ── Q function factories ───────────────────────────────────────────────────────

def make_q1_fn(client: Elasticsearch, cutoff: date):
    """
    Q1 scaled: date_histogram on naive_invoices filtered to created_at < cutoff.
    Same structure as the full-scale Q1 — adds a range upper-bound.
    """
    cutoff_iso = cutoff.isoformat()
    def _run():
        client.search(
            index="naive_invoices",
            query={
                "bool": {
                    "filter": [
                        {"term":  {"status": "paid"}},
                        {"range": {"created_at": {"gte": "now-12M/M", "lt": cutoff_iso}}},
                    ]
                }
            },
            aggs={
                "by_month": {
                    "date_histogram": {
                        "field":             "created_at",
                        "calendar_interval": "month",
                        "format":            "yyyy-MM",
                    },
                    "aggs": {
                        "total_revenue": {"sum": {"field": "total_usd"}},
                        "sub_revenue": {
                            "filter": {"term": {"invoice_type": "subscription"}},
                            "aggs":   {"rev": {"sum": {"field": "total_usd"}}},
                        },
                        "mkt_revenue": {
                            "filter": {"term": {"invoice_type": "marketplace"}},
                            "aggs":   {"rev": {"sum": {"field": "total_usd"}}},
                        },
                    },
                }
            },
            size=0,
        )
    return _run


def make_q2_fn(client: Elasticsearch, invoice_ids: list[str]):
    """
    Q2 scaled: same 4-round-trip pattern as full-scale, pool already pre-filtered
    to invoices created before the cutoff.
    """
    def _run():
        invoice_id = random.choice(invoice_ids)
        inv     = client.get(index="naive_invoices", id=invoice_id)
        user_id = inv["_source"].get("user_id")
        if user_id:
            client.get(index="naive_users", id=user_id)
        lines_resp = client.search(
            index="naive_invoice_lines",
            query={"term": {"invoice_id": invoice_id}},
            size=100,
        )
        product_ids = [
            h["_source"]["product_id"]
            for h in lines_resp["hits"]["hits"]
            if h["_source"].get("product_id")
        ]
        if product_ids:
            client.mget(index="naive_products", ids=product_ids)
    return _run


def make_q3_fn(client: Elasticsearch, user_ids: list[str]):
    """
    Q3 scaled: same search-by-user_id pattern, row-based pool (no date cutoff).
    Thread-safe — single shared client handles 50-thread concurrency.
    """
    def _run():
        user_id = random.choice(user_ids)
        client.search(
            index="naive_sessions",
            query={"term": {"user_id": user_id}},
            sort=[{"last_active_at": {"order": "desc"}}],
            size=1,
        )
    return _run


def make_q4_fn(client: Elasticsearch, product_ids: list[str], cutoff: date):
    """
    Q4 scaled: 3-step orchestration with created_at < cutoff filter on orders.
    Product pool pre-filtered to orders before cutoff.
    """
    cutoff_iso = cutoff.isoformat()
    def _run():
        product_id = random.choice(product_ids)

        step1 = client.search(
            index="naive_order_items",
            query={"term": {"product_id": product_id}},
            aggs={"order_ids": {"terms": {"field": "order_id", "size": 5000}}},
            size=0,
        )
        order_ids = [b["key"] for b in step1["aggregations"]["order_ids"]["buckets"]]
        if not order_ids:
            return

        step2 = client.search(
            index="naive_orders",
            query={
                "bool": {
                    "filter": [
                        {"ids":   {"values": order_ids}},
                        {"terms": {"status": ["confirmed", "shipped", "delivered"]}},
                        {"range": {"created_at": {"lt": cutoff_iso}}},
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
    return _run


def make_q5_fn(client: Elasticsearch, cutoff: date):
    """
    Q5 scaled: multi_match with is_active filter + created_at < cutoff.
    Products index uses created_at for date-range scoping.
    """
    cutoff_iso = cutoff.isoformat()
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
                    "filter": [
                        {"term":  {"is_active": True}},
                        {"range": {"created_at": {"lt": cutoff_iso}}},
                    ],
                }
            },
            size=20,
        )
    return _run


def make_q6_fn(client: Elasticsearch, pairs: list[tuple]):
    """
    Q6 scaled: range + term filter on naive_events, anchor pool pre-filtered
    to occurred_at < cutoff. Window centred on anchor — guaranteed non-empty.
    """
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


def make_q7_fn(client: Elasticsearch, cutoff: date, data_min: date):
    """
    Q7 scaled: random 183-day window within [data_min, cutoff], full
    PostgreSQL-equivalent Q7 (per-tier daily revenue, gap-fill, rolling 7-day
    avg) via client-side tier attribution — see es_revenue.naive_q7.
    """
    max_start = max(0, (cutoff - data_min).days - WINDOW_DAYS_Q7)

    def _run():
        offset = random.randint(0, max_start) if max_start > 0 else 0
        start  = data_min + timedelta(days=offset)
        end    = min(start + timedelta(days=WINDOW_DAYS_Q7 - 1), cutoff)
        return naive_q7(client, start, (end - start).days + 1)

    return _run


# ── runner ─────────────────────────────────────────────────────────────────────

def run_scaled_query(
    query_id:    str,
    scale_pct:   int,
    cutoff:      date,
    client:      Elasticsearch,
    cutoffs:     dict,
    iterations:  int,
    results_dir: str,
) -> dict | None:

    output_path = os.path.join(
        results_dir,
        f"elasticsearch_naive_{query_id.lower()}_scale{scale_pct}.json"
    )

    labels = {
        "Q1": (
            f"Q1 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "date_histogram (monthly) on naive_invoices, created_at < cutoff. "
            "No tier breakdown — tier_id absent from invoice docs."
        ),
        "Q2": (
            f"Q2 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "4 round trips: get(invoice) → get(user) → search(lines) → mget(products). "
            "Invoice pool pre-filtered to created_at < cutoff."
        ),
        "Q3": (
            f"Q3 naive at {scale_pct}% row-based scale "
            f"({scale_pct}% of session user_ids). "
            "No date cutoff — sessions generated in 2025. "
            "search(naive_sessions, user_id, sort=last_active_at)."
        ),
        "Q4": (
            f"Q4 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "3-step orchestration: terms agg(order_items) → "
            "ids+status+date filter(orders, created_at < cutoff) → "
            "terms agg(order_items, exclude product X)."
        ),
        "Q5": (
            f"Q5 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "multi_match on name/description/product_type, default BM25. "
            "is_active=True + created_at < cutoff filters."
        ),
        "Q6": (
            f"Q6 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "range(occurred_at) + term(user_id) on naive_events. "
            "Anchor pool pre-filtered to occurred_at < cutoff. "
            "30-day window centred on anchor — guaranteed non-empty."
        ),
        "Q7": (
            f"Q7 naive at {scale_pct}% scale (cutoff {cutoff}). "
            "Full PostgreSQL-equivalent Q7 on [data_min, cutoff]: PIT scans + "
            "client-side tier attribution, per-tier daily revenue, gap-fill, rolling 7-day avg. "
            f"Data min: {cutoffs['min_date']}."
        ),
    }

    try:
        concurrency = 1

        if query_id == "Q1":
            query_fn = make_q1_fn(client, cutoff)

        elif query_id == "Q2":
            pool     = fetch_invoice_pool(client, cutoff)
            info(f"Q2 pool: {len(pool):,} invoice IDs (before {cutoff})")
            query_fn = make_q2_fn(client, pool)

        elif query_id == "Q3":
            concurrency = 50
            pool        = fetch_user_pool_sessions(client, scale_pct)
            info(f"Q3 pool: {len(pool):,} user IDs ({scale_pct}% row-based scale)")
            query_fn    = make_q3_fn(client, pool)

        elif query_id == "Q4":
            pool     = fetch_product_pool(client, cutoff)
            info(f"Q4 pool: {len(pool):,} product IDs (before {cutoff})")
            query_fn = make_q4_fn(client, pool, cutoff)

        elif query_id == "Q5":
            query_fn = make_q5_fn(client, cutoff)

        elif query_id == "Q6":
            pairs    = fetch_anchor_pool(client, cutoff)
            info(f"Q6 pool: {len(pairs):,} anchor pairs (before {cutoff})")
            query_fn = make_q6_fn(client, pairs)

        elif query_id == "Q7":
            query_fn = make_q7_fn(client, cutoff, cutoffs["min_date"])

        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db="elasticsearch_naive",
            query_id=query_id,
            label=labels[query_id],
            iterations=iterations,
            concurrency=concurrency,
            output_path=output_path,
        )
        ok(f"{query_id} @ {scale_pct}% → {output_path}")
        return result

    except Exception as e:
        fail(f"{query_id} @ {scale_pct}% failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Elasticsearch naive scalability baselines"
    )
    parser.add_argument("--scale",       type=int, choices=[10, 50], default=None)
    parser.add_argument("--iterations",  type=int, default=1000)
    parser.add_argument("--dry-run",     action="store_true", dest="dry_run")
    parser.add_argument("--only",        nargs="+", metavar="Q")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, dest="results_dir")
    args = parser.parse_args()

    iterations = 100 if args.dry_run else args.iterations
    scales     = [10, 50] if args.scale is None else [args.scale]
    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]

    print("\n" + "═" * 60)
    print("  Elasticsearch Naive — Scalability Baselines")
    print("═" * 60)
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")
    print()
    print("  Scale method:")
    print("  • Q1/Q2/Q4/Q5/Q6/Q7: date range cutoff (created_at / occurred_at < cutoff)")
    print("  • Q3: row-based — scale_pct% of session user_ids (sessions have no date range)")
    print("  • Q8: excluded (write throughput measured by thread variation)")

    os.makedirs(args.results_dir, exist_ok=True)

    client = get_client()
    try:
        info_resp = client.info()
        info(f"Connected — ES {info_resp['version']['number']}")
    except Exception as e:
        fail(f"Cannot connect to Elasticsearch: {e}")
        sys.exit(1)

    info("Computing date range cutoffs from naive_events...")
    cutoffs = compute_cutoffs(client)
    print(f"  Dataset range : {cutoffs['min_date']} → {cutoffs['max_date']} "
          f"({cutoffs['range_days']} days)")
    print(f"  10% cutoff    : {cutoffs['cutoff_10pct']}")
    print(f"  50% cutoff    : {cutoffs['cutoff_50pct']}")

    all_results = []
    failed      = []
    total_start = time.perf_counter()

    for scale in scales:
        cutoff = cutoffs[f"cutoff_{scale}pct"]
        print(f"\n{'═'*60}")
        print(f"  Elasticsearch naive — {scale}% scale (cutoff: {cutoff})")
        print(f"{'═'*60}")

        for qid in queries:
            print(f"\n{'─'*60}")
            print(f"  {qid} @ {scale}%")
            print(f"{'─'*60}")
            result = run_scaled_query(
                query_id=qid,
                scale_pct=scale,
                cutoff=cutoff,
                client=client,
                cutoffs=cutoffs,
                iterations=iterations,
                results_dir=args.results_dir,
            )
            if result:
                result["scale_pct"]   = scale
                result["cutoff_date"] = str(cutoff)
                all_results.append(result)
            else:
                failed.append(f"{qid}@{scale}%")

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'═'*60}")
    print(f"  SCALABILITY SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Query':<8} {'Scale':>6} {'p50':>10} {'p95':>10} {'p99':>10}")
    print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for r in all_results:
        lms = r.get("latency_ms", {})
        print(
            f"  {r['query_id']:<8} {str(r.get('scale_pct','?'))+'%':>6} "
            f"{lms.get('p50', 0):>9.2f}ms "
            f"{lms.get('p95', 0):>9.2f}ms "
            f"{lms.get('p99', 0):>9.2f}ms"
        )

    summary_path = os.path.join(args.results_dir, "elasticsearch_naive_scalability_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db": "elasticsearch_naive",
            "cutoffs": {k: str(v) for k, v in cutoffs.items()},
            "benchmarks": all_results,
        }, f, indent=2)
    info(f"Summary saved → {summary_path}")

    print(f"\n  Total wall time : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Completed       : {len(all_results)} / {len(all_results) + len(failed)}")

    if failed:
        warn(f"Failed: {', '.join(failed)}")
        print(f"\n  Re-run failures with:")
        print(f"    python run_scalability.py --only {' '.join(q.split('@')[0] for q in failed)}\n")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All Elasticsearch naive scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()