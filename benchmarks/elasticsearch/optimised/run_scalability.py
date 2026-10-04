"""
benchmarks/elasticsearch/optimised/run_scalability.py — Elasticsearch Optimised Scalability
=============================================================================================
Re-runs Q1–Q7 at 10% and 50% data scale to establish the Elasticsearch optimised
scalability curve for Chart 3.

Scale methodology — identical to all other databases
──────────────────────────────────────────────────────
  10% scale : queries restricted to the first 10% of the dataset date range
  50% scale : queries restricted to the first 50% of the date range
  100% scale: full dataset — already captured by run_benchmarks.py

Cutoffs computed from actual min/max of occurred_at in optimised_events,
consistent with all other databases.

Q3 uses row-based scaling — sessions in 2025, date cutoff returns zero rows.
We sample scale_pct% of distinct session IDs instead (same exception as all DBs).

Q8 excluded — write throughput scaling measured separately.

Optimised SQL/query changes vs naive scalability
─────────────────────────────────────────────────
  Q1: identical to naive (no tier_id embedding — confirmed by loader inspection).

  Q2: single GET by invoice ID (pool pre-filtered to created_at < cutoff).
      1 round trip vs naive's 4. Pool pre-filtered to cutoff.

  Q3: GET by session_id (row-based pool, no date cutoff).
      vs naive's search-by-user_id. Native array cart, no JSON parse.

  Q4: identical 3-step orchestration on optimised indices + created_at < cutoff
      filter on orders. No schema effect.

  Q5: best_fields multi_match with name^3/description^1.5 boosts + custom
      English analyser. created_at < cutoff filter on optimised_products.

  Q6: identical query on optimised_events (composite sort at storage level).
      Anchor pool pre-filtered to occurred_at < cutoff.

  Q7: date_histogram with extended_bounds + min_doc_count=0 for gap-filling.
      Correct rolling average (no missing days). Scoped to [data_min, cutoff].

Usage:
    cd benchmarks/elasticsearch/optimised
    python run_scalability.py                    # both scales, 1000 iterations
    python run_scalability.py --scale 10
    python run_scalability.py --scale 50
    python run_scalability.py --iterations 100
    python run_scalability.py --only Q2 Q3 Q7
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
from benchmarks.elasticsearch.es_revenue import optimised_q7, optimised_tier_names

load_dotenv()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "benchmarks", "elasticsearch", "optimised", "results", "scale")
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

    min_d      = _ms_to_date(resp["aggregations"]["min_ts"]["value"])
    max_d      = _ms_to_date(resp["aggregations"]["max_ts"]["value"])
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
    """Invoice IDs with created_at < cutoff — used for Q2 GET by ID."""
    resp = client.search(
        index="optimised_invoices",
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


def fetch_session_pool(client: Elasticsearch, scale_pct: int, pool_size: int = 1000) -> list[str]:
    """
    Row-based scaling for Q3: sample scale_pct% of distinct session IDs.
    Date cutoff not applicable — sessions were generated in 2025.
    Pool contains session document _ids (session tokens) for direct GET.
    """
    count_resp = client.search(
        index="optimised_sessions",
        aggs={"total": {"value_count": {"field": "user_id"}}},
        size=0,
    )
    # Use cardinality of user_id as proxy for distinct session count
    card_resp = client.search(
        index="optimised_sessions",
        aggs={"distinct": {"cardinality": {"field": "user_id"}}},
        size=0,
    )
    total = card_resp["aggregations"]["distinct"]["value"]
    limit = max(1, int(total * scale_pct / 100))

    resp = client.search(
        index="optimised_sessions",
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        _source=False,
        size=min(limit, 10000),
    )
    ids = [h["_id"] for h in resp["hits"]["hits"]]
    if not ids:
        raise RuntimeError("optimised_sessions is empty")
    return random.choices(ids, k=min(pool_size, len(ids)))


def fetch_product_pool(client: Elasticsearch, cutoff: date, pool_size: int = 1000) -> list[str]:
    """
    Product IDs in confirmed/shipped/delivered orders where created_at < cutoff.
    Two-step: collect valid order IDs from optimised_orders, then agg product IDs.
    """
    orders_resp = client.search(
        index="optimised_orders",
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
    valid_order_ids = [h["_id"] for h in orders_resp["hits"]["hits"]]
    if not valid_order_ids:
        raise RuntimeError(f"No valid orders before {cutoff}")

    items_resp = client.search(
        index="optimised_order_items",
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
    Random (user_id, occurred_at) pairs from optimised_events where occurred_at < cutoff.
    30-day window centred on anchor → guaranteed non-empty result each iteration.
    """
    resp = client.search(
        index="optimised_events",
        query={"function_score": {
            "query": {"range": {"occurred_at": {"lt": cutoff.isoformat()}}},
            "random_score": {},
        }},
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
    Q1 scaled: identical to naive — date_histogram on optimised_invoices
    filtered to created_at < cutoff. No tier_id embedding in this schema.
    """
    cutoff_iso = cutoff.isoformat()
    def _run():
        client.search(
            index="optimised_invoices",
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
    Q2 scaled: single GET by invoice ID.
    Schema effect vs naive: 1 round trip (vs 4). Pool pre-filtered to cutoff.
    """
    def _run():
        client.get(index="optimised_invoices", id=random.choice(invoice_ids))
    return _run


def make_q3_fn(client: Elasticsearch, session_ids: list[str]):
    """
    Q3 scaled: single GET by session_id.
    Row-based pool (no date cutoff). Thread-safe shared client handles 50 threads.
    """
    def _run():
        client.get(index="optimised_sessions", id=random.choice(session_ids))
    return _run


def make_q4_fn(client: Elasticsearch, product_ids: list[str], cutoff: date):
    """
    Q4 scaled: 3-step orchestration with created_at < cutoff on orders.
    No schema effect vs naive.
    """
    cutoff_iso = cutoff.isoformat()
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


def make_q5_fn(client: Elasticsearch, cutoff: date):
    """
    Q5 scaled: best_fields multi_match with boosts + is_active + created_at < cutoff.
    Schema effect: name^3/description^1.5 + custom English analyser vs naive's plain BM25.
    """
    cutoff_iso = cutoff.isoformat()
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
    Q6 scaled: range + term filter on optimised_events. Anchor pool pre-filtered
    to occurred_at < cutoff. 30-day window centred on anchor.
    """
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


def make_q7_fn(client: Elasticsearch, cutoff: date, data_min: date):
    """
    Q7 scaled: random 183-day window within [data_min, cutoff], full
    PostgreSQL-equivalent Q7 in one server-side search (filters(tier) →
    date_histogram + extended_bounds → sum + moving_fn) — see es_revenue.optimised_q7.
    """
    max_start  = max(0, (cutoff - data_min).days - WINDOW_DAYS_Q7)
    tier_names = optimised_tier_names(client)          # startup, untimed

    def _run():
        offset = random.randint(0, max_start) if max_start > 0 else 0
        start  = data_min + timedelta(days=offset)
        end    = min(start + timedelta(days=WINDOW_DAYS_Q7 - 1), cutoff)
        return optimised_q7(client, start, (end - start).days + 1, tier_names)

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
        f"elasticsearch_optimised_{query_id.lower()}_scale{scale_pct}.json"
    )

    labels = {
        "Q1": (
            f"Q1 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "date_histogram (monthly) on optimised_invoices, created_at < cutoff. "
            "No schema effect — identical to naive (tier_id not embedded by loader)."
        ),
        "Q2": (
            f"Q2 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Single get(optimised_invoices) by ID. "
            "Schema effect: embedded lines → 1 round trip (vs 4 in naive). "
            "Invoice pool pre-filtered to created_at < cutoff."
        ),
        "Q3": (
            f"Q3 optimised at {scale_pct}% row-based scale "
            f"({scale_pct}% of session IDs). "
            "Single get(optimised_sessions) by session_id. "
            "Schema effect: native cart array + GET (vs search + JSON parse in naive). "
            "No date cutoff — sessions generated in 2025."
        ),
        "Q4": (
            f"Q4 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "3-step orchestration on optimised indices, "
            "orders filtered to created_at < cutoff. "
            "No schema effect — identical orchestration to naive."
        ),
        "Q5": (
            f"Q5 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "multi_match best_fields: name^3 / description^1.5 / product_type. "
            "Custom English analyser. is_active=True + created_at < cutoff. "
            "Schema effect: field boosts + analyser vs naive plain BM25."
        ),
        "Q6": (
            f"Q6 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "range(occurred_at) + term(user_id) on optimised_events. "
            "Anchor pool pre-filtered to occurred_at < cutoff. "
            "30-day window centred on anchor — guaranteed non-empty. "
            "Schema effect: composite sort optimisation at storage level."
        ),
        "Q7": (
            f"Q7 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Full PostgreSQL-equivalent Q7 on [data_min, cutoff], one server-side search: "
            "filters(tier) → date_histogram(min_doc_count=0, extended_bounds) → sum + moving_fn. "
            f"Data min: {cutoffs['min_date']}."
        ),
    }

    try:
        concurrency = 1

        if query_id == "Q1":
            query_fn = make_q1_fn(client, cutoff)

        elif query_id == "Q2":
            pool = fetch_invoice_pool(client, cutoff)
            info(f"Q2 pool: {len(pool):,} invoice IDs (before {cutoff})")
            query_fn = make_q2_fn(client, pool)

        elif query_id == "Q3":
            concurrency = 50
            pool        = fetch_session_pool(client, scale_pct)
            info(f"Q3 pool: {len(pool):,} session IDs ({scale_pct}% row-based scale)")
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
            db="elasticsearch_optimised",
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
        description="Elasticsearch optimised scalability baselines"
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
    print("  Elasticsearch Optimised — Scalability Baselines")
    print("═" * 60)
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")
    print()
    print("  Schema effects vs naive scalability:")
    print("  • Q1: none (tier_id not embedded — identical to naive)")
    print("  • Q2: 1 GET (vs 4 round trips) — embedded lines")
    print("  • Q3: GET by session_id (vs search by user_id) — row-based scale")
    print("  • Q4: none (same 3-step orchestration)")
    print("  • Q5: name^3/desc^1.5 boosts + custom English analyser")
    print("  • Q6: composite sort at storage level (query unchanged)")
    print("  • Q7: extended_bounds gap-filling → correct rolling average")
    print("  • Q8: excluded (write throughput measured by thread variation)")

    os.makedirs(args.results_dir, exist_ok=True)

    client = get_client()
    try:
        info_resp = client.info()
        info(f"Connected — ES {info_resp['version']['number']}")
    except Exception as e:
        fail(f"Cannot connect to Elasticsearch: {e}")
        sys.exit(1)

    info("Computing date range cutoffs from optimised_events...")
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
        print(f"  Elasticsearch optimised — {scale}% scale (cutoff: {cutoff})")
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

    summary_path = os.path.join(args.results_dir, "elasticsearch_optimised_scalability_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db": "elasticsearch_optimised",
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
        print(f"\n  {GREEN}All Elasticsearch optimised scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()