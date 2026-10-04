"""
benchmarks/neo4j/run_scalability.py — Neo4j Scalability Baselines
==================================================================
Re-runs Q1–Q7 at 10% and 50% data scale for both naive and optimised
schemas to establish the Neo4j scalability curves for Chart 3.

Scale is defined by date range, not row count — identical methodology
to the PostgreSQL run_scalability.py:
  - 10% scale : first 10% of the dataset's event date range
  - 50% scale : first 50% of the date range
  - 100% scale: full dataset — already captured by individual q*.py runs

Cutoff dates are computed from the actual MIN/MAX of Event.occurred_at
in the naive container — same anchor as PostgreSQL for a fair comparison.

Q3 uses row-based scaling (user pool drawn from scale_pct% of total
session users) — same exception as PostgreSQL. Sessions fall within a
single year making date-range scaling inapplicable.

Q7 is the partial Neo4j equivalent (daily aggregation only — no
gap-filling or rolling average). Consistent with the standalone q7 files.

Q4 optimised note: ALSO_BOUGHT edges are precomputed over all data at
load time and cannot be date-filtered without reloading. Scalability for
optimised Q4 is run with the full edge set and this limitation is noted
in the output label and methodology chapter.

Q8 excluded — write throughput scaling is measured by thread variation,
not date range.

Usage:
    python run_scalability.py                    # both scales, 1000 iterations
    python run_scalability.py --scale 10         # 10% only
    python run_scalability.py --scale 50         # 50% only
    python run_scalability.py --iterations 100   # smoke test
    python run_scalability.py --only Q4 Q6       # specific queries only
    python run_scalability.py --schema naive      # naive only
    python run_scalability.py --schema optimised  # optimised only
    python run_scalability.py --dry-run           # 100 iterations, both scales
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
WINDOW_DAYS  = 30    # Q6 window
WINDOW_DAYS7 = 183   # Q7 window

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

TIER_NAMES   = {"1": "Free", "2": "Pro", "3": "Business"}
TIER_PRICING = [
    {"tier_id": "1", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": None,   "monthly_price_usd": 0.00},
    {"tier_id": "2", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 14.99},
    {"tier_id": "2", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,   "monthly_price_usd": 19.99},
    {"tier_id": "3", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 39.99},
    {"tier_id": "3", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,   "monthly_price_usd": 49.99},
]
SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour", "photoshop brushes", "video editing",
    "certificate course", "logo design", "colour palette", "font pack",
    "texture pack", "motion graphics", "social media", "icon set",
    "web design", "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

# ── cutoff computation ────────────────────────────────────────────────────────

def compute_cutoffs(driver) -> dict:
    """
    Compute 10% and 50% date cutoffs from Event.occurred_at range.
    Uses the naive container — same anchor as PostgreSQL scalability.
    """
    cypher = """
    MATCH (e:Event)
    RETURN min(e.occurred_at) AS min_dt, max(e.occurred_at) AS max_dt
    """
    with driver.session() as session:
        row = session.run(cypher).single()
    if not row or not row["min_dt"]:
        raise RuntimeError("No Event nodes found — run the naive loader first.")

    min_date = date.fromisoformat(row["min_dt"][:10])
    max_date = date.fromisoformat(row["max_dt"][:10])
    range_days = (max_date - min_date).days

    return {
        "min_date":     min_date,
        "max_date":     max_date,
        "range_days":   range_days,
        "cutoff_10pct": min_date + timedelta(days=int(range_days * 0.10)),
        "cutoff_50pct": min_date + timedelta(days=int(range_days * 0.50)),
    }

# ── helpers ───────────────────────────────────────────────────────────────────

def get_price_for_tier_at(tier_id, created_at):
    for p in TIER_PRICING:
        if p["tier_id"] != str(tier_id):
            continue
        if p["valid_from"] <= created_at and (p["valid_to"] is None or p["valid_to"] > created_at):
            return p["monthly_price_usd"]
    return 0.0

# ── Q1 ────────────────────────────────────────────────────────────────────────

Q1_SUB_CYPHER = """
MATCH (i:Invoice)-[:FOR_SUBSCRIPTION]->(s:Subscription)-[:ON_TIER]->(t:SubscriptionTier)
WHERE i.status = 'paid' AND i.invoice_type = 'subscription'
  AND i.created_at >= $cutoff_12m AND i.created_at < $cutoff
RETURN i.created_at AS created_at, i.total_usd AS total_usd, t.id AS tier_id
"""
Q1_MKT_CYPHER = """
MATCH (u:User)-[:HAS_INVOICE]->(i:Invoice)
WHERE i.status = 'paid' AND i.invoice_type = 'marketplace'
  AND i.created_at >= $cutoff_12m AND i.created_at < $cutoff
MATCH (u)-[:HAS_SUBSCRIPTION]->(s:Subscription)
WHERE s.started_at <= i.created_at
WITH i, s ORDER BY s.started_at DESC
WITH i, head(collect(s)) AS active_sub
WHERE active_sub IS NOT NULL
RETURN i.created_at AS created_at, i.total_usd AS total_usd, active_sub.tier_id AS tier_id
"""

def make_q1_fn(driver, cutoff: date):
    cutoff_iso   = cutoff.isoformat()
    cutoff_12m   = (cutoff - timedelta(days=365)).isoformat()
    def _run():
        with driver.session() as session:
            sub_rows = session.run(Q1_SUB_CYPHER, cutoff=cutoff_iso, cutoff_12m=cutoff_12m).data()
            mkt_rows = session.run(Q1_MKT_CYPHER, cutoff=cutoff_iso, cutoff_12m=cutoff_12m).data()
        agg = defaultdict(lambda: {"invoice_count": 0, "total_revenue_usd": 0.0, "price_in_effect_usd": 0.0})
        for row in sub_rows + mkt_rows:
            tier_id = str(row["tier_id"])
            key = (row["created_at"][:7], TIER_NAMES.get(tier_id, tier_id))
            agg[key]["invoice_count"] += 1
            agg[key]["total_revenue_usd"] += float(row["total_usd"] or 0)
            agg[key]["price_in_effect_usd"] = get_price_for_tier_at(tier_id, row["created_at"])
        return list(agg.items())
    return _run

# ── Q2 ────────────────────────────────────────────────────────────────────────

Q2_CYPHER = """
MATCH (u:User)-[:HAS_INVOICE]->(i:Invoice {id: $invoice_id})
MATCH (i)-[:HAS_LINE]->(il:InvoiceLine)
OPTIONAL MATCH (il)-[:LINE_FOR_PRODUCT]->(p:Product)
RETURN i.id, i.invoice_type, i.status, i.total_usd, i.created_at,
       u.id AS customer_id, u.full_name, u.email,
       il.id AS line_id, il.description, il.quantity, il.line_total_usd,
       p.id AS product_id, p.name AS product_name, p.product_type
ORDER BY il.created_at
"""
def fetch_invoice_pool(driver, cutoff: date, pool_size=1000) -> list[str]:
    cypher = f"""
    MATCH (i:Invoice)
    WHERE i.created_at < $cutoff
    WITH i, rand() AS r ORDER BY r LIMIT {pool_size}
    RETURN i.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, cutoff=cutoff.isoformat())
        ids = [row["id"] for row in result]
    if not ids:
        raise RuntimeError(f"No invoices before {cutoff}")
    info(f"  Q2 pool: {len(ids)} invoice IDs (cutoff {cutoff})")
    return ids

def make_q2_fn(driver, pool: list[str]):
    def _run():
        with driver.session() as session:
            session.run(Q2_CYPHER, invoice_id=random.choice(pool)).data()
    return _run

# ── Q3 (row-based scaling) ────────────────────────────────────────────────────

Q3_CYPHER = """
MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)
RETURN s.id AS id, s.user_id AS user_id, s.cart AS cart,
       s.ip_address AS ip_address, s.last_active_at AS last_active_at,
       s.expires_at AS expires_at
ORDER BY s.last_active_at DESC LIMIT 1
"""
def fetch_user_pool(driver, scale_pct: int, pool_size=5000) -> list[str]:
    """
    Row-based: sample scale_pct% of total session users.
    pool_size is embedded as a literal — Neo4j does not support integer
    parameters in LIMIT clauses inside WITH chains reliably.
    """
    scaled_pool = max(10, int(pool_size * scale_pct / 100))
    # Embed limit as a literal integer, not a parameter
    cypher = f"""
    MATCH (u:User)-[:HAS_SESSION]->(:Session)
    WITH DISTINCT u, rand() AS r
    ORDER BY r LIMIT {scaled_pool}
    RETURN u.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher)
        ids = [row["id"] for row in result]
    if not ids:
        raise RuntimeError("No users with sessions found")
    info(f"  Q3 pool: {len(ids)} user IDs ({scale_pct}% row-based scaling, limit {scaled_pool})")
    return ids

def make_q3_fn(driver, pool: list[str], schema: str):
    import json as _json
    def _run():
        with driver.session() as session:
            row = session.run(Q3_CYPHER, user_id=random.choice(pool)).single()
            if row and row["cart"] and schema == "naive":
                _json.loads(row["cart"])
    return _run

# ── Q4 naive ──────────────────────────────────────────────────────────────────
#
# Exact Cypher from benchmarks/neo4j/naive/q4_recommendations.py (working version).
# Relationship direction: (Product)<-[:FOR_PRODUCT]-(OrderItem)<-[:CONTAINS]-(Order)
# Date cutoff added to Order filter for scalability — not present in standalone Q4.
# No is_active filter on rec — consistent with standalone file.

Q4_NAIVE_CYPHER = """
MATCH (target:Product {id: $product_id})<-[:FOR_PRODUCT]-(oi1:OrderItem)
      <-[:CONTAINS]-(o:Order)
WHERE o.status IN ['confirmed', 'shipped', 'delivered']
  AND o.created_at < $cutoff
MATCH (o)-[:CONTAINS]->(oi2:OrderItem)-[:FOR_PRODUCT]->(rec:Product)
WHERE rec.id <> target.id
WITH rec, count(DISTINCT o) AS co_purchase_count, count(DISTINCT oi1) AS total_orders
ORDER BY co_purchase_count DESC, rec.name
LIMIT 10
RETURN
    rec.id           AS product_id,
    rec.name         AS product_name,
    rec.product_type AS product_type,
    rec.price_usd    AS price_usd,
    co_purchase_count,
    ROUND(toFloat(co_purchase_count)/total_orders, 4) AS confidence
"""

Q4_NAIVE_POOL_CYPHER = """
MATCH (o:Order)-[:CONTAINS]->(oi1:OrderItem)-[:FOR_PRODUCT]->(p:Product)
WHERE o.status IN ['confirmed', 'shipped', 'delivered']
  AND o.created_at < $cutoff
MATCH (o)-[:CONTAINS]->(oi2:OrderItem)-[:FOR_PRODUCT]->(other:Product)
WHERE other.id <> p.id
WITH DISTINCT p, rand() AS r
ORDER BY r LIMIT $pool_size
RETURN p.id AS id
"""

# Q4 optimised — ALSO_BOUGHT edges precomputed at load time, no date filter possible.
# Exact Cypher from benchmarks/neo4j/optimised/q4_recommendations.py (working version).

Q4_OPT_CYPHER = """
MATCH (target:Product {id: $product_id})-[r:ALSO_BOUGHT]->(rec:Product)
RETURN
    rec.id           AS product_id,
    rec.name         AS product_name,
    rec.product_type AS product_type,
    rec.price_usd    AS price_usd,
    r.count          AS co_purchase_count,
    r.confidence     AS confidence
ORDER BY r.count DESC, rec.name
LIMIT 10
"""

Q4_OPT_POOL_CYPHER = """
MATCH (p:Product)-[:ALSO_BOUGHT]->(:Product)
WITH DISTINCT p, rand() AS r
ORDER BY r LIMIT $pool_size
RETURN p.id AS id
"""

def fetch_product_pool_naive(driver, cutoff: date, pool_size=1000) -> list[str]:
    cypher = f"""
    MATCH (o:Order)-[:CONTAINS]->(oi1:OrderItem)-[:FOR_PRODUCT]->(p:Product)
    WHERE o.status IN ['confirmed', 'shipped', 'delivered']
      AND o.created_at < $cutoff
    MATCH (o)-[:CONTAINS]->(oi2:OrderItem)-[:FOR_PRODUCT]->(other:Product)
    WHERE other.id <> p.id
    WITH DISTINCT p, rand() AS r
    ORDER BY r LIMIT {pool_size}
    RETURN p.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, cutoff=cutoff.isoformat())
        ids = [row["id"] for row in result]
    if not ids:
        raise RuntimeError(f"No products in confirmed orders before {cutoff}")
    info(f"  Q4 naive pool: {len(ids)} product IDs (cutoff {cutoff})")
    return ids

def fetch_product_pool_optimised(driver, pool_size=1000) -> list[str]:
    cypher = f"""
    MATCH (p:Product)-[:ALSO_BOUGHT]->(:Product)
    WITH DISTINCT p, rand() AS r
    ORDER BY r LIMIT {pool_size}
    RETURN p.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher)
        ids = [row["id"] for row in result]
    if not ids:
        raise RuntimeError("No ALSO_BOUGHT edges found")
    info(f"  Q4 optimised pool: {len(ids)} product IDs (full dataset — precomputed edges)")
    return ids

def make_q4_fn(driver, pool: list[str], schema: str, cutoff: date):
    cutoff_iso = cutoff.isoformat()
    def _run():
        product_id = random.choice(pool)
        with driver.session() as session:
            if schema == "naive":
                session.run(Q4_NAIVE_CYPHER, product_id=product_id, cutoff=cutoff_iso).data()
            else:
                session.run(Q4_OPT_CYPHER, product_id=product_id).data()
    return _run

# ── Q5 ────────────────────────────────────────────────────────────────────────

Q5_NAIVE_SINGLE = """
MATCH (p:Product)
WHERE p.is_active = 'True' AND p.created_at < $cutoff
  AND (toLower(p.name) CONTAINS toLower($term)
    OR toLower(p.description) CONTAINS toLower($term))
RETURN p.id, p.name, p.product_type, p.price_usd
ORDER BY p.name LIMIT 20
"""
Q5_OPT_CYPHER = """
CALL db.index.fulltext.queryNodes('product_fulltext', $term)
YIELD node AS p, score
WHERE p.is_active = 'True' AND p.created_at < $cutoff
RETURN p.id, p.name, p.product_type, p.price_usd, score
ORDER BY score DESC LIMIT 20
"""

def make_q5_fn(driver, cutoff: date, schema: str):
    cutoff_iso = cutoff.isoformat()
    def _run():
        term = random.choice(SEARCH_TERMS)
        with driver.session() as session:
            if schema == "naive":
                session.run(Q5_NAIVE_SINGLE, term=term, cutoff=cutoff_iso).data()
            else:
                session.run(Q5_OPT_CYPHER, term=term, cutoff=cutoff_iso).data()
    return _run

# ── Q6 ────────────────────────────────────────────────────────────────────────

Q6_CYPHER = """
MATCH (u:User {id: $user_id})-[:TRIGGERED]->(e:Event)
WHERE e.occurred_at >= $start AND e.occurred_at < $end
RETURN e.id, e.event_type, e.occurred_at, e.product_id, e.session_id, e.metadata
ORDER BY e.occurred_at DESC
"""
Q6_POOL_CYPHER = """
MATCH (u:User)-[:TRIGGERED]->(e:Event)
WHERE e.occurred_at < $cutoff
WITH u.id AS user_id, e.occurred_at AS occurred_at, rand() AS r
ORDER BY r LIMIT $pool_size
RETURN user_id, occurred_at
"""

def fetch_anchor_pool(driver, cutoff: date, pool_size=1000) -> list[tuple]:
    cypher = f"""
    MATCH (u:User)-[:TRIGGERED]->(e:Event)
    WHERE e.occurred_at < $cutoff
    WITH u.id AS user_id, e.occurred_at AS occurred_at, rand() AS r
    ORDER BY r LIMIT {pool_size}
    RETURN user_id, occurred_at
    """
    with driver.session() as session:
        result = session.run(cypher, cutoff=cutoff.isoformat())
        pairs = [(row["user_id"], row["occurred_at"]) for row in result]
    if not pairs:
        raise RuntimeError(f"No events before {cutoff}")
    info(f"  Q6 pool: {len(pairs)} (user_id, anchor) pairs (cutoff {cutoff})")
    return pairs

def anchor_window(anchor_str: str) -> tuple[str, str]:
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except Exception:
        anchor = datetime.fromisoformat(anchor_str.replace("Z", "+00:00"))
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return (anchor - timedelta(days=15)).isoformat(), (anchor + timedelta(days=15)).isoformat()

def make_q6_fn(driver, pairs: list[tuple]):
    def _run():
        user_id, anchor_str = random.choice(pairs)
        start, end = anchor_window(anchor_str)
        with driver.session() as session:
            session.run(Q6_CYPHER, user_id=user_id, start=start, end=end).data()
    return _run

# ── Q7 ────────────────────────────────────────────────────────────────────────

# Full PostgreSQL-equivalent Q7 (daily revenue per tier + gap-fill + rolling
# 7-day avg, all server-side) — same Cypher as benchmarks/neo4j/*/q7_rolling_revenue.py
from benchmarks.neo4j.naive.q7_rolling_revenue import Q7_CYPHER


def make_q7_fn(driver, cutoff: date, data_min: date):
    def _run():
        delta     = (cutoff - data_min).days
        max_start = max(0, delta - WINDOW_DAYS7)
        start     = data_min + timedelta(days=random.randint(0, max_start))
        end       = min(start + timedelta(days=WINDOW_DAYS7 - 1), cutoff)
        with driver.session() as session:
            return session.run(Q7_CYPHER, start=start.isoformat(),
                               end=(end + timedelta(days=1)).isoformat(),
                               n_days=(end - start).days + 1).data()
    return _run

# ── per-query runner ──────────────────────────────────────────────────────────

def run_scaled_query(
    query_id: str,
    schema: str,
    scale_pct: int,
    cutoff: date,
    driver,
    cutoffs: dict,
    iterations: int,
    results_dir: str,
) -> dict | None:

    suffix      = f"scale{scale_pct}"
    output_path = os.path.join(
        results_dir, f"neo4j_{schema}_{query_id.lower()}_{suffix}.json"
    )

    try:
        concurrency = 1
        if query_id == "Q1":
            query_fn = make_q1_fn(driver, cutoff)
            label    = f"Q1 at {scale_pct}% scale (cutoff {cutoff})"
        elif query_id == "Q2":
            pool     = fetch_invoice_pool(driver, cutoff)
            query_fn = make_q2_fn(driver, pool)
            label    = f"Q2 at {scale_pct}% scale (cutoff {cutoff}, pool {len(pool)})"
        elif query_id == "Q3":
            pool        = fetch_user_pool(driver, scale_pct)
            query_fn    = make_q3_fn(driver, pool, schema)
            concurrency = 50
            label       = f"Q3 at {scale_pct}% row-based scale ({len(pool)} users). No date cutoff — sessions in single year."
        elif query_id == "Q4":
            if schema == "naive":
                pool     = fetch_product_pool_naive(driver, cutoff)
                label    = f"Q4 naive at {scale_pct}% scale (cutoff {cutoff}, 2-hop traversal)"
            else:
                pool     = fetch_product_pool_optimised(driver)
                label    = (f"Q4 optimised at {scale_pct}% scale — ALSO_BOUGHT edges are "
                            "precomputed over full dataset, date-range filtering not applicable.")
            query_fn = make_q4_fn(driver, pool, schema, cutoff)
        elif query_id == "Q5":
            query_fn = make_q5_fn(driver, cutoff, schema)
            label    = f"Q5 {schema} at {scale_pct}% scale (cutoff {cutoff})"
        elif query_id == "Q6":
            pool     = fetch_anchor_pool(driver, cutoff)
            query_fn = make_q6_fn(driver, pool)
            label    = f"Q6 at {scale_pct}% scale (cutoff {cutoff}, anchor-centred 30-day windows)"
        elif query_id == "Q7":
            query_fn = make_q7_fn(driver, cutoff, cutoffs["min_date"])
            label    = (f"Q7 {schema} at {scale_pct}% scale (cutoff {cutoff}). "
                        "Full PostgreSQL-equivalent Q7: one Cypher query — per-tier daily revenue, "
                        "range()-based gap-fill, rolling 7-day avg (list slice).")
        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db=f"neo4j_{schema}",
            query_id=query_id,
            label=label,
            iterations=iterations,
            concurrency=concurrency,
            output_path=output_path,
        )
        ok(f"{query_id} {schema} @ {scale_pct}% → {output_path}")
        return result

    except Exception as e:
        fail(f"{query_id} {schema} @ {scale_pct}% failed: {e}")
        import traceback; traceback.print_exc()
        return None

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j scalability baselines")
    parser.add_argument("--scale",      type=int, choices=[10, 50], default=None,
                        help="Run only one scale (10 or 50). Default: both.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run",    action="store_true", dest="dry_run")
    parser.add_argument("--only",       nargs="+", metavar="Q",
                        help="Run only specific queries e.g. --only Q4 Q6")
    parser.add_argument("--schema",     choices=["naive", "optimised", "both"], default="both",
                        help="Which schema to run (default: both)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, dest="results_dir")
    args = parser.parse_args()

    iterations = 100 if args.dry_run else args.iterations
    scales     = [10, 50] if args.scale is None else [args.scale]
    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]
    schemas = ["naive", "optimised"] if args.schema == "both" else [args.schema]

    print("\n" + "═" * 60)
    print("  Neo4j — Scalability Baselines")
    print("═" * 60)
    print(f"  Schemas     : {schemas}")
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Compute cutoffs from naive container (same data, same anchor as PostgreSQL)
    # Cutoffs come from Event.occurred_at. Both containers hold the same Event
    # data (same CSV), so if the naive container is not running (only one Neo4j
    # fits in Docker memory at a time) the optimised container is used instead.
    try:
        naive_driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))
        naive_driver.verify_connectivity()
        cutoff_source = "naive"
    except Exception:
        naive_driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))
        cutoff_source = "optimised"
    cutoffs = compute_cutoffs(naive_driver)
    naive_driver.close()
    info(f"Cutoffs (from {cutoff_source} container): 10% = {cutoffs['cutoff_10pct']}, "
         f"50% = {cutoffs['cutoff_50pct']}  (expected 2024-03-20 / 2025-01-04)")

    print(f"  Dataset range : {cutoffs['min_date']} → {cutoffs['max_date']} ({cutoffs['range_days']} days)")
    print(f"  10% cutoff    : {cutoffs['cutoff_10pct']}")
    print(f"  50% cutoff    : {cutoffs['cutoff_50pct']}")

    all_results = []
    failed      = []
    total_start = time.perf_counter()

    for schema in schemas:
        port   = int(os.getenv("NEO4J_NAIVE_PORT", 7687) if schema == "naive"
                     else os.getenv("NEO4J_OPTIMISED_PORT", 7688))
        # Pool size must cover Q3's concurrency=50 via the harness.
        # All other queries are single-threaded so the extra pool slots are idle.
        driver = get_driver(port=port, max_connection_pool_size=60)
        info(f"Connected to Neo4j {schema} (port {port})")

        for scale in scales:
            cutoff = cutoffs[f"cutoff_{scale}pct"]
            print(f"\n{'═'*60}")
            print(f"  Neo4j {schema} — {scale}% scale (cutoff: {cutoff})")
            print(f"{'═'*60}")

            for qid in queries:
                print(f"\n{'─'*60}")
                print(f"  {qid} | {schema} | {scale}%")
                print(f"{'─'*60}")
                result = run_scaled_query(
                    query_id=qid,
                    schema=schema,
                    scale_pct=scale,
                    cutoff=cutoff,
                    driver=driver,
                    cutoffs=cutoffs,
                    iterations=iterations,
                    results_dir=args.results_dir,
                )
                if result:
                    result["scale_pct"]   = scale
                    result["schema"]      = schema
                    result["cutoff_date"] = str(cutoff)
                    all_results.append(result)
                else:
                    failed.append(f"{qid}_{schema}@{scale}%")

        driver.close()

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'═'*60}")
    print(f"  SCALABILITY SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Query':<6} {'Schema':<12} {'Scale':>6} {'p50':>10} {'p95':>10} {'p99':>10}")
    print(f"  {'─'*6} {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for r in all_results:
        lms = r.get("latency_ms", {})
        print(
            f"  {r['query_id']:<6} {r.get('schema',''):<12} "
            f"{str(r.get('scale_pct','?'))+'%':>6} "
            f"{lms.get('p50', 0):>9.2f}ms "
            f"{lms.get('p95', 0):>9.2f}ms "
            f"{lms.get('p99', 0):>9.2f}ms"
        )

    summary_path = os.path.join(args.results_dir, "neo4j_scalability_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db": "neo4j",
            "cutoffs": {k: str(v) for k, v in cutoffs.items()},
            "benchmarks": all_results,
        }, f, indent=2)
    info(f"Summary saved → {summary_path}")

    print(f"\n  Total wall time : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Completed       : {len(all_results)} / {len(all_results) + len(failed)}")

    if failed:
        warn(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All Neo4j scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()