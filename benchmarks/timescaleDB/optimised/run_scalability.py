"""
benchmarks/timescaledb/optimised/run_scalability.py — TimescaleDB Optimised Scalability
========================================================================================
Re-runs Q1–Q7 at 10% and 50% data scale to establish the TimescaleDB optimised
scalability curve for Chart 3.

Scale methodology — identical to all other databases
──────────────────────────────────────────────────────
  10% scale : queries restricted to the first 10% of the dataset date range
  50% scale : queries restricted to the first 50% of the date range
  100% scale: full dataset — already captured by run_benchmarks.py

Cutoffs are computed from the actual MIN/MAX of events.occurred_at,
consistent with the PostgreSQL, Neo4j, Cassandra, and TimescaleDB naive anchors.

Q3 uses row-based scaling (pool from scale_pct% of session users) — same
exception as all other databases. Sessions were generated in 2025; date cutoff
returns zero rows.

Q8 excluded — write throughput scaling measured by thread variation.

Optimised SQL changes vs naive scalability
───────────────────────────────────────────
  Q1: Reads from daily_revenue_by_tier continuous aggregate for subscription
      revenue (filtered to day < cutoff). Marketplace still raw LATERAL.
      At 10% scale, the aggregate has only ~73 rows for subscription revenue
      (1 row per day per tier × 73 days × 3 tiers) — extremely fast read.

  Q2–Q6: SQL identical to naive scalability — no schema change for these queries.
      Q6 benefits from 1-month chunks and compression but the SQL is the same.

  Q7: time_bucket_gapfill on daily_revenue_by_tier, scoped to [data_min, cutoff].
      At 10% scale reads only ~73 aggregate rows for the subscription component.
      Marketplace still raw LATERAL scoped to the cutoff window.

Usage:
    python run_scalability.py                    # both scales, 1000 iterations
    python run_scalability.py --scale 10
    python run_scalability.py --scale 50
    python run_scalability.py --iterations 100
    python run_scalability.py --only Q1 Q7
    python run_scalability.py --dry-run
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.harness import run_benchmark

load_dotenv()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "benchmarks", "timescaledb", "optimised", "results", "scale")
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

# ── connection ─────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"),
        connect_timeout=10,
    )

# ── cutoff computation ─────────────────────────────────────────────────────────

def compute_cutoffs(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MIN(occurred_at)::date, MAX(occurred_at)::date
            FROM events;
        """)
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("No events found — run timescaledb_optimised_loader.py first.")
    min_date, max_date = row
    range_days = (max_date - min_date).days
    return {
        "min_date":     min_date,
        "max_date":     max_date,
        "range_days":   range_days,
        "cutoff_10pct": min_date + timedelta(days=int(range_days * 0.10)),
        "cutoff_50pct": min_date + timedelta(days=int(range_days * 0.50)),
    }

# ── Q1 optimised — continuous aggregate + marketplace LATERAL ──────────────────
# Subscription revenue: time_bucket('1 month', day) on daily_revenue_by_tier
# filtered to day < cutoff. At 10% scale this scans ~73 aggregate rows vs
# tens of thousands of raw invoices in the naive version.
# Marketplace: raw LATERAL scoped to created_at < cutoff (same as naive Q1).

Q1_SCALED_SQL = """
WITH sub_monthly AS (
    SELECT
        time_bucket('1 month', day)  AS month,
        tier_id,
        tier_name,
        SUM(invoice_count)           AS invoice_count,
        SUM(total_usd)               AS total_usd
    FROM daily_revenue_by_tier
    WHERE day >= NOW() - INTERVAL '12 months'
      AND day <  %s::timestamptz
    GROUP BY 1, 2, 3
),
mkt_monthly AS (
    SELECT
        DATE_TRUNC('month', i.created_at)   AS month,
        active_sub.tier_id,
        SUM(i.total_usd)                    AS total_usd,
        COUNT(*)                            AS invoice_count
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id FROM subscriptions s
        WHERE  s.user_id    = i.user_id
          AND  s.started_at <= i.created_at
        ORDER BY s.started_at DESC LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
      AND i.status       = 'paid'
      AND i.created_at  >= NOW() - INTERVAL '12 months'
      AND i.created_at  <  %s::timestamptz
    GROUP BY 1, 2
),
combined AS (
    SELECT month, tier_id, tier_name,
           SUM(invoice_count) AS invoice_count, SUM(total_usd) AS total_usd
    FROM (
        SELECT month, tier_id, tier_name, invoice_count, total_usd FROM sub_monthly
        UNION ALL
        SELECT m.month, m.tier_id, st.name, m.invoice_count, m.total_usd
        FROM mkt_monthly m
        JOIN subscription_tiers st ON st.id = m.tier_id
    ) r GROUP BY 1, 2, 3
)
SELECT TO_CHAR(c.month, 'YYYY-MM') AS month, c.tier_name,
       stp.monthly_price_usd AS price_in_effect_usd,
       c.invoice_count, ROUND(c.total_usd, 2) AS total_revenue_usd
FROM combined c
JOIN subscription_tier_pricing stp
    ON  stp.tier_id    = c.tier_id
    AND stp.valid_from <= c.month
    AND (stp.valid_to IS NULL OR stp.valid_to > c.month)
ORDER BY month, tier_name;
"""

# ── Q2–Q5 — identical to naive (no schema change) ─────────────────────────────

Q2_SCALED_SQL = """
SELECT i.id, i.invoice_type, i.status, i.subtotal_usd, i.tax_usd,
       i.discount_usd, i.total_usd, i.due_at, i.paid_at, i.created_at,
       u.id AS customer_id, u.full_name, u.email, u.country_code,
       il.id AS line_id, il.description, il.quantity,
       il.unit_price_usd, il.line_total_usd,
       p.id AS product_id, p.name AS product_name,
       p.product_type, p.price_usd, p.attributes
FROM invoices i
JOIN users         u  ON u.id  = i.user_id
JOIN invoice_lines il ON il.invoice_id = i.id
LEFT JOIN products p  ON p.id  = il.product_id
WHERE i.id = %s
  AND i.created_at < %s::timestamptz
ORDER BY il.created_at;
"""

Q3_SCALED_SQL = """
SELECT s.id, s.user_id, s.cart, s.ip_address,
       s.user_agent, s.created_at, s.last_active_at, s.expires_at
FROM sessions s
WHERE s.user_id = %s
ORDER BY s.last_active_at DESC
LIMIT 1;
"""

Q4_SCALED_SQL = """
WITH orders_with_product AS (
    SELECT DISTINCT oi.order_id
    FROM order_items oi
    JOIN orders o ON o.id = oi.order_id
    WHERE oi.product_id = %s
      AND o.status IN ('confirmed', 'shipped', 'delivered')
      AND o.created_at < %s::timestamptz
),
co_purchased AS (
    SELECT oi.product_id, COUNT(DISTINCT oi.order_id) AS co_purchase_count
    FROM order_items oi
    WHERE oi.order_id IN (SELECT order_id FROM orders_with_product)
      AND oi.product_id != %s
    GROUP BY oi.product_id
)
SELECT p.id, p.name, p.product_type, p.price_usd,
       cp.co_purchase_count,
       ROUND(cp.co_purchase_count::NUMERIC /
             NULLIF((SELECT COUNT(*) FROM orders_with_product), 0), 4) AS confidence
FROM co_purchased cp
JOIN products p ON p.id = cp.product_id
WHERE p.is_active = TRUE
ORDER BY cp.co_purchase_count DESC, p.name
LIMIT 10;
"""

Q5_SCALED_SQL = """
SELECT p.id, p.name, p.product_type, p.price_usd, p.attributes,
       ts_rank_cd(p.search_vector, query) AS rank
FROM products p, plainto_tsquery('english', %s) AS query
WHERE p.search_vector @@ query
  AND p.is_active = TRUE
  AND p.created_at < %s::timestamptz
ORDER BY rank DESC
LIMIT 20;
"""

Q6_SCALED_SQL = """
SELECT e.id, e.event_type, e.occurred_at,
       e.product_id, e.session_id, e.metadata
FROM events e
WHERE e.user_id     = %s
  AND e.occurred_at >= %s
  AND e.occurred_at <  %s
ORDER BY e.occurred_at DESC;
"""

# ── Q7 optimised — time_bucket_gapfill on continuous aggregate ─────────────────
# Reads from daily_revenue_by_tier scoped to [data_min, cutoff].
# At 10% scale this processes ~73 aggregate rows for subscription revenue
# (vs hundreds of thousands of raw invoice rows in naive Q7).
# Marketplace still requires raw LATERAL — same limitation as Q1.
# params: (start, end, start, end, start, end) — gapfill bounds, agg WHERE, mkt WHERE

Q7_SCALED_SQL = """
WITH gapfilled AS (
    SELECT
        time_bucket_gapfill('1 day', day, %s::timestamptz, %s::timestamptz) AS day,
        tier_id,
        tier_name,
        COALESCE(SUM(total_usd), 0.0) AS sub_revenue
    FROM daily_revenue_by_tier
    WHERE day >= %s AND day < %s
    GROUP BY 1, 2, 3
),
mkt_daily AS (
    SELECT
        DATE_TRUNC('day', i.created_at)::date AS day,
        active_sub.tier_id,
        SUM(i.total_usd) AS mkt_revenue
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id FROM subscriptions s
        WHERE  s.user_id    = i.user_id
          AND  s.started_at <= i.created_at
        ORDER BY s.started_at DESC LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
      AND i.status       = 'paid'
      AND i.created_at  >= %s AND i.created_at < %s
    GROUP BY 1, 2
),
combined AS (
    SELECT g.day, g.tier_id, g.tier_name,
           g.sub_revenue + COALESCE(m.mkt_revenue, 0.0) AS daily_total
    FROM gapfilled g
    LEFT JOIN mkt_daily m
        ON m.day = g.day::date AND m.tier_id = g.tier_id
)
SELECT day, tier_name,
       ROUND(daily_total, 2) AS daily_revenue_usd,
       ROUND(AVG(daily_total) OVER (
           PARTITION BY tier_id ORDER BY day
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_7day_avg_usd
FROM combined
ORDER BY day, tier_name;
"""

# ── pool helpers ───────────────────────────────────────────────────────────────

def fetch_invoice_id_pool(conn, cutoff, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM (
                SELECT DISTINCT id FROM invoices WHERE created_at < %s
            ) AS s ORDER BY RANDOM() LIMIT %s;
        """, (cutoff, pool_size))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No invoices before {cutoff}")
    return [str(r[0]) for r in rows]


def fetch_user_id_pool_sessions(conn, scale_pct, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM sessions;")
        total = cur.fetchone()[0]
        limit = max(1, int(total * scale_pct / 100))
        cur.execute("""
            SELECT user_id FROM (
                SELECT DISTINCT user_id FROM sessions
            ) AS s ORDER BY RANDOM() LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No sessions found")
    ids = [str(r[0]) for r in rows]
    return random.choices(ids, k=min(pool_size, len(ids)))


def fetch_product_id_pool(conn, cutoff, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT product_id FROM (
                SELECT DISTINCT oi.product_id
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE o.status IN ('confirmed','shipped','delivered')
                  AND o.created_at < %s
            ) AS s ORDER BY RANDOM() LIMIT %s;
        """, (cutoff, pool_size))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No products in orders before {cutoff}")
    return [str(r[0]) for r in rows]


def fetch_anchor_pool_events(conn, cutoff, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id, occurred_at FROM events
            WHERE occurred_at < %s
            ORDER BY RANDOM() LIMIT %s;
        """, (cutoff, pool_size))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No events before {cutoff}")
    return [(str(r[0]), r[1]) for r in rows]

# ── query function factories ───────────────────────────────────────────────────

def make_q1_fn(conn, cutoff):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q1_SCALED_SQL, (cutoff, cutoff))
            cur.fetchall()
    return _run


def make_q2_fn(conn, invoice_ids, cutoff):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q2_SCALED_SQL, (random.choice(invoice_ids), cutoff))
            cur.fetchall()
    return _run


def make_q3_fn(conn, user_ids):
    local = threading.local()
    def _get_conn():
        if not getattr(local, "conn", None) or local.conn.closed:
            local.conn = get_connection()
            local.conn.autocommit = False
        return local.conn
    def _run():
        c = _get_conn()
        with c.cursor() as cur:
            cur.execute(Q3_SCALED_SQL, (random.choice(user_ids),))
            cur.fetchone()
    return _run


def make_q4_fn(conn, product_ids, cutoff):
    def _run():
        pid = random.choice(product_ids)
        with conn.cursor() as cur:
            cur.execute(Q4_SCALED_SQL, (pid, cutoff, pid))
            cur.fetchall()
    return _run


def make_q5_fn(conn, cutoff):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q5_SCALED_SQL, (random.choice(SEARCH_TERMS), cutoff))
            cur.fetchall()
    return _run


def make_q6_fn(conn, pairs):
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start = anchor_dt - timedelta(days=15)
        end   = anchor_dt + timedelta(days=15)
        with conn.cursor() as cur:
            cur.execute(Q6_SCALED_SQL, (user_id, start, end))
            cur.fetchall()
    return _run


def make_q7_fn(conn, cutoff, data_min):
    cutoff_date = cutoff if isinstance(cutoff, date) else cutoff.date()
    max_start   = max(0, (cutoff_date - data_min).days - WINDOW_DAYS_Q7)

    def _run():
        offset = random.randint(0, max_start) if max_start > 0 else 0
        start  = data_min + timedelta(days=offset)
        end    = min(start + timedelta(days=WINDOW_DAYS_Q7 - 1), cutoff_date)
        params = (start, end, start, end, start, end)
        with conn.cursor() as cur:
            cur.execute(Q7_SCALED_SQL, params)
            cur.fetchall()
    return _run

# ── runner ─────────────────────────────────────────────────────────────────────

def run_scaled_query(
    query_id:    str,
    scale_pct:   int,
    cutoff,
    conn,
    cutoffs:     dict,
    iterations:  int,
    results_dir: str,
) -> dict | None:

    output_path = os.path.join(
        results_dir,
        f"timescaledb_optimised_{query_id.lower()}_scale{scale_pct}.json"
    )

    labels = {
        "Q1": (
            f"Q1 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Subscription: time_bucket('1 month') on daily_revenue_by_tier "
            f"(day < {cutoff}) — reads only aggregate rows in window. "
            "Marketplace: raw LATERAL scoped to cutoff."
        ),
        "Q2": (
            f"Q2 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Identical to naive — 4-table JOIN, no TS benefit for point lookups."
        ),
        "Q3": (
            f"Q3 optimised at {scale_pct}% row-based scale "
            f"({scale_pct}% of session users). "
            "No date cutoff — sessions in 2025. Identical to naive."
        ),
        "Q4": (
            f"Q4 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Identical to naive — plain tables, no TS benefit."
        ),
        "Q5": (
            f"Q5 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "Identical to naive — products is a plain table."
        ),
        "Q6": (
            f"Q6 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "1-month chunks: 30-day window touches 1-2 chunks. "
            "Compression segmentby=user_id: contiguous user segments."
        ),
        "Q7": (
            f"Q7 optimised at {scale_pct}% scale (cutoff {cutoff}). "
            "time_bucket_gapfill on daily_revenue_by_tier scoped to cutoff. "
            "Subscription: reads only aggregate rows in window (~very few at 10%). "
            "Marketplace: raw LATERAL scoped to window."
        ),
    }

    try:
        concurrency = 1

        if query_id == "Q1":
            query_fn = make_q1_fn(conn, cutoff)

        elif query_id == "Q2":
            pool     = fetch_invoice_id_pool(conn, cutoff)
            query_fn = make_q2_fn(conn, pool, cutoff)

        elif query_id == "Q3":
            concurrency = 50
            pool        = fetch_user_id_pool_sessions(conn, scale_pct)
            query_fn    = make_q3_fn(conn, pool)

        elif query_id == "Q4":
            pool     = fetch_product_id_pool(conn, cutoff)
            query_fn = make_q4_fn(conn, pool, cutoff)

        elif query_id == "Q5":
            query_fn = make_q5_fn(conn, cutoff)

        elif query_id == "Q6":
            pairs    = fetch_anchor_pool_events(conn, cutoff)
            query_fn = make_q6_fn(conn, pairs)

        elif query_id == "Q7":
            query_fn = make_q7_fn(conn, cutoff, cutoffs["min_date"])

        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db="timescaledb_optimised",
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
        description="TimescaleDB optimised scalability baselines"
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
    print("  TimescaleDB Optimised — Scalability Baselines")
    print("═" * 60)
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")
    print()
    print("  Optimised SQL changes vs naive:")
    print("  • Q1: continuous aggregate for subscription, LATERAL for marketplace")
    print("  • Q7: time_bucket_gapfill on aggregate, LATERAL for marketplace")
    print("  • Q6: 1-month chunks + compression (SQL unchanged)")
    print("  • Q2–Q5: identical to naive")

    os.makedirs(args.results_dir, exist_ok=True)

    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET jit = off; SET work_mem = '64MB';")
    conn.autocommit = False

    info("Computing date range cutoffs from events table...")
    cutoffs = compute_cutoffs(conn)
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
        print(f"  TimescaleDB optimised — {scale}% scale (cutoff: {cutoff})")
        print(f"{'═'*60}")

        for qid in queries:
            print(f"\n{'─'*60}")
            print(f"  {qid} @ {scale}%")
            print(f"{'─'*60}")
            result = run_scaled_query(
                query_id=qid,
                scale_pct=scale,
                cutoff=cutoff,
                conn=conn,
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

    conn.close()
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

    summary_path = os.path.join(
        args.results_dir, "timescaledb_optimised_scalability_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db": "timescaledb_optimised",
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
        print(f"\n  {GREEN}All TimescaleDB optimised scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()