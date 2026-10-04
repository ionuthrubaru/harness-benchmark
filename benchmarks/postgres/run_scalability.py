"""
benchmarks/postgres/run_scalability.py — Phase 2 Scalability Baselines
=======================================================================
Re-runs Q1-Q7 at 10% and 50% data scale to establish the PostgreSQL
scalability curve for Chart 3.

Scale is defined by date range, not row count:
  - 10% scale: queries restricted to the first 10% of the dataset's
               date range (e.g. Jan 2025 → ~Feb 5 2025, ~36 days)
  - 50% scale: queries restricted to the first 50% of the date range
               (e.g. Jan 2025 → ~Jul 2 2025, ~182 days)
  - 100% scale: full year — already captured by run_all.py

No data is regenerated — date range filters are applied at query time
on the existing loaded dataset.

Cutoff dates are computed at startup from the actual MIN/MAX of
occurred_at in the events table, so the script is robust to any
dataset date range.

Queries that don't use occurred_at (Q2: invoices, Q4: orders,
Q5: products) use their respective created_at column instead,
applying the same cutoff logic for consistency.

Q8 (write benchmark) is excluded — throughput scaling is measured
differently (thread count variation) and is not part of Chart 3.

Usage:
    python run_scalability.py                    # both scales, 1000 iterations
    python run_scalability.py --scale 10         # 10% scale only
    python run_scalability.py --scale 50         # 50% scale only
    python run_scalability.py --iterations 100   # smoke test
    python run_scalability.py --only Q6       # specific queries only
    python run_scalability.py --dry-run          # 100 iterations, both scales
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.harness import run_benchmark

load_dotenv()

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── connection ────────────────────────────────────────────────────────────────

from pg_conn import get_connection

# ── cutoff computation ────────────────────────────────────────────────────────

def compute_cutoffs(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                MIN(occurred_at)::date AS min_date,
                MAX(occurred_at)::date AS max_date
            FROM events;
        """)
        row = cur.fetchone()

    if not row or not row[0]:
        raise RuntimeError("No events found — run the data loader first.")

    min_date, max_date = row
    range_days = (max_date - min_date).days

    from datetime import timedelta
    cutoff_10 = min_date + timedelta(days=int(range_days * 0.10))
    cutoff_50 = min_date + timedelta(days=int(range_days * 0.50))

    return {
        "min_date":     min_date,
        "max_date":     max_date,
        "range_days":   range_days,
        "cutoff_10pct": cutoff_10,
        "cutoff_50pct": cutoff_50,
    }

# ── scaled SQL ────────────────────────────────────────────────────────────────

Q1_SCALED_SQL = """
WITH invoice_tiers AS (
    SELECT
        i.id AS invoice_id, i.created_at, i.total_usd, i.invoice_type, sub.tier_id
    FROM invoices i
    JOIN subscriptions sub ON sub.id = i.subscription_id
    WHERE i.invoice_type = 'subscription'
      AND i.status       = 'paid'
      AND i.created_at  >= NOW() - INTERVAL '12 months'
      AND i.created_at  <  %s::timestamptz

    UNION ALL

    SELECT
        i.id AS invoice_id, i.created_at, i.total_usd, i.invoice_type, active_sub.tier_id
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id FROM subscriptions s
        WHERE  s.user_id = i.user_id AND s.started_at <= i.created_at
        ORDER BY s.started_at DESC LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
      AND i.status       = 'paid'
      AND i.created_at  >= NOW() - INTERVAL '12 months'
      AND i.created_at  <  %s::timestamptz
),
monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', it.created_at)  AS month,
        st.name                             AS tier_name,
        stp.monthly_price_usd               AS price_in_effect_usd,
        COUNT(it.invoice_id)                AS invoice_count,
        SUM(it.total_usd)                   AS total_revenue_usd
    FROM invoice_tiers it
    JOIN subscription_tiers st ON st.id = it.tier_id
    JOIN subscription_tier_pricing stp
        ON  stp.tier_id    = it.tier_id
        AND stp.valid_from <= it.created_at
        AND (stp.valid_to IS NULL OR stp.valid_to > it.created_at)
    GROUP BY 1, 2, 3
)
SELECT TO_CHAR(month, 'YYYY-MM') AS month, tier_name,
       price_in_effect_usd, invoice_count,
       ROUND(total_revenue_usd, 2) AS total_revenue_usd
FROM monthly_revenue ORDER BY month, tier_name;
"""

Q2_SCALED_SQL = """
SELECT
    i.id, i.invoice_type, i.status, i.subtotal_usd, i.tax_usd,
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
  AND s.created_at < %s::timestamptz
ORDER BY s.last_active_at DESC
LIMIT 1;
"""

Q4_SCALED_SQL = """
WITH orders_with_product AS (
    SELECT DISTINCT oi.order_id
    FROM   order_items oi
    JOIN   orders o ON o.id = oi.order_id
    WHERE  oi.product_id = %s
      AND  o.status IN ('confirmed', 'shipped', 'delivered')
      AND  o.created_at < %s::timestamptz
),
co_purchased AS (
    SELECT oi.product_id, COUNT(DISTINCT oi.order_id) AS co_purchase_count
    FROM   order_items oi
    WHERE  oi.order_id   IN (SELECT order_id FROM orders_with_product)
      AND  oi.product_id != %s
    GROUP BY oi.product_id
)
SELECT p.id, p.name, p.product_type, p.price_usd,
       cp.co_purchase_count,
       ROUND(cp.co_purchase_count::NUMERIC /
             NULLIF((SELECT COUNT(*) FROM orders_with_product), 0), 4) AS confidence
FROM   co_purchased cp
JOIN   products p ON p.id = cp.product_id
WHERE  p.is_active = TRUE
ORDER BY cp.co_purchase_count DESC, p.name
LIMIT 10;
"""

Q5_SCALED_SQL = """
SELECT p.id, p.name, p.product_type, p.price_usd, p.attributes,
       ts_rank_cd(p.search_vector, query) AS rank
FROM   products p,
       plainto_tsquery('english', %s) AS query
WHERE  p.search_vector @@ query
  AND  p.is_active = TRUE
  AND  p.created_at < %s::timestamptz
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

Q7_SCALED_SQL = """
WITH date_spine AS (
    SELECT generate_series(%s::date, %s::date, '1 day'::interval)::date AS day
),
tier_days AS (
    SELECT ds.day, st.id AS tier_id, st.name AS tier_name
    FROM date_spine ds CROSS JOIN subscription_tiers st
),
daily_revenue AS (
    SELECT DATE_TRUNC('day', i.created_at)::date AS day,
           sub.tier_id, SUM(i.total_usd) AS revenue
    FROM invoices i
    JOIN subscriptions sub ON sub.id = i.subscription_id
    WHERE i.invoice_type = 'subscription' AND i.status = 'paid'
      AND i.created_at >= %s::date
      AND i.created_at <  (%s::date + INTERVAL '1 day')
      AND i.created_at <  %s::timestamptz
    GROUP BY 1, 2

    UNION ALL

    SELECT DATE_TRUNC('day', i.created_at)::date AS day,
           active_sub.tier_id, SUM(i.total_usd) AS revenue
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id FROM subscriptions s
        WHERE  s.user_id = i.user_id AND s.started_at <= i.created_at
        ORDER BY s.started_at DESC LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace' AND i.status = 'paid'
      AND i.created_at >= %s::date
      AND i.created_at <  (%s::date + INTERVAL '1 day')
      AND i.created_at <  %s::timestamptz
    GROUP BY 1, 2
),
filled AS (
    SELECT td.day, td.tier_id, td.tier_name,
           COALESCE(SUM(dr.revenue), 0.00) AS daily_total
    FROM tier_days td
    LEFT JOIN daily_revenue dr
        ON dr.day = td.day AND dr.tier_id = td.tier_id
    GROUP BY td.day, td.tier_id, td.tier_name
)
SELECT day, tier_name,
       ROUND(daily_total, 2) AS daily_revenue_usd,
       ROUND(AVG(daily_total) OVER (
           PARTITION BY tier_id ORDER BY day
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_7day_avg_usd
FROM filled ORDER BY day, tier_name;
"""

# ── ID pool helpers ───────────────────────────────────────────────────────────

import random
from datetime import date, timedelta

WINDOW_DAYS_Q7 = 183

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour", "photoshop brushes", "video editing",
    "certificate course", "logo design", "colour palette", "font pack",
    "texture pack", "motion graphics", "social media", "icon set",
    "web design", "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

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
    """
    Q3 uses row-based scaling — sessions all generated in 2025 so a date
    cutoff returns zero rows. Sample a fraction of distinct session users.
    """
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
        raise RuntimeError("No sessions found in the sessions table")
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
    """
    Sample (user_id, occurred_at) pairs from events before the cutoff.
    Window is centred on the anchor — guaranteed non-empty each iteration.
    Mirrors the fix applied to the standalone q6_events.py.

    Uses ORDER BY RANDOM() rather than TABLESAMPLE — TABLESAMPLE samples
    random heap pages across the whole table, so at small cutoffs (e.g.
    10% scale = 72 days in) almost no pages land in the date slice.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id, occurred_at
            FROM events
            WHERE occurred_at < %s
            ORDER BY RANDOM()
            LIMIT %s;
        """, (cutoff, pool_size))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No events before {cutoff}")
    return [(str(r[0]), r[1]) for r in rows]

# ── query function factories ──────────────────────────────────────────────────

def make_q1_fn(conn, cutoff):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q1_SCALED_SQL, (cutoff, cutoff))
            cur.fetchall()
    return _run

def make_q2_fn(conn, invoice_ids, cutoff):
    def _run():
        iid = random.choice(invoice_ids)
        with conn.cursor() as cur:
            cur.execute(Q2_SCALED_SQL, (iid, cutoff))
            cur.fetchall()
    return _run

def make_q3_fn_no_cutoff(conn, user_ids):
    """
    Q3 scalability variant — pool size controls scale, no date cutoff.
    """
    import threading
    local = threading.local()
    def _get_conn():
        if not getattr(local, "conn", None) or local.conn.closed:
            local.conn = get_connection()
        return local.conn
    def _run():
        c = _get_conn()
        uid = random.choice(user_ids)
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.id, s.user_id, s.cart, s.ip_address,
                       s.user_agent, s.created_at, s.last_active_at, s.expires_at
                FROM sessions s
                WHERE s.user_id = %s
                ORDER BY s.last_active_at DESC
                LIMIT 1;
            """, (uid,))
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
        term = random.choice(SEARCH_TERMS)
        with conn.cursor() as cur:
            cur.execute(Q5_SCALED_SQL, (term, cutoff))
            cur.fetchall()
    return _run

def make_q6_fn(conn, pairs):
    """
    Window centred on a sampled real event — guaranteed non-empty.
    Mirrors q6_events.py anchor-based methodology.
    """
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start = anchor_dt - timedelta(days=15)
        end   = anchor_dt + timedelta(days=15)
        with conn.cursor() as cur:
            cur.execute(Q6_SCALED_SQL, (user_id, start, end))
            cur.fetchall()
    return _run

def make_q7_fn(conn, cutoff):
    cutoff_date = cutoff.date() if hasattr(cutoff, "date") else cutoff
    min_date    = cutoff_date - timedelta(days=cutoff_date.toordinal() - date.min.toordinal())

    # Recompute data_min from DB once at factory time
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(occurred_at)::date FROM events WHERE occurred_at < %s;",
                    (cutoff,))
        row = cur.fetchone()
    data_min = row[0] if row and row[0] else date(2024, 1, 1)

    def _run():
        delta     = (cutoff_date - data_min).days
        max_start = max(0, delta - WINDOW_DAYS_Q7)
        start     = data_min + timedelta(days=random.randint(0, max_start))
        end       = min(start + timedelta(days=WINDOW_DAYS_Q7 - 1), cutoff_date)
        params    = (start, end, start, end, cutoff, start, end, cutoff)
        with conn.cursor() as cur:
            cur.execute(Q7_SCALED_SQL, params)
            cur.fetchall()
    return _run

# ── runner ────────────────────────────────────────────────────────────────────

def run_scaled_query(
    query_id: str,
    scale_pct: int,
    cutoff,
    conn,
    iterations: int,
    results_dir: str,
) -> dict | None:
    suffix      = f"scale{scale_pct}"
    output_path = os.path.join(results_dir, f"postgres_{query_id.lower()}_{suffix}.json")

    if query_id == "Q6":
        label = (
            f"Q6 at {scale_pct}% scale — events cutoff: {cutoff}. "
            "Window centred on a sampled real event — guaranteed non-empty. "
            "Mirrors standalone q6_events.py anchor-based methodology."
        )
    elif query_id == "Q3":
        label = (
            f"Q3 at {scale_pct}% scale — row-based scaling: user pool drawn from "
            f"{scale_pct}% of total session users. No date cutoff applied. "
            "Sessions were generated in 2025 making date-range scaling inapplicable."
        )
    else:
        label = (
            f"{query_id} at {scale_pct}% scale — date range cutoff: {cutoff}. "
            "Same query as full-scale baseline with additional created_at/occurred_at filter."
        )

    try:
        if query_id == "Q1":
            query_fn    = make_q1_fn(conn, cutoff)
            concurrency = 1
        elif query_id == "Q2":
            pool        = fetch_invoice_id_pool(conn, cutoff)
            query_fn    = make_q2_fn(conn, pool, cutoff)
            concurrency = 1
        elif query_id == "Q3":
            pool        = fetch_user_id_pool_sessions(conn, scale_pct)
            query_fn    = make_q3_fn_no_cutoff(conn, pool)
            concurrency = 50
        elif query_id == "Q4":
            pool        = fetch_product_id_pool(conn, cutoff)
            query_fn    = make_q4_fn(conn, pool, cutoff)
            concurrency = 1
        elif query_id == "Q5":
            query_fn    = make_q5_fn(conn, cutoff)
            concurrency = 1
        elif query_id == "Q6":
            pool        = fetch_anchor_pool_events(conn, cutoff)
            query_fn    = make_q6_fn(conn, pool)
            concurrency = 1
        elif query_id == "Q7":
            query_fn    = make_q7_fn(conn, cutoff)
            concurrency = 1
        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db="postgres",
            query_id=query_id,
            label=label,
            iterations=iterations,
            concurrency=concurrency,
            output_path=output_path,
        )
        ok(f"{query_id} @ {scale_pct}% complete → {output_path}")
        return result

    except Exception as e:
        fail(f"{query_id} @ {scale_pct}% failed: {e}")
        return None

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL scalability baselines for Chart 3"
    )
    parser.add_argument(
        "--scale", type=int, choices=[10, 50], default=None,
        help="Run only one scale (10 or 50). Default: run both.",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Iterations per query per scale (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="100 iterations, both scales — quick smoke test",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="Q",
        help="Run only specific queries e.g. --only Q1 Q6",
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR, dest="results_dir",
        help=f"Directory to save results (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    iterations = 100 if args.dry_run else args.iterations
    scales     = [10, 50] if args.scale is None else [args.scale]
    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]

    print("\n" + "═" * 60)
    print("  Phase 2 — PostgreSQL Scalability Baselines")
    print("═" * 60)
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    conn = get_connection()

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
        print(f"  Running at {scale}% scale (cutoff: {cutoff})")
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
    print(f"  SCALABILITY RESULTS SUMMARY")
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

    summary_path = os.path.join(args.results_dir, "postgres_scalability_summary.json")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db": "postgres",
        "cutoffs": {
            "min_date":     str(cutoffs["min_date"]),
            "max_date":     str(cutoffs["max_date"]),
            "range_days":   cutoffs["range_days"],
            "cutoff_10pct": str(cutoffs["cutoff_10pct"]),
            "cutoff_50pct": str(cutoffs["cutoff_50pct"]),
        },
        "benchmarks": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    info(f"Combined summary saved → {summary_path}")

    print(f"\n  Total wall time : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Completed       : {len(all_results)} / {len(all_results) + len(failed)}")

    if failed:
        warn(f"Failed: {', '.join(failed)}")
        print(f"\n  Re-run failures with:")
        print(f"    python run_scalability.py --only {' '.join(q.split('@')[0] for q in failed)}\n")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()