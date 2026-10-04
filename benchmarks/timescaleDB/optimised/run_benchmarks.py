"""
benchmarks/timescaledb/optimised/run_benchmarks.py — TimescaleDB Optimised Q1–Q7
==================================================================================
Runs all seven read benchmarks against the optimised TimescaleDB schema in one
file. Structured identically to run_scalability.py for consistency — each query
has its own SQL, factory function, dry-run printer, and labelled harness call.

Optimised schema features used per query
──────────────────────────────────────────
  Q1: time_bucket('1 month', day) on daily_revenue_by_tier continuous aggregate
      for subscription revenue + small raw LATERAL for marketplace tier attribution.
      Avoids scanning raw invoices for the subscription component.

  Q2: Identical to naive — 4-table JOIN. No TimescaleDB benefit for point lookups.
      Included for completeness and to confirm parity with naive Q2.

  Q3: Identical to naive — sessions is a plain table. Included for completeness.

  Q4: Identical to naive — order_items/orders are plain tables. Included for
      completeness.

  Q5: Identical to naive — products is a plain table. Included for completeness.

  Q6: Same SQL as naive — WHERE (user_id, occurred_at range). Schema effect:
      1-month chunks (vs naive 7-day) means a 30-day window touches 1-2 chunks
      instead of ≤5. Compression with segmentby=user_id means events for one
      user are contiguous within a chunk — Q6 reads a single contiguous segment.

  Q7: TimescaleDB killer feature.
      time_bucket_gapfill('1 day', day) on daily_revenue_by_tier — reads from
      the pre-materialised continuous aggregate instead of raw invoices. Gap-filling
      is native (no generate_series). Rolling average via window function on the
      gapfilled output. The subscription component requires no invoice scan at all.
      Marketplace revenue is handled via a small separate raw LATERAL query and
      merged in Python (same limitation as Q1 — LATERAL cannot go inside the
      continuous aggregate).

Schema effect vs naive for each query
───────────────────────────────────────
  Q1: continuous aggregate replaces raw invoice scan for subscription revenue
  Q2: none (not a time-series query)
  Q3: none (not a time-series query)
  Q4: none (not a time-series query)
  Q5: none (not a time-series query)
  Q6: 1-month chunks + compression → fewer chunks + contiguous user segments
  Q7: continuous aggregate + time_bucket_gapfill → no raw invoice scan, native gap-fill

Usage:
    cd benchmarks/timescaledb/optimised
    python run_benchmarks.py                     # all Q1–Q7, 1000 iterations each
    python run_benchmarks.py --only Q7           # single query
    python run_benchmarks.py --only Q1 Q6 Q7     # subset
    python run_benchmarks.py --iterations 100    # quick smoke test
    python run_benchmarks.py --dry-run           # run each query once, print results
    python run_benchmarks.py --dry-run --only Q7 # dry-run single query
"""

import argparse
import os
import random
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.harness import run_benchmark

load_dotenv()

RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "benchmarks", "timescaledb", "optimised", "results"
)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — Monthly revenue by tier (last 12 months)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Subscription revenue: time_bucket('1 month', day) on daily_revenue_by_tier.
#   Reads from the continuous aggregate — no raw invoice scan for subscriptions.
#   1,258 rows in the aggregate → tiny scan vs hundreds of thousands of invoices.
#
# Marketplace revenue: raw LATERAL query (cannot be pre-aggregated — tier
#   attribution requires a correlated subquery on subscriptions).
#   This is the same LATERAL as PostgreSQL Q1, scoped to the last 12 months.
#
# Schema effect: subscription component reads ~12 aggregate rows per tier
#   instead of scanning raw invoices. Marketplace is unchanged.

Q1_SQL = """
WITH sub_monthly AS (
    -- Read from continuous aggregate: pre-materialised daily subscription revenue.
    -- time_bucket groups daily rows into monthly buckets — tiny scan.
    SELECT
        time_bucket('1 month', day)  AS month,
        tier_id,
        tier_name,
        SUM(invoice_count)           AS invoice_count,
        SUM(total_usd)               AS total_usd
    FROM daily_revenue_by_tier
    WHERE day >= NOW() - INTERVAL '12 months'
    GROUP BY 1, 2, 3
),
mkt_monthly AS (
    -- Marketplace invoices: raw LATERAL for temporal tier attribution.
    -- LATERAL cannot be used inside continuous aggregate SELECT lists,
    -- so this component still scans raw invoices. Scoped to last 12 months.
    SELECT
        DATE_TRUNC('month', i.created_at)   AS month,
        active_sub.tier_id,
        SUM(i.total_usd)                    AS total_usd,
        COUNT(*)                            AS invoice_count
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id
        FROM   subscriptions s
        WHERE  s.user_id    = i.user_id
          AND  s.started_at <= i.created_at
        ORDER BY s.started_at DESC
        LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
      AND i.status       = 'paid'
      AND i.created_at  >= NOW() - INTERVAL '12 months'
    GROUP BY 1, 2
),
combined AS (
    SELECT month, tier_id, tier_name,
           SUM(invoice_count) AS invoice_count,
           SUM(total_usd)     AS total_usd
    FROM (
        SELECT month, tier_id, tier_name, invoice_count, total_usd
        FROM sub_monthly
        UNION ALL
        SELECT m.month, m.tier_id, st.name AS tier_name,
               m.invoice_count, m.total_usd
        FROM mkt_monthly m
        JOIN subscription_tiers st ON st.id = m.tier_id
    ) combined_raw
    GROUP BY 1, 2, 3
)
SELECT
    TO_CHAR(c.month, 'YYYY-MM')     AS month,
    c.tier_name,
    stp.monthly_price_usd           AS price_in_effect_usd,
    c.invoice_count,
    ROUND(c.total_usd, 2)           AS total_revenue_usd
FROM combined c
JOIN subscription_tier_pricing stp
    ON  stp.tier_id    = c.tier_id
    AND stp.valid_from <= c.month
    AND (stp.valid_to IS NULL OR stp.valid_to > c.month)
ORDER BY month, tier_name;
"""

def make_q1_fn(conn):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q1_SQL)
            cur.fetchall()
    return _run

def dry_run_q1(conn):
    print("\n  DRY RUN — Q1 optimised (continuous aggregate + marketplace LATERAL)\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q1_SQL)
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No rows — is daily_revenue_by_tier refreshed?")
        return
    print(f"  {len(rows)} rows returned.\n")
    print(f"  {'Month':<10} {'Tier':<12} {'Price':>10} {'Invoices':>10} {'Revenue':>14}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*14}")
    for row in rows[:10]:
        print(f"  {str(row['month']):<10} {str(row['tier_name']):<12} "
              f"{str(row['price_in_effect_usd']):>10} "
              f"{row['invoice_count']:>10} "
              f"{float(row['total_revenue_usd']):>14.2f}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more rows")

# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — Full invoice fetch (identical to naive)
# ═══════════════════════════════════════════════════════════════════════════════

Q2_SQL = """
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
ORDER BY il.created_at;
"""

def fetch_invoice_pool(conn, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM invoices ORDER BY RANDOM() LIMIT %s", (pool_size,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No invoices found.")
    ids = [str(r[0]) for r in rows]
    print(f"  Q2 pool: {len(ids):,} invoice IDs")
    return ids

def make_q2_fn(conn, invoice_ids):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q2_SQL, (random.choice(invoice_ids),))
            cur.fetchall()
    return _run

def dry_run_q2(conn, invoice_ids):
    iid = invoice_ids[0]
    print(f"\n  DRY RUN — Q2 optimised (identical to naive) — invoice {iid}\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q2_SQL, (iid,))
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No rows returned.")
        return
    r = rows[0]
    print(f"  Invoice : {r['id']}  {r['invoice_type']}  {r['status']}")
    print(f"  Customer: {r['full_name']} <{r['email']}>")
    print(f"  Total   : {r['total_usd']}  Lines: {len(rows)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — Session & cart lookup (identical to naive, 50 threads)
# ═══════════════════════════════════════════════════════════════════════════════

Q3_SQL = """
SELECT s.id, s.user_id, s.cart, s.ip_address,
       s.user_agent, s.created_at, s.last_active_at, s.expires_at
FROM sessions s
WHERE s.user_id = %s
ORDER BY s.last_active_at DESC
LIMIT 1;
"""

def fetch_user_pool(conn, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id FROM (SELECT DISTINCT user_id FROM sessions) AS s
            ORDER BY RANDOM() LIMIT %s
        """, (pool_size,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No sessions found.")
    ids = [str(r[0]) for r in rows]
    print(f"  Q3 pool: {len(ids):,} user IDs")
    return ids

def make_q3_fn(user_ids):
    local = threading.local()
    def _get_conn():
        if not getattr(local, "conn", None) or local.conn.closed:
            local.conn = get_connection()
        return local.conn
    def _run():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(Q3_SQL, (random.choice(user_ids),))
            cur.fetchone()
    return _run

def dry_run_q3(conn, user_ids):
    uid = user_ids[0]
    print(f"\n  DRY RUN — Q3 optimised (identical to naive) — user {uid}\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q3_SQL, (uid,))
        row = cur.fetchone()
    if not row:
        print("  ⚠  No session found.")
        return
    print(f"  Session: {row['id']}  last_active: {row['last_active_at']}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — Co-purchase recommendations (identical to naive)
# ═══════════════════════════════════════════════════════════════════════════════

Q4_SQL = """
WITH orders_with_product AS (
    SELECT DISTINCT oi.order_id
    FROM   order_items oi
    JOIN   orders o ON o.id = oi.order_id
    WHERE  oi.product_id = %s
      AND  o.status IN ('confirmed', 'shipped', 'delivered')
),
co_purchased AS (
    SELECT oi.product_id, COUNT(DISTINCT oi.order_id) AS co_purchase_count
    FROM   order_items oi
    WHERE  oi.order_id   IN (SELECT order_id FROM orders_with_product)
      AND  oi.product_id != %s
    GROUP BY oi.product_id
)
SELECT p.id, p.name, p.product_type, p.price_usd, cp.co_purchase_count,
       ROUND(cp.co_purchase_count::NUMERIC /
             NULLIF((SELECT COUNT(*) FROM orders_with_product), 0), 4) AS confidence
FROM   co_purchased cp
JOIN   products p ON p.id = cp.product_id
WHERE  p.is_active = TRUE
ORDER BY cp.co_purchase_count DESC, p.name
LIMIT 10;
"""

def fetch_product_pool(conn, pool_size=500):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT product_id FROM (
                SELECT DISTINCT oi.product_id FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE o.status IN ('confirmed','shipped','delivered')
            ) AS s ORDER BY RANDOM() LIMIT %s
        """, (pool_size,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No products in confirmed orders.")
    ids = [str(r[0]) for r in rows]
    print(f"  Q4 pool: {len(ids):,} product IDs")
    return ids

def make_q4_fn(conn, product_ids):
    def _run():
        pid = random.choice(product_ids)
        with conn.cursor() as cur:
            cur.execute(Q4_SQL, (pid, pid))
            cur.fetchall()
    return _run

def dry_run_q4(conn, product_ids):
    pid = product_ids[0]
    print(f"\n  DRY RUN — Q4 optimised (identical to naive) — product {pid}\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q4_SQL, (pid, pid))
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No recommendations found.")
        return
    print(f"  {len(rows)} recommendation(s):")
    for r in rows[:5]:
        print(f"    {str(r['name'])[:40]:<40} count={r['co_purchase_count']}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q5 — Full-text search (identical to naive)
# ═══════════════════════════════════════════════════════════════════════════════

Q5_SQL = """
SELECT p.id, p.name, p.product_type, p.price_usd, p.attributes,
       ts_rank_cd(p.search_vector, query) AS rank
FROM   products p,
       plainto_tsquery('english', %s) AS query
WHERE  p.search_vector @@ query
  AND  p.is_active = TRUE
ORDER BY rank DESC
LIMIT 20;
"""

def make_q5_fn(conn):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q5_SQL, (random.choice(SEARCH_TERMS),))
            cur.fetchall()
    return _run

def dry_run_q5(conn):
    term = SEARCH_TERMS[0]
    print(f"\n  DRY RUN — Q5 optimised (identical to naive) — term: '{term}'\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q5_SQL, (term,))
        rows = cur.fetchall()
    print(f"  {len(rows)} result(s) for '{term}'")
    for r in rows[:5]:
        print(f"    {str(r['name'])[:40]:<40} rank={float(r['rank']):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q6 — User events in 30-day window
# ═══════════════════════════════════════════════════════════════════════════════
#
# SQL is identical to naive. Schema effect comes from the storage changes:
#   • 1-month chunks: a 30-day window touches 1-2 chunks (vs ≤5 with 7-day chunks)
#   • Compression with segmentby=user_id: all events for a user within a chunk
#     are stored contiguously. The (user_id, occurred_at) scan reads a single
#     contiguous compressed segment rather than scattered heap pages.

Q6_SQL = """
SELECT e.id, e.event_type, e.occurred_at, e.product_id, e.session_id, e.metadata
FROM events e
WHERE e.user_id     = %s
  AND e.occurred_at >= %s
  AND e.occurred_at <  %s
ORDER BY e.occurred_at DESC;
"""

def fetch_anchor_pool(conn, pool_size=1000):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, occurred_at FROM events ORDER BY RANDOM() LIMIT %s",
            (pool_size,),
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No events found.")
    print(f"  Q6 pool: {len(rows):,} anchor pairs")
    return list(rows)

def make_q6_fn(conn, pairs):
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start = anchor_dt - timedelta(days=15)
        end   = anchor_dt + timedelta(days=15)
        with conn.cursor() as cur:
            cur.execute(Q6_SQL, (user_id, start, end))
            cur.fetchall()
    return _run

def dry_run_q6(conn, pairs):
    user_id, anchor_dt = pairs[0]
    start = anchor_dt - timedelta(days=15)
    end   = anchor_dt + timedelta(days=15)
    print(f"\n  DRY RUN — Q6 optimised — user {user_id}")
    print(f"  Window: {start} → {end}\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q6_SQL, (user_id, start, end))
        rows = cur.fetchall()
    print(f"  {len(rows)} event(s) returned")
    for r in rows[:5]:
        print(f"    {str(r['event_type']):<25} {r['occurred_at']}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q7 — 7-day rolling revenue average (TimescaleDB killer feature)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reads from daily_revenue_by_tier continuous aggregate — no raw invoice scan.
# time_bucket_gapfill fills days with zero subscription revenue natively.
# Rolling average computed via window function on the gapfilled output.
#
# Marketplace revenue: still requires a small raw LATERAL query (same limitation
# as Q1 — LATERAL cannot be expressed inside a continuous aggregate). Added as a
# separate CTE and merged with the aggregate result in the final SELECT.
#
# Schema effect vs naive:
#   Naive:     generate_series + raw invoices scan + LATERAL + window fn (~100ms+)
#   Optimised: time_bucket_gapfill on 1,258-row aggregate + small LATERAL (~ms)

Q7_SQL = """
WITH gapfilled AS (
    -- Subscription revenue from continuous aggregate, gap-filled natively.
    -- time_bucket_gapfill fills missing days with NULL → COALESCE to 0.
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
    -- Marketplace revenue: raw LATERAL for temporal tier attribution.
    -- Small scan scoped to the 6-month window — unavoidable (see module note).
    SELECT
        DATE_TRUNC('day', i.created_at)::date   AS day,
        active_sub.tier_id,
        SUM(i.total_usd)                        AS mkt_revenue
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id
        FROM   subscriptions s
        WHERE  s.user_id    = i.user_id
          AND  s.started_at <= i.created_at
        ORDER BY s.started_at DESC
        LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
      AND i.status       = 'paid'
      AND i.created_at  >= %s AND i.created_at < %s
    GROUP BY 1, 2
),
combined AS (
    SELECT
        g.day,
        g.tier_id,
        g.tier_name,
        g.sub_revenue + COALESCE(m.mkt_revenue, 0.0) AS daily_total
    FROM gapfilled g
    LEFT JOIN mkt_daily m
        ON m.day     = g.day::date
       AND m.tier_id = g.tier_id
)
SELECT
    day,
    tier_name,
    ROUND(daily_total, 2) AS daily_revenue_usd,
    ROUND(
        AVG(daily_total) OVER (
            PARTITION BY tier_id
            ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_7day_avg_usd
FROM combined
ORDER BY day, tier_name;
"""

def load_data_date_range(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MIN(day)::date, MAX(day)::date
            FROM daily_revenue_by_tier;
        """)
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("daily_revenue_by_tier is empty — run the optimised loader first.")
    return row[0], row[1]

def random_window(data_min: date, data_max: date):
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS_Q7)
    start     = data_min + timedelta(days=random.randint(0, max_start))
    end       = start + timedelta(days=WINDOW_DAYS_Q7 - 1)
    return start, end

def make_q7_fn(conn, data_min: date, data_max: date):
    def _run():
        start, end = random_window(data_min, data_max)
        # params order: gapfill(start, end), WHERE day>= start < end,
        #               mkt WHERE created_at >= start < end
        params = (
            start, end,   # time_bucket_gapfill bounds
            start, end,   # WHERE day >= %s AND day < %s
            start, end,   # mkt_daily WHERE created_at >= %s AND < %s
        )
        with conn.cursor() as cur:
            cur.execute(Q7_SQL, params)
            cur.fetchall()
    return _run

def dry_run_q7(conn, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  DRY RUN — Q7 optimised (time_bucket_gapfill on continuous aggregate)")
    print(f"  Window: {start} → {end} ({WINDOW_DAYS_Q7} days)\n")
    params = (start, end, start, end, start, end)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q7_SQL, params)
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No rows returned.")
        return
    print(f"  {len(rows)} rows ({len(rows) // 3} days × 3 tiers)\n")
    print(f"  {'Day':<12} {'Tier':<12} {'Daily':>12} {'7d Avg':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for row in rows[:12]:
        print(
            f"  {str(row['day']):<12} {str(row['tier_name']):<12} "
            f"{float(row['daily_revenue_usd']):>12.2f} "
            f"{float(row['rolling_7day_avg_usd']):>12.2f}"
        )
    if len(rows) > 12:
        print(f"  ... and {len(rows) - 12} more rows")

# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_query(
    query_id:    str,
    conn,
    iterations:  int,
    dry:         bool,
    pools:       dict,
    results_dir: str,
):
    output_path = os.path.join(results_dir, f"timescaledb_optimised_{query_id}.json")

    print(f"\n{'═'*60}")
    print(f"  {query_id} — TimescaleDB Optimised")
    print(f"{'═'*60}")

    try:
        if query_id == "Q1":
            if dry:
                dry_run_q1(conn); return
            run_benchmark(
                query_fn=make_q1_fn(conn),
                db="timescaledb_optimised", query_id="Q1",
                label=(
                    "Monthly revenue by tier (last 12 months). "
                    "Subscription: time_bucket('1 month', day) on daily_revenue_by_tier "
                    "continuous aggregate — no raw invoice scan. "
                    "Marketplace: raw LATERAL temporal attribution (unavoidable). "
                    "1-month invoice chunks."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q2":
            pool = pools.get("invoices") or fetch_invoice_pool(conn)
            pools["invoices"] = pool
            if dry:
                dry_run_q2(conn, pool); return
            run_benchmark(
                query_fn=make_q2_fn(conn, pool),
                db="timescaledb_optimised", query_id="Q2",
                label=(
                    "Full invoice fetch via 4-table JOIN. "
                    "Identical to naive — no TimescaleDB benefit for point lookups. "
                    f"Pool of {len(pool):,} invoice IDs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q3":
            pool = pools.get("users") or fetch_user_pool(conn)
            pools["users"] = pool
            if dry:
                dry_run_q3(conn, pool); return
            run_benchmark(
                query_fn=make_q3_fn(pool),
                db="timescaledb_optimised", query_id="Q3",
                label=(
                    "Active session + cart retrieval under 50 concurrent threads. "
                    "Identical to naive — sessions is a plain table. "
                    f"Pool of {len(pool):,} user IDs."
                ),
                iterations=iterations, concurrency=50, output_path=output_path,
            )

        elif query_id == "Q4":
            pool = pools.get("products") or fetch_product_pool(conn)
            pools["products"] = pool
            if dry:
                dry_run_q4(conn, pool); return
            run_benchmark(
                query_fn=make_q4_fn(conn, pool),
                db="timescaledb_optimised", query_id="Q4",
                label=(
                    "Top-10 co-purchase recommendations via 2-hop JOIN. "
                    "Identical to naive — order_items/orders are plain tables. "
                    f"Pool of {len(pool):,} product IDs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q5":
            if dry:
                dry_run_q5(conn); return
            run_benchmark(
                query_fn=make_q5_fn(conn),
                db="timescaledb_optimised", query_id="Q5",
                label=(
                    "Full-text search via tsvector GIN index + ts_rank_cd. "
                    "Identical to naive — products is a plain table. "
                    f"{len(SEARCH_TERMS)} search terms, random per iteration."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q6":
            pool = pools.get("anchors") or fetch_anchor_pool(conn)
            pools["anchors"] = pool
            if dry:
                dry_run_q6(conn, pool); return
            run_benchmark(
                query_fn=make_q6_fn(conn, pool),
                db="timescaledb_optimised", query_id="Q6",
                label=(
                    f"All events for a user in a {WINDOW_DAYS_Q6}-day window. "
                    "SQL identical to naive. Schema effect: 1-month chunks "
                    "(30-day window touches 1-2 chunks vs ≤5 with 7-day chunks) + "
                    "compression segmentby=user_id (contiguous user segments). "
                    f"Pool of {len(pool):,} anchor pairs."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        elif query_id == "Q7":
            data_min, data_max = pools.get("date_range") or load_data_date_range(conn)
            pools["date_range"] = (data_min, data_max)
            if dry:
                dry_run_q7(conn, data_min, data_max); return
            run_benchmark(
                query_fn=make_q7_fn(conn, data_min, data_max),
                db="timescaledb_optimised", query_id="Q7",
                label=(
                    f"7-day rolling revenue average per tier over {WINDOW_DAYS_Q7} days. "
                    "time_bucket_gapfill('1 day', day) on daily_revenue_by_tier "
                    "continuous aggregate — native gap-fill, no raw invoice scan for "
                    "subscription component. Marketplace via small raw LATERAL. "
                    "Rolling average via window function. "
                    f"Aggregate range: {data_min} → {data_max}."
                ),
                iterations=iterations, concurrency=1, output_path=output_path,
            )

        ok(f"{query_id} complete → {output_path}")

    except Exception as e:
        fail(f"{query_id} failed: {e}")
        import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TimescaleDB optimised Q1–Q7 benchmarks"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run",    action="store_true", dest="dry_run")
    parser.add_argument("--only",       nargs="+", metavar="Q")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, dest="results_dir")
    args = parser.parse_args()

    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]
    iterations = args.iterations

    print("\n" + "═" * 60)
    print("  TimescaleDB Optimised — Q1–Q7 Benchmarks")
    print("═" * 60)
    print(f"  Queries    : {queries}")
    print(f"  Iterations : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir: {args.results_dir}")
    print()
    print("  Schema features:")
    print("  • Q1/Q7: daily_revenue_by_tier continuous aggregate")
    print("  • Q6/Q7: 1-month hypertable chunks")
    print("  • Q6   : compression segmentby=user_id (contiguous user segments)")
    print("  • Q2–Q5: identical to naive (plain tables, no TS benefit)")

    os.makedirs(args.results_dir, exist_ok=True)

    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET jit = off; SET work_mem = '64MB';")
    conn.autocommit = False

    pools      = {}   # shared pools so they are built once and reused
    total_start = time.perf_counter()

    for qid in queries:
        run_query(
            query_id=qid,
            conn=conn,
            iterations=iterations,
            dry=args.dry_run,
            pools=pools,
            results_dir=args.results_dir,
        )

    conn.close()
    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  {GREEN}Done.{RESET}\n")


if __name__ == "__main__":
    main()