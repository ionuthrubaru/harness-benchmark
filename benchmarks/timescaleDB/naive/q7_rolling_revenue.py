"""
benchmarks/timescaledb/naive/q7_rolling_revenue.py — TimescaleDB Naive: Q7
===========================================================================
Q7: 7-day rolling average of daily revenue per subscription tier
    over a 6-month window, with gap-filling for days with zero activity.

This is TimescaleDB's killer query — and the naive schema shows a
partial engine effect even without the continuous aggregate.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
The same CTE structure runs unchanged on TimescaleDB:
  - generate_series date spine for gap-filling
  - daily aggregation from raw invoices (subscription + marketplace)
  - LATERAL temporal tier attribution for marketplace invoices
  - window function for 7-day rolling average

Engine effect for Q7 (naive) — chunk pruning on invoices
──────────────────────────────────────────────────────────
invoices is a hypertable partitioned by created_at with 7-day chunks.
The WHERE clauses `i.created_at >= %s AND i.created_at < %s` trigger
chunk pruning: only the ~26 chunks covering the 6-month window are
opened. PostgreSQL scans the full invoices heap and relies on the
B-tree index on created_at to filter.

For a 2-year dataset (~24 months = ~104 7-day chunks), a 6-month
query touches ~26 chunks (~25% of the total). TimescaleDB skips ~78
chunks entirely at planning time. Whether this beats PostgreSQL's
index scan depends on heap clustering — documented in results.

Naive vs optimised comparison
───────────────────────────────
Naive:     generate_series + raw invoices scan + gap-fill + window fn
           (identical to PostgreSQL, just with chunk pruning added)
Optimised: time_bucket_gapfill() on daily_revenue_by_tier continuous
           aggregate — no raw invoices scan, no generate_series,
           native gap-filling and rolling average on pre-materialised data

The latency difference between naive and optimised Q7 is the primary
schema effect measurement for the TimescaleDB chapter and the central
demonstration of the continuous aggregate advantage.

Window anchoring
─────────────────
Identical to PostgreSQL Q7: 6-month window start chosen randomly within
the actual min/max of paid invoice created_at, queried at startup.

Usage:
    cd benchmarks/timescaledb/naive
    python q7_rolling_revenue.py                   # 1000 iterations
    python q7_rolling_revenue.py --iterations 100  # quick smoke test
    python q7_rolling_revenue.py --explain         # EXPLAIN ANALYZE
    python q7_rolling_revenue.py --dry-run         # run once, print sample
"""

import argparse
import os
import random
import sys
from datetime import date, timedelta

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

WINDOW_MONTHS = 6
WINDOW_DAYS   = 183

# ── data range ────────────────────────────────────────────────────────────────

def load_data_date_range(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MIN(created_at)::date, MAX(created_at)::date
            FROM invoices WHERE status = 'paid';
        """)
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("No paid invoices — run timescaledb_naive_loader.py first.")
    return row[0], row[1]

# ── query — identical to PostgreSQL Q7 ────────────────────────────────────────

Q7_SQL = """
WITH date_spine AS (

    SELECT generate_series(
        %s::date,
        %s::date,
        '1 day'::interval
    )::date AS day

),

tier_days AS (

    SELECT
        ds.day,
        st.id   AS tier_id,
        st.name AS tier_name
    FROM date_spine ds
    CROSS JOIN subscription_tiers st

),

daily_revenue AS (

    SELECT
        DATE_TRUNC('day', i.created_at)::date AS day,
        sub.tier_id,
        SUM(i.total_usd)                       AS revenue
    FROM invoices i
    JOIN subscriptions sub ON sub.id = i.subscription_id
    WHERE i.invoice_type = 'subscription'
      AND i.status       = 'paid'
      AND i.created_at  >= %s::date
      AND i.created_at  <  (%s::date + INTERVAL '1 day')
    GROUP BY 1, 2

    UNION ALL

    SELECT
        DATE_TRUNC('day', i.created_at)::date AS day,
        active_sub.tier_id,
        SUM(i.total_usd)                       AS revenue
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
      AND i.created_at  >= %s::date
      AND i.created_at  <  (%s::date + INTERVAL '1 day')
    GROUP BY 1, 2

),

filled AS (

    SELECT
        td.day,
        td.tier_id,
        td.tier_name,
        COALESCE(SUM(dr.revenue), 0.00) AS daily_total
    FROM tier_days td
    LEFT JOIN daily_revenue dr
        ON  dr.day     = td.day
        AND dr.tier_id = td.tier_id
    GROUP BY td.day, td.tier_id, td.tier_name

)

SELECT
    day,
    tier_name,
    ROUND(daily_total, 2)                        AS daily_revenue_usd,
    ROUND(
        AVG(daily_total) OVER (
            PARTITION BY tier_id
            ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    )                                            AS rolling_7day_avg_usd
FROM filled
ORDER BY day, tier_name;
"""

Q7_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q7_SQL

# ── helpers ───────────────────────────────────────────────────────────────────

def random_window(data_min: date, data_max: date):
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)
    start     = data_min + timedelta(days=random.randint(0, max_start))
    end       = start + timedelta(days=WINDOW_DAYS - 1)
    return start, end


def query_params(start: date, end: date) -> tuple:
    return (start, end, start, end, start, end)

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, data_min: date, data_max: date):
    def _run():
        start, end = random_window(data_min, data_max)
        with conn.cursor() as cur:
            cur.execute(Q7_SQL, query_params(start, end))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  DRY RUN — Q7 naive rolling revenue")
    print(f"  Window: {start} → {end} ({WINDOW_DAYS} days)\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q7_SQL, query_params(start, end))
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No revenue data in this window.")
        return
    print(f"  {len(rows)} rows ({len(rows) // 3} days × 3 tiers)\n")
    headers = ["day", "tier_name", "daily_revenue_usd", "rolling_7day_avg_usd"]
    col_w   = [12, 12, 20, 22]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "  ".join("─" * w for w in col_w))
    for row in rows[:12]:
        print("  " + "  ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_w)))
    if len(rows) > 12:
        print(f"  ... and {len(rows) - 12} more rows")


def explain(conn, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  EXPLAIN ANALYZE — Q7 naive ({start} → {end}):\n")
    print("  Note: look for 'Chunks excluded by runtime exclusion'")
    print("        in the invoices scan nodes — confirms chunk pruning.\n")
    with conn.cursor() as cur:
        cur.execute(Q7_EXPLAIN_SQL, query_params(start, end))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q7 rolling revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q7 Rolling Revenue Benchmark")
    print("=" * 60)
    print("  Schema : naive (invoices hypertable, 7-day chunks)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : chunk pruning on invoices created_at range")
    print(f"  Window : {WINDOW_DAYS} days (~{WINDOW_MONTHS} months)")

    conn = get_connection()
    try:
        data_min, data_max = load_data_date_range(conn)
        print(f"  Invoice range: {data_min} → {data_max}")

        if args.explain:
            explain(conn, data_min, data_max)
            return
        if args.dry_run:
            dry_run(conn, data_min, data_max)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, data_min, data_max),
            db="timescaledb_naive",
            query_id="Q7",
            label=(
                f"7-day rolling average of daily revenue per tier over {WINDOW_DAYS} days. "
                "SQL identical to PostgreSQL baseline: generate_series gap-fill + "
                "raw invoices scan + LATERAL temporal tier attribution + window function. "
                "TimescaleDB engine effect: chunk pruning on invoices hypertable — "
                "6-month window touches ~26 of ~104 7-day chunks. "
                "Naive: no continuous aggregate (that is the optimised schema). "
                f"Invoice range: {data_min} → {data_max}."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()