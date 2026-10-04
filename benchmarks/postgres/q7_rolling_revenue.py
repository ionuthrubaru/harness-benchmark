"""
benchmarks/postgres/q7_rolling_revenue.py — PostgreSQL Baseline: Q7
====================================================================
Q7: 7-day rolling average of daily revenue per subscription tier
    over a 6-month window, with gap-filling for days with zero activity.

    This is the PostgreSQL baseline for the TimescaleDB comparison (Q7).
    TimescaleDB will serve the same result using time_bucket_gapfill()
    targeting a pre-computed continuous aggregate.

Killer feature demonstrated (TimescaleDB side):
    time_bucket_gapfill() + continuous aggregates — TimescaleDB stores
    pre-aggregated daily revenue buckets as a materialised view that is
    incrementally refreshed. Gap-filling is a native operation. The
    rolling average reads from the aggregate, not the raw invoices table.
    PostgreSQL must scan raw invoices, aggregate by day, fill gaps with
    generate_series, then compute the rolling window — all at query time
    on every execution.

Window anchoring
────────────────
The 6-month window start is chosen at random within the actual min/max
of paid invoice created_at, queried once at startup. This replaces the
previous hardcoded DATASET_START/DATASET_END constants (date(2024,1,1)
and date(2025,12,31)), which could generate windows that miss the data
entirely if the dataset date range shifts.

Usage:
    cd benchmarks/postgres
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

# -- window constants ---------------------------------------------------------

WINDOW_MONTHS = 6
WINDOW_DAYS   = 183          # ~6 months in days

# -- connection ---------------------------------------------------------------

from pg_conn import get_connection

# -- data range loader --------------------------------------------------------

def load_data_date_range(conn):
    """
    Query the actual min/max of paid invoice created_at so the random
    window is always anchored within real data — no hardcoded dates.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MIN(created_at)::date, MAX(created_at)::date
            FROM invoices WHERE status = 'paid';
        """)
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("No paid invoices found — run the data loader first.")
    return row[0], row[1]

# -- query --------------------------------------------------------------------

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

# -- helpers ------------------------------------------------------------------

def random_window(data_min: date, data_max: date):
    """
    Return a (window_start, window_end) pair anchored to a random date
    within the actual invoice date range. The window is always WINDOW_DAYS
    long and always falls within the data boundaries.
    """
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)
    start     = data_min + timedelta(days=random.randint(0, max_start))
    end       = start + timedelta(days=WINDOW_DAYS - 1)
    return start, end

def query_params(start: date, end: date) -> tuple:
    """
    Q7 references the window boundaries 3 times each in the SQL
    (date_spine + two UNION ALL branches). Returns the full params tuple.
    """
    return (start, end, start, end, start, end)

# -- benchmark factory --------------------------------------------------------

def make_query_fn(conn, data_min: date, data_max: date):
    def _run():
        start, end = random_window(data_min, data_max)
        with conn.cursor() as cur:
            cur.execute(Q7_SQL, query_params(start, end))
            cur.fetchall()
    return _run

# -- helper modes -------------------------------------------------------------

def dry_run(conn, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  DRY RUN -- Q7 rolling revenue")
    print(f"  Window: {start} to {end} ({WINDOW_DAYS} days)\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q7_SQL, query_params(start, end))
        rows = cur.fetchall()

    if not rows:
        print("  No revenue data found in this window.")
        return

    total_rows = len(rows)
    print(f"  {total_rows} rows returned ({total_rows // 3} days x 3 tiers)\n")

    headers = ["day", "tier_name", "daily_revenue_usd", "rolling_7day_avg_usd"]
    col_w   = [12, 12, 20, 22]

    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    sep_line    = "  " + "  ".join("─" * w for w in col_w)
    print(header_line)
    print(sep_line)

    sample = rows[:9]
    if total_rows > 18:
        sample += [None]
        sample += rows[-9:]

    for row in sample:
        if row is None:
            print(f"  {'...'}")
            continue
        print("  " + "  ".join(
            str(row[h]).ljust(w) for h, w in zip(headers, col_w)
        ))


def explain(conn, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  EXPLAIN ANALYZE -- Q7 ({start} to {end}):\n")
    with conn.cursor() as cur:
        cur.execute(Q7_EXPLAIN_SQL, query_params(start, end))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# -- entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Q7 rolling revenue benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Execute once and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q7_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q7_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL -- Q7 Rolling Revenue Benchmark")
    print("=" * 50)
    print(f"  Window size  : {WINDOW_DAYS} days (~{WINDOW_MONTHS} months)")

    conn = get_connection()

    try:
        data_min, data_max = load_data_date_range(conn)
        print(f"  Invoice range: {data_min} → {data_max}")
        print(f"  Window start : random within [{data_min}, {data_max - timedelta(days=WINDOW_DAYS)}]")

        if args.explain:
            explain(conn, data_min, data_max)
            return

        if args.dry_run:
            dry_run(conn, data_min, data_max)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, data_min, data_max),
            db="postgres",
            query_id="Q7",
            label=(
                f"7-day rolling average of daily revenue per subscription tier "
                f"over a {WINDOW_DAYS}-day (~{WINDOW_MONTHS}-month) window with "
                "generate_series gap-filling. Subscription + marketplace revenue "
                "both included with temporal tier attribution (mirrors Q1 logic). "
                "Random window within actual invoice date range per iteration — "
                "no hardcoded dataset bounds."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()