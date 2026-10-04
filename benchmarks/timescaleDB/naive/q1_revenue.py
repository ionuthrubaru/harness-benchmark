"""
benchmarks/timescaledb/naive/q1_revenue.py — TimescaleDB Naive: Q1
===================================================================
Q1: Monthly revenue by subscription tier, last 12 months.
    Includes marketplace invoices attributed to the user's active
    subscription tier at the time of purchase.
    Temporal JOIN on subscription_tier_pricing captures the correct
    monthly price in effect at invoice creation time.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
TimescaleDB is a PostgreSQL extension — the same CTE structure, LATERAL
subquery, and temporal JOIN predicate run without modification. This is a
deliberate property of the naive schema comparison: the SQL is held constant
so any latency difference is attributable purely to the engine.

Engine effect for Q1 (naive)
─────────────────────────────
TimescaleDB's invoices hypertable is partitioned by created_at into 7-day
chunks (naive default). The WHERE clause `i.created_at >= NOW() - INTERVAL
'12 months'` triggers chunk pruning: TimescaleDB reads only the ~52 chunks
covering the last 12 months, skipping older chunks entirely. PostgreSQL reads
the full invoices heap and relies on the B-tree index on created_at.

For a 2-year dataset, chunk pruning eliminates ~50% of storage I/O vs a full
heap scan. Whether this translates to a measurable latency difference over
PostgreSQL's index scan depends on the index hit rate — documented in results.

Usage:
    cd benchmarks/timescaledb/naive
    python q1_revenue.py                   # 1000 iterations
    python q1_revenue.py --iterations 100  # quick smoke test
    python q1_revenue.py --explain         # EXPLAIN ANALYZE
    python q1_revenue.py --dry-run         # run once, print results
"""

import argparse
import os
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

# ── query ─────────────────────────────────────────────────────────────────────
# Identical to PostgreSQL Q1. TimescaleDB executes this with chunk pruning
# on the invoices hypertable — no SQL change required.

Q1_SQL = """
WITH invoice_tiers AS (

    SELECT
        i.id            AS invoice_id,
        i.created_at,
        i.total_usd,
        i.invoice_type,
        sub.tier_id
    FROM invoices i
    JOIN subscriptions sub ON sub.id = i.subscription_id
    WHERE i.invoice_type = 'subscription'
      AND i.status       = 'paid'
      AND i.created_at  >= NOW() - INTERVAL '12 months'

    UNION ALL

    SELECT
        i.id            AS invoice_id,
        i.created_at,
        i.total_usd,
        i.invoice_type,
        active_sub.tier_id
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

),

monthly_revenue AS (

    SELECT
        DATE_TRUNC('month', it.created_at)  AS month,
        st.name                             AS tier_name,
        stp.monthly_price_usd               AS price_in_effect_usd,
        COUNT(it.invoice_id)                AS invoice_count,
        SUM(it.total_usd)                   AS total_revenue_usd
    FROM invoice_tiers it
    JOIN subscription_tiers st
        ON st.id = it.tier_id
    JOIN subscription_tier_pricing stp
        ON  stp.tier_id    = it.tier_id
        AND stp.valid_from <= it.created_at
        AND (stp.valid_to IS NULL OR stp.valid_to > it.created_at)
    GROUP BY 1, 2, 3

)

SELECT
    TO_CHAR(month, 'YYYY-MM') AS month,
    tier_name,
    price_in_effect_usd,
    invoice_count,
    ROUND(total_revenue_usd, 2) AS total_revenue_usd
FROM monthly_revenue
ORDER BY month, tier_name;
"""

Q1_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q1_SQL

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn):
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q1_SQL)
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn):
    print("\n  DRY RUN — Q1 naive result sample (first 10 rows):\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q1_SQL)
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  No rows returned — is the naive schema loaded?")
        return
    headers = list(rows[0].keys())
    col_w = 22
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("─" * col_w for _ in headers))
    for row in rows[:10]:
        print("  " + "  ".join(str(row[h]).ljust(col_w) for h in headers))
    print(f"\n  {len(rows)} total rows returned.")


def explain(conn):
    print("\n  EXPLAIN ANALYZE — Q1 naive:\n")
    with conn.cursor() as cur:
        cur.execute(Q1_EXPLAIN_SQL)
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q1 revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "timescaledb_naive_Q1.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q1 Monthly Revenue Benchmark")
    print("=" * 60)
    print("  Schema : naive (hypertable on invoices, 7-day chunks)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : chunk pruning on created_at >= NOW() - 12 months")

    conn = get_connection()
    try:
        if args.explain:
            explain(conn)
            return
        if args.dry_run:
            dry_run(conn)
            return
        run_benchmark(
            query_fn=make_query_fn(conn),
            db="timescaledb_naive",
            query_id="Q1",
            label=(
                "Monthly revenue by subscription tier (last 12 months). "
                "SQL identical to PostgreSQL baseline. "
                "TimescaleDB engine effect: chunk pruning on invoices hypertable "
                "(created_at >= NOW() - 12 months skips older 7-day chunks). "
                "Naive schema: 7-day chunk interval (TimescaleDB default)."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()