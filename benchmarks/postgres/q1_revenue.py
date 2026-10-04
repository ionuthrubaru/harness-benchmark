"""
benchmarks/postgres/q1_revenue.py — PostgreSQL Baseline: Q1
============================================================
Q1: Monthly revenue by subscription tier, last 12 months.
    Includes marketplace invoices attributed to the user's active
    subscription tier at the time of purchase.
    Temporal JOIN on subscription_tier_pricing captures the correct
    monthly price in effect at invoice creation time (old vs new prices).

Killer feature demonstrated:
    Multi-table temporal JOIN — the relational model handles historic
    pricing naturally via valid_from / valid_to range predicates.

Usage:
    cd benchmarks\postgres
    python q1_revenue.py                   # 1000 iterations, save results
    python q1_revenue.py --iterations 100  # quick smoke test
    python q1_revenue.py --explain         # print EXPLAIN ANALYZE, no benchmark
    python q1_revenue.py --dry-run         # run query once, print results
"""

import argparse
import os
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── allow `from harness import ...` regardless of working directory ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

# ── connection ────────────────────────────────────────────────────────────────

from pg_conn import get_connection

# ── query ─────────────────────────────────────────────────────────────────────
#
# Design notes
# ────────────
# The query is built in two CTEs:
#
#   1. invoice_tiers
#      Resolves the tier for every paid invoice in the last 12 months.
#
#      - Subscription invoices  → tier comes directly from the linked subscription.
#      - Marketplace invoices   → tier resolved via LATERAL subquery that picks
#        the most-recently-started subscription for that user as of invoice date.
#        LIMIT 1 + ORDER BY started_at DESC is a clean, index-friendly pattern.
#
#   2. monthly_revenue
#      Groups by (month, tier) and sums invoice totals.
#      The JOIN onto subscription_tier_pricing is the temporal JOIN:
#      it captures which price band (old or new) was in effect at invoice time.
#      This makes the per-tier monthly_price_usd column academically interesting —
#      invoices before June 2024 carry the old price, those after carry the new.
#
# The final SELECT formats the month as YYYY-MM for readability and orders
# chronologically so charts can be built directly from the output.

Q1_SQL = """
WITH invoice_tiers AS (

    -- ── Subscription invoices ──────────────────────────────────────────────
    -- Tier is known directly from the subscription record.
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

    -- ── Marketplace invoices ───────────────────────────────────────────────
    -- Attribute to whichever subscription was most recently active for this
    -- user at the time the invoice was created.
    -- LATERAL + ORDER BY started_at DESC LIMIT 1 is index-friendly and avoids
    -- a correlated subquery that would re-plan for every row.
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
        -- Temporal JOIN: pick the price row whose validity window contains
        -- the invoice date. This is the relational elegance Q1 demonstrates.
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

# EXPLAIN ANALYZE wrapper — same query, prefixed for plan output
Q1_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q1_SQL

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn):
    """
    Return a zero-argument callable that executes Q1 on the given connection.
    Reuses a single persistent connection across all iterations — this mirrors
    how a real application connection pool behaves and avoids measuring
    connection-setup overhead as part of query latency.
    """
    def _run():
        with conn.cursor() as cur:
            cur.execute(Q1_SQL)
            cur.fetchall()   # ensure full result materialisation is timed
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn):
    """Execute once and print the first 10 result rows for sanity checking."""
    print("\n  DRY RUN — Q1 result sample (first 10 rows):\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q1_SQL)
        rows = cur.fetchall()
    if not rows:
        print("  ⚠  Query returned no rows — is the database populated?")
        return
    headers = list(rows[0].keys())
    col_w = 22
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("─" * col_w for _ in headers))
    for row in rows[:10]:
        print("  " + "  ".join(str(row[h]).ljust(col_w) for h in headers))
    print(f"\n  ... {len(rows)} total rows returned")


def explain(conn):
    """Print EXPLAIN ANALYZE output — useful for index and plan inspection."""
    print("\n  EXPLAIN ANALYZE — Q1:\n")
    with conn.cursor() as cur:
        cur.execute(Q1_EXPLAIN_SQL)
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Q1 revenue benchmark")
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
        default=os.path.join("results", "postgres_q1_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q1_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q1 Monthly Revenue Benchmark")
    print("=" * 50)

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
            db="postgres",
            query_id="Q1",
            label=(
                "Monthly revenue by subscription tier (last 12 months), "
                "with temporal JOIN on pricing history and marketplace "
                "attribution to user's active subscription at purchase time."
            ),
            iterations=args.iterations,
            concurrency=1,        # Q1 is a single-threaded analytical query
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()