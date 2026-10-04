"""
benchmarks/neo4j/naive/q7_rolling_revenue.py — Neo4j Naive: Q7
==============================================================
Q7: 7-day rolling average of daily revenue per subscription tier over a
    6-month window, with gap-filling for days with zero activity.

Full equivalent of PostgreSQL Q7, computed server-side in ONE Cypher query:

  1. Daily revenue per tier — UNION ALL of the subscription-invoice path
     (Invoice→Subscription→Tier) and the marketplace path (user's most
     recent subscription started_at <= invoice created_at), exactly as the
     two UNION ALL branches of the PostgreSQL daily_revenue CTE.
  2. Gap-filling — range(0, n_days-1) is the Cypher equivalent of
     generate_series; every (tier, day) pair is emitted, with 0.0 for days
     without revenue (PostgreSQL: tier_days CROSS JOIN + LEFT JOIN).
  3. Rolling 7-day average — list slicing daily[o-6 .. o] per tier, the
     equivalent of AVG() OVER (PARTITION BY tier ORDER BY day
     ROWS BETWEEN 6 PRECEDING AND CURRENT ROW). The first six days average
     over the rows available, as in PostgreSQL.

Output: one row per (day, tier) — n_days x 3 tiers — with
daily_revenue_usd and rolling_7day_avg_usd rounded to 2 dp, ordered by
day, tier_name. Identical shape to PostgreSQL Q7.

No client-side post-processing: the Python driver only consumes the rows.
Cypher is identical in the optimised schema — engine effect only.

Usage:
    python q7_rolling_revenue.py
    python q7_rolling_revenue.py --iterations 100
    python q7_rolling_revenue.py --dry-run
"""


import argparse
import os
import random
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

WINDOW_DAYS = 183   # ~6 months (same as PostgreSQL Q7)
LEVEL       = "Naive"

# ── Cypher ────────────────────────────────────────────────────────────────────
# Parameters:
#   $start   window start date, 'YYYY-MM-DD' (inclusive)
#   $end     day AFTER the window end, 'YYYY-MM-DD' (exclusive)
#   $n_days  number of days in the window (WINDOW_DAYS)
# created_at is stored as an ISO-8601 UTC string, so string range comparison
# and substring(…, 0, 10) are exact.

Q7_CYPHER = """
CALL {
    MATCH (i:Invoice)-[:FOR_SUBSCRIPTION]->(:Subscription)-[:ON_TIER]->(t:SubscriptionTier)
    WHERE i.status       = 'paid'
      AND i.invoice_type = 'subscription'
      AND i.created_at  >= $start
      AND i.created_at  <  $end
    RETURN t.id AS tier_id, substring(i.created_at, 0, 10) AS day, toFloat(i.total_usd) AS amount

    UNION ALL

    MATCH (u:User)-[:HAS_INVOICE]->(i:Invoice)
    WHERE i.status       = 'paid'
      AND i.invoice_type = 'marketplace'
      AND i.created_at  >= $start
      AND i.created_at  <  $end
    MATCH (u)-[:HAS_SUBSCRIPTION]->(s:Subscription)
    WHERE s.started_at <= i.created_at
    WITH i, s
    ORDER BY s.started_at DESC
    WITH i, head(collect(s)) AS active_sub
    WHERE active_sub IS NOT NULL
    RETURN active_sub.tier_id AS tier_id, substring(i.created_at, 0, 10) AS day, toFloat(i.total_usd) AS amount
}
// 1. daily revenue per (tier, day offset)
WITH tier_id,
     duration.inDays(date($start), date(day)).days AS offset,
     sum(amount) AS revenue
WITH tier_id, collect([offset, revenue]) AS points
WITH collect({tier_id: tier_id, points: points}) AS by_tier
// 2. gap-fill: every tier x every day in the window (generate_series equivalent)
MATCH (t:SubscriptionTier)
WITH t, [b IN by_tier WHERE b.tier_id = t.id | b.points] AS found
WITH t, CASE WHEN size(found) = 0 THEN [] ELSE found[0] END AS points
WITH t, [o IN range(0, $n_days - 1) |
           coalesce(head([p IN points WHERE p[0] = o | p[1]]), 0.0)] AS daily
// 3. rolling 7-day average (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
UNWIND range(0, size(daily) - 1) AS o
WITH t, o, daily[o] AS revenue,
     daily[CASE WHEN o < 6 THEN 0 ELSE o - 6 END .. o + 1] AS win
RETURN
    toString(date($start) + duration({days: o}))            AS day,
    t.name                                                  AS tier_name,
    round(revenue, 2)                                       AS daily_revenue_usd,
    round(reduce(acc = 0.0, x IN win | acc + x) / size(win), 2) AS rolling_7day_avg_usd
ORDER BY day, tier_name
"""

# ── date range loader ─────────────────────────────────────────────────────────

def load_data_date_range(driver) -> tuple[date, date]:
    cypher = """
    MATCH (i:Invoice)
    WHERE i.status = 'paid'
    RETURN min(i.created_at) AS min_dt, max(i.created_at) AS max_dt
    """
    with driver.session() as session:
        row = session.run(cypher).single()
    if not row or not row["min_dt"]:
        raise RuntimeError("No paid invoices found — run the loader first.")
    return date.fromisoformat(row["min_dt"][:10]), date.fromisoformat(row["max_dt"][:10])


def random_window(data_min: date, data_max: date) -> tuple[str, str]:
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)
    start     = data_min + timedelta(days=random.randint(0, max_start))
    end       = start + timedelta(days=WINDOW_DAYS - 1)
    return start.isoformat(), (end + timedelta(days=1)).isoformat()


# ── query function ────────────────────────────────────────────────────────────

def run_q7(session, start: str, end: str) -> list[dict]:
    return session.run(Q7_CYPHER, start=start, end=end, n_days=WINDOW_DAYS).data()


def make_query_fn(driver, data_min: date, data_max: date):
    def _run():
        start, end = random_window(data_min, data_max)
        with driver.session() as session:
            return run_q7(session, start, end)
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, data_min: date, data_max: date):
    start, end = random_window(data_min, data_max)
    print(f"\n  DRY RUN — Neo4j Naive Q7")
    print(f"  Window: {start} to {end} (exclusive), {WINDOW_DAYS} days\n")

    with driver.session() as session:
        rows = run_q7(session, start, end)

    if not rows:
        print("  ⚠  No rows returned.")
        return
    n_zero = sum(1 for r in rows if r["daily_revenue_usd"] == 0)
    print(f"  {len(rows)} rows (expected {WINDOW_DAYS} days x 3 tiers = {WINDOW_DAYS * 3}); "
          f"{n_zero} gap-filled zero-revenue rows\n")
    print(f"  {'Day':<12} {'Tier':<12} {'Daily revenue':>15} {'7-day avg':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*15} {'─'*12}")
    for row in rows[:12]:
        print(
            f"  {row['day']:<12} {row['tier_name']:<12} "
            f"{row['daily_revenue_usd']:>15,.2f} {row['rolling_7day_avg_usd']:>12,.2f}"
        )
    if len(rows) > 12:
        print(f"  ... {len(rows) - 12} more rows")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Naive Q7 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "neo4j_naive_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Naive — Q7 Rolling Revenue Benchmark")
    print("=" * 55)
    print(f"  Window size : {WINDOW_DAYS} days (~6 months)")
    print(f"  Method      : one Cypher query — daily revenue + gap-fill + rolling 7-day avg")

    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))

    try:
        data_min, data_max = load_data_date_range(driver)
        print(f"  Invoice range: {data_min} → {data_max}")

        if args.dry_run:
            dry_run(driver, data_min, data_max)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, data_min, data_max),
            db="neo4j_naive",
            query_id="Q7",
            label=(
                f"7-day rolling average of daily revenue per tier over {WINDOW_DAYS} days, "
                "gap-filled. Single Cypher query, all server-side: UNION ALL of "
                "subscription path and marketplace most-recent-subscription path, "
                "range()-based gap-fill (generate_series equivalent) for 3 tiers x "
                f"{WINDOW_DAYS} days, rolling avg via list slice daily[o-6..o] "
                "(ROWS BETWEEN 6 PRECEDING AND CURRENT ROW). "
                "Full PostgreSQL-equivalent output; no client-side post-processing."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()