"""
benchmarks/neo4j/optimised/q1_revenue.py — Neo4j Optimised: Q1
===============================================================
Q1: Monthly revenue by subscription tier, last 12 months.

Optimised schema: superset of naive — same relationships, same indexes.
The schema additions for Q1 specifically are none: full-text index,
composite event index, ALSO_BOUGHT edges, and native cart list do not
affect this query. The Cypher is therefore identical to the naive version.

This is an academically honest result: when the schema optimisation does
not target a specific query, there is no schema effect to measure —
only the engine effect (Neo4j vs PostgreSQL) is visible. This is
documented in CHANGES.md and discussed in the methodology chapter.

The benchmark is included for completeness of the Q1–Q8 matrix and to
confirm that the optimised schema does not degrade Q1 performance.

Usage:
    python q1_revenue.py
    python q1_revenue.py --iterations 100
    python q1_revenue.py --dry-run
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

# ── static reference data — identical to naive ────────────────────────────────

TIER_PRICING = [
    {"tier_id": "1", "valid_from": "2023-01-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": 0.00},
    {"tier_id": "2", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 14.99},
    {"tier_id": "2", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": 19.99},
    {"tier_id": "3", "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 39.99},
    {"tier_id": "3", "valid_from": "2024-06-01T00:00:00+00:00", "valid_to": None,
     "monthly_price_usd": 49.99},
]

TIER_NAMES = {"1": "Free", "2": "Pro", "3": "Business"}

# ── Cypher queries — identical to naive ───────────────────────────────────────
# Intentionally unchanged: the schema optimisations do not affect Q1.
# Any performance difference vs naive reflects measurement variance,
# not a schema effect.

Q1_SUBSCRIPTION_CYPHER = """
MATCH (i:Invoice)-[:FOR_SUBSCRIPTION]->(s:Subscription)-[:ON_TIER]->(t:SubscriptionTier)
WHERE i.status       = 'paid'
  AND i.invoice_type = 'subscription'
  AND i.created_at  >= $cutoff
  AND i.created_at  <  $now
RETURN
    i.created_at    AS created_at,
    i.total_usd     AS total_usd,
    t.id            AS tier_id
"""

Q1_MARKETPLACE_CYPHER = """
MATCH (u:User)-[:HAS_INVOICE]->(i:Invoice)
WHERE i.status       = 'paid'
  AND i.invoice_type = 'marketplace'
  AND i.created_at  >= $cutoff
  AND i.created_at  <  $now
MATCH (u)-[:HAS_SUBSCRIPTION]->(s:Subscription)
WHERE s.started_at <= i.created_at
WITH i, s
ORDER BY s.started_at DESC
WITH i, head(collect(s)) AS active_sub
WHERE active_sub IS NOT NULL
RETURN
    i.created_at        AS created_at,
    i.total_usd         AS total_usd,
    active_sub.tier_id  AS tier_id
"""

# ── helpers — identical to naive ─────────────────────────────────────────────

def get_price_for_tier_at(tier_id: str, created_at: str) -> float:
    for p in TIER_PRICING:
        if p["tier_id"] != str(tier_id):
            continue
        if p["valid_from"] <= created_at and (
            p["valid_to"] is None or p["valid_to"] > created_at
        ):
            return p["monthly_price_usd"]
    return 0.0


def format_month(created_at: str) -> str:
    return created_at[:7]


# ── query function — identical logic to naive ─────────────────────────────────

def make_query_fn(driver):
    def _run():
        now    = datetime.now(timezone.utc).isoformat()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        with driver.session() as session:
            sub_result = session.run(
                Q1_SUBSCRIPTION_CYPHER,
                cutoff=cutoff,
                now=now,
            )
            sub_rows = sub_result.data()

            mkt_result = session.run(
                Q1_MARKETPLACE_CYPHER,
                cutoff=cutoff,
                now=now,
            )
            mkt_rows = mkt_result.data()

        agg: dict[tuple, dict] = defaultdict(
            lambda: {"invoice_count": 0, "total_revenue_usd": 0.0,
                     "price_in_effect_usd": 0.0}
        )

        for row in sub_rows + mkt_rows:
            tier_id    = str(row["tier_id"])
            created_at = row["created_at"]
            month      = format_month(created_at)
            tier_name  = TIER_NAMES.get(tier_id, tier_id)
            price      = get_price_for_tier_at(tier_id, created_at)
            key        = (month, tier_name)
            agg[key]["invoice_count"]       += 1
            agg[key]["total_revenue_usd"]   += float(row["total_usd"] or 0)
            agg[key]["price_in_effect_usd"]  = price

        return [
            {
                "month":               k[0],
                "tier_name":           k[1],
                "price_in_effect_usd": v["price_in_effect_usd"],
                "invoice_count":       v["invoice_count"],
                "total_revenue_usd":   round(v["total_revenue_usd"], 2),
            }
            for k, v in sorted(agg.items())
        ]

    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver):
    print("\n  DRY RUN — Neo4j Optimised Q1 result sample:\n")
    rows = make_query_fn(driver)()
    if not rows:
        print("  ⚠  No rows returned — is the optimised DB populated?")
        return
    headers = ["month", "tier_name", "price_in_effect_usd",
                "invoice_count", "total_revenue_usd"]
    col_w = 22
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("─" * col_w for _ in headers))
    for row in rows[:10]:
        print("  " + "  ".join(str(row[h]).ljust(col_w) for h in headers))
    print(f"\n  ... {len(rows)} total rows returned")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Optimised Q1 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_optimised_Q1.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Optimised — Q1 Monthly Revenue Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))

    try:
        if args.dry_run:
            dry_run(driver)
            return

        run_benchmark(
            query_fn=make_query_fn(driver),
            db="neo4j_optimised",
            query_id="Q1",
            label=(
                "Monthly revenue by subscription tier (last 12 months). "
                "Cypher identical to naive — schema optimisations (full-text "
                "index, composite event index, ALSO_BOUGHT, native cart) do "
                "not affect Q1. Measures engine effect only vs naive. "
                "Temporal pricing join on 5 in-memory pricing rows."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()