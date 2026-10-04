"""
benchmarks/mongodb/optimised/q7_rolling_revenue.py — MongoDB Optimised: Q7
============================================================================
Q7: 7-day rolling average of daily revenue per subscription tier over
    6 months, with gap-filling for days with zero activity.

Requires: MongoDB 5.1+ ($densify, $setWindowFields)
          MongoDB 5.3+ / 6.x / 7.x (as confirmed for this benchmark)

Optimised schema changes vs naive:
  - total_usd stored as native float → server-side $sum works correctly
  - subscription_id available on invoice documents → $lookup to
    subscriptions for tier attribution (no Python-side join)
  - Fully server-side pipeline: $group → $densify → $lookup →
    $setWindowFields → $project

Pipeline stages:
  1. $match   : paid invoices in 6-month window
  2. $addFields: truncate created_at string to date string (YYYY-MM-DD)
  3. $group   : sum daily revenue per (day, subscription_id)
  4. $lookup  : subscriptions → get tier_id per subscription_id
  5. $lookup  : subscription_tiers → get tier name per tier_id
  6. $group   : re-group by (day, tier_name) summing revenue
  7. $densify : fill missing days per tier partition (gap-fill)
  8. $fill    : set revenue=0 on densified gap rows
  9. $setWindowFields: compute 7-day rolling average per tier partition
 10. $sort    : (day, tier_name)

This is the most complex pipeline in the benchmark suite. The schema
effect is large: naive performed all gap-filling and windowing in Python
after fetching thousands of raw invoice documents. Optimised returns
~549 pre-aggregated rows with rolling averages already computed.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive — server-side aggregation
  eliminates data transfer overhead; gap-fill + rolling window are
  native MongoDB 5.1+ operations.

  Compare with TimescaleDB Q7 optimised (time_bucket_gapfill) for the
  definitive time-series gap-fill comparison.

Usage:
    python q7_rolling_revenue.py
    python q7_rolling_revenue.py --iterations 100
    python q7_rolling_revenue.py --dry-run
"""

import argparse
import os
import sys
import random
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

WINDOW_DAYS    = 7     # rolling average window
LOOKBACK_DAYS  = 183   # ~6 months — matches naive Q7

# ── date helpers ──────────────────────────────────────────────────────────────

def load_data_date_range(db) -> tuple[datetime, datetime]:
    """
    Find the actual min/max created_at of paid invoices.
    Used to anchor random windows within the real data range so every
    iteration returns data — identical approach to naive Q7.
    """
    result = db["invoices"].aggregate([
        {"$match": {"status": "paid"}},
        {"$group": {
            "_id":      None,
            "min_date": {"$min": "$created_at"},
            "max_date": {"$max": "$created_at"},
        }}
    ])
    row = next(result, None)
    if not row:
        raise RuntimeError("No paid invoices found — is the optimised DB populated?")
    parse = lambda s: datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return parse(row["min_date"]), parse(row["max_date"])

def random_window(data_min: datetime, data_max: datetime) -> tuple[str, str]:
    """
    Pick a random LOOKBACK_DAYS window within the actual invoice date range.
    Mirrors naive Q7: random start within [data_min, data_max - LOOKBACK_DAYS].
    """
    span_sec   = int((data_max - data_min).total_seconds())
    max_offset = max(0, span_sec - LOOKBACK_DAYS * 86400)
    offset_sec = random.randint(0, max_offset)
    start      = data_min + timedelta(seconds=offset_sec)
    end        = start + timedelta(days=LOOKBACK_DAYS)
    return start.isoformat(), end.isoformat()

# ── core Q7 logic (timed portion) ─────────────────────────────────────────────

def run_q7(db, data_min: datetime, data_max: datetime) -> list[dict]:
    """
    Full server-side pipeline using $densify + $setWindowFields.

    Requires MongoDB 5.1+. Will raise OperationFailure if the server
    does not support these stages — fail-loud as per benchmark design.

    Window is a random LOOKBACK_DAYS range within the actual data, matching
    the naive Q7 methodology so results are directly comparable.

    Returns list of dicts with keys:
      day, tier_name, daily_revenue_usd, rolling_7d_avg_usd
    """
    iso_start, iso_end = random_window(data_min, data_max)

    pipeline = [
        # ── Stage 1: filter paid invoices in window ─────────────────────────
        {"$match": {
            "status":     "paid",
            "created_at": {"$gte": iso_start, "$lte": iso_end},
        }},

        # ── Stage 2: convert ISO string → BSON Date, truncate to day ─────────
        # $densify requires a numeric or Date field — strings are rejected.
        # $dateFromString parses the ISO 8601 string stored by the loader.
        # $dateTrunc strips the time component so all events on the same
        # calendar day map to the same Date value.
        {"$addFields": {
            "day_date": {
                "$dateTrunc": {
                    "date": {"$dateFromString": {"dateString": "$created_at"}},
                    "unit": "day",
                }
            },
        }},

        # ── Stage 3: daily revenue per subscription_id ───────────────────────
        {"$group": {
            "_id": {
                "day":             "$day_date",
                "subscription_id": "$subscription_id",
            },
            "daily_revenue": {"$sum": "$total_usd"},
        }},

        # ── Stage 4: join to subscriptions for tier_id ───────────────────────
        {"$lookup": {
            "from":         "subscriptions",
            "localField":   "_id.subscription_id",
            "foreignField": "_id",
            "as":           "sub_docs",
        }},
        {"$addFields": {
            "tier_id": {"$arrayElemAt": ["$sub_docs.tier_id", 0]},
        }},
        {"$project": {"sub_docs": 0}},

        # ── Stage 5: join to subscription_tiers for name ─────────────────────
        {"$lookup": {
            "from":         "subscription_tiers",
            "localField":   "tier_id",
            "foreignField": "_id",
            "as":           "tier_docs",
        }},
        {"$addFields": {
            "tier_name": {"$arrayElemAt": ["$tier_docs.name", 0]},
        }},
        {"$project": {"tier_docs": 0, "tier_id": 0}},

        # ── Stage 6: re-group by (day, tier_name) ────────────────────────────
        {"$group": {
            "_id": {
                "day":       "$_id.day",
                "tier_name": "$tier_name",
            },
            "daily_revenue": {"$sum": "$daily_revenue"},
        }},

        # ── Stage 7: $densify — fill missing days per tier partition ─────────
        # field must be a Date (not a string) — fixed in Stage 2 above.
        {
            "$densify": {
                "field": "_id.day",
                "partitionByFields": ["_id.tier_name"],
                "range": {
                    "step":   1,
                    "unit":   "day",
                    "bounds": "partition",
                },
            }
        },

        # ── Stage 8: $fill — set revenue=0 on gap-filled rows ────────────────
        {"$fill": {
            "sortBy":      {"_id.day": 1},
            "partitionBy": "$_id.tier_name",
            "output": {
                "daily_revenue": {"value": 0},
            },
        }},

        # ── Stage 9: $setWindowFields — 7-day rolling average ────────────────
        {"$setWindowFields": {
            "partitionBy": "$_id.tier_name",
            "sortBy":      {"_id.day": 1},
            "output": {
                "rolling_7d_avg": {
                    "$avg": "$daily_revenue",
                    "window": {"documents": [-6, 0]},
                },
            },
        }},

        # ── Stage 10: project clean output fields ────────────────────────────
        {"$project": {
            "_id":                0,
            "day":                {"$dateToString": {"format": "%Y-%m-%d", "date": "$_id.day"}},
            "tier_name":          "$_id.tier_name",
            "daily_revenue_usd":  {"$round": ["$daily_revenue", 2]},
            "rolling_7d_avg_usd": {"$round": ["$rolling_7d_avg", 2]},
        }},

        {"$sort": {"day": 1, "tier_name": 1}},
    ]

    return list(db["invoices"].aggregate(pipeline))

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, data_min, data_max):
    def _run():
        run_q7(db, data_min, data_max)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, data_min, data_max):
    print("\n  DRY RUN — MongoDB Optimised Q7 result sample:\n")
    try:
        rows = run_q7(db, data_min, data_max)
    except Exception as exc:
        print(f"  ✗ Pipeline failed: {exc}")
        print("  Check MongoDB version — $densify/$setWindowFields require 5.1+")
        return

    if not rows:
        print("  ⚠  No rows returned — is the optimised DB populated?")
        return

    headers = ["day", "tier_name", "daily_revenue_usd", "rolling_7d_avg_usd"]
    col_w   = 22
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("─" * col_w for _ in headers))
    for row in rows[:15]:
        print("  " + "  ".join(str(row.get(h, "")).ljust(col_w) for h in headers))
    print(f"\n  ... {len(rows)} total rows returned (server-side gap-filled)")
    print(f"  All aggregation, gap-fill, and rolling window computed server-side.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q7 — rolling revenue benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q7 Rolling Revenue Benchmark")
    print("=" * 55)
    print("  Requires: MongoDB 5.1+ ($densify + $setWindowFields)")

    db = get_db(schema="optimised")

    print("  Loading invoice date range...")
    data_min, data_max = load_data_date_range(db)
    print(f"  Invoice range: {data_min.date()} → {data_max.date()}")
    print(f"  Window size  : {LOOKBACK_DAYS} days (random start each iteration)\n")

    if args.dry_run:
        dry_run(db, data_min, data_max)
        return

    run_benchmark(
        query_fn=make_query_fn(db, data_min, data_max),
        db="mongodb_optimised",
        query_id="Q7",
        label=(
            f"7-day rolling average of daily revenue per subscription tier "
            f"over a {LOOKBACK_DAYS}-day (~6-month) window, with gap-filling. "
            "Optimised: fully server-side pipeline using $densify (gap-fill) "
            "+ $setWindowFields (rolling window). Requires MongoDB 5.1+. "
            "Random window per iteration within actual data range — "
            "directly comparable to naive Q7 methodology."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()