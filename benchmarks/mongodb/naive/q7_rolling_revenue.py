"""
benchmarks/mongodb/naive/q7_rolling_revenue.py — MongoDB Naive: Q7
===================================================================
Q7: 7-day rolling average of daily revenue per subscription tier
    over a 6-month window, with gap-filling for days with zero activity.

Naive schema: all fields are stored as they came from the PostgreSQL
export — total_usd is a string, not a native float.

Pipeline is identical to the optimised version except for one cast:
    Naive:     {"$sum": {"$toDouble": "$total_usd"}}
    Optimised: {"$sum": "$total_usd"}               (native float)

Using the same server-side pipeline on both naive and optimised is
methodologically correct: PostgreSQL Q7 does all computation inside the
database engine (generate_series + window functions), so MongoDB must do
the same for the comparison to be valid. Python-side gap-fill or rolling
averages would measure Python performance, not MongoDB's engine.

Schema effect = optimised minus naive = cost of $toDouble cast on every
invoice document across 1000 iterations. Expected to be small but
consistent and directly attributable to the type mismatch.

Pipeline stages:
  1. $match           : paid invoices in random 6-month window
  2. $addFields       : parse ISO string -> BSON Date, truncate to day
  3. $group           : sum daily revenue per (day, subscription_id)
                        using $toDouble because total_usd is a string
  4. $lookup          : subscriptions -> tier_id
  5. $lookup          : subscription_tiers -> tier name
  6. $group           : re-group by (day, tier_name)
  7. $densify         : fill missing days per tier partition
  8. $fill            : set revenue=0 on gap rows
  9. $setWindowFields : 7-day rolling average per tier
 10. $project         : clean output

Requires MongoDB 5.1+ ($densify, $setWindowFields).

Usage:
    python q7_rolling_revenue.py
    python q7_rolling_revenue.py --iterations 100
    python q7_rolling_revenue.py --dry-run
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

WINDOW_DAYS = 183   # ~6 months

# -- date helpers -------------------------------------------------------------

def load_data_date_range(db) -> tuple:
    """
    Query the actual min/max of paid invoices from the DB.
    No hardcoded dates -- works correctly at all data scales.
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
        raise RuntimeError("No paid invoices found -- run the naive loader first.")
    parse = lambda s: datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return parse(row["min_date"]), parse(row["max_date"])


def random_window(data_min, data_max):
    """
    Pick a random WINDOW_DAYS window anchored within the actual data range.
    Guarantees every iteration covers real invoices regardless of data scale.
    """
    span_sec   = int((data_max - data_min).total_seconds())
    max_offset = max(0, span_sec - WINDOW_DAYS * 86400)
    offset_sec = random.randint(0, max_offset)
    start      = data_min + timedelta(seconds=offset_sec)
    end        = start + timedelta(days=WINDOW_DAYS)
    return start.isoformat(), end.isoformat()

# -- core Q7 logic (timed portion) --------------------------------------------

def run_q7(db, data_min, data_max):
    """
    Fully server-side pipeline using $densify + $setWindowFields.
    $toDouble cast required -- total_usd is stored as a string in naive schema.
    """
    iso_start, iso_end = random_window(data_min, data_max)

    pipeline = [
        # Stage 1: paid invoices in window
        {"$match": {
            "status":     "paid",
            "created_at": {"$gte": iso_start, "$lte": iso_end},
        }},

        # Stage 2: parse ISO string -> BSON Date truncated to day.
        # Required because $densify rejects string fields.
        {"$addFields": {
            "day_date": {
                "$dateTrunc": {
                    "date": {"$dateFromString": {"dateString": "$created_at"}},
                    "unit": "day",
                }
            },
        }},

        # Stage 3: daily revenue per (day, subscription_id).
        # $toDouble required -- total_usd is stored as a string in naive schema.
        {"$group": {
            "_id": {
                "day":             "$day_date",
                "subscription_id": "$subscription_id",
            },
            "daily_revenue": {"$sum": {"$toDouble": "$total_usd"}},
        }},

        # Stage 4: join subscriptions for tier_id
        {"$lookup": {
            "from":         "subscriptions",
            "localField":   "_id.subscription_id",
            "foreignField": "_id",
            "as":           "sub_docs",
        }},
        {"$addFields": {"tier_id": {"$arrayElemAt": ["$sub_docs.tier_id", 0]}}},
        {"$project": {"sub_docs": 0}},

        # Stage 5: join subscription_tiers for name
        {"$lookup": {
            "from":         "subscription_tiers",
            "localField":   "tier_id",
            "foreignField": "_id",
            "as":           "tier_docs",
        }},
        {"$addFields": {"tier_name": {"$arrayElemAt": ["$tier_docs.name", 0]}}},
        {"$project": {"tier_docs": 0, "tier_id": 0}},

        # Stage 6: re-group by (day, tier_name)
        {"$group": {
            "_id": {
                "day":       "$_id.day",
                "tier_name": "$tier_name",
            },
            "daily_revenue": {"$sum": "$daily_revenue"},
        }},

        # Stage 7: gap-fill missing days per tier partition
        {"$densify": {
            "field": "_id.day",
            "partitionByFields": ["_id.tier_name"],
            "range": {"step": 1, "unit": "day", "bounds": "partition"},
        }},

        # Stage 8: zero-fill gap rows
        {"$fill": {
            "sortBy":      {"_id.day": 1},
            "partitionBy": "$_id.tier_name",
            "output": {"daily_revenue": {"value": 0}},
        }},

        # Stage 9: 7-day rolling average per tier
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

        # Stage 10: clean output
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

# -- benchmark factory --------------------------------------------------------

def make_query_fn(db, data_min, data_max):
    def _run():
        run_q7(db, data_min, data_max)
    return _run

# -- dry run ------------------------------------------------------------------

def dry_run(db, data_min, data_max):
    print("\n  DRY RUN -- MongoDB Naive Q7 rolling revenue sample:\n")
    try:
        rows = run_q7(db, data_min, data_max)
    except Exception as exc:
        print(f"  Pipeline failed: {exc}")
        print("  Check MongoDB version -- $densify/$setWindowFields require 5.1+")
        return
    if not rows:
        print("  No rows returned -- is the database populated?")
        return
    total = len(rows)
    print(f"  {total} rows returned ({total // 3} days x 3 tiers)\n")
    headers = ["day", "tier_name", "daily_revenue_usd", "rolling_7d_avg_usd"]
    col_w   = [12, 12, 20, 22]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "  ".join("-" * w for w in col_w))
    sample = rows[:9] + ([None] + rows[-9:] if total > 18 else [])
    for row in sample:
        if row is None:
            print("  ...")
            continue

        def get_field(row, field):
            if field in row:
                return row[field]
            if field == "tier_name" and "_id" in row:
                return row["_id"].get("tier_name")
            if field == "day" and "_id" in row:
                return row["_id"].get("day")
            return None

        print("  " + "  ".join(str(get_field(row, h)).ljust(w) for h, w in zip(headers, col_w)))

# -- entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q7 -- rolling revenue benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive -- Q7 Rolling Revenue Benchmark")
    print("=" * 55)
    print(f"  Window size : {WINDOW_DAYS} days (~6 months)")
    print("  Requires    : MongoDB 5.1+ ($densify + $setWindowFields)")
    print("  Note        : $toDouble cast applied -- total_usd stored as string")

    db = get_db()

    print("  Loading invoice date range...")
    data_min, data_max = load_data_date_range(db)
    print(f"  Invoice range : {data_min.date()} -> {data_max.date()}")
    print(f"  Window start  : random within data range\n")

    if args.dry_run:
        dry_run(db, data_min, data_max)
        return

    run_benchmark(
        query_fn=make_query_fn(db, data_min, data_max),
        db="mongodb_naive",
        query_id="Q7",
        label=(
            f"7-day rolling average of daily revenue per subscription tier "
            f"over a {WINDOW_DAYS}-day (~6-month) window, with gap-filling. "
            "Naive: fully server-side pipeline ($densify + $setWindowFields). "
            "$toDouble cast on total_usd required -- stored as string in naive schema. "
            "Random window within actual data range per iteration."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()