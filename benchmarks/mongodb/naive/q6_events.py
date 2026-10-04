"""
benchmarks/mongodb/naive/q6_events.py — MongoDB Naive: Q6
==========================================================
Q6: Retrieve all activity events for a specific user in a
    30-day window, ordered by time.

Naive implementation notes:
    The events collection is a flat mirror of the PostgreSQL schema.
    The composite index on (user_id, occurred_at DESC) is present —
    it was created by the naive loader to mirror the PostgreSQL index
    idx_events_user_time.

    Each iteration executes a single find() with:
      - user_id equality filter
      - occurred_at range filter (BETWEEN window_start AND window_end)
      - sort by occurred_at descending

    This is structurally identical to the PostgreSQL Q6 query. The
    difference is that all values are stored as ISO 8601 strings —
    MongoDB will use a string-range comparison on the index rather
    than a native datetime comparison. This is a naive constraint
    (the optimised loader will store datetimes as BSON Date objects).

    Date window logic is identical to the PostgreSQL baseline:
      - Dataset is fixed to 2025
      - Window start drawn randomly from 2025-01-01 to 2025-11-30
      - Window end = start + 30 days
    This ensures every iteration falls within the dataset range and
    returns at least some rows for active users.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive result.
    This file measures the naive (engine-only) side.

Usage:
    cd benchmarks/mongodb/naive
    python q6_events.py                   # 1000 iterations
    python q6_events.py --iterations 100  # quick smoke test
    python q6_events.py --dry-run         # run once, print result sample
    python q6_events.py --pool-size 500   # use 500 user IDs (default: 1000)
"""

import argparse
import os
import random
import sys
from datetime import date, timedelta, timezone, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── dataset date range (mirrors PostgreSQL baseline exactly) ──────────────────

WINDOW_DAYS   = 30
DATASET_START = date(2025, 1, 1)
DATASET_END   = date(2025, 12, 31) - timedelta(days=WINDOW_DAYS)

def load_data_date_range(db) -> tuple[datetime, datetime]:
    result = db["events"].aggregate([
        {"$group": {
            "_id":      None,
            "min_date": {"$min": "$occurred_at"},
            "max_date": {"$max": "$occurred_at"},
        }}
    ])
    row = next(result, None)
    if not row:
        raise RuntimeError("No events found — run the naive loader first.")
    parse = lambda s: datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return parse(row["min_date"]), parse(row["max_date"])

# ── date helpers ──────────────────────────────────────────────────────────────

def random_window(data_min: datetime, data_max: datetime) -> tuple[str, str]:
    data_span  = int((data_max - data_min).total_seconds())
    max_offset = max(0, data_span - WINDOW_DAYS * 86400)
    offset_sec = random.randint(0, max_offset)
    start      = data_min + timedelta(seconds=offset_sec)
    end        = start + timedelta(days=WINDOW_DAYS)
    return start.isoformat(), end.isoformat()
# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_user_anchor_pool(db, pool_size: int) -> list[tuple[str, str]]:
    """
    Sample (user_id, occurred_at) pairs directly from the events collection.
    The window start is set to occurred_at - 15 days, so the anchor event
    always falls in the middle of the window — guaranteed non-empty result.
    """
    pipeline = [
        {"$sample": {"size": pool_size}},
        {"$project": {"_id": 0, "user_id": 1, "occurred_at": 1}},
    ]
    pairs = [(d["user_id"], d["occurred_at"])
             for d in db["events"].aggregate(pipeline)]
    if not pairs:
        raise RuntimeError(
            "No events found in MongoDB. "
            "Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(pairs)} (user_id, anchor) pairs from events.")
    return pairs

def anchor_window(anchor_iso: str) -> tuple[str, str]:
    """
    Build a 30-day window centred on the anchor event.
    anchor is placed ~15 days in, guaranteeing it falls inside the window.
    """
    anchor = datetime.fromisoformat(anchor_iso.strip().replace("Z", "+00:00"))
    start  = anchor - timedelta(days=15)
    end    = start + timedelta(days=WINDOW_DAYS)
    return start.isoformat(), end.isoformat()

# ── core Q6 logic ─────────────────────────────────────────────────────────────

def run_q6(db, user_id: str, window_start: str, window_end: str) -> list[dict]:
    return list(db["events"].find(
        {
            "user_id":    user_id,
            "occurred_at": {"$gte": window_start, "$lt": window_end},
        },
        {
            "_id": 1, "event_type": 1, "occurred_at": 1,
            "product_id": 1, "session_id": 1, "metadata": 1,
        }
    ).sort("occurred_at", -1))

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, pairs: list[tuple[str, str]]):
    def _run():
        user_id, anchor = random.choice(pairs)
        start, end = anchor_window(anchor)
        run_q6(db, user_id, start, end)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, pairs: list[tuple[str, str]]):
    user_id, anchor = pairs[0]
    start, end = anchor_window(anchor)
    print(f"\n  DRY RUN — MongoDB Naive Q6 events for user {user_id}")
    print(f"  Window: {start} to {end}\n")
    rows = run_q6(db, user_id, start, end)
    if not rows:
        print("  ⚠  Still no events — check the loader ran correctly.")
        return
    print(f"  {len(rows)} event(s) returned:\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<32} {'Product ID':<38}")
    print(f"  {'─'*4} {'─'*25} {'─'*32} {'─'*38}")
    for i, row in enumerate(rows[:20], 1):
        print(
            f"  {i:<4} {str(row.get('event_type', '')):<25} "
            f"{str(row.get('occurred_at', '')):<32} "
            f"{str(row.get('product_id', 'N/A')):<38}"
        )
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more rows")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q6 — user events benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000, dest="pool_size",
        help="Number of user IDs to pre-fetch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Fetch events for one user in one window then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q6.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q6.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q6 User Events Benchmark")
    print("=" * 55)
    print(f"  Window size  : {WINDOW_DAYS} days")
    print(f"  Dataset range: {DATASET_START} to "
          f"{DATASET_END + timedelta(days=WINDOW_DAYS)}")

    db = get_db()
    pairs = fetch_user_anchor_pool(db, args.pool_size)

    if args.dry_run:
        dry_run(db, pairs)
        return

    run_benchmark(
        query_fn=make_query_fn(db, pairs),
        db="mongodb_naive",
        query_id="Q6",
        label=(
            f"All events for a user in a {WINDOW_DAYS}-day window, ordered "
            "by occurred_at DESC. Naive: ISO 8601 string range comparison "
            "on composite index (user_id, occurred_at DESC). "
            "Window anchored on a sampled real event — guaranteed non-empty. "
            f"Pool of {args.pool_size} (user_id, anchor) pairs."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()