"""
benchmarks/mongodb/optimised/q6_events.py — MongoDB Optimised: Q6
==================================================================
Q6: All activity events for a specific user in a 30-day window,
    ordered by time.

Optimised schema changes vs naive:
  - Compound index on (user_id, occurred_at DESC) — same as naive,
    but explicitly ensured during optimised load (may differ in
    index hint eligibility)
  - metadata stored as native BSON subdocument → returned as Python
    dict directly, no json.loads()
  - occurred_at still stored as ISO string (same as naive — datetime
    conversion was not applied during load; not needed for range
    comparisons which work correctly on ISO strings)

Schema effect for Q6 is modest: the query path (compound index range
scan) is identical to naive. The only gain is elimination of per-event
json.loads() for metadata. The larger Q6 story is Cassandra vs MongoDB —
at 5M+ events, Cassandra's wide-row partition model beats both.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive — primarily metadata
  deserialisation elimination.

Usage:
    python q6_events.py
    python q6_events.py --iterations 100
    python q6_events.py --dry-run
"""

import argparse
import os
import random
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

WINDOW_DAYS = 30

# ── sample user IDs ───────────────────────────────────────────────────────────

def load_user_ids(db, sample_size: int = 200) -> list[str]:
    ids = [d["_id"] for d in db["users"].find({}, {"_id": 1}).limit(sample_size * 4)]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    return ids

# ── data date range (loaded once, used to anchor windows) ────────────────────

def load_data_date_range(db) -> tuple[datetime, datetime]:
    """
    Find the actual min/max occurred_at in the events collection.
    The 30-day window is anchored within this range, not relative to now(),
    so the query always lands on real data regardless of when the benchmark runs.
    """
    result = db["events"].aggregate([
        {"$group": {
            "_id":      None,
            "min_date": {"$min": "$occurred_at"},
            "max_date": {"$max": "$occurred_at"},
        }}
    ])
    row = next(result, None)
    if not row:
        raise RuntimeError("No events found — is the optimised DB populated?")
    parse = lambda s: datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return parse(row["min_date"]), parse(row["max_date"])

# ── core Q6 logic (timed portion) ─────────────────────────────────────────────

def run_q6(db, user_ids: list[str],
           data_min: datetime, data_max: datetime) -> list[dict]:
    """
    Compound index scan on (user_id, occurred_at).
    Optimised: metadata is a native BSON subdocument — no json.loads().

    Window is anchored inside the actual data date range so the query
    always hits real events (synthetic data is from 2024, not 2026).
    A random start is chosen so each iteration samples a different window,
    avoiding cache bias — identical to the PostgreSQL and naive MongoDB approach.
    """
    user_id    = random.choice(user_ids)
    data_span  = int((data_max - data_min).total_seconds())
    # Pick a random start that still leaves room for a full WINDOW_DAYS window
    max_offset = max(0, data_span - WINDOW_DAYS * 86400)
    offset_sec = random.randint(0, max_offset)
    start      = data_min + timedelta(seconds=offset_sec)
    end        = start + timedelta(days=WINDOW_DAYS)

    events = list(
        db["events"].find(
            {
                "user_id":    user_id,
                "occurred_at": {"$gte": start.isoformat(), "$lte": end.isoformat()},
            },
            {
                "event_type": 1,
                "product_id": 1,
                "occurred_at": 1,
                "metadata":   1,   # native BSON dict — no deserialisation
                "session_id": 1,
            }
        ).sort("occurred_at", -1)
    )
    return events

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, user_ids, data_min, data_max):
    def _run():
        run_q6(db, user_ids, data_min, data_max)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, user_ids, data_min, data_max):
    print("\n  DRY RUN — MongoDB Optimised Q6 result sample:\n")
    events = run_q6(db, user_ids, data_min, data_max)
    print(f"  Events returned : {len(events)} (30-day window)")
    if events:
        e = events[0]
        meta = e.get("metadata", {})
        print(f"\n  First event:")
        print(f"    event_type   : {e.get('event_type')}")
        print(f"    occurred_at  : {e.get('occurred_at')}")
        print(f"    metadata type: {type(meta).__name__}  ← should be 'dict', not 'str'")
        print(f"    metadata keys: {list(meta.keys()) if isinstance(meta, dict) else '(not a dict)'}")
    print(f"\n  Queries issued: 1 (compound index range scan on events)")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q6 — user events benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q6.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q6 User Events Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    print("  Sampling user IDs...")
    user_ids = load_user_ids(db)
    print(f"  Loaded {len(user_ids):,} user IDs for random sampling.")

    print("  Loading data date range from events collection...")
    data_min, data_max = load_data_date_range(db)
    print(f"  Events range: {data_min.date()} → {data_max.date()}\n")

    if args.dry_run:
        dry_run(db, user_ids, data_min, data_max)
        return

    run_benchmark(
        query_fn=make_query_fn(db, user_ids, data_min, data_max),
        db="mongodb_optimised",
        query_id="Q6",
        label=(
            f"All user events in a {WINDOW_DAYS}-day window, ordered by time. "
            "Optimised: metadata stored as native BSON subdocument (no json.loads). "
            "Compound index on (user_id, occurred_at DESC). "
            "Schema effect modest — query path identical to naive. "
            "Cassandra Q6 is the definitive comparison at 5M+ events."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()