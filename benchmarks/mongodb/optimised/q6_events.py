"""
benchmarks/mongodb/optimised/q6_events.py — MongoDB Optimised: Q6
==================================================================
Q6: All activity events for a specific user in a 30-day window,
    ordered by time.

Optimised schema changes vs naive:
  - Compound index on (user_id, occurred_at DESC) — same as naive,
    ensured during the optimised load
  - metadata stored as native BSON subdocument → returned as Python
    dict directly, no json.loads()
  - occurred_at still stored as ISO string (same as naive; range
    comparisons work correctly on ISO strings)

Revision 2026-11: the access pattern is now identical to the naive driver
and to every other engine. Each iteration picks a (user_id, occurred_at)
anchor sampled from the events collection and reads the 30-day window
centred on it (anchor - 15 days, $gte, to anchor + 15 days, $lt), so every
result is non-empty. The earlier version drew users from the first 800
documents of the users collection and a uniformly random 30-day window over
the whole range, which often returned no events; it remains available as
--legacy-random-window.

Academic context:
  Engine effect = naive MongoDB result minus PostgreSQL baseline.
  Schema effect = optimised minus naive, on the same access path.

Usage:
    python q6_events.py
    python q6_events.py --iterations 100
    python q6_events.py --dry-run
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

WINDOW_DAYS = 30
PROJECTION = {"event_type": 1, "product_id": 1, "occurred_at": 1, "metadata": 1, "session_id": 1}


# ── anchor pool (equivalent driver, identical to the naive driver) ───────────

def fetch_user_anchor_pool(db, pool_size: int) -> list[tuple[str, str]]:
    """(user_id, occurred_at) pairs sampled from the events collection."""
    pipeline = [
        {"$sample": {"size": pool_size}},
        {"$project": {"_id": 0, "user_id": 1, "occurred_at": 1}},
    ]
    pairs = [(d["user_id"], d["occurred_at"]) for d in db["events"].aggregate(pipeline)]
    if not pairs:
        raise RuntimeError("No events found — is the optimised DB populated?")
    print(f"  Loaded pool of {len(pairs)} (user_id, anchor) pairs from events.")
    return pairs


def anchor_window(anchor_iso: str) -> tuple[str, str]:
    """30-day window centred on the anchor event, exactly as in the naive driver."""
    anchor = datetime.fromisoformat(anchor_iso.strip().replace("Z", "+00:00"))
    start = anchor - timedelta(days=15)
    end = start + timedelta(days=WINDOW_DAYS)
    return start.isoformat(), end.isoformat()


def run_q6(db, user_id: str, window_start: str, window_end: str) -> list[dict]:
    """Compound index range scan; metadata is a native BSON subdocument."""
    return list(db["events"].find(
        {"user_id": user_id, "occurred_at": {"$gte": window_start, "$lt": window_end}},
        PROJECTION,
    ).sort("occurred_at", -1))


def make_query_fn(db, pairs):
    def _run():
        user_id, anchor = random.choice(pairs)
        start, end = anchor_window(anchor)
        run_q6(db, user_id, start, end)
    return _run


# ── legacy driver (not equivalent), kept for reproducing the earlier cell ────

def load_user_ids(db, sample_size: int = 200) -> list[str]:
    ids = [d["_id"] for d in db["users"].find({}, {"_id": 1}).limit(sample_size * 4)]
    if len(ids) > sample_size:
        ids = random.sample(ids, sample_size)
    return ids


def load_data_date_range(db) -> tuple[datetime, datetime]:
    row = next(db["events"].aggregate([
        {"$group": {"_id": None, "min_date": {"$min": "$occurred_at"}, "max_date": {"$max": "$occurred_at"}}}
    ]), None)
    if not row:
        raise RuntimeError("No events found — is the optimised DB populated?")
    parse = lambda s: datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return parse(row["min_date"]), parse(row["max_date"])


def make_legacy_query_fn(db, user_ids, data_min, data_max):
    def _run():
        user_id = random.choice(user_ids)
        max_offset = max(0, int((data_max - data_min).total_seconds()) - WINDOW_DAYS * 86400)
        start = data_min + timedelta(seconds=random.randint(0, max_offset))
        end = start + timedelta(days=WINDOW_DAYS)
        list(db["events"].find(
            {"user_id": user_id, "occurred_at": {"$gte": start.isoformat(), "$lte": end.isoformat()}},
            PROJECTION,
        ).sort("occurred_at", -1))
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, pairs):
    print("\n  DRY RUN — MongoDB Optimised Q6 result sample:\n")
    user_id, anchor = pairs[0]
    start, end = anchor_window(anchor)
    events = run_q6(db, user_id, start, end)
    print(f"  User {user_id}, window {start} → {end}: {len(events)} events")
    if events:
        meta = events[0].get("metadata", {})
        print(f"  metadata type: {type(meta).__name__}  ← should be 'dict', not 'str'")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MongoDB Optimised Q6 — user events benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size",
                        help="Number of (user_id, anchor) pairs to pre-fetch (default: 1000, as in the naive driver)")
    parser.add_argument("--legacy-random-window", action="store_true", dest="legacy",
                        help="Earlier, non-equivalent driver: random 30-day window for one of 200 users")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--output", type=str, default=os.path.join("results", "mongodb_optimised_Q6.json"))
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q6 User Events Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    if args.legacy:
        user_ids = load_user_ids(db)
        data_min, data_max = load_data_date_range(db)
        query_fn = make_legacy_query_fn(db, user_ids, data_min, data_max)
        label = ("LEGACY, not equivalent: random 30-day window over the whole range for one of "
                 f"{len(user_ids)} users taken from the first 800 user documents (often empty). "
                 "Optimised: metadata stored as native BSON subdocument.")
    else:
        pairs = fetch_user_anchor_pool(db, args.pool_size)
        if args.dry_run:
            dry_run(db, pairs)
            return
        query_fn = make_query_fn(db, pairs)
        label = (f"All events for a user in a {WINDOW_DAYS}-day window, ordered by occurred_at DESC "
                 "(equivalent driver, revision 2026-11). Window anchored on a sampled real event, "
                 "guaranteed non-empty, exactly as in the naive driver. Optimised: metadata stored as "
                 "native BSON subdocument (no json.loads); compound index (user_id, occurred_at DESC). "
                 f"Pool of {args.pool_size} (user_id, anchor) pairs.")

    run_benchmark(
        query_fn=query_fn,
        db="mongodb_optimised",
        query_id="Q6",
        label=label,
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
