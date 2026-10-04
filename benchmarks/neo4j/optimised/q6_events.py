"""
benchmarks/neo4j/optimised/q6_events.py — Neo4j Optimised: Q6
==============================================================
Q6: Retrieve all activity events for a specific user in a
    30-day window, ordered by time.

Optimised schema: composite index on Event(user_id, occurred_at)
created in the optimised loader. This allows Neo4j to satisfy both
the equality filter on user_id and the range filter on occurred_at
from a single index scan — equivalent to PostgreSQL's composite
idx_events_user_time index.

Cypher is identical to naive — the schema effect is entirely in the
index: the planner can now push both predicates into the index lookup
rather than filtering occurred_at in memory after a user_id scan.

Usage:
    python q6_events.py
    python q6_events.py --iterations 100
    python q6_events.py --dry-run
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

WINDOW_DAYS = 30

# ── Cypher — identical to naive ───────────────────────────────────────────────
# The composite index (user_id, occurred_at) is used automatically by
# the Neo4j planner when it sees both predicates. No query change needed.

Q6_CYPHER = """
MATCH (u:User {id: $user_id})-[:TRIGGERED]->(e:Event)
WHERE e.occurred_at >= $start
  AND e.occurred_at <  $end
RETURN
    e.id            AS event_id,
    e.event_type    AS event_type,
    e.occurred_at   AS occurred_at,
    e.product_id    AS product_id,
    e.session_id    AS session_id,
    e.metadata      AS metadata
ORDER BY e.occurred_at DESC
"""

# ── anchor pool ───────────────────────────────────────────────────────────────

def fetch_anchor_pool(driver, pool_size: int) -> list[tuple]:
    cypher = """
    MATCH (u:User)-[:TRIGGERED]->(e:Event)
    WITH u.id AS user_id, e.occurred_at AS occurred_at, rand() AS r
    ORDER BY r
    LIMIT $pool_size
    RETURN user_id, occurred_at
    """
    with driver.session() as session:
        result = session.run(cypher, pool_size=pool_size)
        pairs = [(row["user_id"], row["occurred_at"]) for row in result]

    if not pairs:
        raise RuntimeError(
            "No Event nodes found. Run the optimised loader before benchmarking."
        )
    print(f"  Loaded pool of {len(pairs)} (user_id, anchor) pairs from events.")
    return pairs


def anchor_window(anchor_str: str) -> tuple[str, str]:
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except Exception:
        anchor = datetime.fromisoformat(anchor_str.replace("Z", "+00:00"))
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    start = anchor - timedelta(days=15)
    end   = anchor + timedelta(days=15)
    return start.isoformat(), end.isoformat()


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, pairs: list[tuple]):
    def _run():
        user_id, anchor_str = random.choice(pairs)
        start, end = anchor_window(anchor_str)
        with driver.session() as session:
            result = session.run(
                Q6_CYPHER, user_id=user_id, start=start, end=end
            )
            result.data()
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, pairs: list[tuple]):
    user_id, anchor_str = pairs[0]
    start, end = anchor_window(anchor_str)
    print(f"\n  DRY RUN — Neo4j Optimised Q6 events for user {user_id}")
    print(f"  Window: {start} to {end}\n")

    with driver.session() as session:
        result = session.run(
            Q6_CYPHER, user_id=user_id, start=start, end=end
        )
        rows = result.data()

    if not rows:
        print("  ⚠  No events returned — check the loader ran correctly.")
        return

    print(f"  {len(rows)} event(s) returned:\n")
    print(
        f"  {'#':<4} {'Event type':<25} {'Occurred at':<30} {'Product ID':<38}"
    )
    print(f"  {'─'*4} {'─'*25} {'─'*30} {'─'*38}")
    for i, row in enumerate(rows[:20], 1):
        print(
            f"  {i:<4} {str(row['event_type']):<25} "
            f"{str(row['occurred_at']):<30} "
            f"{str(row['product_id'] or 'N/A'):<38}"
        )
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more rows")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Optimised Q6 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size",  type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_optimised_Q6.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Optimised — Q6 User Events Benchmark")
    print("=" * 55)
    print(f"  Window size : {WINDOW_DAYS} days (centred on anchor event)")

    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))

    try:
        pairs = fetch_anchor_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, pairs)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, pairs),
            db="neo4j_optimised",
            query_id="Q6",
            label=(
                f"All events for a user in a {WINDOW_DAYS}-day window ordered "
                "by occurred_at DESC. Optimised: composite index on "
                "(user_id, occurred_at) — both predicates pushed into single "
                "index scan. Schema effect vs naive: eliminates in-memory "
                f"occurred_at filter. Pool of {len(pairs)} (user_id, anchor) pairs."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()