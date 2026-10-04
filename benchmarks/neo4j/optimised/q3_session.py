"""
benchmarks/neo4j/optimised/q3_session.py — Neo4j Optimised: Q3
===============================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

Optimised schema: cart stored as a native Neo4j list property.
The list items are JSON strings (one per cart item), but the outer
structure is a native list — no json.loads() on the cart itself.
This is the schema effect for Q3: eliminating the outer deserialisation
step that the naive implementation requires.

Cypher is otherwise identical to naive — same traversal pattern,
same ORDER BY / LIMIT 1. Any performance difference vs naive is
attributable to the cart storage change.

Usage:
    python q3_session.py
    python q3_session.py --iterations 100
    python q3_session.py --concurrency 1
    python q3_session.py --dry-run
"""

import argparse
import json
import os
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

# ── Cypher — identical to naive ───────────────────────────────────────────────

Q3_CYPHER = """
MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)
RETURN
    s.id             AS session_id,
    s.user_id        AS user_id,
    s.cart           AS cart,
    s.ip_address     AS ip_address,
    s.user_agent     AS user_agent,
    s.created_at     AS created_at,
    s.last_active_at AS last_active_at,
    s.expires_at     AS expires_at
ORDER BY s.last_active_at DESC
LIMIT 1
"""

# ── ID pool ───────────────────────────────────────────────────────────────────

def fetch_user_id_pool(driver, pool_size: int) -> list[str]:
    cypher = """
    MATCH (u:User)-[:HAS_SESSION]->(:Session)
    WITH DISTINCT u, rand() AS r
    ORDER BY r
    LIMIT $pool_size
    RETURN u.id AS id
    """
    with driver.session() as session:
        result = session.run(cypher, pool_size=pool_size)
        ids = [row["id"] for row in result]

    if not ids:
        raise RuntimeError(
            "No Session nodes found. Run the optimised loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} user IDs with sessions.")
    return ids


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, user_ids: list[str]):
    """
    Cart is a native Neo4j list — no outer json.loads() required.
    Each list item is a JSON string (one per cart entry); those are
    left as-is since the benchmark measures retrieval cost, not
    per-item deserialisation. This is the schema effect being measured.
    """
    def _run():
        user_id = random.choice(user_ids)
        with driver.session() as session:
            result = session.run(Q3_CYPHER, user_id=user_id)
            row = result.single()
            if row:
                _ = row["cart"]   # native list — no json.loads() needed
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Neo4j Optimised Q3 result for user {user_id}:\n")
    with driver.session() as session:
        result = session.run(Q3_CYPHER, user_id=user_id)
        row = result.single()

    if not row:
        print("  ⚠  No session found for this user.")
        return

    print(f"  Session ID     : {row['session_id']}")
    print(f"  User ID        : {row['user_id']}")
    print(f"  IP Address     : {row['ip_address']}")
    print(f"  Last active    : {row['last_active_at']}")
    print(f"  Expires at     : {row['expires_at']}")

    cart = row["cart"]
    if not cart:
        print(f"\n  Cart           : (empty)")
    else:
        print(f"\n  Cart ({len(cart)} item(s)):  [stored as native list — optimised]")
        print(f"  {'Product ID':<38} {'Name':<30} {'Qty':>4} {'Price':>10}")
        print(f"  {'─'*38} {'─'*30} {'─'*4} {'─'*10}")
        for item in cart:
            if isinstance(item, str):
                item = json.loads(item)
            print(
                f"  {str(item.get('product_id', 'N/A')):<38} "
                f"{str(item.get('product_name', 'N/A')):<30} "
                f"{str(item.get('quantity', '?')):>4} "
                f"{str(item.get('price_usd', '?')):>10}"
            )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Optimised Q3 benchmark")
    parser.add_argument("--iterations",  type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--pool-size",   type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_optimised_Q3.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Optimised — Q3 Session & Cart Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))

    try:
        user_ids = fetch_user_id_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, user_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, user_ids),
            db="neo4j_optimised",
            query_id="Q3",
            label=(
                "Session + cart retrieval by user_id under "
                f"{args.concurrency} concurrent threads. "
                "Cart stored as native Neo4j list — no outer json.loads(). "
                "Schema effect vs naive: eliminates JSON string deserialisation. "
                f"Random user from pool of {len(user_ids)} IDs."
            ),
            iterations=args.iterations,
            concurrency=args.concurrency,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
