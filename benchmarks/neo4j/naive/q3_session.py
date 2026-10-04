"""
benchmarks/neo4j/naive/q3_session.py — Neo4j Naive: Q3
=======================================================
Q3: Retrieve a user's active session and full cart contents
    under 50 concurrent requests.

Naive schema: Session node with cart stored as a JSON string property.
The benchmark must call json.loads() on the cart field to materialise
cart items — equivalent cost to PostgreSQL's JSONB deserialisation.

Query pattern:
  MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)
  RETURN s ORDER BY s.last_active_at DESC LIMIT 1

The Neo4j Python driver maintains an internal connection pool.
Each concurrent iteration calls driver.session() which draws from
that pool — this mirrors a real application connection-pool model
without requiring per-thread connection management.

Schema effect vs optimised:
  cart is a JSON string → requires json.loads() to access items.
  Optimised stores cart as a native Neo4j list property — no
  deserialisation needed.

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

# ── Cypher ────────────────────────────────────────────────────────────────────

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
    """
    Pre-fetch user IDs that have at least one Session node.
    rand() ordering gives a random sample without APOC dependency.
    """
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
            "No Session nodes found. Run the naive loader before benchmarking."
        )
    print(f"  Loaded pool of {len(ids)} user IDs with sessions.")
    return ids


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, user_ids: list[str]):
    """
    Returns a zero-argument callable for the harness.
    driver.session() draws from the internal connection pool —
    safe to call concurrently from multiple threads.
    The cart JSON string is deserialised on every call — this is
    the naive schema cost being measured.
    """
    def _run():
        user_id = random.choice(user_ids)
        with driver.session() as session:
            result = session.run(Q3_CYPHER, user_id=user_id)
            row = result.single()
            if row and row["cart"]:
                # Naive cost: JSON string must be deserialised
                json.loads(row["cart"])
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, user_ids: list[str]):
    user_id = user_ids[0]
    print(f"\n  DRY RUN — Neo4j Naive Q3 result for user {user_id}:\n")
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

    cart_raw = row["cart"]
    if not cart_raw:
        print(f"\n  Cart           : (empty)")
        return

    try:
        cart = json.loads(cart_raw)
    except (json.JSONDecodeError, TypeError):
        cart = []

    if not cart:
        print(f"\n  Cart           : (empty)")
    else:
        print(f"\n  Cart ({len(cart)} item(s)):  [stored as JSON string — naive]")
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
    parser = argparse.ArgumentParser(description="Neo4j Naive Q3 benchmark")
    parser.add_argument("--iterations",  type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--pool-size",   type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_naive_Q3.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Naive — Q3 Session & Cart Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))

    try:
        user_ids = fetch_user_id_pool(driver, args.pool_size)

        if args.dry_run:
            dry_run(driver, user_ids)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, user_ids),
            db="neo4j_naive",
            query_id="Q3",
            label=(
                "Session + cart retrieval by user_id under "
                f"{args.concurrency} concurrent threads. "
                "Cart stored as JSON string property — requires json.loads(). "
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