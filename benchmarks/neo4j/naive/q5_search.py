"""
benchmarks/neo4j/naive/q5_search.py — Neo4j Naive: Q5
======================================================
Q5: Full-text product search with relevance ranking across
    name, description, and product_type.

Naive schema: no full-text index. Search is performed via
toLower() CONTAINS predicate — a sequential scan of all
Product nodes. No relevance ranking is possible without a
full-text index; results are ordered by name as a stable
fallback.

This is deliberately slow and academically valuable — it
demonstrates the cost of full-text search without native
index support and motivates the optimised schema.

Same SEARCH_TERMS pool as the PostgreSQL baseline — the
query distribution is identical so the comparison is fair.

Usage:
    python q5_search.py
    python q5_search.py --iterations 100
    python q5_search.py --dry-run
    python q5_search.py --term "brushes"
"""

import argparse
import os
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.neo4j.neo4j_conn import get_driver

load_dotenv()

# ── search term pool — identical to PostgreSQL baseline ───────────────────────

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour",
    "photoshop brushes", "video editing", "certificate course", "logo design",
    "colour palette", "font pack", "texture pack", "motion graphics",
    "social media", "icon set", "web design", "canva template",
    "vector illustration", "beginner design", "digital course",
    "design assets", "procreate brushes",
]

# ── Cypher ────────────────────────────────────────────────────────────────────
#
# toLower() CONTAINS performs a full node scan — no index is used.
# Multi-word terms are split and each word is checked individually
# (AND logic) so "photoshop brushes" matches products containing
# both tokens. This is approximate but honest for the naive case.
#
# No relevance score is available without a full-text index;
# results ordered by name for determinism.

Q5_CYPHER_SINGLE = """
MATCH (p:Product)
WHERE p.is_active = 'True'
  AND (
      toLower(p.name)        CONTAINS toLower($term)
   OR toLower(p.description) CONTAINS toLower($term)
   OR toLower(p.product_type) CONTAINS toLower($term)
  )
RETURN
    p.id            AS product_id,
    p.name          AS product_name,
    p.product_type  AS product_type,
    p.price_usd     AS price_usd,
    p.attributes    AS product_attributes,
    0.0             AS rank
ORDER BY p.name
LIMIT 20
"""

# Multi-word: split into tokens, require all tokens to match name OR description
# Built dynamically in make_query_fn for multi-word terms.

def build_cypher(term: str) -> tuple[str, dict]:
    """
    For single-word terms: use the simple CONTAINS query.
    For multi-word terms: split and AND the tokens together.
    Returns (cypher, params).
    """
    tokens = term.lower().split()
    if len(tokens) == 1:
        return Q5_CYPHER_SINGLE, {"term": term}

    # Build WHERE clause: each token must appear in name OR description
    conditions = " AND ".join([
        f"(toLower(p.name) CONTAINS $t{i} OR toLower(p.description) CONTAINS $t{i})"
        for i in range(len(tokens))
    ])
    params = {f"t{i}": tok for i, tok in enumerate(tokens)}

    cypher = f"""
MATCH (p:Product)
WHERE p.is_active = 'True'
  AND {conditions}
RETURN
    p.id            AS product_id,
    p.name          AS product_name,
    p.product_type  AS product_type,
    p.price_usd     AS price_usd,
    p.attributes    AS product_attributes,
    0.0             AS rank
ORDER BY p.name
LIMIT 20
"""
    return cypher, params


# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, terms: list[str]):
    def _run():
        term = random.choice(terms)
        cypher, params = build_cypher(term)
        with driver.session() as session:
            result = session.run(cypher, **params)
            result.data()
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, term: str):
    print(f"\n  DRY RUN — Neo4j Naive Q5 search for: '{term}'\n")
    cypher, params = build_cypher(term)
    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = result.data()

    if not rows:
        print(f"  ⚠  No results found for '{term}'.")
        return

    print(f"  {len(rows)} result(s) returned (max 20):  [sequential CONTAINS scan — naive]\n")
    print(f"  {'#':<3} {'Product name':<35} {'Type':<16} {'Price':>8}")
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row['product_name']):<35} "
            f"{str(row['product_type']):<16} "
            f"{str(row['price_usd']):>8}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Naive Q5 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--term", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_naive_Q5.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Naive — Q5 Full-Text Search Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))
    term = args.term if args.term else random.choice(SEARCH_TERMS)

    try:
        if args.dry_run:
            dry_run(driver, term)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, SEARCH_TERMS),
            db="neo4j_naive",
            query_id="Q5",
            label=(
                "Full-text product search via toLower() CONTAINS — sequential "
                "node scan, no full-text index. No relevance ranking available. "
                f"Random term per iteration from pool of {len(SEARCH_TERMS)} queries."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()