"""
benchmarks/neo4j/optimised/q5_search.py — Neo4j Optimised: Q5
==============================================================
Q5: Full-text product search with relevance ranking across
    name, description, and product_type.

Optimised schema: native Neo4j full-text index on Product nodes
with the `english` analyser (stemming + stop words), created in
the optimised loader:

    CREATE FULLTEXT INDEX product_search
    FOR (p:Product) ON EACH [p.name, p.description]
    OPTIONS {indexConfig: {"fulltext.analyzer": "english"}}

Query uses db.index.fulltext.queryNodes() — returns results with
a relevance score based on Lucene's BM25 scoring, equivalent to
Elasticsearch's default scorer and significantly more sophisticated
than PostgreSQL's ts_rank_cd.

Schema effect vs naive:
  - Sequential CONTAINS scan → index-backed BM25 query
  - No ranking → relevance-ranked results
  - Multi-word terms handled natively, no Python token splitting

Same SEARCH_TERMS pool as PostgreSQL baseline and naive.

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

# ── search term pool — identical to PostgreSQL baseline and naive ─────────────

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
# db.index.fulltext.queryNodes() uses the product_search full-text index.
# The index was created with the "english" analyser — terms are stemmed
# before indexing and at query time, so "brushes" matches "brush" etc.
#
# score is the Lucene BM25 relevance score — equivalent to ts_rank_cd
# in PostgreSQL but with IDF weighting, which ts_rank_cd lacks.
#
# CALL syntax is required for full-text index queries in Neo4j.
# Results filtered to is_active = true after retrieval.

Q5_CYPHER = """
CALL db.index.fulltext.queryNodes('product_fulltext', $term)
YIELD node AS p, score
WHERE p.is_active = 'True'
RETURN
    p.id            AS product_id,
    p.name          AS product_name,
    p.product_type  AS product_type,
    p.price_usd     AS price_usd,
    p.attributes    AS product_attributes,
    score           AS rank
ORDER BY score DESC
LIMIT 20
"""

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(driver, terms: list[str]):
    def _run():
        term = random.choice(terms)
        with driver.session() as session:
            result = session.run(Q5_CYPHER, term=term)
            result.data()
    return _run


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(driver, term: str):
    print(f"\n  DRY RUN — Neo4j Optimised Q5 search for: '{term}'\n")
    with driver.session() as session:
        result = session.run(Q5_CYPHER, term=term)
        rows = result.data()

    if not rows:
        print(f"  ⚠  No results found for '{term}'.")
        print("  Check the full-text index 'product_search' exists on the optimised DB.")
        return

    print(f"  {len(rows)} result(s) returned (max 20):  [BM25 full-text index — optimised]\n")
    print(f"  {'#':<3} {'Product name':<35} {'Type':<16} {'Price':>8} {'Score':>8}")
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row['product_name']):<35} "
            f"{str(row['product_type']):<16} "
            f"{str(row['price_usd']):>8} "
            f"{float(row['rank']):>8.4f}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neo4j Optimised Q5 benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--term", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "neo4j_optimised_Q5.json",
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Neo4j Optimised — Q5 Full-Text Search Benchmark")
    print("=" * 55)

    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))
    term = args.term if args.term else random.choice(SEARCH_TERMS)

    try:
        if args.dry_run:
            dry_run(driver, term)
            return

        run_benchmark(
            query_fn=make_query_fn(driver, SEARCH_TERMS),
            db="neo4j_optimised",
            query_id="Q5",
            label=(
                "Full-text product search via native full-text index "
                "('product_search') with English analyser and BM25 scoring. "
                "Schema effect vs naive: sequential CONTAINS scan eliminated, "
                "relevance ranking enabled. "
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