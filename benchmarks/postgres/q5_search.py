"""
benchmarks/postgres/q5_search.py — PostgreSQL Baseline: Q5
===========================================================
Q5: Full-text product search with relevance ranking across
    name, description, and product_type.

    This is the PostgreSQL baseline for the Elasticsearch comparison (Q5).
    Elasticsearch will serve the same queries using BM25 scoring with
    custom analysers and field-level boosting.

Killer feature demonstrated (Elasticsearch side):
    BM25 ranking + custom analysers — Elasticsearch applies term
    frequency / inverse document frequency scoring natively, with
    English stemming, synonym expansion, and per-field boost weights
    (name^3, description^1.5). PostgreSQL's tsvector uses a simpler
    ranking model (ts_rank based on term frequency only) with no
    synonym support and no field-level IDF weighting.

PostgreSQL baseline design notes
──────────────────────────────────
The search_vector column on products is a pre-computed tsvector:
    setweight(to_tsvector('english', name),        'A')  -- highest weight
    setweight(to_tsvector('english', description), 'B')
    setweight(to_tsvector('english', product_type),'C')  -- lowest weight

The column is maintained by a trigger (trg_products_search_vector)
defined in schema.sql — no manual updates needed.

A GIN index (idx_products_search) on search_vector means the WHERE
clause is an index scan, not a sequential scan. This gives PostgreSQL
its best possible performance for tsvector search.

ts_rank_cd is used instead of ts_rank — ts_rank_cd accounts for
document length (cover density), producing more stable rankings
across products with very different description lengths.

A random search term is chosen each iteration from a realistic pool
of domain-relevant queries. This prevents the PostgreSQL plan cache
and OS page cache from serving identical results on every iteration.

Usage:
    python q5_search.py                   # 1000 iterations
    python q5_search.py --iterations 100  # quick smoke test
    python q5_search.py --explain         # EXPLAIN ANALYZE for one term
    python q5_search.py --dry-run         # run once, print result sample
    python q5_search.py --term "brushes"  # search a specific term
"""

import argparse
import os
import random
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

# ── connection ────────────────────────────────────────────────────────────────

from pg_conn import get_connection

# ── search term pool ──────────────────────────────────────────────────────────
#
# These terms reflect realistic user search behaviour on a digital
# marketplace selling courses, digital assets, and merch.
# A mix of:
#   - single-word queries   ("brushes", "typography")
#   - multi-word phrases    ("photoshop brushes", "beginner python")
#   - product-type terms    ("course", "digital asset", "merch")
#   - attribute terms       ("4K", "procreate", "certificate")
#   - broad discovery terms ("design", "illustration", "video")
#
# Elasticsearch will use the same pool in Phase 3/4, keeping the
# query distribution identical so the comparison is fair.

SEARCH_TERMS = [
    # Single-word queries
    "brushes",
    "typography",
    "illustration",
    "photography",
    "animation",
    "branding",
    "mockup",
    "watercolour",
    # Multi-word phrases
    "photoshop brushes",
    "video editing",
    "certificate course",
    "logo design",
    "colour palette",
    "font pack",
    "texture pack",
    "motion graphics",
    "social media",
    "icon set",
    "web design",
    "canva template",
    "vector illustration",
    "beginner design",
    "digital course",
    "design assets",
    "procreate brushes",
    # Deliberately excluded — tsvector strips these tokens entirely,
    # producing zero-row results that would bias latency downward:
    #   "4K assets"       — "4K" is not a valid tsvector lexeme
    #   "ui kit"          — "ui" stripped as a 2-char non-word token
    #   "lightroom preset"— "lightroom" not in English dictionary
    #   "after effects"   — "effects" stems fine but "after" is a stop word
    #   "t-shirt design"  — hyphenated token handling is inconsistent
    # These ARE valid Elasticsearch queries — the exclusion gap is itself
    # a dissertation finding worth noting in the analysis.
]

# ── query ─────────────────────────────────────────────────────────────────────
#
# Design notes
# ────────────
# plainto_tsquery is used instead of to_tsquery because it handles
# natural-language multi-word input gracefully (no syntax errors on
# phrases like "photoshop brushes"). Elasticsearch's match query is
# the equivalent — both tokenise and normalise the input automatically.
#
# ts_rank_cd(search_vector, query) scores each result by how well it
# matches the query, weighted by the A/B/C field weights assigned at
# index time. Results are ordered by rank descending so the most
# relevant products appear first.
#
# We limit to 20 results — a realistic first-page result set. Returning
# a fixed page size keeps result materialisation cost stable across
# iterations regardless of corpus size.
#
# The WHERE clause hits idx_products_search (GIN index on search_vector)
# — this is the fast path PostgreSQL offers for tsvector queries.

Q5_SQL = """
SELECT
    p.id,
    p.name,
    p.product_type,
    p.price_usd,
    p.attributes,
    ts_rank_cd(p.search_vector, query) AS rank
FROM
    products p,
    plainto_tsquery('english', %s) AS query
WHERE
    p.search_vector @@ query
  AND p.is_active = TRUE
ORDER BY rank DESC
LIMIT 20;
"""

Q5_EXPLAIN_SQL = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)\n" + Q5_SQL

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(conn, terms: list[str]):
    """
    Return a zero-argument callable that executes Q5 with a randomly
    chosen search term on every call. Random term selection mirrors
    realistic search traffic distribution and prevents plan/page cache
    from artificially deflating latency.
    """
    def _run():
        term = random.choice(terms)
        with conn.cursor() as cur:
            cur.execute(Q5_SQL, (term,))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, term: str):
    """Execute one search and print results for sanity checking."""
    print(f"\n  DRY RUN — Q5 full-text search for: '{term}'\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q5_SQL, (term,))
        rows = cur.fetchall()

    if not rows:
        print(f"  ⚠  No results found for '{term}'.")
        print("  Is the database populated and are search_vectors populated?")
        return

    print(f"  {len(rows)} result(s) returned (max 20):\n")
    print(
        f"  {'#':<3} {'Product name':<35} {'Type':<16} "
        f"{'Price':>8} {'Rank':>8}"
    )
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row['name']):<35} "
            f"{str(row['product_type']):<16} "
            f"{str(row['price_usd']):>8} "
            f"{float(row['rank']):>8.4f}"
        )


def explain(conn, term: str):
    """Print EXPLAIN ANALYZE for one search query."""
    print(f"\n  EXPLAIN ANALYZE — Q5 (term: '{term}'):\n")
    with conn.cursor() as cur:
        cur.execute(Q5_EXPLAIN_SQL, (term,))
        rows = cur.fetchall()
    for row in rows:
        print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Q5 full-text search benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--term", type=str, default=None,
        help="Fixed search term for --dry-run and --explain (default: random)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print EXPLAIN ANALYZE output then exit (no benchmark run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Execute one search and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q5_baseline.json"),
        help="Path to save JSON results (default: results/postgres_q5_baseline.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q5 Full-Text Search Benchmark")
    print("=" * 50)

    conn = get_connection()

    # Resolve the term to use for single-shot modes
    term = args.term if args.term else random.choice(SEARCH_TERMS)

    try:
        if args.explain:
            explain(conn, term)
            return

        if args.dry_run:
            dry_run(conn, term)
            return

        run_benchmark(
            query_fn=make_query_fn(conn, SEARCH_TERMS),
            db="postgres",
            query_id="Q5",
            label=(
                "Full-text product search using tsvector + plainto_tsquery "
                "with ts_rank_cd scoring. GIN index on search_vector column. "
                "Field weights: name(A) > description(B) > product_type(C). "
                f"Random term chosen per iteration from pool of "
                f"{len(SEARCH_TERMS)} domain-relevant queries."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()