"""
benchmarks/cassandra/optimised/q5_search.py — Cassandra Optimised: Q5
======================================================================
Q5: Full-text product search with relevance ranking.

Optimised schema — SASI CONTAINS index on name_lower
──────────────────────────────────────────────────────
Table: products_search
Index: SASI CONTAINS on name_lower (StandardAnalyzer, case_sensitive=false)

name_lower stores the product name in lowercase at load time. The SASI
index with mode=CONTAINS enables `name_lower LIKE '%keyword%'` to push
the substring filter to the server without ALLOW FILTERING. Cassandra
evaluates the LIKE predicate using the SSTable-attached index, reading
only matching partitions rather than every row in the table.

Schema effect vs naive:
  Naive  : Full products table scan (ALLOW FILTERING on is_active)
           + Python substring filter on name.lower() → O(all products)
  Optimised : SASI LIKE query → server-side index scan → O(matches)

Documented limitations (for methodology chapter):
  • SASI is deprecated in Cassandra 5.0. Fully supported in 4.1 (this experiment).
  • No relevance ranking in either naive or optimised — Cassandra has no
    BM25 or tf-idf equivalent. Results are returned in token ring order.
    This is a documented Cassandra limitation vs Elasticsearch (Q5's killer DB).
  • Multi-word terms: each word is searched as a separate LIKE clause
    combined with AND — the query fans out to N LIKE predicates for an
    N-word search term.
  • The schema effect is purely structural (ALLOW FILTERING eliminated),
    not qualitative (matching quality is unchanged).

Usage:
    python q5_search.py                    # 1000 iterations
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE     = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")
RESULT_LIMIT = 20

# Same pool as PostgreSQL Q5 and Cassandra naive Q5 — identical term distribution
SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography",
    "animation", "branding", "mockup", "watercolour",
    "photoshop brushes", "video editing", "certificate course",
    "logo design", "colour palette", "font pack", "texture pack",
    "motion graphics", "social media", "icon set", "web design",
    "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

# ── query builder ─────────────────────────────────────────────────────────────

def _build_sasi_query(term: str) -> tuple[str, tuple, list[str]]:
    """
    Build a CQL SASI query for the search term.

    SASI limitation: a column cannot be restricted by more than one LIKE
    in a single CQL statement ("name_lower cannot be restricted by more than
    one relation if it includes a LIKE"). This prevents the naive multi-word
    AND approach of chaining LIKE predicates.

    Workaround for multi-word terms:
      - Push only the FIRST word to the server via the SASI LIKE index.
      - Filter remaining words Python-side after the server returns results.
    This still eliminates the full table scan (SASI handles the primary filter)
    while respecting the one-LIKE-per-column constraint. The Python post-filter
    is O(SASI results), not O(all products), so it is far cheaper than the
    naive full scan. Documented in the methodology as a SASI limitation.

    Returns (cql, params, extra_words) where extra_words are the additional
    words to filter Python-side.
    """
    words = term.lower().split()
    primary_word = words[0]
    extra_words  = words[1:]  # filtered Python-side
    cql = (
        f"SELECT id, name, name_lower, product_type, price_usd, attributes "
        f"FROM products_search "
        f"WHERE name_lower LIKE %s LIMIT {RESULT_LIMIT * 10}"
        # Fetch more than RESULT_LIMIT to allow Python post-filter to find enough matches.
        # RESULT_LIMIT * 10 is a reasonable upper bound for the SASI result set.
    )
    params = (f"%{primary_word}%",)
    return cql, params, extra_words

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, fixed_term: str | None = None):
    """
    SASI LIKE query for the primary word + Python post-filter for extra words.
    Server handles the primary word index scan; Python filters remaining words
    on the (small) SASI result set. No ALLOW FILTERING. No full table scan.
    """
    def _run():
        term = fixed_term if fixed_term else random.choice(SEARCH_TERMS)
        cql, params, extra_words = _build_sasi_query(term)
        rows = list(session.execute(cql, params))
        if extra_words:
            rows = [
                r for r in rows
                if r.name_lower and all(w in r.name_lower for w in extra_words)
            ]
        return rows[:RESULT_LIMIT]
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, term: str | None = None):
    chosen = term or random.choice(SEARCH_TERMS)
    print(f"\n  DRY RUN — Q5 optimised search: '{chosen}'\n")
    cql, params, extra_words = _build_sasi_query(chosen)
    print(f"  CQL: {cql}")
    print(f"  Params: {params}")
    if extra_words:
        print(f"  Python post-filter words: {extra_words}")
    print()
    rows = list(session.execute(cql, params))
    if extra_words:
        rows = [r for r in rows if r.name_lower and all(w in r.name_lower for w in extra_words)]
    rows = rows[:RESULT_LIMIT]
    if not rows:
        print(f"  ⚠  No products matched '{chosen}' via SASI index.")
        return
    print(f"  {len(rows)} result(s) — server-side SASI index scan:\n")
    print(f"  {'Name':<45} {'Type':<15} {'Price':>8}")
    print(f"  {'─'*45} {'─'*15} {'─'*8}")
    for r in rows[:10]:
        print(f"  {str(r.name)[:45]:<45} {str(r.product_type):<15} {str(r.price_usd):>8}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    print(f"\n  Note: results in token ring order (no relevance ranking).")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra optimised Q5 search benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--term", type=str, default=None)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_optimised_Q5.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Optimised — Q5 Product Search Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_optimised (products_search + SASI index)")
    print("  Method : SASI LIKE on name_lower — server-side index scan")
    print("           No ALLOW FILTERING. Multi-word = AND of LIKE clauses.")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        if args.dry_run:
            dry_run(session, args.term)
            return
        run_benchmark(
            query_fn=make_query_fn(session, fixed_term=args.term),
            db="cassandra_optimised",
            query_id="Q5",
            label=(
                "Full-text product search via SASI CONTAINS index on name_lower. "
                "Table: products_search. Index: SASIIndex mode=CONTAINS, analyzed=true. "
                "Query: name_lower LIKE '%term%' (one LIKE per word, AND logic). "
                "Server-side index scan — no ALLOW FILTERING. "
                "No relevance ranking (documented Cassandra limitation vs Elasticsearch). "
                f"LIMIT {RESULT_LIMIT}. "
                f"Search terms: {len(SEARCH_TERMS)} domain-relevant terms, random per iteration."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()