"""
benchmarks/cassandra/naive/q5_search.py — Cassandra Naive: Q5
==============================================================
Q5: Full-text product search with relevance ranking.

Naive schema limitation — CQL LIKE requires SASI; no SASI on naive schema
──────────────────────────────────────────────────────────────────────────
PostgreSQL Q5 uses a GIN-indexed tsvector with ts_rank_cd scoring.
Elasticsearch Q5 uses BM25 with field-level boosting. In the Cassandra
naive schema, neither is available:

  • CQL LIKE (substring matching) is only valid on columns with a SASI
    index configured with mode='CONTAINS'. The naive schema creates no
    SASI index — adding one would begin to approximate the optimised
    schema and violate the engine-vs-schema isolation principle.

  • Without SASI, there is no way to push text filtering to the server
    at all. The only honest naive approach is:
      1. Fetch all active products with ALLOW FILTERING on is_active.
      2. Filter Python-side: keyword.lower() in row.name.lower().
      3. Return up to 20 matches (mirroring the PostgreSQL LIMIT 20).

  This means naive Q5 always performs a full table scan of products
  regardless of the search term. There is no relevance ranking — results
  are returned in partition key order (Murmur3 hash order), not by match
  quality.

The schema effect for Q5:
  • Naive : full products scan + Python-side substring filter (no DB-level text search)
  • Optimised : SASI CONTAINS index on products_search.name_lower → single
               index scan with the substring filter pushed to the server

The quality gap (no ranking in naive, no ranking in optimised) is documented
as a Cassandra limitation compared to Elasticsearch. This is an intentional
"awkward implementation" that demonstrates why Elasticsearch exists.

Search term pool
─────────────────
Uses the same SEARCH_TERMS as PostgreSQL Q5 (single-word and multi-word
terms from a digital marketplace domain). Multi-word terms are checked
via Python `all(word in name for word in term.split())` — a naive
multi-word AND match on the product name only, which is the closest
equivalent to tsvector's multi-token AND matching.

Usage:
    python q5_search.py                    # 1000 iterations
    python q5_search.py --iterations 100   # quick smoke test
    python q5_search.py --dry-run          # run once, print result sample
    python q5_search.py --term "brushes"   # search a specific term
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

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")
RESULT_LIMIT = 20

# ── search term pool ──────────────────────────────────────────────────────────
# Same pool as PostgreSQL Q5 — keeps query distribution identical across DBs.

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography",
    "animation", "branding", "mockup", "watercolour",
    "photoshop brushes", "video editing", "certificate course",
    "logo design", "colour palette", "font pack", "texture pack",
    "motion graphics", "social media", "icon set", "web design",
    "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, fixed_term: str | None = None):
    """
    Each call:
      1. Full scan of products WHERE is_active = true (ALLOW FILTERING)
      2. Python-side substring filter on name.lower()
      3. Return up to RESULT_LIMIT matches (no ranking — partition key order)

    Note: CQL LIKE is not available without a SASI index on the column.
    The naive schema has no SASI index on products.name.
    Text matching is entirely Python-side after a full products table scan.
    """
    def _run():
        term = fixed_term if fixed_term else random.choice(SEARCH_TERMS)
        term_lower = term.lower()
        words = term_lower.split()

        # Full products scan (ALLOW FILTERING on is_active)
        rows = list(session.execute(
            "SELECT id, name, product_type, price_usd, attributes "
            "FROM products WHERE is_active = true ALLOW FILTERING"
        ))

        # Python-side text filter: all words must appear in name (case-insensitive)
        # Multi-word: "photoshop brushes" → name must contain "photoshop" AND "brushes"
        results = []
        for row in rows:
            if row.name and all(w in row.name.lower() for w in words):
                results.append(row)
            if len(results) >= RESULT_LIMIT:
                break

        return results

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, term: str | None = None):
    chosen = term or random.choice(SEARCH_TERMS)
    print(f"\n  DRY RUN — Q5 naive search: '{chosen}'\n")
    fn = make_query_fn(session, fixed_term=chosen)
    results = fn()
    if not results:
        print(f"  ⚠  No products matched '{chosen}' in name field.")
        return
    print(f"  {len(results)} result(s):\n")
    print(f"  {'Name':<45} {'Type':<15} {'Price':>8}")
    print(f"  {'─'*45} {'─'*15} {'─'*8}")
    for r in results[:10]:
        print(f"  {str(r.name)[:45]:<45} {str(r.product_type):<15} {str(r.price_usd):>8}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
    print(f"\n  Note: results in partition key (hash) order — no relevance ranking.")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q5 search benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--term", type=str, default=None,
                        help="Fix a single search term (default: random from pool)")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q5.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q5 Product Search Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_naive")
    print("  Method : ALLOW FILTERING full products scan + Python substring filter")
    print("  Note   : CQL LIKE requires SASI index (not present in naive schema).")
    print("           All text matching is Python-side. No relevance ranking.")

    cluster, session = get_session(keyspace=KEYSPACE)
    try:
        if args.dry_run:
            dry_run(session, args.term)
            return

        run_benchmark(
            query_fn=make_query_fn(session, fixed_term=args.term),
            db="cassandra_naive",
            query_id="Q5",
            label=(
                f"Full-text product search. CQL LIKE requires SASI index — not present "
                "in naive schema. Method: ALLOW FILTERING full scan of products "
                "(is_active=true) + Python-side substring match on name.lower(). "
                "No relevance ranking — results in partition key order. "
                f"LIMIT {RESULT_LIMIT} matches returned. "
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