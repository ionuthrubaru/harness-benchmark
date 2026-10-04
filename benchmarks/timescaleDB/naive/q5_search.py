"""
benchmarks/timescaledb/naive/q5_search.py — TimescaleDB Naive: Q5
==================================================================
Q5: Full-text product search with relevance ranking across name,
    description, and product_type.

SQL is identical to the PostgreSQL baseline
────────────────────────────────────────────
TimescaleDB adds nothing for Q5 — products is a plain PostgreSQL table
(not a hypertable). The tsvector GIN index, plainto_tsquery, and
ts_rank_cd ranking run identically. Latency should match PostgreSQL
closely, confirming TimescaleDB doesn't penalise non-time-series queries.

Engine effect for Q5 (naive)
─────────────────────────────
None expected. products is not a hypertable — no chunk pruning, no
time-series optimisations. The query plan is identical to PostgreSQL.

Search term pool
─────────────────
Identical to PostgreSQL Q5 — same 25 terms, same exclusion rationale.
Keeping the term distribution identical across all databases ensures the
comparison is fair (same mix of common and rare terms per iteration).

Usage:
    cd benchmarks/timescaledb/naive
    python q5_search.py                    # 1000 iterations
    python q5_search.py --iterations 100   # quick smoke test
    python q5_search.py --explain          # EXPLAIN ANALYZE
    python q5_search.py --dry-run          # run once, print results
    python q5_search.py --term "brushes"   # fixed search term
"""

import argparse
import os
import random
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

load_dotenv()

from benchmarks.timescaleDB.timescaledb_conn import get_connection

# ── search term pool — identical to PostgreSQL Q5 ─────────────────────────────

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography",
    "animation", "branding", "mockup", "watercolour",
    "photoshop brushes", "video editing", "certificate course",
    "logo design", "colour palette", "font pack", "texture pack",
    "motion graphics", "social media", "icon set", "web design",
    "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

# ── query — identical to PostgreSQL Q5 ────────────────────────────────────────

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

def make_query_fn(conn, terms: list[str], fixed_term: str | None = None):
    def _run():
        term = fixed_term if fixed_term else random.choice(terms)
        with conn.cursor() as cur:
            cur.execute(Q5_SQL, (term,))
            cur.fetchall()
    return _run

# ── helper modes ──────────────────────────────────────────────────────────────

def dry_run(conn, term: str):
    print(f"\n  DRY RUN — Q5 naive search for: '{term}'\n")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(Q5_SQL, (term,))
        rows = cur.fetchall()
    if not rows:
        print(f"  ⚠  No results for '{term}' — check search_vector is populated.")
        return
    print(f"  {len(rows)} result(s) (max 20):\n")
    print(f"  {'#':<3} {'Product name':<35} {'Type':<16} {'Price':>8} {'Rank':>8}")
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row['name'])[:35]:<35} "
            f"{str(row['product_type']):<16} "
            f"{str(row['price_usd']):>8} "
            f"{float(row['rank']):.4f}"
        )


def explain(conn, term: str):
    print(f"\n  EXPLAIN ANALYZE — Q5 naive (term: '{term}'):\n")
    with conn.cursor() as cur:
        cur.execute(Q5_EXPLAIN_SQL, (term,))
        for row in cur.fetchall():
            print(" ", row[0])

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimescaleDB naive Q5 search benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--term", type=str, default=None,
                        help="Fix a single search term (default: random from pool)")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("benchmarks", "timescaleDB", "naive", "results", "timescaledb_naive_Q5.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB Naive — Q5 Product Search Benchmark")
    print("=" * 60)
    print("  Schema : naive (products is a plain table, not hypertable)")
    print("  SQL    : identical to PostgreSQL baseline")
    print("  Engine : GIN tsvector index — no TimescaleDB-specific effect")

    conn = get_connection()
    try:
        term_for_dry = args.term or random.choice(SEARCH_TERMS)
        if args.explain:
            explain(conn, term_for_dry)
            return
        if args.dry_run:
            dry_run(conn, term_for_dry)
            return
        run_benchmark(
            query_fn=make_query_fn(conn, SEARCH_TERMS, fixed_term=args.term),
            db="timescaledb_naive",
            query_id="Q5",
            label=(
                "Full-text product search using tsvector GIN index + ts_rank_cd. "
                "SQL identical to PostgreSQL baseline. "
                "products is a plain PostgreSQL table — no hypertable, no chunk pruning. "
                "TimescaleDB engine effect expected to be zero for this query. "
                f"{len(SEARCH_TERMS)} domain-relevant search terms, random per iteration."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()