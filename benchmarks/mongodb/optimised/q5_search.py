"""
benchmarks/mongodb/optimised/q5_search.py — MongoDB Optimised: Q5
==================================================================
Q5: Full-text product search with relevance ranking.

Optimised schema changes vs naive:
  - Text index created with field weights: name=10, description=3,
    product_type=1 (vs naive: equal weighting across all fields)
  - is_active stored as bool → $match {"is_active": True} works correctly
    (naive stored "True" as string — filter was a no-op or wrong)
  - attributes stored as native BSON subdocument → returned directly as
    Python dict, no json.loads()
  - $meta "textScore" sort returns relevance-ranked results

Query structure vs naive:
  Naive:  $text search without weights, is_active filter broken (string),
          attributes returned as string
  Optimised: $text search with per-field weights, correct bool filter,
          $meta textScore sort, native attributes dict

Academic context:
  MongoDB $text search is BM25-like but NOT true BM25 (that is
  Elasticsearch's domain). The optimised version is as good as MongoDB
  can do natively. The schema effect here is: field weighting + correct
  is_active filtering. Elasticsearch Q5 optimised is the comparison point
  for true BM25 with custom analysers.

Usage:
    python q5_search.py
    python q5_search.py --iterations 100
    python q5_search.py --dry-run
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

TOP_N = 20

SEARCH_TERMS = [
    "photoshop brushes",
    "python course",
    "design assets",
    "video editing",
    "typography",
    "beginner tutorial",
    "premium bundle",
    "illustration kit",
    "merch cotton",
    "digital download",
    "advanced techniques",
    "procreate pack",
    "web development",
    "certificate course",
    "3d models",
]

# ── core Q5 logic (timed portion) ─────────────────────────────────────────────

def run_q5(db) -> list[dict]:
    """
    $text search with textScore sort.

    Optimised: field weights (name=10, description=3, product_type=1) mean
    a match in name ranks much higher than a match in description.
    is_active is a bool so the filter is correct.
    attributes is a native dict in the returned document.
    """
    term = random.choice(SEARCH_TERMS)
    cursor = db["products"].find(
        {
            "$text": {"$search": term},
            "is_active": True,
        },
        {
            "score":        {"$meta": "textScore"},
            "name":         1,
            "product_type": 1,
            "price_usd":    1,
            "description":  1,
            "attributes":   1,
        }
    ).sort([("score", {"$meta": "textScore"})]).limit(TOP_N)
    return list(cursor)

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db):
    def _run():
        run_q5(db)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db):
    print("\n  DRY RUN — MongoDB Optimised Q5 result sample:\n")
    results = run_q5(db)
    if not results:
        print("  ⚠  No products returned — check text index on optimised DB")
        return
    print(f"  {'score':<8} {'name':<40} {'type':<16} {'price_usd'}")
    print(f"  {'─'*8} {'─'*40} {'─'*16} {'─'*9}")
    for row in results[:10]:
        print(
            f"  {row.get('score', 0):<8.4f} "
            f"{str(row.get('name','')):<40} "
            f"{str(row.get('product_type','')):<16} "
            f"{row.get('price_usd')}"
        )
    if results:
        print(f"\n  attributes type: {type(results[0].get('attributes')).__name__}  "
              f"← should be 'dict', not 'str'")
    print(f"\n  Total results: {len(results)}")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised Q5 — full-text search benchmark"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_optimised_Q5.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Optimised — Q5 Full-Text Search Benchmark")
    print("=" * 55)

    db = get_db(schema="optimised")

    if args.dry_run:
        dry_run(db)
        return

    run_benchmark(
        query_fn=make_query_fn(db),
        db="mongodb_optimised",
        query_id="Q5",
        label=(
            f"Full-text product search, top {TOP_N} by relevance. "
            "Optimised: text index with field weights (name=10, description=3, "
            "product_type=1); $meta textScore sort; is_active as bool (filter "
            "correct); attributes as native BSON dict. "
            "Note: MongoDB $text is not true BM25 — Elasticsearch Q5 is the "
            "authoritative comparison point for relevance ranking."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()