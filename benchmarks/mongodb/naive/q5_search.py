"""
benchmarks/mongodb/naive/q5_search.py — MongoDB Naive: Q5
==========================================================
Q5: Full-text product search with relevance ranking across
    name, description, and product_type.

Naive implementation notes:
    MongoDB's built-in $text search is used against a text index on
    (name, description). This is the naive equivalent of PostgreSQL's
    tsvector + GIN index.

    Naive constraints (no optimised features used):
      - No custom analyser — default English analyser only
      - No field-level boosting (name^3, description^1.5 etc.)
      - No synonym expansion
      - Results sorted by textScore (MongoDB's built-in TF/IDF proxy)
      - Limit 20 results — consistent with the PostgreSQL baseline

    Prerequisites:
        A text index must exist on the products collection before running
        this benchmark. If the loader has not already created it, run once:
            db.products.createIndex({ name: "text", description: "text" })
        The loader's create_indexes() function includes this index — it
        will be present if the loader was run. If not, create it manually.

    The same SEARCH_TERMS pool used by the PostgreSQL baseline is reused
    here so the query distribution is identical and the comparison is fair.

Academic context:
    Engine effect = naive MongoDB result minus PostgreSQL baseline.
    Schema effect = optimised MongoDB result minus naive result.
    This file measures the naive (engine-only) side.

Usage:
    cd benchmarks/mongodb/naive
    python q5_search.py                   # 1000 iterations
    python q5_search.py --iterations 100  # quick smoke test
    python q5_search.py --dry-run         # run once, print result sample
    python q5_search.py --term "brushes"  # search a specific term
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

# ── search term pool (identical to PostgreSQL baseline) ───────────────────────

SEARCH_TERMS = [
    "brushes",
    "typography",
    "illustration",
    "photography",
    "animation",
    "branding",
    "mockup",
    "watercolour",
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
]

# ── core Q5 logic ─────────────────────────────────────────────────────────────

def run_q5(db, term: str) -> list[dict]:
    """
    Execute a full-text search against the products collection.
    Uses $text with $meta textScore sorting — MongoDB's built-in
    relevance proxy (TF/IDF based, no BM25, no field boosting).
    Mirrors the PostgreSQL ts_rank_cd ordering.
    """
    return list(db["products"].find(
        {
            "$text":      {"$search": term},
            "is_active":  "True",
        },
        {
            "_id":         1,
            "name":        1,
            "product_type":1,
            "price_usd":   1,
            "attributes":  1,
            "score":       {"$meta": "textScore"},
        }
    ).sort([("score", {"$meta": "textScore"})]).limit(20))

# ── benchmark factory ─────────────────────────────────────────────────────────

def make_query_fn(db, terms: list[str]):
    def _run():
        term = random.choice(terms)
        run_q5(db, term)
    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(db, term: str):
    print(f"\n  DRY RUN — MongoDB Naive Q5 search for: '{term}'\n")
    rows = run_q5(db, term)
    if not rows:
        print(f"  ⚠  No results for '{term}'.")
        print("  Check that the text index exists on products (name + description).")
        return
    print(f"  {len(rows)} result(s) returned (max 20):\n")
    print(
        f"  {'#':<3} {'Product name':<35} {'Type':<16} "
        f"{'Price':>8} {'Score':>8}"
    )
    print(f"  {'─'*3} {'─'*35} {'─'*16} {'─'*8} {'─'*8}")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {str(row.get('name', '')):<35} "
            f"{str(row.get('product_type', '')):<16} "
            f"{str(row.get('price_usd', '')):>8} "
            f"{float(row.get('score', 0)):>8.4f}"
        )

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Naive Q5 — full-text search benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of measured iterations (default: 1000)",
    )
    parser.add_argument(
        "--term", type=str, default=None,
        help="Fixed search term for --dry-run (default: random from pool)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Execute one search and print result sample then exit",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_naive_Q5.json"),
        help="Path to save JSON results (default: results/mongodb_naive_Q5.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q5 Full-Text Search Benchmark")
    print("=" * 55)

    db = get_db()

    # Verify text index exists before running
    index_info = db["products"].index_information()
    has_text_index = any(
        any(v == "text" for _, v in idx.get("key", []))
        for idx in index_info.values()
    )
    if not has_text_index:
        print(
            "\n  ✘ No text index found on products collection.\n"
            "  Create it first:\n"
            '    db.products.createIndex({ name: "text", description: "text" })\n'
        )
        sys.exit(1)

    term = args.term if args.term else random.choice(SEARCH_TERMS)

    if args.dry_run:
        dry_run(db, term)
        return

    run_benchmark(
        query_fn=make_query_fn(db, SEARCH_TERMS),
        db="mongodb_naive",
        query_id="Q5",
        label=(
            "Full-text product search using MongoDB $text with textScore "
            "sorting. Default English analyser, no field boosting, no "
            "synonyms — naive port of PostgreSQL tsvector baseline. "
            f"Random term chosen per iteration from pool of "
            f"{len(SEARCH_TERMS)} domain-relevant queries."
        ),
        iterations=args.iterations,
        concurrency=1,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()