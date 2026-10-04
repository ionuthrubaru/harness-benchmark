"""
analysis/check_q1_q7_equivalence.py — result-equivalence check for Q1 and Q7
============================================================================
Runs Q1 and Q7 once on PostgreSQL (the reference) and on each engine that
now implements the full query, for the SAME window, and compares the result
rows. Use it as evidence that the benchmarked queries return the same answer.

  Q1: PostgreSQL vs Elasticsearch naive, Elasticsearch optimised
      (anchored to es_revenue.Q1_AS_OF, i.e. NOW() of the PG baseline run)
  Q7: PostgreSQL vs Neo4j naive, Neo4j optimised,
      Elasticsearch naive, Elasticsearch optimised  (3 random windows)

Tolerance: 0.05 USD per value. Elasticsearch stores total_usd as `float`
(32-bit), so server-side sums can differ from NUMERIC by a few cents; the
row set (months / days / tiers / prices / counts) must match exactly.

Usage (all containers up, run from the repo root):
    python analysis/check_q1_q7_equivalence.py
    python analysis/check_q1_q7_equivalence.py --skip neo4j_optimised es_naive
"""

import argparse
import os
import random
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "postgres"))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

TOL = 0.05
WINDOW_DAYS = 183


def _sql(path, name):
    src = open(path, encoding="utf-8").read()
    a = src.index(name + ' = """') + len(name + ' = """')
    return src[a:src.index('"""', a)]


def pg_rows(start=None, q="Q1", as_of=None):
    from pg_conn import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if q == "Q1":
                sql = _sql(os.path.join(ROOT, "benchmarks/postgres/q1_revenue.py"), "Q1_SQL")
                cur.execute(sql.replace("NOW()", "%s::timestamptz"), (as_of,) * sql.count("NOW()"))
                return [(m, t, round(float(p), 2), int(c), float(r)) for m, t, p, c, r in cur.fetchall()]
            sql = _sql(os.path.join(ROOT, "benchmarks/postgres/q7_rolling_revenue.py"), "Q7_SQL")
            end = start + timedelta(days=WINDOW_DAYS - 1)
            cur.execute(sql, (start, end, start, end, start, end))
            return [(str(d), t, float(a), float(b)) for d, t, a, b in cur.fetchall()]
    finally:
        conn.close()


def compare(name, ref, got, n_key):
    ref = sorted(ref)
    got = sorted(got)
    keys_ref = [r[:n_key] for r in ref]
    keys_got = [r[:n_key] for r in got]
    if keys_ref != keys_got:
        missing = sorted(set(keys_ref) - set(keys_got))[:5]
        extra = sorted(set(keys_got) - set(keys_ref))[:5]
        print(f"    ✘ {name}: row keys differ ({len(ref)} vs {len(got)} rows)"
              f"  missing={missing} extra={extra}")
        return False
    worst = 0.0
    for a, b in zip(ref, got):
        for x, y in zip(a[n_key:], b[n_key:]):
            worst = max(worst, abs(float(x) - float(y)))
    status = "✔" if worst <= TOL else "✘"
    print(f"    {status} {name}: {len(got)} rows, max abs diff {worst:.4f}")
    return worst <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    help="neo4j_naive neo4j_optimised es_naive es_optimised")
    ap.add_argument("--windows", type=int, default=3)
    args = ap.parse_args()
    skip = set(args.skip)
    ok = True

    from benchmarks.elasticsearch import es_revenue as R
    as_of = R.Q1_AS_OF

    es = None
    if not {"es_naive", "es_optimised"} <= skip:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
                           request_timeout=300)

    # ── Q1 ────────────────────────────────────────────────────────────────────
    print(f"\n  Q1 (12 months to {as_of.isoformat()})")
    ref = pg_rows(q="Q1", as_of=as_of)
    print(f"    PostgreSQL: {len(ref)} rows")
    q1 = lambda rows: [(r["month"], r["tier_name"], r["price_in_effect_usd"],
                        r["invoice_count"], r["total_revenue_usd"]) for r in rows]
    if "es_naive" not in skip:
        ok &= compare("Elasticsearch naive", ref, q1(R.naive_q1(es, as_of)), 4)
    if "es_optimised" not in skip:
        ok &= compare("Elasticsearch optimised", ref, q1(R.optimised_q1(es, as_of)), 4)

    # ── Q7 ────────────────────────────────────────────────────────────────────
    drivers = {}
    from benchmarks.neo4j.neo4j_conn import get_driver
    if "neo4j_naive" not in skip:
        drivers["Neo4j naive"] = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))
    if "neo4j_optimised" not in skip:
        drivers["Neo4j optimised"] = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "neo4j_q7", os.path.join(ROOT, "benchmarks/neo4j/naive/q7_rolling_revenue.py"))
    neo_q7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neo_q7)

    tier_names = R.optimised_tier_names(es) if es and "es_optimised" not in skip else None
    random.seed(7)
    dmin, dmax = date(2024, 1, 1), date(2025, 12, 31)
    for k in range(args.windows):
        start = dmin + timedelta(days=random.randint(0, (dmax - dmin).days - WINDOW_DAYS))
        end_excl = start + timedelta(days=WINDOW_DAYS)
        print(f"\n  Q7 window {start} → {start + timedelta(days=WINDOW_DAYS - 1)}")
        ref = pg_rows(start=start, q="Q7")
        print(f"    PostgreSQL: {len(ref)} rows")
        q7 = lambda rows: [(r["day"], r["tier_name"], r["daily_revenue_usd"],
                            r["rolling_7day_avg_usd"]) for r in rows]
        for name, drv in drivers.items():
            with drv.session() as s:
                ok &= compare(name, ref, q7(neo_q7.run_q7(s, start.isoformat(),
                                                          end_excl.isoformat())), 2)
        if "es_naive" not in skip:
            ok &= compare("Elasticsearch naive", ref, q7(R.naive_q7(es, start, WINDOW_DAYS)), 2)
        if "es_optimised" not in skip:
            ok &= compare("Elasticsearch optimised", ref,
                          q7(R.optimised_q7(es, start, WINDOW_DAYS, tier_names)), 2)

    for d in drivers.values():
        d.close()
    print("\n  RESULT:", "ALL EQUIVALENT ✔" if ok else "DIFFERENCES FOUND ✘")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
