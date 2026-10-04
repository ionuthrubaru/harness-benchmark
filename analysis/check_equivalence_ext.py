"""
analysis/check_equivalence_ext.py — equivalence check, remaining engines
========================================================================
Extends analysis/check_q1_q7_equivalence.py (PostgreSQL vs Elasticsearch /
Neo4j Q7) to the engines not yet verified row by row:

  implemented here:   TimescaleDB naive  (Q1, Q7) — runs the PostgreSQL SQL
                      on the TimescaleDB connection, so the check is exact.
  via driver hook:    MongoDB (naive/optimised), Cassandra (naive/optimised),
                      Neo4j Q1 (naive/optimised), TimescaleDB optimised.

Driver hook convention. For each engine below, the script imports the Q1/Q7
driver module and calls, if present:

    compute_rows(kind: str, start: date, end: date, as_of=None) -> list[tuple]
      Q1 rows: (month 'YYYY-MM', tier_name, price_in_effect: float,
                invoice_count: int, total_revenue: float)
      Q7 rows: (day 'YYYY-MM-DD', tier_name, daily_revenue: float,
                rolling_7d_avg: float)

Exposing compute_rows in a driver is a small refactor: move the body of the
timed query_fn into compute_rows and have query_fn call it. Until a driver
exposes it, the script reports SKIP for that engine instead of failing.

STATUS: written off-line against the repository at commit 2e5deb6 and NOT yet
executed against live containers (the audit environment has no databases).
Run order: docker compose up postgres timescaledb (plus the engine under
test), then:  python analysis/check_equivalence_ext.py
"""

import importlib
import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "postgres"))

# reuse the reference implementation and the comparator from the base script
base = importlib.import_module("analysis.check_q1_q7_equivalence")

WINDOW_DAYS = 183
DRIVER_MODULES = {
    "mongodb_naive":        "benchmarks.mongodb.naive.q1_revenue",
    "mongodb_naive_q7":     "benchmarks.mongodb.naive.q7_rolling_revenue",
    "mongodb_optimised":    "benchmarks.mongodb.optimised.q1_revenue",
    "mongodb_optimised_q7": "benchmarks.mongodb.optimised.q7_rolling_revenue",
    "cassandra_naive":      "benchmarks.cassandra.naive.q1_revenue",
    "cassandra_naive_q7":   "benchmarks.cassandra.naive.q7_rolling_revenue",
    "cassandra_optimised":  "benchmarks.cassandra.optimised.q1_revenue",
    "cassandra_optimised_q7": "benchmarks.cassandra.optimised.q7_rolling_revenue",
    "neo4j_naive_q1":       "benchmarks.neo4j.naive.q1_revenue",
    "neo4j_optimised_q1":   "benchmarks.neo4j.optimised.q1_revenue",
    "timescaledb_optimised": "benchmarks.timescaleDB.optimised.run_benchmarks",
}


def ts_naive_rows(kind, start=None, as_of=None):
    """TimescaleDB naive uses the PostgreSQL SQL verbatim; run it on the
    TimescaleDB connection (port 5433) and reuse the base row shaping."""
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    conn = psycopg2.connect(
        host="localhost", port=5433,
        user=os.getenv("TIMESCALE_USER"), password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"), connect_timeout=10,
    )
    try:
        return base.pg_rows(start=start, q=kind, as_of=as_of, conn=conn) \
            if "conn" in base.pg_rows.__code__.co_varnames \
            else _run_pg_sql_on(conn, kind, start, as_of)
    finally:
        conn.close()


def _run_pg_sql_on(conn, kind, start, as_of):
    """Fallback if base.pg_rows does not accept an external connection:
    execute the same SQL files the base script loads, on `conn`."""
    raise SystemExit(
        "base.pg_rows does not accept a connection argument; add `conn=None` "
        "to pg_rows in check_q1_q7_equivalence.py (3-line change) and retry.")


def hook_rows(modname, kind, start, as_of):
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        return None, f"import failed ({e})"
    fn = getattr(mod, "compute_rows", None)
    if fn is None:
        return None, "compute_rows() not exposed yet (see module docstring above)"
    return fn(kind, start, start + timedelta(days=WINDOW_DAYS - 1), as_of=as_of), None


def main():
    as_of = getattr(importlib.import_module(
        "benchmarks.elasticsearch.es_revenue"), "Q1_AS_OF", None)
    failures, skips = 0, 0

    # Q1, anchored window
    ref_q1 = base.pg_rows(q="Q1", as_of=as_of)
    # Q7, one deterministic window (extend with --windows like the base script)
    q7_start = date(2025, 3, 1)
    ref_q7 = base.pg_rows(start=q7_start, q="Q7")

    print("TimescaleDB naive (PostgreSQL SQL on the TimescaleDB connection):")
    for kind, ref, st in (("Q1", ref_q1, None), ("Q7", ref_q7, q7_start)):
        got = ts_naive_rows(kind, start=st, as_of=as_of)
        failures += 0 if base.compare(f"TimescaleDB naive {kind}", ref, got,
                                      n_key=2 if kind == "Q1" else 2) else 1

    for label, modname in DRIVER_MODULES.items():
        kind = "Q7" if label.endswith("q7") else "Q1"
        ref, st = (ref_q7, q7_start) if kind == "Q7" else (ref_q1, None)
        rows, err = hook_rows(modname, kind, st or date(2025, 3, 1), as_of)
        if err:
            print(f"  SKIP {label} {kind}: {err}")
            skips += 1
            continue
        failures += 0 if base.compare(f"{label} {kind}", ref, rows,
                                      n_key=2) else 1

    print(f"\nRESULT: {'ALL CHECKED EQUIVALENT' if failures == 0 else f'{failures} FAILURES'}"
          f" ({skips} engines skipped pending compute_rows hooks)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
