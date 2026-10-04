"""
benchmarks/timescaledb/timescaledb_conn.py — Shared TimescaleDB connection helper
==================================================================================
All TimescaleDB benchmark scripts (Q1–Q8, run_scalability) import
get_connection() from here.

TimescaleDB runs on port 5433 (not 5432) to avoid conflicting with the
PostgreSQL baseline container. Both use the same psycopg2 driver — TimescaleDB
is PostgreSQL with time-series extensions installed on top.

Session-level settings — identical rationale to pg_conn.py
────────────────────────────────────────────────────────────
  jit = off
      Disables JIT compilation for the same reason as the PostgreSQL baseline:
      JIT fires on early iterations of a repeated benchmark loop and adds
      ~500ms compilation overhead that does not reflect steady-state query cost.
      Disabling it produces stable, repeatable latency across all 1000 iterations.

  work_mem = 64MB
      Prevents sort/aggregate spill to disk for Q1 and Q7. TimescaleDB Q7
      with time_bucket_gapfill() operates on a continuous aggregate (already
      pre-aggregated), so its memory pressure is low. However, Q1 and naive Q7
      (which scan raw invoices) can still spill without this setting.
      Kept consistent with the PostgreSQL baseline for a fair comparison.

TimescaleDB-specific note on timescaledb.max_background_workers
────────────────────────────────────────────────────────────────
TimescaleDB uses background workers for continuous aggregate refresh and
compression jobs. These run server-side and are not affected by this
connection helper. The default max_background_workers = 8 is sufficient
for a single-node Docker instance with one database.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Session-level settings — same as pg_conn.py, applied identically
# so that any latency difference between PostgreSQL and TimescaleDB
# is attributable to the engine, not to different session configuration.
_BENCHMARK_SETTINGS = [
    "SET jit = off",
    "SET work_mem = '64MB'",
]


def get_connection():
    """
    Open a psycopg2 connection to TimescaleDB with benchmark-optimised
    session settings applied. Import this function in every Q1–Q8 script
    instead of defining get_connection() locally.

    Returns a psycopg2 connection with autocommit=False (the default).
    The connection is to the TimescaleDB container on port 5433.
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,                              # TimescaleDB port (not 5432)
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"),
        connect_timeout=10,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        for setting in _BENCHMARK_SETTINGS:
            cur.execute(setting)
    conn.autocommit = False
    return conn