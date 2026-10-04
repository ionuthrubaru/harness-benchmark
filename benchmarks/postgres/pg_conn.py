"""
benchmarks/postgres/pg_conn.py — Shared PostgreSQL connection helper
====================================================================
All Q1-Q7 benchmark scripts import get_connection() from here.

Session-level settings applied on every connection:
  - jit = off        : Disables JIT compilation. JIT is designed for
                       one-off long-running analytics queries — it spends
                       hundreds of ms compiling the query plan to native
                       code. For a benchmark that runs the same query
                       1000 times in a loop, JIT fires on early iterations
                       and adds ~500ms overhead per run. Disabling it
                       produces stable, repeatable latency that reflects
                       the engine's actual query execution cost.

  - work_mem = 64MB  : Increases the per-sort-operation memory from the
                       default 4MB. Without this, Q1 and Q7 sort operations
                       spill to disk (external merge sort on temp storage),
                       adding significant I/O cost on every iteration.
                       64MB keeps sorts in RAM for the dataset sizes used
                       in this benchmark.

Both settings are documented in the methodology as session-level benchmark
configuration, not permanent server changes. They are academically
defensible because they eliminate measurement artifacts (JIT compilation
overhead, disk spill I/O) that do not reflect steady-state query
performance under repeated load.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Session-level settings applied to every benchmark connection
_BENCHMARK_SETTINGS = [
    "SET jit = off",
    "SET work_mem = '64MB'",
]


def get_connection():
    """
    Open a psycopg2 connection to PostgreSQL with benchmark-optimised
    session settings applied. Import this function in every Q1-Q7 script
    instead of defining get_connection() locally.
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
        connect_timeout=10,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        for setting in _BENCHMARK_SETTINGS:
            cur.execute(setting)
    conn.autocommit = False
    return conn