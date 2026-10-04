"""
benchmarks/elasticsearch/elasticsearch_conn.py — Shared Elasticsearch client helper
====================================================================================
All Elasticsearch benchmark and loader scripts import get_client() from here.

Connection settings applied to every client:
  - URL from ELASTICSEARCH_URL env var (default: http://localhost:9200)
  - Security disabled on the Docker container — no auth headers needed
  - request_timeout=30s: generous for aggregation queries on a cold JVM;
    ES date_histogram + sum aggs on a large invoices index can be slow on
    the first call before Lucene's field data cache is warm.
  - retry_on_timeout=True + max_retries=3: handles transient GC pauses
    that briefly kill connections during long benchmark runs (ES 8.x JVM
    can pause for 50–200ms under full heap pressure, dropping TCP connections).
  - sniff_on_start=False: single-node Docker deployment — no node discovery
    needed, and sniffing adds ~100ms startup latency per client creation.

Thread safety:
  The Elasticsearch client is internally connection-pool-backed and
  thread-safe. A single shared client instance works for all single-
  threaded read benchmarks (Q1–Q7).

  For Q8 (concurrent writes), create one client per thread at thread
  startup, matching the one-connection-per-thread pattern used in the
  PostgreSQL, MongoDB, and Cassandra Q8 implementations. This avoids
  connection pool contention under 100-thread concurrency.
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")


def get_client() -> Elasticsearch:
    """
    Return a configured Elasticsearch client for benchmark use.

    Call once at module level for single-threaded benchmarks:
        es = get_client()

    Call once per thread for Q8 concurrent writes:
        def worker():
            es = get_client()
            ...
    """
    return Elasticsearch(
        _ES_URL,
        request_timeout=30,
        retry_on_timeout=True,
        max_retries=3,
        sniff_on_start=False,
    )