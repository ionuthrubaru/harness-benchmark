"""
benchmarks/elasticsearch/naive/q8_write.py — Elasticsearch Naive: Q8
======================================================================
Q8: Concurrent event ingestion benchmark.

    100 threads each index individual events from a pre-generated CSV
    as fast as possible until all 1M rows are exhausted.
    Each thread executes one index() call per event — the realistic
    production pattern for an event-driven application.

Dissertation angle:
    Does specialisation confer a write advantage under realistic
    production conditions? This benchmark isolates the engine's write
    path under genuine concurrent load. The identical CSV and the same
    100-thread pattern is used across all databases so the only variable
    is the engine.

Design decisions
────────────────
  - Single index() per event (not bulk API) — mirrors how a real
    application writes events: one user action, one immediate write.
    The bulk API is a data migration pattern, not a production write path,
    and using it here would not be comparable with the single-INSERT
    pattern used in PostgreSQL, Cassandra, and MongoDB Q8.

  - Shared ES client across all threads — unlike PostgreSQL where each
    thread needs its own connection, the elasticsearch-py client is
    thread-safe (urllib3 connection pool handles concurrent requests).
    The pool size is set to n_threads so no thread ever waits for a
    free connection slot.

  - CSV pre-sliced at startup — the 1M rows are divided into 100 equal
    slices. Each thread owns its slice and works through it sequentially.
    CSV exhaustion is the natural stop condition.

  - Staging index (naive_events_q8) — separate from naive_events so the
    benchmark does not pollute the data used by Q1–Q7 reads. No FK
    constraints (consistent with the PostgreSQL staging table approach).

  - Docker stats sampled every second in a background thread — CPU and
    memory recorded alongside latency data.

  - Per-index latency recorded for every one of the 1M documents — full
    distribution stored in the output JSON.

Output JSON includes:
    total_events, wall_time_s, throughput_events_per_sec,
    latency_ms (p50/p95/p99/mean/std_dev/min/max),
    resource_stats (cpu_pct + memory_mb timeseries),
    raw_timings_ms (full 1M latency list)

Usage:
    python q8_write.py                        # full 1M event benchmark
    python q8_write.py --rows 10000           # smoke test with 10K rows
    python q8_write.py --threads 10           # reduce concurrency
    python q8_write.py --no-resource-stats    # skip Docker stats
    python q8_write.py --dry-run              # index 100 rows, print summary
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_THREADS  = 100
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "events_q8.csv"
)
STAGING_INDEX    = "naive_events_q8"
DOCKER_CONTAINER = "dissertation_elasticsearch"
RESOURCE_SAMPLE_S = 1.0


# ── ES client ──────────────────────────────────────────────────────────────────

def get_client(n_threads: int) -> Elasticsearch:
    """
    Single shared thread-safe ES client.
    connections_per_node set to n_threads so every concurrent thread
    can hold an active HTTP connection without queuing.
    """
    return Elasticsearch(
        "http://localhost:9200",
        connections_per_node=n_threads,
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


# ── staging index setup ────────────────────────────────────────────────────────

STAGING_MAPPING = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,          # no replica overhead during write bench
        "refresh_interval":   "30s",      # reduce refresh cost — matches production
    },
    "mappings": {
        "dynamic": False,                 # explicit mapping — no auto-detection overhead
        "properties": {
            "user_id":      {"type": "keyword"},
            "event_type":   {"type": "keyword"},
            "product_id":   {"type": "keyword"},
            "session_id":   {"type": "keyword"},
            "metadata":     {"type": "object",  "enabled": False},  # no indexing of metadata
            "occurred_at":  {"type": "date"},
        },
    },
}


def setup_staging_index(client: Elasticsearch) -> None:
    """
    Create or reset the staging index.
    Deletes any existing naive_events_q8 to start with a clean slate
    (consistent with PostgreSQL's TRUNCATE TABLE events_q8).
    """
    if client.indices.exists(index=STAGING_INDEX):
        client.indices.delete(index=STAGING_INDEX)
        print(f"  Deleted existing index: {STAGING_INDEX}")
    client.indices.create(index=STAGING_INDEX, body=STAGING_MAPPING)
    print(f"  Created staging index: {STAGING_INDEX}")
    print(f"    shards=1, replicas=0, refresh_interval=30s, metadata not indexed")


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(csv_path: str, max_rows: int | None = None) -> list[dict]:
    """
    Load events_q8.csv into a list of document dicts ready for ES index().
    Empty strings are converted to None for nullable fields.
    max_rows limits load size for smoke tests.
    """
    print(f"  Loading CSV: {csv_path}")
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            doc = {
                "user_id":     row["user_id"]    or None,
                "event_type":  row["event_type"] or None,
                "product_id":  row["product_id"] or None,
                "session_id":  row["session_id"] or None,
                "metadata":    row["metadata"]   or "{}",
                "occurred_at": row["occurred_at"] or None,
            }
            # Remove None values — ES skips missing fields cleanly
            doc = {k: v for k, v in doc.items() if v is not None}
            docs.append((row["id"], doc))   # (document_id, body)

    print(f"  Loaded {len(docs):,} rows from CSV.")
    return docs


def slice_rows(rows: list, n_threads: int) -> list[list]:
    """Divide rows into n_threads equal slices (remainder distributed to early slices)."""
    size   = len(rows) // n_threads
    slices = []
    start  = 0
    for i in range(n_threads):
        end = start + size + (1 if i < len(rows) % n_threads else 0)
        slices.append(rows[start:end])
        start = end
    return slices


# ── statistics ─────────────────────────────────────────────────────────────────

def compute_stats(timings_ms: list[float]) -> dict:
    if not timings_ms:
        return {}
    s = sorted(timings_ms)
    n = len(s)

    def pct(p):
        k = max(0, min(math.ceil((p / 100) * n) - 1, n - 1))
        return round(s[k], 4)

    mean     = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "p50":     pct(50),
        "p95":     pct(95),
        "p99":     pct(99),
        "mean":    round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min":     round(s[0], 4),
        "max":     round(s[-1], 4),
    }


# ── Docker resource monitor ────────────────────────────────────────────────────

def start_resource_monitor(container: str, stop_event: threading.Event) -> list[dict]:
    """
    Background thread: sample Docker container CPU + memory every
    RESOURCE_SAMPLE_S seconds. Non-fatal if docker SDK is unavailable.
    """
    samples = []

    def _sample():
        try:
            import docker
            dc = docker.from_env()
            c  = dc.containers.get(container)
            while not stop_event.is_set():
                stats        = c.stats(stream=False)
                cpu_delta    = (stats["cpu_stats"]["cpu_usage"]["total_usage"]
                                - stats["precpu_stats"]["cpu_usage"]["total_usage"])
                system_delta = (stats["cpu_stats"]["system_cpu_usage"]
                                - stats["precpu_stats"]["system_cpu_usage"])
                n_cpus       = stats["cpu_stats"].get("online_cpus", 1)
                cpu_pct      = (cpu_delta / system_delta) * n_cpus * 100.0 if system_delta > 0 else 0.0
                mem_mb       = stats["memory_stats"].get("usage", 0) / (1024 * 1024)
                samples.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cpu_pct":   round(cpu_pct, 2),
                    "memory_mb": round(mem_mb, 2),
                })
                time.sleep(RESOURCE_SAMPLE_S)
        except Exception as e:
            samples.append({"error": str(e)})

    threading.Thread(target=_sample, daemon=True).start()
    return samples


# ── worker thread ──────────────────────────────────────────────────────────────

def worker(
    thread_id:        int,
    client:           Elasticsearch,       # shared thread-safe client
    rows:             list[tuple],         # (doc_id, doc_body) pairs
    all_timings:      list[float],
    timings_lock:     threading.Lock,
    progress_counter: list[int],
    progress_lock:    threading.Lock,
    barrier:          threading.Barrier,
    errors:           list[str],
):
    """
    Index every document in `rows` into STAGING_INDEX, one index() call
    at a time. Records per-document latency in milliseconds.

    All threads synchronise at the barrier before starting timed work —
    ensures genuine concurrent load rather than staggered writes.
    """
    local_timings = []

    barrier.wait()   # all threads start simultaneously

    try:
        for doc_id, doc_body in rows:
            t0 = time.perf_counter()
            client.index(index=STAGING_INDEX, id=doc_id, document=doc_body)
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            local_timings.append(elapsed_ms)
    except Exception as e:
        with timings_lock:
            errors.append(f"Thread {thread_id}: {e}")

    with timings_lock:
        all_timings.extend(local_timings)

    with progress_lock:
        progress_counter[0] += len(local_timings)


# ── progress printer ───────────────────────────────────────────────────────────

def progress_printer(total: int, counter: list[int], stop_event: threading.Event):
    """Print a progress line every 5 seconds until stop_event is set."""
    start = time.perf_counter()
    while not stop_event.is_set():
        time.sleep(5)
        done    = counter[0]
        elapsed = time.perf_counter() - start
        tps     = done / elapsed if elapsed > 0 else 0
        pct     = (done / total) * 100
        print(f"  [{elapsed:>6.0f}s] {done:>9,} / {total:,} "
              f"({pct:>5.1f}%)  {tps:>8,.0f} events/sec")


# ── main benchmark ─────────────────────────────────────────────────────────────

def run_q8(
    csv_path:       str,
    n_threads:      int,
    max_rows:       int | None,
    output_path:    str,
    resource_stats: bool,
):
    # ── client + staging index ─────────────────────────────────────────────────
    client = get_client(n_threads)
    try:
        info = client.info()
        print(f"  Connected — ES {info['version']['number']}")
    except Exception as e:
        print(f"  Cannot connect to Elasticsearch: {e}")
        sys.exit(1)

    setup_staging_index(client)

    # ── load + slice CSV ───────────────────────────────────────────────────────
    rows      = load_csv(csv_path, max_rows)
    total     = len(rows)
    n_threads = min(n_threads, total)
    slices    = slice_rows(rows, n_threads)
    del rows

    print(f"\n  Threads    : {n_threads}")
    print(f"  Total rows : {total:,}")
    print(f"  Rows/thread: {len(slices[0]):,} (±1 for remainder)")
    print(f"  Index      : {STAGING_INDEX}")

    # ── shared state ───────────────────────────────────────────────────────────
    all_timings      = []
    timings_lock     = threading.Lock()
    errors           = []
    progress_counter = [0]
    progress_lock    = threading.Lock()
    barrier          = threading.Barrier(n_threads)
    stop_progress    = threading.Event()
    stop_resources   = threading.Event()

    # ── resource monitor ───────────────────────────────────────────────────────
    resource_samples = []
    if resource_stats:
        resource_samples = start_resource_monitor(DOCKER_CONTAINER, stop_resources)
        print(f"  Resource monitor started (container: {DOCKER_CONTAINER})")
    else:
        print("  Resource monitoring disabled.")

    # ── progress printer ───────────────────────────────────────────────────────
    progress_thread = threading.Thread(
        target=progress_printer,
        args=(total, progress_counter, stop_progress),
        daemon=True,
    )
    progress_thread.start()

    # ── launch workers ─────────────────────────────────────────────────────────
    # No per-thread connection needed — single shared ES client is thread-safe.
    # This is the key structural difference from PostgreSQL Q8 (which opens
    # 100 separate psycopg2 connections). The ES urllib3 pool handles all
    # concurrent HTTP requests internally.
    print(f"\n  Starting benchmark at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}...")
    wall_start = time.perf_counter()

    threads = []
    for i, slice_ in enumerate(slices):
        t = threading.Thread(
            target=worker,
            args=(
                i, client, slice_, all_timings, timings_lock,
                progress_counter, progress_lock, barrier, errors,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_elapsed = time.perf_counter() - wall_start

    stop_progress.set()
    stop_resources.set()

    # ── results ────────────────────────────────────────────────────────────────
    actual_events = len(all_timings)
    throughput    = actual_events / wall_elapsed if wall_elapsed > 0 else 0
    stats         = compute_stats(all_timings)

    print(f"\n  {'═' * 46}")
    print(f"  Benchmark complete")
    print(f"  {'─' * 46}")
    print(f"  {'Events indexed':<28} {actual_events:>12,}")
    print(f"  {'Wall time':<28} {wall_elapsed:>11.2f}s")
    print(f"  {'Throughput':<28} {throughput:>10,.0f} events/sec")
    print(f"  {'p50 latency':<28} {stats.get('p50', 0):>11.2f} ms")
    print(f"  {'p95 latency':<28} {stats.get('p95', 0):>11.2f} ms")
    print(f"  {'p99 latency':<28} {stats.get('p99', 0):>11.2f} ms")
    print(f"  {'mean ± std':<28} "
          f"{stats.get('mean', 0):.2f} ± {stats.get('std_dev', 0):.2f} ms")
    print(f"  {'─' * 46}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    # ── save JSON ──────────────────────────────────────────────────────────────
    result = {
        "db":       "elasticsearch_naive",
        "query_id": "Q8",
        "label": (
            f"Concurrent event ingestion — {actual_events:,} single index() calls "
            f"across {n_threads} threads. Shared thread-safe ES client "
            f"(connections_per_node={n_threads}). "
            f"Documents written to {STAGING_INDEX} staging index "
            "(no FK constraints, refresh_interval=30s, replicas=0). "
            "CSV: events_q8.csv (pre-generated, fixed seed, same as all other DBs)."
        ),
        "timestamp":                 datetime.now(timezone.utc).isoformat(),
        "n_threads":                 n_threads,
        "total_events":              actual_events,
        "wall_time_s":               round(wall_elapsed, 3),
        "throughput_events_per_sec": round(throughput, 2),
        "errors":                    errors,
        "latency_ms":                stats,
        "resource_stats":            resource_samples,
        "raw_timings_ms":            [round(t, 4) for t in all_timings],
    }

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {output_path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Elasticsearch naive Q8 concurrent write benchmark"
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"Number of concurrent threads (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Limit total rows (default: all 1M). Use 10000 for smoke test.",
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV_PATH,
        help="Path to events_q8.csv",
    )
    parser.add_argument(
        "--no-resource-stats", action="store_true", dest="no_resource_stats",
        help="Disable Docker resource monitoring",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Index only 100 rows and print summary (no output file saved)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "results", "elasticsearch_naive_q8_write.json"
        ),
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Elasticsearch Naive — Q8 Concurrent Write Benchmark")
    print("=" * 50)

    run_q8(
        csv_path=args.csv,
        n_threads=args.threads,
        max_rows=100 if args.dry_run else args.rows,
        output_path=args.output if not args.dry_run else os.devnull,
        resource_stats=not args.no_resource_stats,
    )


if __name__ == "__main__":
    main()