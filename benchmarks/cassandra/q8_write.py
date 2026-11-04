"""
benchmarks/cassandra/q8_write.py — Cassandra Q8: Concurrent Event Ingestion
=============================================================================
Q8: 100 threads each insert individual events from events_q8.csv as fast
    as possible until all 1M rows are exhausted. One INSERT per event.

This benchmark is run once against the naive keyspace. There is no
optimised variant — Q8 has a single level per database. See
dissertation_todo.md: micro-batching is an application-level pattern
change, not a schema-level change, and would muddy the methodology.

Cassandra driver model vs PostgreSQL/Neo4j
───────────────────────────────────────────
PostgreSQL Q8: one psycopg2 connection per thread (driver is not
    thread-safe at the connection level).
Neo4j Q8: one driver Session per thread (session is not thread-safe).
Cassandra Q8: one Session shared across all threads.
    The Cassandra Python driver Session is fully thread-safe. It manages
    an internal connection pool (one TCP connection per host by default,
    but the pool handles concurrent requests via async I/O internally).
    There is no benefit to creating one Session per thread — doing so
    would create 100 independent connection pools and saturate the server.
    The correct Cassandra pattern is one Session for the process, shared
    across all threads. This is the driver's intended use and the pattern
    recommended by DataStax.

Write target — events table in cassandra_naive keyspace
─────────────────────────────────────────────────────────
Events are inserted into the cassandra_naive.events table (the same
table used by naive Q6). This table has id as sole partition key,
matching the production-style write pattern: one event = one partition.
A separate staging table is not needed because Cassandra has no FK
constraints — inserts into events never fail due to missing user_id or
product_id references.

The events_q8.csv contains pre-generated UUIDs. A small number of
duplicate UUIDs across the naive events table and events_q8.csv is
possible but statistically negligible (UUID4 collision probability at
these row counts is ~10^-22). Cassandra's last-write-wins semantics mean
duplicates are silently overwritten with identical data — no error.

autocommit / transaction model
────────────────────────────────
Cassandra has no multi-row transactions and no concept of autocommit.
Every INSERT is immediately durable once the coordinator receives the
required number of replica acknowledgements (LOCAL_ONE here = 1 replica).
This is equivalent to autocommit=True in PostgreSQL Q8 and matches the
"one user action, one immediate write" production pattern.

Prepared statement
───────────────────
The INSERT is prepared once at startup and shared across all threads.
PreparedStatement objects are thread-safe in the Cassandra driver.
Preparation sends the statement to the server once for parsing and
caching; subsequent executions send only the bound values, reducing
per-insert overhead. This is the correct production pattern and mirrors
the parameterised queries used in PostgreSQL Q8.

Output JSON:
    db, query_id, label, timestamp, n_threads, total_events,
    wall_time_s, throughput_events_per_sec, errors,
    latency_ms (p50/p95/p99/mean/std_dev/min/max),
    resource_stats (cpu_pct + memory_mb timeseries),
    raw_timings_ms (full per-insert latency list)

Usage:
    cd benchmarks/cassandra
    python q8_write.py                        # full 1M event benchmark
    python q8_write.py --rows 10000           # smoke test with 10K rows
    python q8_write.py --threads 10           # reduce concurrency
    python q8_write.py --no-resource-stats    # skip Docker stats
    python q8_write.py --dry-run              # 100 rows, print summary only
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, os.path.dirname(__file__))
from cassandra_conn import get_session

load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_THREADS   = 100
DEFAULT_CSV_PATH  = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "events_q8.csv"
)
DOCKER_CONTAINER  = "dissertation_cassandra"
RESOURCE_SAMPLE_S = 1.0
KEYSPACE          = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")

# ── INSERT statement ──────────────────────────────────────────────────────────
#
# Inserts directly into cassandra_naive.events — the same table as naive Q6.
# No staging table needed: Cassandra has no FK constraints so there is no
# referential integrity to violate. The naive events table schema:
#   CREATE TABLE events (
#       id uuid PRIMARY KEY, user_id uuid, event_type text,
#       product_id uuid, session_id text, metadata text, occurred_at timestamp
#   )
# product_id and session_id are nullable — absent in most events_q8.csv rows.

INSERT_CQL = """
INSERT INTO events (id, user_id, event_type, product_id, session_id, metadata, occurred_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str, max_rows: int | None = None) -> list[tuple]:
    """
    Load events_q8.csv into a list of tuples with correct Python types.
    UUIDs are parsed to uuid.UUID objects — the Cassandra driver requires
    native uuid.UUID, not strings, for uuid columns.
    Timestamps are parsed to timezone-aware datetime objects.
    Empty product_id and session_id become None.
    """
    print(f"  Loading CSV: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            pid = row["product_id"].strip()
            sid = row["session_id"].strip()
            rows.append((
                uuid.UUID(row["id"]),
                uuid.UUID(row["user_id"]),
                row["event_type"] or None,
                uuid.UUID(pid) if pid else None,
                sid if sid else None,
                row["metadata"] or "{}",
                datetime.fromisoformat(row["occurred_at"].replace("Z", "+00:00")),
            ))
    print(f"  Loaded {len(rows):,} rows from CSV.")
    return rows


def slice_rows(rows: list[tuple], n_threads: int) -> list[list[tuple]]:
    """Divide rows into n_threads equal slices, remainder distributed to early slices."""
    size   = len(rows) // n_threads
    slices = []
    start  = 0
    for i in range(n_threads):
        end = start + size + (1 if i < len(rows) % n_threads else 0)
        slices.append(rows[start:end])
        start = end
    return slices

# ── Docker resource stats ─────────────────────────────────────────────────────

def start_resource_monitor(container: str, stop_event: threading.Event) -> list[dict]:
    samples = []

    def _sample():
        try:
            import docker
            client = docker.from_env()
            c = client.containers.get(container)
            while not stop_event.is_set():
                stats    = c.stats(stream=False)
                cpu_d    = (stats["cpu_stats"]["cpu_usage"]["total_usage"]
                            - stats["precpu_stats"]["cpu_usage"]["total_usage"])
                sys_d    = (stats["cpu_stats"]["system_cpu_usage"]
                            - stats["precpu_stats"]["system_cpu_usage"])
                n_cpus   = stats["cpu_stats"].get("online_cpus", 1)
                cpu_pct  = (cpu_d / sys_d) * n_cpus * 100.0 if sys_d > 0 else 0.0
                mem_mb   = stats["memory_stats"].get("usage", 0) / (1024 * 1024)
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

# ── statistics ────────────────────────────────────────────────────────────────

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
        "p50":     pct(50),  "p95":     pct(95),  "p99":     pct(99),
        "mean":    round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min":     round(s[0], 4), "max": round(s[-1], 4),
    }

# ── worker thread ─────────────────────────────────────────────────────────────

def worker(
    thread_id:        int,
    session,                        # shared thread-safe Cassandra Session
    prepared,                       # shared PreparedStatement (thread-safe)
    rows:             list[tuple],
    all_timings:      list[float],
    timings_lock:     threading.Lock,
    progress_counter: list[int],
    progress_lock:    threading.Lock,
    barrier:          threading.Barrier,
    errors:           list[str],
):
    """
    Insert every row in `rows`, one INSERT at a time.
    Records per-insert latency in milliseconds.

    Unlike PostgreSQL Q8 (one connection per thread) and Neo4j Q8
    (one session per thread), all Cassandra threads share the same
    Session object. The driver's internal connection pool handles
    concurrent execute() calls from multiple threads transparently.
    No per-thread setup is needed.

    barrier.wait() synchronises all threads to start simultaneously,
    ensuring genuine concurrent write pressure on the server.
    """
    # Wait at barrier — all threads start together
    barrier.wait()

    local_timings = []
    try:
        for row in rows:
            t0 = time.perf_counter()
            session.execute(prepared, row)
            local_timings.append((time.perf_counter() - t0) * 1_000)
    except Exception as e:
        with timings_lock:
            errors.append(f"Thread {thread_id}: {e}")

    with timings_lock:
        all_timings.extend(local_timings)
    with progress_lock:
        progress_counter[0] += len(local_timings)

# ── progress printer ──────────────────────────────────────────────────────────

def progress_printer(total: int, counter: list[int], stop_event: threading.Event):
    start = time.perf_counter()
    while not stop_event.is_set():
        time.sleep(5)
        done    = counter[0]
        elapsed = time.perf_counter() - start
        tps     = done / elapsed if elapsed > 0 else 0
        pct     = (done / total) * 100
        print(f"  [{elapsed:>6.0f}s] {done:>9,} / {total:,} "
              f"({pct:>5.1f}%)  {tps:>8,.0f} events/sec")

# ── main benchmark ────────────────────────────────────────────────────────────

def run_q8(
    csv_path:       str,
    n_threads:      int,
    max_rows:       int | None,
    output_path:    str,
    resource_stats: bool,
):
    # ── connect + prepare ──────────────────────────────────────────────────────
    # Higher request_timeout for Q8: individual inserts are fast, but under
    # 100-thread load the coordinator may queue briefly. 30s is ample.
    cluster, session = get_session(keyspace=KEYSPACE, request_timeout=30.0)
    prepared = session.prepare(INSERT_CQL)
    print(f"  Keyspace : {KEYSPACE}")
    print(f"  Table    : events")
    print(f"  Prepared INSERT statement ready.")

    # ── load + slice CSV ───────────────────────────────────────────────────────
    rows      = load_csv(csv_path, max_rows)
    total     = len(rows)
    n_threads = min(n_threads, total)
    slices    = slice_rows(rows, n_threads)
    del rows

    print(f"\n  Threads    : {n_threads}")
    print(f"  Total rows : {total:,}")
    print(f"  Rows/thread: {len(slices[0]):,} (±1 for remainder)")

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
    # No per-thread connection setup needed — all threads share one Session.
    # Threads are launched directly, unlike PostgreSQL Q8 which opens
    # connections sequentially first to avoid thundering herd on the DB.
    # Cassandra's driver pool handles concurrent requests internally.
    print(f"\n  Starting benchmark at "
          f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}...")
    wall_start = time.perf_counter()

    threads = []
    for i, slice_ in enumerate(slices):
        t = threading.Thread(
            target=worker,
            args=(
                i, session, prepared, slice_,
                all_timings, timings_lock,
                progress_counter, progress_lock,
                barrier, errors,
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
    print(f"  {'Events inserted':<28} {actual_events:>12,}")
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
        "db":       "cassandra",
        "query_id": "Q8",
        "label": (
            "Concurrent event ingestion — 1M single INSERTs across "
            f"{n_threads} threads into cassandra_naive.events. "
            "One shared thread-safe Session (Cassandra driver model). "
            "Prepared statement shared across threads. "
            "No per-thread connection setup — driver pool handles concurrency. "
            "autocommit equivalent: every INSERT is immediately durable "
            "at LOCAL_ONE consistency. No FK constraints — no staging table needed. "
            "CSV: events_q8.csv (pre-generated, fixed seed)."
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

    if output_path and output_path != os.devnull:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved → {output_path}")

    cluster.shutdown()

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cassandra Q8 concurrent write benchmark"
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
        help="Insert 100 rows and print summary — no output file saved",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_Q8.json"),
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra — Q8 Concurrent Write Benchmark")
    print("=" * 60)
    print(f"  Keyspace    : {KEYSPACE}")
    print(f"  Target      : events table (naive schema)")
    print(f"  Driver model: shared Session across all threads")
    print(f"  Consistency : LOCAL_ONE (immediate durability, single replica)")

    max_rows = 100 if args.dry_run else args.rows

    run_q8(
        csv_path=args.csv,
        n_threads=args.threads,
        max_rows=max_rows,
        output_path=args.output if not args.dry_run else "",
        resource_stats=not args.no_resource_stats,
    )


if __name__ == "__main__":
    main()