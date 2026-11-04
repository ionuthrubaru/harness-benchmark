"""
benchmarks/timescaledb/q8_write.py — TimescaleDB Q8: Concurrent Event Ingestion
=================================================================================
Q8: 100 threads each insert individual events from events_q8.csv as fast
    as possible until all 1M rows are exhausted. One INSERT per event.

This benchmark is run once and serves both naive and optimised schemas —
Q8 has a single level per database (no naive/optimised split). See
dissertation_todo.md: micro-batching is an application-level pattern
change, not a schema-level change, and would muddy the methodology.

TimescaleDB write path vs PostgreSQL
──────────────────────────────────────
TimescaleDB intercepts INSERTs to hypertables and routes each row to the
correct time chunk based on the partition column (occurred_at). This is
transparent to the client — the INSERT syntax is identical to PostgreSQL.

The routing overhead is the key measurement: does TimescaleDB's chunk
dispatch add meaningful latency vs a plain PostgreSQL heap INSERT?

For a single-node instance with well-populated chunk cache:
  - Routing to an existing open chunk adds ~microseconds of overhead
  - Creating a new chunk (first INSERT into a new time window) adds
    ~10-50ms but happens at most once per chunk interval (7 days for
    the naive schema, 1 month for optimised)
  - At 1M events distributed across ~104 7-day chunks (~10k events/chunk)
    the chunk-creation overhead is amortised to near-zero

Staging table — events_q8 as a hypertable
───────────────────────────────────────────
The staging table is created as a hypertable on occurred_at. This is the
correct approach for a write benchmark on TimescaleDB — using a plain
table would measure plain PostgreSQL INSERT performance, not TimescaleDB's
write path. The hypertable chunk interval matches the naive schema (7 days).

Like the PostgreSQL staging table, events_q8 has no FK constraints —
mirroring the production pattern of async validation for high-throughput
event ingestion.

ON CONFLICT (id, occurred_at) DO NOTHING
──────────────────────────────────────────
The TimescaleDB hypertable PK is (id, occurred_at) — not just (id) — because
TimescaleDB requires the partition column in all unique constraints. The
ON CONFLICT clause must match: ON CONFLICT (id, occurred_at) DO NOTHING.

One connection per thread — identical to PostgreSQL Q8
────────────────────────────────────────────────────────
psycopg2 connections are not thread-safe. One connection per thread is
opened sequentially in the main thread before launching workers to avoid
the thundering herd problem. Identical pattern to PostgreSQL Q8.

Output JSON matches PostgreSQL Q8 format exactly for cross-DB comparison.

Usage:
    cd benchmarks/timescaledb
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
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_THREADS  = 100
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "events_q8.csv"
)
DOCKER_CONTAINER  = "dissertation_timescaledb"
RESOURCE_SAMPLE_S = 1.0

# ── connection ─────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"),
        connect_timeout=10,
    )

# ── staging table ──────────────────────────────────────────────────────────────
#
# Created as a TimescaleDB hypertable on occurred_at — measures the actual
# TimescaleDB write path including chunk routing. Using a plain table would
# bypass the hypertable overhead and not represent TimescaleDB's write cost.
#
# Chunk interval: 7 days (naive schema default). The optimised schema uses
# 1-month chunks but Q8 is a single-level benchmark — naive chunk interval
# is used for consistency with the naive data schema being benchmarked.

CREATE_STAGING_TABLE = """
CREATE TABLE IF NOT EXISTS events_q8 (
    id              UUID            NOT NULL,
    user_id         UUID            NOT NULL,
    event_type      VARCHAR(50)     NOT NULL,
    product_id      UUID,
    session_id      VARCHAR(64),
    metadata        JSONB           NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ     NOT NULL,
    PRIMARY KEY (id, occurred_at)
);
"""

CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'events_q8',
    'occurred_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);
"""

# ON CONFLICT must reference (id, occurred_at) — the full composite PK.
# TimescaleDB requires the partition column in all unique constraints,
# so ON CONFLICT (id) would fail as there is no unique index on id alone.
INSERT_SQL = """
INSERT INTO events_q8 (id, user_id, event_type, product_id, session_id, metadata, occurred_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id, occurred_at) DO NOTHING;
"""

# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(csv_path: str, max_rows: int | None = None) -> list[tuple]:
    print(f"  Loading CSV: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append((
                row["id"]          or None,
                row["user_id"]     or None,
                row["event_type"]  or None,
                row["product_id"]  or None,
                row["session_id"]  or None,
                row["metadata"]    or "{}",
                row["occurred_at"] or None,
            ))
    print(f"  Loaded {len(rows):,} rows from CSV.")
    return rows


def slice_rows(rows: list[tuple], n_threads: int) -> list[list[tuple]]:
    size   = len(rows) // n_threads
    slices = []
    start  = 0
    for i in range(n_threads):
        end = start + size + (1 if i < len(rows) % n_threads else 0)
        slices.append(rows[start:end])
        start = end
    return slices

# ── Docker resource stats ──────────────────────────────────────────────────────

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
        "p50": pct(50), "p95": pct(95), "p99": pct(99),
        "mean": round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min": round(s[0], 4), "max": round(s[-1], 4),
    }

# ── worker thread ──────────────────────────────────────────────────────────────

def worker(
    thread_id:        int,
    conn,
    rows:             list[tuple],
    all_timings:      list[float],
    timings_lock:     threading.Lock,
    progress_counter: list[int],
    progress_lock:    threading.Lock,
    barrier:          threading.Barrier,
    errors:           list[str],
):
    """
    Insert every row in `rows` one INSERT at a time.
    One connection per thread — identical to PostgreSQL Q8.
    barrier.wait() synchronises all threads to start simultaneously.
    """
    barrier.wait()

    if conn is None:
        return

    local_timings = []
    try:
        cur = conn.cursor()
        for row in rows:
            t0 = time.perf_counter()
            cur.execute(INSERT_SQL, row)
            local_timings.append((time.perf_counter() - t0) * 1_000)
        cur.close()
    except Exception as e:
        with timings_lock:
            errors.append(f"Thread {thread_id}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    with timings_lock:
        all_timings.extend(local_timings)
    with progress_lock:
        progress_counter[0] += len(local_timings)

# ── progress printer ───────────────────────────────────────────────────────────

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

# ── main benchmark ─────────────────────────────────────────────────────────────

def run_q8(
    csv_path:       str,
    n_threads:      int,
    max_rows:       int | None,
    output_path:    str,
    resource_stats: bool,
):
    # ── setup staging table ────────────────────────────────────────────────────
    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_STAGING_TABLE)
        cur.execute(CREATE_HYPERTABLE)
        cur.execute("TRUNCATE TABLE events_q8;")
    conn.close()
    print("  Staging table events_q8 ready (hypertable, truncated).")

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

    # ── open connections sequentially ─────────────────────────────────────────
    # Avoids thundering herd on TimescaleDB's connection accept queue —
    # identical rationale to PostgreSQL Q8.
    print(f"  Opening {n_threads} connections sequentially...")
    connections = []
    for i in range(n_threads):
        try:
            c = get_connection()
            c.autocommit = True
            connections.append(c)
        except Exception as e:
            errors.append(f"Thread {i} (connection failed): {e}")
            connections.append(None)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_threads} connections opened...")
    opened = sum(1 for c in connections if c is not None)
    print(f"  {opened}/{n_threads} connections ready.")

    if opened == 0:
        print("  No connections could be opened — aborting.")
        return

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
    print(f"\n  Starting benchmark at "
          f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}...")
    wall_start = time.perf_counter()

    threads = []
    for i, (slice_, conn) in enumerate(zip(slices, connections)):
        t = threading.Thread(
            target=worker,
            args=(
                i, conn, slice_, all_timings, timings_lock,
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
        "db":       "timescaledb",
        "query_id": "Q8",
        "label": (
            "Concurrent event ingestion — 1M single INSERTs across "
            f"{n_threads} threads into events_q8 hypertable (7-day chunks). "
            "One connection per thread (opened sequentially, autocommit=True). "
            "TimescaleDB routes each INSERT to the correct time chunk via the "
            "occurred_at partition column. ON CONFLICT (id, occurred_at) DO NOTHING "
            "(hypertable PK must include partition column). "
            "Measures TimescaleDB hypertable write path overhead vs plain PostgreSQL. "
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

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TimescaleDB Q8 concurrent write benchmark"
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
        default=os.path.join("benchmarks", "timescaleDB", "timescaledb_Q8.json"),
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB — Q8 Concurrent Write Benchmark")
    print("=" * 60)
    print(f"  Target      : events_q8 hypertable (7-day chunks)")
    print(f"  Connections : one per thread (sequential open, autocommit)")
    print(f"  Write path  : TimescaleDB hypertable INSERT with chunk routing")

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