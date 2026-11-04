"""
benchmarks/postgres/q8_write.py — PostgreSQL Baseline: Q8
==========================================================
Q8: Concurrent event ingestion benchmark.

    100 threads each insert individual events from a pre-generated CSV
    as fast as possible until all 1M rows are exhausted.
    Each thread executes one INSERT per event — the realistic production
    pattern for an event-driven application.

Dissertation angle:
    Does specialisation confer a write advantage under realistic
    production conditions? This benchmark isolates the engine's write
    path under genuine concurrent load. All specialised DBs in Phase 3
    and 4 will use the identical CSV and the same 100-thread pattern,
    so the only variable is the database engine.

Design decisions
────────────────
  - Single INSERT per event (not batch) — mirrors how a real application
    writes events: one user action, one immediate write. Batching would
    measure a data migration pattern, not a production write path.
  - CSV pre-sliced at startup — the 1M rows are divided into 100 equal
    slices of 10K rows. Each thread owns its slice and works through it
    sequentially. When all threads finish their slice the benchmark ends.
    No shared counter needed — CSV exhaustion is the natural stop condition.
  - One connection per thread — consistent with Q3 and with how a real
    application connection pool allocates connections.
  - Docker stats sampled every second in a background thread — CPU and
    memory are recorded as a timeseries alongside the latency data.
  - Per-insert latency recorded for every one of the 1M inserts —
    the full distribution is stored in the output JSON. At 1M floats
    this is ~8MB on disk, which is acceptable for dissertation data.

Output JSON includes:
    total_events, total_time_s, throughput_events_per_sec,
    latency_ms (p50/p95/p99/mean/std_dev/min/max),
    resource_stats (cpu_pct + memory_mb timeseries),
    raw_timings_ms (full 1M latency list)

Usage:
    python q8_write.py                        # full 1M event benchmark
    python q8_write.py --rows 10000           # smoke test with 10K rows
    python q8_write.py --threads 10           # reduce concurrency
    python q8_write.py --no-resource-stats    # skip Docker stats (faster)
    python q8_write.py --dry-run              # insert 100 rows, print summary
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

DEFAULT_THREADS     = 100
DEFAULT_CSV_PATH    = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "events_q8.csv"
)
DOCKER_CONTAINER    = "dissertation_postgres"
RESOURCE_SAMPLE_S   = 1.0      # Docker stats sample interval in seconds

# ── connection ────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
        connect_timeout=10,
    )

# ── INSERT statement ──────────────────────────────────────────────────────────
#
# We INSERT into a staging table (events_q8) rather than the live events
# table so the benchmark does not pollute the data used by Q1-Q7 reads.
# The staging table has the same schema as events but without the FK
# constraints on user_id / product_id / session_id — this mirrors what
# a real high-throughput event ingestion service would do (write fast,
# validate async). The table is created below if it does not exist.

CREATE_STAGING_TABLE = """
CREATE TABLE IF NOT EXISTS events_q8 (
    id              UUID            PRIMARY KEY,
    user_id         UUID            NOT NULL,
    event_type      VARCHAR(50)     NOT NULL,
    product_id      UUID,
    session_id      VARCHAR(64),
    metadata        JSONB           NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ     NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO events_q8 (id, user_id, event_type, product_id, session_id, metadata, occurred_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str, max_rows: int | None = None) -> list[tuple]:
    """
    Load the events CSV into a list of row tuples.
    Converts empty strings to None for nullable columns (product_id, session_id).
    Limits to max_rows if specified (for smoke tests).
    """
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
                row["product_id"]  or None,   # NULL for non-product events
                row["session_id"]  or None,   # NULL for some event types
                row["metadata"]    or "{}",
                row["occurred_at"] or None,
            ))
    print(f"  Loaded {len(rows):,} rows from CSV.")
    return rows


def slice_rows(rows: list[tuple], n_threads: int) -> list[list[tuple]]:
    """
    Divide rows into n_threads equal slices.
    Any remainder rows are distributed to the first slices.
    """
    size   = len(rows) // n_threads
    slices = []
    start  = 0
    for i in range(n_threads):
        # Distribute remainder rows one-by-one to early slices
        end = start + size + (1 if i < len(rows) % n_threads else 0)
        slices.append(rows[start:end])
        start = end
    return slices

# ── Docker resource stats ─────────────────────────────────────────────────────

def start_resource_monitor(container: str, stop_event: threading.Event) -> list[dict]:
    """
    Start a background thread that samples Docker container CPU and memory
    every RESOURCE_SAMPLE_S seconds until stop_event is set.
    Returns the shared list that the background thread appends to —
    the caller reads it after stop_event is set.
    """
    samples = []

    def _sample():
        try:
            import docker
            client = docker.from_env()
            c = client.containers.get(container)
            while not stop_event.is_set():
                stats = c.stats(stream=False)
                # CPU % calculation (same formula Docker CLI uses)
                cpu_delta    = (stats["cpu_stats"]["cpu_usage"]["total_usage"]
                                - stats["precpu_stats"]["cpu_usage"]["total_usage"])
                system_delta = (stats["cpu_stats"]["system_cpu_usage"]
                                - stats["precpu_stats"]["system_cpu_usage"])
                n_cpus       = stats["cpu_stats"].get("online_cpus", 1)
                cpu_pct      = (cpu_delta / system_delta) * n_cpus * 100.0 if system_delta > 0 else 0.0

                # Memory in MB
                mem_usage = stats["memory_stats"].get("usage", 0)
                mem_mb    = mem_usage / (1024 * 1024)

                samples.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cpu_pct":   round(cpu_pct, 2),
                    "memory_mb": round(mem_mb, 2),
                })
                time.sleep(RESOURCE_SAMPLE_S)
        except Exception as e:
            # Non-fatal — benchmark continues without resource stats
            samples.append({"error": str(e)})

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
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
        "p50":     pct(50),
        "p95":     pct(95),
        "p99":     pct(99),
        "mean":    round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min":     round(s[0], 4),
        "max":     round(s[-1], 4),
    }

# ── worker thread ─────────────────────────────────────────────────────────────

def worker(
    thread_id: int,
    conn,                          # pre-opened connection passed from main thread
    rows: list[tuple],
    all_timings: list[float],
    timings_lock: threading.Lock,
    progress_counter: list[int],   # single-element list used as mutable int
    progress_lock: threading.Lock,
    barrier: threading.Barrier,
    errors: list[str],
):
    """
    Insert every row in `rows` into events_q8, one INSERT at a time.
    Records per-insert latency in milliseconds.

    Connections are opened sequentially in the main thread before any
    worker starts — this avoids the thundering herd problem where 100
    threads all connect simultaneously and overwhelm PostgreSQL's
    connection acceptance queue.

    barrier.wait() is called unconditionally so a thread with a bad
    connection never deadlocks the others.
    """
    local_timings = []

    # Always reach the barrier regardless of connection state
    barrier.wait()

    if conn is None:
        return

    try:
        cur = conn.cursor()
        for row in rows:
            t0 = time.perf_counter()
            cur.execute(INSERT_SQL, row)
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            local_timings.append(elapsed_ms)
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

# ── progress printer ──────────────────────────────────────────────────────────

def progress_printer(
    total: int,
    counter: list[int],
    stop_event: threading.Event,
):
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

# ── main benchmark ────────────────────────────────────────────────────────────

def run_q8(
    csv_path: str,
    n_threads: int,
    max_rows: int | None,
    output_path: str,
    resource_stats: bool,
):
    # ── setup ──────────────────────────────────────────────────────────────────
    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_STAGING_TABLE)
        cur.execute("TRUNCATE TABLE events_q8;")   # clean slate each run
    conn.close()
    print("  Staging table events_q8 ready (truncated).")

    # ── load + slice CSV ───────────────────────────────────────────────────────
    rows      = load_csv(csv_path, max_rows)
    total     = len(rows)
    # Cap threads to row count — no point spinning up 100 threads for 100 rows
    n_threads = min(n_threads, total)
    slices    = slice_rows(rows, n_threads)
    del rows   # free memory — slices hold references

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

    # ── open connections sequentially ────────────────────────────────────────
    # Opening all connections in the main thread before launching workers
    # avoids the thundering herd problem: 100 simultaneous psycopg2 connect()
    # calls overwhelm PostgreSQL's connection acceptance queue and cause
    # "server closed the connection unexpectedly" errors even when
    # max_connections is high enough.
    print(f"  Opening {n_threads} connections sequentially...")
    connections = []
    for i in range(n_threads):
        try:
            c = get_connection()
            c.autocommit = True
            connections.append(c)
        except Exception as e:
            errors.append(f"Thread {i} (connection failed): {e}")
            connections.append(None)   # placeholder keeps index alignment
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_threads} connections opened...")
    opened = sum(1 for c in connections if c is not None)
    print(f"  {opened}/{n_threads} connections ready.")

    if opened == 0:
        print("  No connections could be opened — aborting.")
        return

    # ── start resource monitor ─────────────────────────────────────────────────
    # Started AFTER connections are open so its output doesn't interleave
    # with the connection progress messages.
    resource_samples = []
    if resource_stats:
        resource_samples = start_resource_monitor(DOCKER_CONTAINER, stop_resources)
        print(f"  Resource monitor started (container: {DOCKER_CONTAINER})")
    else:
        print("  Resource monitoring disabled.")

    # ── start progress printer ─────────────────────────────────────────────────
    # Started AFTER connections are open — progress only reflects actual inserts.
    progress_thread = threading.Thread(
        target=progress_printer,
        args=(total, progress_counter, stop_progress),
        daemon=True,
    )
    progress_thread.start()

    # ── launch workers ─────────────────────────────────────────────────────────
    print(f"\n  Starting benchmark at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}...")
    wall_start = time.perf_counter()

    threads = []
    for i, (slice_, conn) in enumerate(zip(slices, connections)):
        t = threading.Thread(
            target=worker,
            args=(
                i, conn, slice_, all_timings, timings_lock,
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

    # ── stop background threads ────────────────────────────────────────────────
    stop_progress.set()
    stop_resources.set()

    # ── results ────────────────────────────────────────────────────────────────
    actual_events = len(all_timings)
    throughput    = actual_events / wall_elapsed if wall_elapsed > 0 else 0
    stats         = compute_stats(all_timings)

    print(f"\n  {'═' * 46}")
    print(f"  {'Benchmark complete':}")
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
        "db":                    "postgres",
        "query_id":              "Q8",
        "label":                 (
            "Concurrent event ingestion — 1M single INSERTs across "
            f"{n_threads} threads. One connection per thread. "
            "autocommit=True (each INSERT commits immediately). "
            "Inserts into events_q8 staging table (no FK constraints). "
            "CSV: events_q8.csv (pre-generated, fixed seed)."
        ),
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "n_threads":             n_threads,
        "total_events":          actual_events,
        "wall_time_s":           round(wall_elapsed, 3),
        "throughput_events_per_sec": round(throughput, 2),
        "errors":                errors,
        "latency_ms":            stats,
        "resource_stats":        resource_samples,
        "raw_timings_ms":        [round(t, 4) for t in all_timings],
    }

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {output_path}")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Q8 concurrent write benchmark"
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
        help="Path to events_q8.csv (default: ../../data/events_q8.csv)",
    )
    parser.add_argument(
        "--no-resource-stats", action="store_true", dest="no_resource_stats",
        help="Disable Docker resource monitoring (useful if docker SDK not installed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Insert only 100 rows and print summary (no output file saved)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "postgres_q8_write_baseline.json"),
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  PostgreSQL — Q8 Concurrent Write Benchmark")
    print("=" * 50)

    max_rows = 100 if args.dry_run else args.rows

    run_q8(
        csv_path=args.csv,
        n_threads=args.threads,
        max_rows=max_rows,
        output_path=args.output if not args.dry_run else os.devnull,
        resource_stats=not args.no_resource_stats,
    )


if __name__ == "__main__":
    main()