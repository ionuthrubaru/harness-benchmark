"""
benchmarks/mongodb/naive/q8_write.py — MongoDB Naive: Q8
=========================================================
Q8: Concurrent event ingestion benchmark.

    100 threads each insert individual events from the pre-generated
    events_q8.csv as fast as possible until all 1M rows are exhausted.
    One insert_one() call per event — mirrors the PostgreSQL baseline
    exactly so the only variable is the database engine.

Design decisions (mirrors PostgreSQL baseline)
──────────────────────────────────────────────
  - Single insert_one() per event — not bulk_write().
  - CSV pre-sliced at startup — 1M rows divided into 100 equal slices.
  - Each thread creates its own MongoClient with maxPoolSize=1.
    The URI is built once in the main thread and passed to workers —
    avoids load_dotenv() re-read issues inside threads.
  - write_concern=1 (acknowledged write) — equivalent to autocommit=True.
  - Inserts go into events_q8 collection (no live data pollution).
  - Progress counter updated per-insert inside the loop (not at end).
  - Docker stats sampled every second in a background thread.

Usage:
    python q8_write.py                        # full 1M event benchmark
    python q8_write.py --rows 10000           # smoke test with 10K rows
    python q8_write.py --threads 10           # reduce concurrency
    python q8_write.py --no-resource-stats    # skip Docker stats
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

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_THREADS   = 100
DEFAULT_CSV_PATH  = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "events_q8.csv"
)
DOCKER_CONTAINER  = "dissertation_mongodb"
RESOURCE_SAMPLE_S = 1.0
COLLECTION_NAME   = "events_q8"

# ── URI built once in main thread ─────────────────────────────────────────────

def build_uri() -> tuple[str, str]:
    """Return (mongo_uri, db_name) — resolved from env before any threads start."""
    user     = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    db_name  = os.getenv("MONGO_DB_NAIVE")
    if not all([user, password, db_name]):
        raise RuntimeError("MONGO_USER / MONGO_PASSWORD / MONGO_DB_NAIVE not set in .env")
    return f"mongodb://{user}:{password}@localhost:27017/", db_name

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str, max_rows: int | None = None) -> list[dict]:
    print(f"  Loading CSV: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append({
                "_id":         row["id"]          or None,
                "user_id":     row["user_id"]     or None,
                "event_type":  row["event_type"]  or None,
                "product_id":  row["product_id"]  or None,
                "session_id":  row["session_id"]  or None,
                "metadata":    row["metadata"]    or "{}",
                "occurred_at": row["occurred_at"] or None,
            })
    print(f"  Loaded {len(rows):,} rows from CSV.")
    return rows


def slice_rows(rows: list[dict], n_threads: int) -> list[list[dict]]:
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
    mongo_uri: str,
    db_name: str,
    rows: list[dict],
    all_timings: list[float],
    timings_lock: threading.Lock,
    progress_counter: list[int],
    progress_lock: threading.Lock,
    barrier: threading.Barrier,
    errors: list[str],
):
    """
    Each thread creates its own MongoClient using the URI resolved in the
    main thread — avoids load_dotenv() re-read issues inside threads.
    maxPoolSize=1 gives each thread exactly one connection.
    Progress counter updated per-insert so progress printer works correctly.
    """
    local_timings = []

    barrier.wait()

    try:
        from pymongo import MongoClient
        client = MongoClient(
            mongo_uri,
            maxPoolSize=1,
            w=1,
            serverSelectionTimeoutMS=30_000,
        )
        col = client[db_name][COLLECTION_NAME]

        for doc in rows:
            t0 = time.perf_counter()
            col.insert_one(doc)
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            local_timings.append(elapsed_ms)
            with progress_lock:
                progress_counter[0] += 1

        client.close()

    except Exception as e:
        with timings_lock:
            errors.append(f"Thread {thread_id}: {e}")

    with timings_lock:
        all_timings.extend(local_timings)

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
    csv_path: str,
    n_threads: int,
    max_rows: int | None,
    output_path: str,
    resource_stats: bool,
):
    mongo_uri, db_name = build_uri()

    # ── setup: drop + recreate collection ─────────────────────────────────────
    from pymongo import MongoClient
    setup_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30_000)
    db = setup_client[db_name]
    db.drop_collection(COLLECTION_NAME)
    db.create_collection(COLLECTION_NAME)
    setup_client.close()
    print(f"  Collection '{COLLECTION_NAME}' dropped and recreated (clean slate).")

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

    resource_samples = []
    if resource_stats:
        resource_samples = start_resource_monitor(DOCKER_CONTAINER, stop_resources)
        print(f"  Resource monitor started (container: {DOCKER_CONTAINER})")
    else:
        print("  Resource monitoring disabled.")

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
    for i, slice_ in enumerate(slices):
        t = threading.Thread(
            target=worker,
            args=(
                i, mongo_uri, db_name, slice_,
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

    if output_path == os.devnull:
        return

    result = {
        "db":                        "mongodb_naive",
        "query_id":                  "Q8",
        "label": (
            "Concurrent event ingestion — single insert_one() per event "
            f"across {n_threads} threads. Each thread creates its own "
            "MongoClient (maxPoolSize=1). write_concern=1 (acknowledged). "
            "Inserts into events_q8 collection (no FK constraints). "
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
        description="MongoDB Naive Q8 — concurrent write benchmark"
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--no-resource-stats", action="store_true", dest="no_resource_stats")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "mongodb_Q8.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MongoDB Naive — Q8 Concurrent Write Benchmark")
    print("=" * 55)

    run_q8(
        csv_path=args.csv,
        n_threads=args.threads,
        max_rows=100 if args.dry_run else args.rows,
        output_path=args.output if not args.dry_run else os.devnull,
        resource_stats=not args.no_resource_stats,
    )


if __name__ == "__main__":
    main()