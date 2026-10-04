"""
benchmarks/harness.py — Benchmarking Harness
=============================================
Core utility used by every benchmark in Phases 2–4.

Responsibilities:
  - Run a query function N times and record per-run latency in milliseconds
  - Discard the first 10 runs as warm-up
  - Compute p50, p95, p99, mean, std dev, min, max
  - Support concurrent execution via threading (for Q3, Q8, etc.)
  - Save results to JSON with full metadata and raw timings

Usage (single-threaded):
    from harness import run_benchmark

    def my_query():
        # execute your query here
        pass

    run_benchmark(
        query_fn=my_query,
        db="postgres",
        query_id="Q1",
        iterations=1000,
        concurrency=1,
        output_path="results/postgres_q1_baseline.json",
    )

Usage (concurrent):
    run_benchmark(
        query_fn=my_query,
        db="redis",
        query_id="Q3",
        iterations=1000,
        concurrency=50,                # 50 concurrent threads
        output_path="results/redis_q3_baseline.json",
    )
"""

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable

# ── constants ─────────────────────────────────────────────────────────────────

WARMUP_RUNS = 10          # runs discarded before timing begins
DEFAULT_ITERATIONS = 1000 # measured runs (excludes warm-up)

# ── statistics helpers ────────────────────────────────────────────────────────

def _percentile(sorted_data: list[float], pct: float) -> float:
    """
    Compute a percentile from a pre-sorted list using nearest-rank method.
    pct should be in range 0–100.
    """
    if not sorted_data:
        return 0.0
    k = math.ceil((pct / 100) * len(sorted_data)) - 1
    k = max(0, min(k, len(sorted_data) - 1))
    return round(sorted_data[k], 4)


def _compute_stats(timings_ms: list[float]) -> dict:
    """
    Given a list of raw latencies in milliseconds, return a stats dict.
    Input list is NOT required to be sorted — this function sorts it internally.
    """
    if not timings_ms:
        raise ValueError("No timings to compute statistics on.")

    sorted_t = sorted(timings_ms)
    n = len(sorted_t)
    mean = sum(sorted_t) / n
    variance = sum((x - mean) ** 2 for x in sorted_t) / n

    return {
        "p50":     _percentile(sorted_t, 50),
        "p95":     _percentile(sorted_t, 95),
        "p99":     _percentile(sorted_t, 99),
        "mean":    round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "min":     round(sorted_t[0], 4),
        "max":     round(sorted_t[-1], 4),
    }

# ── single-threaded runner ────────────────────────────────────────────────────

def _run_single(query_fn: Callable, total_runs: int) -> list[float]:
    """
    Execute query_fn (total_runs + WARMUP_RUNS) times sequentially.
    Returns only the measured timings (warm-up discarded).
    """
    # warm-up — not timed
    for _ in range(WARMUP_RUNS):
        query_fn()

    # measured runs
    timings = []
    for _ in range(total_runs):
        start = time.perf_counter()
        query_fn()
        elapsed_ms = (time.perf_counter() - start) * 1_000
        timings.append(elapsed_ms)

    return timings

# ── concurrent runner ─────────────────────────────────────────────────────────

def _run_concurrent(query_fn: Callable, total_runs: int, concurrency: int) -> list[float]:
    """
    Execute query_fn across `concurrency` threads.

    Each thread runs its share of (total_runs / concurrency) measured iterations,
    preceded by WARMUP_RUNS warm-up calls. All threads start simultaneously via
    a threading.Barrier so the concurrency pressure is real.

    Returns the combined list of all measured timings across all threads.
    """
    runs_per_thread = max(1, total_runs // concurrency)
    barrier = threading.Barrier(concurrency)
    all_timings: list[float] = []
    lock = threading.Lock()

    def worker():
        # warm-up before the barrier — avoids polluting the timed window
        for _ in range(WARMUP_RUNS):
            query_fn()

        # all threads block here until every worker is ready
        barrier.wait()

        local_timings = []
        for _ in range(runs_per_thread):
            start = time.perf_counter()
            query_fn()
            elapsed_ms = (time.perf_counter() - start) * 1_000
            local_timings.append(elapsed_ms)

        with lock:
            all_timings.extend(local_timings)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return all_timings

# ── public API ────────────────────────────────────────────────────────────────

def run_benchmark(
    query_fn: Callable,
    db: str,
    query_id: str,
    iterations: int = DEFAULT_ITERATIONS,
    concurrency: int = 1,
    output_path: str | None = None,
    label: str | None = None,
) -> dict:
    """
    Run a benchmark and return (and optionally save) the result dict.

    Parameters
    ----------
    query_fn    : zero-argument callable that executes one query/operation.
    db          : database name for metadata, e.g. "postgres", "redis".
    query_id    : benchmark identifier, e.g. "Q1", "Q3", "Q8_naive".
    iterations  : number of *measured* runs (warm-up runs are additional).
    concurrency : number of concurrent threads. Use 1 for sequential runs.
    output_path : if provided, result JSON is written to this path.
    label       : optional human-readable description saved in the result.

    Returns
    -------
    dict with keys: db, query_id, timestamp, iterations, warmup_runs,
                    concurrency, label, latency_ms, raw_timings_ms
    """
    print(f"\n  Running {db.upper()} {query_id}"
          f" — {iterations} iterations"
          f" × {concurrency} thread(s)"
          f" (+{WARMUP_RUNS} warm-up)")

    wall_start = time.perf_counter()

    if concurrency <= 1:
        timings = _run_single(query_fn, iterations)
    else:
        timings = _run_concurrent(query_fn, iterations, concurrency)

    wall_elapsed = time.perf_counter() - wall_start

    stats = _compute_stats(timings)

    result = {
        "db":             db,
        "query_id":       query_id,
        "label":          label or "",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "iterations":     len(timings),          # actual measured runs
        "warmup_runs":    WARMUP_RUNS,
        "concurrency":    concurrency,
        "wall_time_s":    round(wall_elapsed, 3),
        "latency_ms":     stats,
        "raw_timings_ms": [round(t, 4) for t in timings],
    }

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"  {'─' * 46}")
    print(f"  {'Wall time':<20} {result['wall_time_s']:.2f}s")
    print(f"  {'p50':<20} {stats['p50']:.2f} ms")
    print(f"  {'p95':<20} {stats['p95']:.2f} ms")
    print(f"  {'p99':<20} {stats['p99']:.2f} ms")
    print(f"  {'mean ± std':<20} {stats['mean']:.2f} ± {stats['std_dev']:.2f} ms")
    print(f"  {'min / max':<20} {stats['min']:.2f} / {stats['max']:.2f} ms")
    print(f"  {'─' * 46}")

    # ── save to file ──────────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {output_path}")

    return result