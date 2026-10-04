"""
benchmarks/cassandra/naive/q6_events.py — Cassandra Naive: Q6
==============================================================
Q6: Retrieve all activity events for a specific user in a 30-day
    window, ordered by time.

This is Cassandra's killer query — and the naive schema demonstrates
exactly why the killer query requires the right schema to be fast.

Naive schema penalty — paginated full table scan + Python-side filter
──────────────────────────────────────────────────────────────────────
The naive events table has id (UUID) as its sole partition key.

The first approach was ALLOW FILTERING with a WHERE user_id + occurred_at
predicate. This fails at production scale regardless of server configuration:
  - tombstone_failure_threshold must be raised to 10M (2M tombstones
    accumulate from nullable columns across ~1M rows)
  - MAX_HEAP_SIZE must be raised to 4GB (full scan materialises ~1M rows
    in JVM heap)
  - read_request_timeout must be raised above 120s (coordinator blocks
    on the full-scan + filter evaluation, exceeding the server timeout)
  Even after all three changes, ALLOW FILTERING on the events table with
  a WHERE clause triggers a single long-running blocking read on the
  coordinator that exceeds the server timeout.

The correct naive fallback — and what a naive developer would actually
reach for after hitting these failures — is a paginated full table scan
with no WHERE clause, filtering Python-side:

  SELECT id, user_id, event_type, occurred_at, product_id, session_id,
         metadata FROM events

The driver fetches 10,000 rows per page. Each page completes in
milliseconds. The coordinator never blocks on a single long read. The
total data read is identical (every row in the table), but chunked into
manageable pages rather than one giant blocking operation.

This is still genuinely naive: every iteration reads the entire events
table (~1M rows) to return the events for one user in one 30-day window.
The per-iteration cost is O(total dataset), not O(user events in window).
The contrast with the optimised schema remains the schema effect:
  - Naive  : full table scan, ~1M rows read, Python filter, Python sort
  - Optimised : 1-2 partition reads, bounded by user's monthly event count

The ALLOW FILTERING failure is documented separately as a finding:
the naive schema requires progressive server reconfiguration (tombstone
threshold, heap, server timeout) just to attempt the query, and still
fails. The paginated approach is the naive implementation that actually
produces measurable latency for the benchmark comparison.

ORDER BY limitation
────────────────────
CQL ORDER BY requires a clustering column. The naive events table has
none. Results are sorted Python-side by occurred_at DESC.

Anchor pool design
───────────────────
(user_id, occurred_at) pairs sampled from the events table with no WHERE
clause. The 30-day window is centred ±15 days on the anchor occurred_at,
guaranteeing at least one real event per window — identical to the
PostgreSQL Q6 methodology.

Usage:
    python q6_events.py                   # 1000 iterations
    python q6_events.py --iterations 100  # quick smoke test
    python q6_events.py --dry-run         # run once, print result sample
    python q6_events.py --pool-size 500   # smaller anchor pool
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import json
import math
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE     = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")
WINDOW_DAYS  = 30

# ── pool helper ───────────────────────────────────────────────────────────────

def fetch_anchor_pool(session, pool_size: int) -> list[tuple]:
    """
    Sample (user_id, occurred_at) pairs from the events table.
    No WHERE clause → no ALLOW FILTERING. Returns the first pool_size rows
    Cassandra encounters across token ranges, then shuffled in Python.
    """
    rows = list(session.execute(
        f"SELECT user_id, occurred_at FROM events LIMIT {pool_size}"
    ))
    if not rows:
        raise RuntimeError("No events found — run cassandra_naive_loader.py first.")
    pairs = [(r.user_id, r.occurred_at) for r in rows if r.user_id and r.occurred_at]
    random.shuffle(pairs)
    print(f"  Anchor pool: {len(pairs):,} (user_id, occurred_at) pairs loaded.")
    return pairs


def anchor_window(anchor_dt) -> tuple:
    """30-day window centred on anchor_dt (±15 days). Mirrors PostgreSQL Q6."""
    from datetime import timezone
    if anchor_dt.tzinfo is None:
        anchor_dt = anchor_dt.replace(tzinfo=timezone.utc)
    start = anchor_dt - timedelta(days=15)
    end   = anchor_dt + timedelta(days=15)
    return start, end

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, pairs: list[tuple]):
    """
    Each call:
      1. Paginated full table scan of events (no WHERE clause)
         — reads every row in the table, 10,000 rows per page.
         Each page is a separate driver request; no single blocking
         read that would trigger the server-side read_request_timeout.
      2. Python-side filter: keep rows matching (user_id, start, end)
      3. Python-side sort by occurred_at DESC

    Total data read per iteration: ~1M rows (entire events table).
    Rows matching the 30-day window for the sampled user: typically O(10s).
    This is the naive penalty: O(dataset) work to answer an O(user window)
    question. The optimised schema answers the same question in O(user window).

    Why not ALLOW FILTERING?
    ALLOW FILTERING with WHERE user_id = ? AND occurred_at BETWEEN ? AND ?
    forces Cassandra to evaluate the filter server-side in a single blocking
    read across all partitions. At ~1M rows this exceeds the server-side
    read_request_timeout regardless of how high it is set — the coordinator
    cannot complete the scan within any reasonable timeout. Paginated full
    scan avoids this by chunking the work into 10k-row pages, each completing
    in milliseconds, with Python applying the filter after the fact.
    """
    def _run():
        user_id, anchor_dt = random.choice(pairs)
        start, end = anchor_window(anchor_dt)

        # Paginated full table scan — no WHERE, no ALLOW FILTERING.
        # The driver automatically fetches pages of session.default_fetch_size
        # (10,000 rows) until the table is exhausted.
        matching = []
        for row in session.execute(
            "SELECT id, user_id, event_type, occurred_at, "
            "product_id, session_id, metadata FROM events"
        ):
            if row.occurred_at is None or row.user_id != user_id:
                continue
            # Cassandra returns timezone-naive datetimes; anchor_window()
            # produces timezone-aware. Normalise by stripping tzinfo for
            # comparison — the data is stored as UTC so this is safe.
            occ = (row.occurred_at.replace(tzinfo=None)
                   if row.occurred_at.tzinfo is not None
                   else row.occurred_at)
            start_n = start.replace(tzinfo=None)
            end_n   = end.replace(tzinfo=None)
            if start_n <= occ < end_n:
                matching.append(row)

        # Sort Python-side — no clustering column available
        matching.sort(
            key=lambda r: r.occurred_at or datetime.min,
            reverse=True,
        )
        return matching

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, pairs: list[tuple]):
    user_id, anchor_dt = pairs[0]
    start, end = anchor_window(anchor_dt)
    print(f"\n  DRY RUN — Q6 naive events for user {user_id}")
    print(f"  Window: {start} → {end}\n")

    fn = make_query_fn(session, [(user_id, anchor_dt)])
    rows = fn()
    if not rows:
        print("  ⚠  No events in this window — try a different anchor.")
        return
    print(f"  {len(rows)} event(s) returned (sorted Python-side):\n")
    print(f"  {'#':<4} {'Event type':<25} {'Occurred at':<32} {'Product ID'}")
    print(f"  {'─'*4} {'─'*25} {'─'*32} {'─'*36}")
    for i, r in enumerate(rows[:15], 1):
        print(
            f"  {i:<4} {str(r.event_type):<25} "
            f"{str(r.occurred_at):<32} "
            f"{str(r.product_id) if r.product_id else 'N/A'}"
        )
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more rows")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q6 events benchmark")
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help=(
            "Measured iterations (default: 50 for naive Q6). "
            "Each iteration reads ~1M rows (~30-60s). 1000 iterations = 8+ hours "
            "and causes JVM GC pressure leading to ConnectionShutdown errors. "
            "50 iterations provides sufficient p50/p95/p99 coverage. "
            "Documented in the methodology as a constraint of the naive full-scan."
        ),
    )
    parser.add_argument("--pool-size", type=int, default=1000, dest="pool_size")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q6.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q6 User Events Benchmark")
    print("=" * 60)
    print("  Schema  : cassandra_naive")
    print("  Method  : Paginated full table scan of events (no WHERE clause)")
    print("            + Python-side filter (user_id + date range)")
    print("            + Python-side sort by occurred_at DESC")
    print("  Note    : ALLOW FILTERING with WHERE clause causes server-side")
    print("            ReadTimeout regardless of timeout config at this scale.")
    print("            Paginated full scan avoids coordinator blocking.")
    print(f"  Window  : {WINDOW_DAYS} days centred on anchor event")
    print("  ⚠  Full table scan (~1M rows) per iteration.")
    print("     ALLOW FILTERING + WHERE causes server ReadTimeout at this scale;")
    print("     paginated full scan is used instead. Same total I/O cost.")

    # request_timeout=300s: naive Q6 issues a full ALLOW FILTERING scan on
    # ~1M event rows. At 1536M heap the scan completes but can take 60-120s
    # on a cold page cache. The cassandra_conn default of 30s is insufficient.
    # Timeout must be set at profile construction — session.default_timeout
    # cannot be assigned post-construction when ExecutionProfile is in use.
    cluster, session = get_session(keyspace=KEYSPACE, request_timeout=300.0)
    # Reduce fetch_size from default 10,000 to 2,000 rows per page.
    # Under continuous repeated full-table scans, 10k-row pages cause JVM GC
    # pressure that leads to mid-stream ConnectionShutdown (CRC mismatch) errors.
    # 2,000 rows per page reduces per-page heap allocation by 5×.
    session.default_fetch_size = 2_000
    try:
        pairs = fetch_anchor_pool(session, args.pool_size)

        if args.dry_run:
            dry_run(session, pairs)
            return

        # Custom timing loop — see docstring for full explanation.
        # Cannot use standard harness: back-to-back full scans of ~1M rows cause
        # JVM GC pressure → ConnectionShutdown (CRC mismatch). Fix: reconnect
        # session per iteration so GC can collect between scans.
        label = (
            f"All events for a user in a {WINDOW_DAYS}-day window. "
            "Naive schema: id is sole PK — no partition key on user_id or time. "
            "ALLOW FILTERING with WHERE causes server ReadTimeout at ~1M rows. "
            "Two-pass naive scan: pass 1 reads all events (id, user_id, occurred_at "
            "only — 3 cols vs 7) to find matching rows; pass 2 does PK lookups "
            "for ~20-30 matches. Full table scan penalty fully measured. "
            "Session reconnected per iteration to prevent GC ConnectionShutdown. "
            f"Warmup=10. Iterations={args.iterations} (not 1000). "
            f"Window centred ±15 days on sampled anchor event. "
            f"Pool of {args.pool_size} (user_id, anchor) pairs."
        )

        print(f"\n  Running CASSANDRA_NAIVE Q6 (custom reconnect loop) — "
              f"{args.iterations} iterations + 10 warm-up")

        def _single_scan():
            """
            Two-pass approach to minimise data transfer during the full table scan:

            Pass 1 — scan ALL events, selecting only 3 lightweight columns
                     (id, user_id, occurred_at). This is the minimum needed to
                     identify matching rows. Reduces per-page transfer by ~65%
                     vs selecting all 7 columns (event_type, product_id,
                     session_id, metadata are not needed for filtering).

            Pass 2 — PK lookup for each matching row (~20-30 rows typical).
                     Fetches full event details by id. Fast single-partition reads.

            The full table scan penalty is still fully measured — pass 1 reads
            every row in the events table. The schema effect is unchanged.
            """
            c2, s2 = get_session(keyspace=KEYSPACE, request_timeout=300.0)
            s2.default_fetch_size = 20_000   # larger pages are fine with reconnect
            try:
                user_id, anchor_dt = random.choice(pairs)
                start, end = anchor_window(anchor_dt)
                start_n = start.replace(tzinfo=None)
                end_n   = end.replace(tzinfo=None)

                # Pass 1: minimal column scan — id + user_id + occurred_at only
                matching_ids = []
                for row in s2.execute(
                    "SELECT id, user_id, occurred_at FROM events"
                ):
                    if row.user_id != user_id or row.occurred_at is None:
                        continue
                    occ = (row.occurred_at.replace(tzinfo=None)
                           if row.occurred_at.tzinfo else row.occurred_at)
                    if start_n <= occ < end_n:
                        matching_ids.append((row.id, row.occurred_at))

                results = []
                # Pass 2: single query instead of a loop of ~20-30 reads
                if matching_ids:
                    id_list = [eid for eid, _ in matching_ids]
                    placeholders = ", ".join(["%s"] * len(id_list))
                    rows = s2.execute(
                        f"SELECT id, event_type, occurred_at, product_id, "
                        f"session_id, metadata FROM events WHERE id IN ({placeholders})",
                        id_list,
                    )
                    results = list(rows)

                results.sort(key=lambda r: r.occurred_at or datetime.min, reverse=True)
                return results
            finally:
                c2.shutdown()

        import time as _time
        print("  Warm-up (10 scans)...")
        for i in range(10):
            _single_scan()
            print(f"    warm-up {i+1}/10 done")

        timings = []
        wall_start = _time.perf_counter()
        for i in range(args.iterations):
            t0 = _time.perf_counter()
            _single_scan()
            timings.append((_time.perf_counter() - t0) * 1_000)
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = _time.perf_counter() - wall_start
                print(f"  [{elapsed:>6.0f}s] {i+1:>3}/{args.iterations} done  "
                      f"last={timings[-1]:.0f}ms")

        wall_elapsed = _time.perf_counter() - wall_start
        s = sorted(timings)
        n = len(s)
        mean = sum(s) / n
        variance = sum((x - mean) ** 2 for x in s) / n
        def _pct(p):
            k = max(0, min(math.ceil((p / 100) * n) - 1, n - 1))
            return round(s[k], 4)
        stats = {
            "p50": _pct(50), "p95": _pct(95), "p99": _pct(99),
            "mean": round(mean, 4),
            "std_dev": round(math.sqrt(variance), 4),
            "min": round(s[0], 4), "max": round(s[-1], 4),
        }
        print(f"  {'─'*46}")
        print(f"  {'Wall time':<20} {wall_elapsed:.2f}s")
        print(f"  {'p50':<20} {stats['p50']:.2f} ms")
        print(f"  {'p95':<20} {stats['p95']:.2f} ms")
        print(f"  {'p99':<20} {stats['p99']:.2f} ms")
        print(f"  {'mean ± std':<20} {stats['mean']:.2f} ± {stats['std_dev']:.2f} ms")
        print(f"  {'─'*46}")

        result = {
            "db": "cassandra_naive", "query_id": "Q6", "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": n, "warmup_runs": 10, "concurrency": 1,
            "wall_time_s": round(wall_elapsed, 3),
            "latency_ms": stats,
            "raw_timings_ms": [round(t, 4) for t in timings],
            "note": "Custom loop: session reconnected per iteration (GC relief).",
        }
        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {args.output}")
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()