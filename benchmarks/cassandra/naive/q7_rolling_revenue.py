"""
benchmarks/cassandra/naive/q7_rolling_revenue.py — Cassandra Naive: Q7
=======================================================================
Q7: 7-day rolling average of daily revenue per subscription tier over
    a 6-month window, with gap-filling for days with zero activity.

Naive schema — full scan + Python aggregation + Python gap-fill
────────────────────────────────────────────────────────────────
TimescaleDB Q7 uses time_bucket_gapfill() on a continuous aggregate.
PostgreSQL Q7 uses generate_series + window functions. In the Cassandra
naive schema, none of these constructs exist. Each iteration:

  1. ALLOW FILTERING scan on invoices — filter paid invoices in a date
     range. Neither status nor created_at is a partition key.

  2. Full scan on subscriptions — for marketplace invoice tier attribution.
     Same pattern as Q1 naive. The subscription data is re-fetched every
     iteration to avoid inter-iteration state.

  3. Python-side tier attribution — using the same temporal logic as Q1
     (find most recent subscription started_at <= invoice created_at for
     marketplace invoices; use subscription_id directly for subscription
     invoices).

  4. Python daily aggregation — group by (date, tier_id), sum total_usd.

  5. Python gap-filling — iterate over every calendar day in the 6-month
     window, insert zero-revenue days where no invoices exist.

  6. Python rolling average — 7-day rolling mean over the gap-filled
     daily series, per tier.

All of steps 3–6 are client-side computation. In TimescaleDB these are
native server-side operations running on pre-aggregated continuous
aggregates. The total wall time for naive Q7 includes both the full
table scans (steps 1–2) and the Python computation (steps 3–6).

Window design
──────────────
A random 6-month start date is chosen within the actual min/max range
of paid invoices (computed once at startup). This matches the PostgreSQL
Q7 methodology and ensures every iteration covers a real data window.
The min/max is found via a startup full scan of invoices — a one-time
cost not included in the measured latency.

Usage:
    python q7_rolling_revenue.py                   # 1000 iterations
    python q7_rolling_revenue.py --iterations 100  # quick smoke test
    python q7_rolling_revenue.py --dry-run         # run once, print sample
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.harness import run_benchmark

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE      = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")
WINDOW_DAYS   = 182    # ~6 months
ROLLING_DAYS  = 7
TIER_NAMES    = {1: "Free", 2: "Pro", 3: "Business"}

# ── startup helpers ───────────────────────────────────────────────────────────

def fetch_date_range(session) -> tuple[date, date]:
    """
    Compute min and max created_at of paid invoices.
    Full scan with ALLOW FILTERING — startup cost, not benchmarked.
    """
    rows = list(session.execute(
        "SELECT created_at FROM invoices WHERE status = 'paid' ALLOW FILTERING"
    ))
    dates = [r.created_at.date() for r in rows if r.created_at]
    if not dates:
        raise RuntimeError("No paid invoices found — run cassandra_naive_loader.py first.")
    return min(dates), max(dates)


def build_subscriptions_index(session) -> dict:
    """
    Full scan of subscriptions → user_id → sorted [(started_at, tier_id)].
    Startup cost — not benchmarked.
    """
    subs = list(session.execute("SELECT id, user_id, tier_id, started_at FROM subscriptions"))
    user_subs = defaultdict(list)
    for s in subs:
        if s.user_id and s.started_at and s.tier_id is not None:
            started = s.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            user_subs[str(s.user_id)].append((started, s.tier_id))
    for uid in user_subs:
        user_subs[uid].sort(key=lambda x: x[0])
    return dict(user_subs)


def build_sub_id_tier_index(session) -> dict:
    """
    Full scan of subscriptions → subscription_id → tier_id.
    Startup cost — not benchmarked.
    """
    subs = list(session.execute("SELECT id, tier_id FROM subscriptions"))
    return {str(s.id): s.tier_id for s in subs if s.id and s.tier_id is not None}

# ── helpers ───────────────────────────────────────────────────────────────────

def _rolling_avg(daily_series: list[Decimal], window: int) -> list[Decimal]:
    """7-day rolling mean over a list of daily totals (gap-filled)."""
    avgs = []
    for i in range(len(daily_series)):
        start = max(0, i - window + 1)
        chunk = daily_series[start : i + 1]
        avgs.append(sum(chunk) / Decimal(len(chunk)))
    return avgs


def _gap_fill(daily_totals: dict, start_d: date, end_d: date) -> list[tuple]:
    """
    Produce a complete daily series from start_d to end_d (inclusive),
    inserting Decimal('0') for days with no revenue.
    Returns list of (date, total_usd).
    """
    result = []
    current = start_d
    while current <= end_d:
        result.append((current, daily_totals.get(current, Decimal("0"))))
        current += timedelta(days=1)
    return result

# ── query function ────────────────────────────────────────────────────────────

def make_query_fn(session, data_min: date, data_max: date,
                  user_subs: dict, sub_id_to_tier: dict):
    """
    Each call performs:
      1. ALLOW FILTERING scan on invoices (paid, date range)
      2. Python tier attribution (using startup-built sub indexes)
      3. Python daily aggregation per tier
      4. Python gap-filling per tier
      5. Python 7-day rolling average per tier

    Sub indexes (user_subs, sub_id_to_tier) are passed in from startup
    so they are not re-fetched per iteration. This mirrors PostgreSQL Q7
    where the subscription table is joined via the query plan (data is
    available to the engine) rather than re-loaded each call.
    """
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)

    def _run():
        # Random 6-month window within dataset range
        offset = random.randint(0, max_start) if max_start > 0 else 0
        start_d = data_min + timedelta(days=offset)
        end_d   = min(start_d + timedelta(days=WINDOW_DAYS - 1), data_max)

        # Timestamps for CQL parameters (timezone-aware)
        start_ts = datetime.combine(start_d, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_ts   = datetime.combine(end_d,   datetime.max.time()).replace(tzinfo=timezone.utc)

        # ── Step 1: paid invoices in window (ALLOW FILTERING) ─────────────────
        invoices = list(session.execute(
            "SELECT id, user_id, invoice_type, total_usd, subscription_id, created_at "
            "FROM invoices "
            "WHERE status = 'paid' AND created_at >= %s AND created_at <= %s "
            "ALLOW FILTERING",
            (start_ts, end_ts),
        ))

        # ── Steps 2–3: tier attribution + daily aggregation ───────────────────
        # daily[tier_id][date] = Decimal total
        daily = defaultdict(lambda: defaultdict(Decimal))

        for inv in invoices:
            if not inv.created_at:
                continue
            created_at = inv.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            if inv.invoice_type == "subscription" and inv.subscription_id:
                tier_id = sub_id_to_tier.get(str(inv.subscription_id))
            else:
                tier_id = None
                uid_str = str(inv.user_id) if inv.user_id else None
                if uid_str:
                    for started_at, tid in user_subs.get(uid_str, []):
                        if started_at <= created_at:
                            tier_id = tid
                        else:
                            break

            if tier_id is None:
                continue

            daily[tier_id][created_at.date()] += inv.total_usd or Decimal("0")

        # ── Steps 4–5: gap-fill + rolling average per tier ────────────────────
        results = []
        for tier_id in sorted(TIER_NAMES.keys()):
            tier_daily = _gap_fill(daily[tier_id], start_d, end_d)
            totals     = [t for _, t in tier_daily]
            rolling    = _rolling_avg(totals, ROLLING_DAYS)

            for i, (d, total) in enumerate(tier_daily):
                results.append({
                    "day":              str(d),
                    "tier_name":        TIER_NAMES[tier_id],
                    "daily_revenue":    total,
                    "rolling_7d_avg":   rolling[i],
                })

        return results

    return _run

# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(session, data_min, data_max, user_subs, sub_id_to_tier):
    print(f"\n  DRY RUN — Q7 naive rolling revenue\n")
    fn = make_query_fn(session, data_min, data_max, user_subs, sub_id_to_tier)
    results = fn()
    if not results:
        print("  ⚠  No results — is data loaded and date range valid?")
        return
    print(f"  {len(results)} rows (days × tiers).")
    print(f"\n  {'Day':<12} {'Tier':<12} {'Daily':>12} {'7d Avg':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for r in results[:15]:
        print(
            f"  {r['day']:<12} {r['tier_name']:<12} "
            f"{float(r['daily_revenue']):>12.2f} "
            f"{float(r['rolling_7d_avg']):>12.2f}"
        )
    if len(results) > 15:
        print(f"  ... {len(results) - 15} more rows")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra naive Q7 rolling revenue benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("results", "cassandra_naive_Q7.json"),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra Naive — Q7 Rolling Revenue Benchmark")
    print("=" * 60)
    print("  Schema : cassandra_naive")
    print("  Method : ALLOW FILTERING scan (invoices) + Python aggregation")
    print("           + Python gap-fill + Python 7-day rolling average")

    # request_timeout=120.0s: naive Q7 scans invoices (ALLOW FILTERING) + full subscriptions table.
    # 30s default may be exceeded on cold cache runs.
    cluster, session = get_session(keyspace=KEYSPACE, request_timeout=120.0)
    try:
        print("\n  Computing invoice date range (startup scan)...")
        data_min, data_max = fetch_date_range(session)
        print(f"  Date range: {data_min} → {data_max}")

        print("  Building subscription indexes (startup scan)...")
        user_subs      = build_subscriptions_index(session)
        sub_id_to_tier = build_sub_id_tier_index(session)
        print(f"  Subscription indexes ready ({len(user_subs):,} users).")

        if args.dry_run:
            dry_run(session, data_min, data_max, user_subs, sub_id_to_tier)
            return

        run_benchmark(
            query_fn=make_query_fn(session, data_min, data_max, user_subs, sub_id_to_tier),
            db="cassandra_naive",
            query_id="Q7",
            label=(
                "7-day rolling revenue average per tier, 6-month window. "
                "ALLOW FILTERING scan on invoices (status + date range). "
                "Tier attribution via Python lookup on pre-built subscription indexes. "
                "Daily aggregation, gap-filling, and rolling average all Python-side. "
                "Gap-fill inserts Decimal(0) for days with no revenue. "
                f"Window: {WINDOW_DAYS} days, randomly placed within dataset range "
                f"({data_min} → {data_max})."
            ),
            iterations=args.iterations,
            concurrency=1,
            output_path=args.output,
        )
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()