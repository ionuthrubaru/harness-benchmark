"""
benchmarks/cassandra/run_scalability.py — Cassandra Scalability Baselines
==========================================================================
Re-runs Q1–Q7 at 10% and 50% data scale for both naive and optimised
schemas to establish the Cassandra scalability curves for Chart 3.

Scale methodology — date range cutoff, identical to PostgreSQL and Neo4j
─────────────────────────────────────────────────────────────────────────
  10% scale : queries restricted to the first 10% of the event date range
  50% scale : queries restricted to the first 50% of the event date range
  100% scale: full dataset — already captured by individual q*.py runs

Cutoffs are computed from the actual MIN/MAX of occurred_at in the
cassandra_naive.events table (sampled via paginated full scan), consistent
with the PostgreSQL and Neo4j scalability anchors.

Per-query scaling approach
───────────────────────────
  Q1  naive : ALLOW FILTERING scan on invoices with created_at < cutoff
  Q1  opt   : Fan-out only to months within the cutoff window
  Q2  both  : Invoice ID pool filtered to created_at < cutoff
  Q3  both  : Row-based scaling — user pool from scale_pct% of total users
              Sessions fall within a single year; date-range scaling inapplicable
  Q4  naive : Python filter of qualifying orders by created_at < cutoff
  Q4  opt   : Pre-computed at load time — date-range scaling not applicable
              (also_bought reflects full dataset; noted in methodology)
  Q5  both  : Product pool / results filtered to created_at < cutoff
  Q6  naive : Python filter of paginated full scan to occurred_at < cutoff
              Anchor pool restricted to events before the cutoff
  Q6  opt   : Anchor pool restricted to occurred_at < cutoff;
              query fans out to only year_month partitions before cutoff
  Q7  both  : Date window bounded to [data_min, cutoff]

Q8 excluded — write throughput scaling is measured by thread variation.

Cassandra-specific notes
─────────────────────────
  - get_session() returns (cluster, session); cluster.shutdown() in finally
  - cassandra.util.Date != Python date — converted via _to_pydate()
  - Naive full scans use request_timeout=300s; optimised uses 30s default
  - ALLOW FILTERING is used for pool-building startup queries (not benchmarked)
  - Q6 naive full scan is the same paginated approach as the standalone script

Usage:
    python run_scalability.py                    # both schemas, both scales, 1000 iters
    python run_scalability.py --scale 10
    python run_scalability.py --scale 50
    python run_scalability.py --schema naive
    python run_scalability.py --schema optimised
    python run_scalability.py --only Q6
    python run_scalability.py --iterations 100
    python run_scalability.py --dry-run          # 100 iters, both scales
"""

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.harness import run_benchmark
from benchmarks.cassandra.cassandra_conn import get_session

load_dotenv()

KEYSPACE_NAIVE     = os.getenv("CASSANDRA_KEYSPACE_NAIVE",      "cassandra_naive")
KEYSPACE_OPTIMISED = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED",  "cassandra_optimised")

RESULTS_NAIVE     = os.path.join(PROJECT_ROOT, "benchmarks", "cassandra", "naive",     "results", "scale")
RESULTS_OPTIMISED = os.path.join(PROJECT_ROOT, "benchmarks", "cassandra", "optimised", "results", "scalability")

WINDOW_DAYS  = 30
WINDOW_DAYS7 = 182
ROLLING_DAYS = 7
TIER_IDS     = [1, 2, 3]
TIER_NAMES   = {1: "Free", 2: "Pro", 3: "Business"}
PRICING = [
    (1, datetime(2023, 1, 1, tzinfo=timezone.utc), None,                                       Decimal("0.00")),
    (2, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("14.99")),
    (2, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("19.99")),
    (3, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("39.99")),
    (3, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("49.99")),
]
SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour", "photoshop brushes", "video editing",
    "certificate course", "logo design", "colour palette", "font pack",
    "texture pack", "motion graphics", "social media", "icon set",
    "web design", "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]
COMPLETED_STATUSES    = {"confirmed", "shipped", "delivered"}
Q6_NAIVE_ITERATIONS   = 30   # each scan ~2 min; 30 = ~60 min per scale level

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── type helpers ──────────────────────────────────────────────────────────────

def _to_pydate(d) -> date:
    """Convert cassandra.util.Date to Python datetime.date."""
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))

def _dt_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _resolve_price(tier_id: int, at: datetime) -> Decimal:
    at = _dt_to_utc(at)
    for t_id, valid_from, valid_to, price in PRICING:
        if t_id != tier_id:
            continue
        if valid_from <= at and (valid_to is None or valid_to > at):
            return price
    return Decimal("0")

# ── cutoff computation ────────────────────────────────────────────────────────

def compute_cutoffs(session) -> dict:
    """
    Derive 10% and 50% date cutoffs from cassandra_naive.events.
    Paginated full scan — startup cost only, not benchmarked.
    Consistent with PostgreSQL and Neo4j cutoff anchors.
    """
    info("Scanning cassandra_naive.events for date range (paginated)...")
    min_dt = None
    max_dt = None
    count  = 0
    for row in session.execute("SELECT occurred_at FROM events"):
        if row.occurred_at is None:
            continue
        occ = _dt_to_utc(row.occurred_at)
        if min_dt is None or occ < min_dt:
            min_dt = occ
        if max_dt is None or occ > max_dt:
            max_dt = occ
        count += 1

    if not min_dt:
        raise RuntimeError("No events found in cassandra_naive.events.")

    info(f"Scanned {count:,} events.")
    min_date   = min_dt.date()
    max_date   = max_dt.date()
    range_days = (max_date - min_date).days

    return {
        "min_date":     min_date,
        "max_date":     max_date,
        "range_days":   range_days,
        "cutoff_10pct": min_date + timedelta(days=int(range_days * 0.10)),
        "cutoff_50pct": min_date + timedelta(days=int(range_days * 0.50)),
    }

# ── Q1 ────────────────────────────────────────────────────────────────────────

def make_q1_naive_fn(session, cutoff: date):
    since_dt  = datetime.combine(cutoff - timedelta(days=365), datetime.min.time()).replace(tzinfo=timezone.utc)
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)

    # Build subscription indexes at setup time (startup cost, not benchmarked)
    subs      = list(session.execute("SELECT id, user_id, tier_id, started_at FROM subscriptions"))
    sub_id_to_tier = {str(s.id): s.tier_id for s in subs if s.id and s.tier_id}
    user_subs = defaultdict(list)
    for s in subs:
        if s.user_id and s.started_at and s.tier_id:
            user_subs[str(s.user_id)].append((_dt_to_utc(s.started_at), s.tier_id))
    for uid in user_subs:
        user_subs[uid].sort(key=lambda x: x[0])

    def _run():
        invoices = list(session.execute(
            "SELECT id, user_id, invoice_type, total_usd, subscription_id, created_at "
            "FROM invoices WHERE status = 'paid' "
            "AND created_at >= %s AND created_at < %s ALLOW FILTERING",
            (since_dt, cutoff_dt),
        ))
        revenue = defaultdict(lambda: {"count": 0, "total": Decimal("0")})
        for inv in invoices:
            if not inv.created_at:
                continue
            created_at = _dt_to_utc(inv.created_at)
            if inv.invoice_type == "subscription" and inv.subscription_id:
                tier_id = sub_id_to_tier.get(str(inv.subscription_id))
            else:
                tier_id = None
                uid = str(inv.user_id) if inv.user_id else None
                if uid:
                    for started_at, tid in user_subs.get(uid, []):
                        if started_at <= created_at:
                            tier_id = tid
                        else:
                            break
            if tier_id is None:
                continue
            ym = created_at.strftime("%Y-%m")
            revenue[(ym, tier_id)]["count"] += 1
            revenue[(ym, tier_id)]["total"] += inv.total_usd or Decimal("0")
        return list(revenue.items())
    return _run

def make_q1_optimised_fn(session, cutoff: date):
    """Fan-out only to months within [12 months before cutoff, cutoff]."""
    def _run():
        end_ym   = cutoff.strftime("%Y-%m")
        start_ym = (cutoff - timedelta(days=365)).strftime("%Y-%m")
        results  = []
        current  = datetime.strptime(start_ym, "%Y-%m").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end_ym, "%Y-%m").replace(tzinfo=timezone.utc)
        while current <= end_dt:
            ym = current.strftime("%Y-%m")
            for tier_id in TIER_IDS:
                rows = list(session.execute(
                    "SELECT invoice_id, total_usd, tier_name, monthly_price_usd_at_time "
                    "FROM invoices_by_month_tier WHERE year_month = %s AND tier_id = %s",
                    (ym, tier_id),
                ))
                if rows:
                    results.append({"month": ym, "tier_id": tier_id, "count": len(rows),
                                    "total": sum(r.total_usd for r in rows if r.total_usd)})
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return results
    return _run

# ── Q2 ────────────────────────────────────────────────────────────────────────

def fetch_invoice_pool_naive(session, cutoff: date, pool_size=1000) -> list:
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    rows = list(session.execute(
        f"SELECT id FROM invoices WHERE created_at < %s LIMIT {pool_size} ALLOW FILTERING",
        (cutoff_dt,),
    ))
    ids = [r.id for r in rows if r.id]
    random.shuffle(ids)
    return ids

def fetch_invoice_pool_optimised(session, cutoff: date, pool_size=1000) -> list:
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    rows = list(session.execute(
        f"SELECT invoice_id FROM invoices_full "
        f"WHERE invoice_created_at < %s LIMIT {pool_size} ALLOW FILTERING",
        (cutoff_dt,),
    ))
    ids = list({r.invoice_id for r in rows if r.invoice_id})
    random.shuffle(ids)
    return ids

def make_q2_naive_fn(session, pool: list):
    def _run():
        inv_id = random.choice(pool)
        inv_row = session.execute(
            "SELECT id, user_id, invoice_type, status, total_usd FROM invoices WHERE id = %s",
            (inv_id,),
        ).one()
        if not inv_row:
            return None
        lines = list(session.execute(
            "SELECT id, product_id, line_total_usd FROM invoice_lines "
            "WHERE invoice_id = %s ALLOW FILTERING",
            (inv_id,),
        ))
        user_row = session.execute(
            "SELECT id, full_name FROM users WHERE id = %s",
            (inv_row.user_id,),
        ).one()
        return {"invoice": inv_row, "customer": user_row, "lines": lines}
    return _run

def make_q2_optimised_fn(session, pool: list):
    def _run():
        inv_id = random.choice(pool)
        rows = list(session.execute(
            "SELECT invoice_id, line_id, total_usd, customer_full_name, "
            "product_name FROM invoices_full WHERE invoice_id = %s",
            (inv_id,),
        ))
        return rows
    return _run

# ── Q3 ────────────────────────────────────────────────────────────────────────

def fetch_user_pool_naive(session, scale_pct: int, total_pool=5000) -> list:
    rows = list(session.execute(f"SELECT user_id FROM sessions LIMIT {total_pool}"))
    ids = list({r.user_id for r in rows if r.user_id})
    n = max(1, int(len(ids) * scale_pct / 100))
    return ids[:n]

def fetch_user_pool_optimised(session, scale_pct: int, total_pool=5000) -> list:
    rows = list(session.execute(f"SELECT user_id FROM sessions_by_user LIMIT {total_pool}"))
    ids = list({r.user_id for r in rows if r.user_id})
    n = max(1, int(len(ids) * scale_pct / 100))
    return ids[:n]

def make_q3_naive_fn(session, pool: list):
    def _run():
        user_id = random.choice(pool)
        rows = list(session.execute(
            "SELECT id, user_id, cart, last_active_at FROM sessions "
            "WHERE user_id = %s ALLOW FILTERING",
            (user_id,),
        ))
        if not rows:
            return None
        rows.sort(key=lambda r: r.last_active_at or 0, reverse=True)
        return rows[0]
    return _run

def make_q3_optimised_fn(session, pool: list):
    def _run():
        user_id = random.choice(pool)
        return session.execute(
            "SELECT id, user_id, cart, last_active_at FROM sessions_by_user "
            "WHERE user_id = %s LIMIT 1",
            (user_id,),
        ).one()
    return _run

# ── Q4 ────────────────────────────────────────────────────────────────────────

def fetch_product_pool_naive(session, cutoff: date, pool_size=500) -> list:
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    rows = list(session.execute(
        f"SELECT id FROM products WHERE is_active = true AND created_at < %s "
        f"LIMIT {pool_size} ALLOW FILTERING",
        (cutoff_dt,),
    ))
    ids = [r.id for r in rows if r.id]
    random.shuffle(ids)
    return ids

def fetch_product_pool_optimised(session, pool_size=500) -> list:
    rows = list(session.execute(f"SELECT product_id FROM also_bought LIMIT {pool_size}"))
    ids = list({r.product_id for r in rows if r.product_id})
    random.shuffle(ids)
    return ids

def make_q4_naive_fn(session, pool: list, cutoff: date):
    """
    Q4 naive — ALLOW FILTERING on order_items + Python-side co-purchase counting.

    Scalability optimisation vs standalone q4_recommendations.py
    ─────────────────────────────────────────────────────────────
    The standalone script re-scans the full order_items table on every iteration
    (the "least-bad naive approach" for a single benchmark run). For the scalability
    run with 1000 iterations this would take 10+ hours (36s/iter × 1000).

    The full order_items scan is a static startup cost — the table does not change
    between iterations. Pre-loading it once at make_q4_naive_fn creation time and
    reusing it per iteration is what any real developer would do. The naive schema
    penalty being measured (no CQL join → ALLOW FILTERING + Python co-purchase
    aggregation) is fully preserved. Only the repeated re-scan of static data
    is eliminated.

    Per-iteration cost (unchanged vs standalone):
      1. ALLOW FILTERING on order_items by product_id → candidate order_ids
      2. PK lookups on orders → filter by status AND created_at < cutoff
      3. Python co-count from pre-loaded all_items dict (startup cost, not per-iter)
      4. PK lookups on products for top-10 results
    """
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)

    # Pre-load full order_items table ONCE — startup cost, not benchmarked.
    # Stores as dict: order_id_str → [product_id_str, ...] for O(1) per-order lookup.
    info("Q4 naive: pre-loading order_items table (startup cost)...")
    all_items_raw = list(session.execute("SELECT order_id, product_id FROM order_items"))
    # Group by order_id for fast per-qualifying-order lookup
    items_by_order = defaultdict(list)
    for item in all_items_raw:
        if item.order_id and item.product_id:
            items_by_order[str(item.order_id)].append(str(item.product_id))
    info(f"Q4 naive: pre-loaded {len(all_items_raw):,} order_items rows "
         f"({len(items_by_order):,} unique orders).")

    def _run():
        product_id = random.choice(pool)
        pid_str = str(product_id)

        # Step 1: ALLOW FILTERING — find orders containing this product
        order_id_rows = list(session.execute(
            "SELECT order_id FROM order_items WHERE product_id = %s ALLOW FILTERING",
            (product_id,),
        ))
        candidate_ids = {str(r.order_id) for r in order_id_rows if r.order_id}

        # Step 2: PK lookups — filter by status and cutoff date
        qualifying = set()
        for oid_str in candidate_ids:
            try:
                oid = uuid.UUID(oid_str)
            except ValueError:
                continue
            o = session.execute(
                "SELECT id, status, created_at FROM orders WHERE id = %s",
                (oid,),
            ).one()
            if (o and o.status in COMPLETED_STATUSES
                    and o.created_at and _dt_to_utc(o.created_at) < cutoff_dt):
                qualifying.add(oid_str)

        if not qualifying:
            return []

        # Step 3: co-purchase counts from pre-loaded dict (O(qualifying_orders))
        co_counts = defaultdict(int)
        for oid_str in qualifying:
            for co_pid_str in items_by_order.get(oid_str, []):
                if co_pid_str != pid_str:
                    co_counts[co_pid_str] += 1

        sorted_p = sorted(co_counts.items(), key=lambda x: x[1], reverse=True)

        # Step 4: PK lookups for top-10 product details
        results = []
        for co_pid_str, count in sorted_p[:10]:
            try:
                pid = uuid.UUID(co_pid_str)
            except ValueError:
                continue
            prod = session.execute(
                "SELECT id, name, product_type, price_usd, is_active "
                "FROM products WHERE id = %s",
                (pid,),
            ).one()
            if prod and prod.is_active:
                results.append({"product_id": prod.id, "count": count})
        return results

    return _run

def make_q4_optimised_fn(session, pool: list):
    def _run():
        product_id = random.choice(pool)
        return list(session.execute(
            "SELECT co_product_id, co_product_name, co_purchase_count, confidence "
            "FROM also_bought WHERE product_id = %s LIMIT 10",
            (product_id,),
        ))
    return _run

# ── Q5 ────────────────────────────────────────────────────────────────────────

def make_q5_naive_fn(session, cutoff: date):
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    def _run():
        term  = random.choice(SEARCH_TERMS)
        words = term.lower().split()
        rows = list(session.execute(
            "SELECT id, name, product_type, price_usd, created_at "
            "FROM products WHERE is_active = true ALLOW FILTERING"
        ))
        results = []
        for row in rows:
            if not row.name:
                continue
            if row.created_at and _dt_to_utc(row.created_at) >= cutoff_dt:
                continue
            if all(w in row.name.lower() for w in words):
                results.append(row)
            if len(results) >= 20:
                break
        return results
    return _run

def make_q5_optimised_fn(session, cutoff: date):
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    def _run():
        term         = random.choice(SEARCH_TERMS)
        words        = term.lower().split()
        primary_word = words[0]
        extra_words  = words[1:]
        rows = list(session.execute(
            "SELECT id, name, name_lower, product_type, price_usd, created_at "
            "FROM products_search WHERE name_lower LIKE %s LIMIT 200",
            (f"%{primary_word}%",),
        ))
        results = []
        for row in rows:
            if row.created_at and _dt_to_utc(row.created_at) >= cutoff_dt:
                continue
            if extra_words and not all(w in (row.name_lower or "") for w in extra_words):
                continue
            results.append(row)
            if len(results) >= 20:
                break
        return results
    return _run

# ── Q6 ────────────────────────────────────────────────────────────────────────

def fetch_anchor_pool_naive(session, cutoff: date, pool_size=1000) -> list:
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    rows = list(session.execute(f"SELECT user_id, occurred_at FROM events LIMIT {pool_size * 5}"))
    pairs = [
        (r.user_id, _dt_to_utc(r.occurred_at))
        for r in rows
        if r.user_id and r.occurred_at and _dt_to_utc(r.occurred_at) < cutoff_dt
    ]
    random.shuffle(pairs)
    if not pairs:
        warn(f"Q6 naive anchor pool empty before cutoff {cutoff.isoformat()} — "
             "no events at this scale level. Scalability run will skip.")
    return pairs[:pool_size]

def fetch_anchor_pool_optimised(session, cutoff: date, pool_size=1000) -> list:
    """
    Sample anchor pairs from events_by_user_month where occurred_at < cutoff.
    At 10% scale the cutoff is very early; the fixed LIMIT may not return enough
    events that fall before the cutoff. Use ALLOW FILTERING with occurred_at < cutoff
    directly on the optimised table — this is a startup pool-building query, not
    a benchmarked query, so ALLOW FILTERING here is acceptable.
    We scan a larger initial set and filter, accepting that the pool may be smaller
    than requested at very tight cutoffs.
    """
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    # Fetch a large sample and filter Python-side to events before the cutoff.
    # At 10% scale ~10% of events fall before the cutoff, so fetch 20x pool_size.
    rows = list(session.execute(
        f"SELECT user_id, occurred_at FROM events_by_user_month LIMIT {pool_size * 20}"
    ))
    pairs = [
        (r.user_id, _dt_to_utc(r.occurred_at))
        for r in rows
        if r.user_id and r.occurred_at and _dt_to_utc(r.occurred_at) < cutoff_dt
    ]
    random.shuffle(pairs)

    # If the pool is still thin after 20x fetch, fall back to the naive events
    # table. events rows are distributed in Murmur3 hash order so a large LIMIT
    # is more likely to hit early-date events than events_by_user_month which
    # is partitioned by (user_id, year_month) — early months may be sparse in
    # the token ring sample. This fallback is only for pool-building (startup),
    # never for benchmarked queries.
    if len(pairs) < 10:
        warn(f"Optimised Q6 pool thin ({len(pairs)} pairs before {cutoff}) — "
             f"falling back to naive events table for anchor pool.")
        naive_cluster, naive_session = get_session(
            keyspace=KEYSPACE_NAIVE, request_timeout=300.0
        )
        naive_session.default_fetch_size = 5_000
        try:
            naive_rows = list(naive_session.execute(
                f"SELECT user_id, occurred_at FROM events LIMIT {pool_size * 50}"
            ))
            pairs = [
                (r.user_id, _dt_to_utc(r.occurred_at))
                for r in naive_rows
                if r.user_id and r.occurred_at and _dt_to_utc(r.occurred_at) < cutoff_dt
            ]
            random.shuffle(pairs)
        finally:
            naive_cluster.shutdown()

    if not pairs:
        raise RuntimeError(
            f"No events found before cutoff {cutoff} in either keyspace. "
            "Check that the cutoff date falls within the dataset range."
        )

    info(f"Optimised Q6 anchor pool: {len(pairs)} pairs before {cutoff}")
    return pairs[:pool_size]

def _anchor_window(anchor_dt: datetime):
    return anchor_dt - timedelta(days=15), anchor_dt + timedelta(days=15)

def _q6_naive_single_scan(pairs: list, cutoff: date) -> list:
    """
    One Q6 naive iteration with its own session — allows JVM GC between scans.
    Two-pass: pass 1 scans minimal columns (id, user_id, occurred_at) to find
    matching rows; pass 2 does PK lookups for full details.
    Cannot use make_query_fn + harness loop: sustained back-to-back full-table
    scans cause JVM GC pressure → ConnectionShutdown (CRC mismatch).
    """
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)
    c2, s2 = get_session(keyspace=KEYSPACE_NAIVE, request_timeout=300.0)
    s2.default_fetch_size = 5_000
    try:
        user_id, anchor_dt = random.choice(pairs)
        start, end = _anchor_window(anchor_dt)
        end = min(end, cutoff_dt)
        start_n = start.replace(tzinfo=None)
        end_n   = end.replace(tzinfo=None)

        # Pass 1: minimal columns only
        matching_ids = []
        for row in s2.execute("SELECT id, user_id, occurred_at FROM events"):
            if row.user_id != user_id or row.occurred_at is None:
                continue
            occ = row.occurred_at.replace(tzinfo=None) if row.occurred_at.tzinfo else row.occurred_at
            if start_n <= occ < end_n:
                matching_ids.append((row.id, row.occurred_at))

        # Pass 2: PK lookups for full details
        results = []
        for eid, _ in matching_ids:
            full = s2.execute(
                "SELECT id, event_type, occurred_at, product_id, session_id, metadata "
                "FROM events WHERE id = %s", (eid,),
            ).one()
            if full:
                results.append(full)

        results.sort(key=lambda r: r.occurred_at or datetime.min, reverse=True)
        return results
    finally:
        c2.shutdown()


def make_q6_naive_fn(session, pairs: list, cutoff: date):
    # session arg kept for API consistency but not used — reconnect per iteration
    def _run():
        return _q6_naive_single_scan(pairs, cutoff)
    return _run

def _months_in_range(start: datetime, end: datetime) -> list:
    months  = []
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    end_m   = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while current <= end_m:
        months.append(current.strftime("%Y-%m"))
        current = current.replace(month=current.month + 1) if current.month < 12 \
                  else current.replace(year=current.year + 1, month=1)
    return months

def make_q6_optimised_fn(session, pairs: list):
    def _run():
        if not pairs:
            return []   # empty anchor pool — skip gracefully
        user_id, anchor_dt = random.choice(pairs)
        start, end = _anchor_window(anchor_dt)
        months = _months_in_range(start, end)
        all_rows = []
        for ym in months:
            all_rows.extend(session.execute(
                "SELECT id, event_type, occurred_at, product_id "
                "FROM events_by_user_month "
                "WHERE user_id = %s AND year_month = %s "
                "AND occurred_at >= %s AND occurred_at < %s",
                (user_id, ym, start, end),
            ))
        if len(months) > 1:
            all_rows.sort(key=lambda r: r.occurred_at or datetime.min, reverse=True)
        return all_rows
    return _run

# ── Q7 ────────────────────────────────────────────────────────────────────────

def _gap_fill(daily: dict, start_d: date, end_d: date):
    result, current = [], start_d
    while current <= end_d:
        result.append((current, daily.get(current, Decimal("0"))))
        current += timedelta(days=1)
    return result

def _rolling_avg(series, window):
    avgs = []
    for i in range(len(series)):
        chunk = series[max(0, i - window + 1): i + 1]
        avgs.append(sum(chunk) / Decimal(len(chunk)))
    return avgs

def make_q7_naive_fn(session, cutoff: date, data_min: date,
                     user_subs: dict, sub_id_to_tier: dict):
    max_start = max(0, (cutoff - data_min).days - WINDOW_DAYS7)
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)

    def _run():
        offset  = random.randint(0, max_start) if max_start > 0 else 0
        start_d = data_min + timedelta(days=offset)
        end_d   = min(start_d + timedelta(days=WINDOW_DAYS7 - 1), cutoff)
        start_ts = datetime.combine(start_d, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_ts   = datetime.combine(end_d,   datetime.max.time()).replace(tzinfo=timezone.utc)
        invoices = list(session.execute(
            "SELECT id, user_id, invoice_type, total_usd, subscription_id, created_at "
            "FROM invoices WHERE status = 'paid' "
            "AND created_at >= %s AND created_at <= %s ALLOW FILTERING",
            (start_ts, end_ts),
        ))
        daily = defaultdict(lambda: defaultdict(Decimal))
        for inv in invoices:
            if not inv.created_at:
                continue
            ca = _dt_to_utc(inv.created_at)
            if inv.invoice_type == "subscription" and inv.subscription_id:
                tier_id = sub_id_to_tier.get(str(inv.subscription_id))
            else:
                tier_id = None
                uid = str(inv.user_id) if inv.user_id else None
                if uid:
                    for started_at, tid in user_subs.get(uid, []):
                        if started_at <= ca:
                            tier_id = tid
                        else:
                            break
            if tier_id is None:
                continue
            daily[tier_id][ca.date()] += inv.total_usd or Decimal("0")
        results = []
        for tier_id in TIER_IDS:
            filled  = _gap_fill(daily[tier_id], start_d, end_d)
            rolling = _rolling_avg([t for _, t in filled], ROLLING_DAYS)
            for i, (d, total) in enumerate(filled):
                results.append({"day": str(d), "tier_name": TIER_NAMES[tier_id],
                                 "daily_revenue": total, "rolling_7d_avg": rolling[i]})
        return results
    return _run

def make_q7_optimised_fn(session, cutoff: date, data_min: date):
    max_start = max(0, (cutoff - data_min).days - WINDOW_DAYS7)

    def _run():
        offset  = random.randint(0, max_start) if max_start > 0 else 0
        start_d = data_min + timedelta(days=offset)
        end_d   = min(start_d + timedelta(days=WINDOW_DAYS7 - 1), cutoff)
        results = []
        for tier_id in TIER_IDS:
            rows = list(session.execute(
                "SELECT paid_date, total_usd, tier_name "
                "FROM invoices_by_tier "
                "WHERE tier_id = %s AND paid_date >= %s AND paid_date <= %s",
                (tier_id, start_d, end_d),
            ))
            daily = defaultdict(Decimal)
            for r in rows:
                if r.paid_date and r.total_usd:
                    daily[_to_pydate(r.paid_date)] += r.total_usd
            filled  = _gap_fill(daily, start_d, end_d)
            rolling = _rolling_avg([t for _, t in filled], ROLLING_DAYS)
            for i, (d, total) in enumerate(filled):
                results.append({"day": str(d), "tier_name": TIER_NAMES[tier_id],
                                 "daily_revenue": total, "rolling_7d_avg": rolling[i]})
        return results
    return _run

# ── runner ────────────────────────────────────────────────────────────────────

def run_scaled_query(
    query_id:    str,
    schema:      str,
    scale_pct:   int,
    cutoff:      date,
    session,
    cutoffs:     dict,
    iterations:  int,
    results_dir: str,
) -> dict | None:

    suffix      = f"scale{scale_pct}"
    db_label    = f"cassandra_{schema}"
    fname       = f"cassandra_{schema}_{query_id.lower()}_{suffix}.json"
    output_path = os.path.join(results_dir, fname)

    try:
        concurrency = 1

        if query_id == "Q1":
            if schema == "naive":
                query_fn = make_q1_naive_fn(session, cutoff)
            else:
                query_fn = make_q1_optimised_fn(session, cutoff)
            label = f"Q1 {schema} at {scale_pct}% scale (cutoff {cutoff})"

        elif query_id == "Q2":
            if schema == "naive":
                pool = fetch_invoice_pool_naive(session, cutoff)
                query_fn = make_q2_naive_fn(session, pool)
            else:
                pool = fetch_invoice_pool_optimised(session, cutoff)
                query_fn = make_q2_optimised_fn(session, pool)
            label = f"Q2 {schema} at {scale_pct}% scale (cutoff {cutoff}, pool {len(pool)})"

        elif query_id == "Q3":
            concurrency = 50
            if schema == "naive":
                pool = fetch_user_pool_naive(session, scale_pct)
                query_fn = make_q3_naive_fn(session, pool)
            else:
                pool = fetch_user_pool_optimised(session, scale_pct)
                query_fn = make_q3_optimised_fn(session, pool)
            label = (f"Q3 {schema} at {scale_pct}% row-based scale "
                     f"({len(pool)} users). No date cutoff — sessions in single year.")

        elif query_id == "Q4":
            if schema == "naive":
                pool = fetch_product_pool_naive(session, cutoff)
                query_fn = make_q4_naive_fn(session, pool, cutoff)
                label = (f"Q4 naive at {scale_pct}% scale (cutoff {cutoff}). "
                         "ALLOW FILTERING (order_items by product_id) + PK lookups "
                         "(orders, cutoff-filtered) + Python co-count from pre-loaded "
                         "order_items dict (startup cost). Full per-iter scan eliminated "
                         "for scalability feasibility — naive schema penalty preserved.")
            else:
                pool = fetch_product_pool_optimised(session)
                query_fn = make_q4_optimised_fn(session, pool)
                label = (f"Q4 optimised at {scale_pct}% scale — also_bought precomputed "
                         "over full dataset; date-range filtering not applicable.")

        elif query_id == "Q5":
            if schema == "naive":
                query_fn = make_q5_naive_fn(session, cutoff)
            else:
                query_fn = make_q5_optimised_fn(session, cutoff)
            label = f"Q5 {schema} at {scale_pct}% scale (cutoff {cutoff})"

        elif query_id == "Q6":
            if schema == "naive":
                # Q6 naive uses a custom timing loop — see _q6_naive_single_scan.
                # Cannot use run_benchmark: sustained full-table scans cause JVM GC
                # pressure → ConnectionShutdown. Session reconnects per iteration.
                # Iterations capped at Q6_NAIVE_ITERATIONS (30) regardless of
                # --iterations flag: each scan takes ~2 min; 30 = ~60 min per scale.
                pairs = fetch_anchor_pool_naive(session, cutoff)
                label = (f"Q6 naive at {scale_pct}% scale (cutoff {cutoff}). "
                         "Two-pass paginated full scan (minimal cols + PK lookups). "
                         f"Iterations={Q6_NAIVE_ITERATIONS} (not {iterations}): "
                         "each scan ~2 min; standard 1000 = 33+ hours.")

                import time as _time
                print(f"  Q6 naive scalability: {Q6_NAIVE_ITERATIONS} iterations "
                      f"(warmup=2) @ {scale_pct}% scale")
                for i in range(2):
                    _q6_naive_single_scan(pairs, cutoff)
                    print(f"    warm-up {i+1}/2 done")

                timings = []
                w_start = _time.perf_counter()
                for i in range(Q6_NAIVE_ITERATIONS):
                    t0 = _time.perf_counter()
                    _q6_naive_single_scan(pairs, cutoff)
                    timings.append((_time.perf_counter() - t0) * 1_000)
                    if (i + 1) % 5 == 0 or i == 0:
                        elapsed = _time.perf_counter() - w_start
                        print(f"  [{elapsed:>6.0f}s] {i+1}/{Q6_NAIVE_ITERATIONS} "
                              f"last={timings[-1]:.0f}ms")
                wall_elapsed = _time.perf_counter() - w_start

                s_t = sorted(timings)
                n   = len(s_t)
                mean = sum(s_t) / n
                var  = sum((x - mean) ** 2 for x in s_t) / n
                def _pct(p):
                    k = max(0, min(math.ceil((p / 100) * n) - 1, n - 1))
                    return round(s_t[k], 4)
                stats = {
                    "p50": _pct(50), "p95": _pct(95), "p99": _pct(99),
                    "mean": round(mean, 4),
                    "std_dev": round(math.sqrt(var), 4),
                    "min": round(s_t[0], 4), "max": round(s_t[-1], 4),
                }
                result = {
                    "db": "cassandra_naive", "query_id": "Q6",
                    "label": label, "scale_pct": scale_pct,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iterations": n, "warmup_runs": 2, "concurrency": 1,
                    "wall_time_s": round(wall_elapsed, 3),
                    "latency_ms": stats,
                    "raw_timings_ms": [round(t, 4) for t in timings],
                }
                os.makedirs(results_dir, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                ok(f"Q6 naive @ {scale_pct}% → {output_path}")
                return result

            else:
                pairs = fetch_anchor_pool_optimised(session, cutoff)
                if not pairs:
                    # Cutoff-restricted pool is empty (cutoff predates most events).
                    # Fall back to unrestricted pool — methodologically defensible for
                    # optimised Q6: the query reads 1-2 bounded partitions regardless
                    # of total dataset size. Its cost is independent of scale, so the
                    # anchor point does not need to respect the date cutoff.
                    warn(f"Q6 optimised @ {scale_pct}%: pool empty before cutoff "
                         f"{cutoff} — falling back to unrestricted anchor pool.")
                    rows = list(session.execute(
                        "SELECT user_id, occurred_at FROM events_by_user_month LIMIT 5000"
                    ))
                    pairs = [
                        (r.user_id, _dt_to_utc(r.occurred_at))
                        for r in rows if r.user_id and r.occurred_at
                    ]
                    random.shuffle(pairs)
                    pairs = pairs[:1000]
                    label_suffix = (f" Anchor pool unrestricted (cutoff {cutoff} predates "
                                    "most events; optimised Q6 cost is cutoff-independent).")
                else:
                    label_suffix = ""
                query_fn = make_q6_optimised_fn(session, pairs)
                label = (f"Q6 optimised at {scale_pct}% scale (cutoff {cutoff}). "
                         "1-2 partition reads, anchor pool restricted to events before cutoff."
                         + label_suffix)

        elif query_id == "Q7":
            data_min = cutoffs["min_date"]
            if schema == "naive":
                subs = list(session.execute(
                    "SELECT id, user_id, tier_id, started_at FROM subscriptions"
                ))
                sub_id_to_tier = {str(s.id): s.tier_id for s in subs if s.id}
                user_subs = defaultdict(list)
                for s in subs:
                    if s.user_id and s.started_at and s.tier_id:
                        user_subs[str(s.user_id)].append((_dt_to_utc(s.started_at), s.tier_id))
                for uid in user_subs:
                    user_subs[uid].sort(key=lambda x: x[0])
                query_fn = make_q7_naive_fn(session, cutoff, data_min,
                                            dict(user_subs), sub_id_to_tier)
            else:
                query_fn = make_q7_optimised_fn(session, cutoff, data_min)
            label = f"Q7 {schema} at {scale_pct}% scale (cutoff {cutoff})"

        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db=db_label,
            query_id=query_id,
            label=label,
            iterations=iterations,
            concurrency=concurrency,
            output_path=output_path,
        )
        ok(f"{query_id} {schema} @ {scale_pct}% → {output_path}")
        return result

    except Exception as e:
        fail(f"{query_id} {schema} @ {scale_pct}% failed: {e}")
        import traceback; traceback.print_exc()
        return None

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cassandra scalability baselines")
    parser.add_argument("--scale",      type=int, choices=[10, 50], default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run",    action="store_true", dest="dry_run")
    parser.add_argument("--only",       nargs="+", metavar="Q")
    parser.add_argument("--schema",     choices=["naive", "optimised", "both"], default="both")
    parser.add_argument("--results-dir-naive",     type=str, default=RESULTS_NAIVE,
                        dest="results_dir_naive")
    parser.add_argument("--results-dir-optimised", type=str, default=RESULTS_OPTIMISED,
                        dest="results_dir_optimised")
    args = parser.parse_args()

    iterations = 100 if args.dry_run else args.iterations
    scales     = [10, 50] if args.scale is None else [args.scale]
    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]
    schemas = ["naive", "optimised"] if args.schema == "both" else [args.schema]

    print("\n" + "═" * 60)
    print("  Cassandra — Scalability Baselines")
    print("═" * 60)
    print(f"  Schemas     : {schemas}")
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")

    os.makedirs(args.results_dir_naive, exist_ok=True)
    os.makedirs(args.results_dir_optimised, exist_ok=True)

    # Compute cutoffs from naive keyspace (same anchor as PostgreSQL / Neo4j)
    # Naive Q6 needs 300s timeout for its full scan
    naive_cluster, naive_session = get_session(
        keyspace=KEYSPACE_NAIVE, request_timeout=300.0
    )
    try:
        cutoffs = compute_cutoffs(naive_session)
    finally:
        # Keep naive_session open only for cutoff computation; reopen per schema below
        naive_cluster.shutdown()

    print(f"  Dataset range : {cutoffs['min_date']} → {cutoffs['max_date']} "
          f"({cutoffs['range_days']} days)")
    print(f"  10% cutoff    : {cutoffs['cutoff_10pct']}")
    print(f"  50% cutoff    : {cutoffs['cutoff_50pct']}")

    all_results = []
    failed      = []
    total_start = time.perf_counter()

    for schema in schemas:
        keyspace = KEYSPACE_NAIVE if schema == "naive" else KEYSPACE_OPTIMISED
        timeout  = 300.0 if schema == "naive" else 30.0
        results_dir = (args.results_dir_naive if schema == "naive"
                       else args.results_dir_optimised)

        cluster, session = get_session(keyspace=keyspace, request_timeout=timeout)
        info(f"Connected to {keyspace} (timeout={timeout}s)")

        try:
            for scale in scales:
                cutoff = cutoffs[f"cutoff_{scale}pct"]
                print(f"\n{'═'*60}")
                print(f"  Cassandra {schema} — {scale}% scale (cutoff: {cutoff})")
                print(f"{'═'*60}")

                for qid in queries:
                    print(f"\n{'─'*60}")
                    print(f"  {qid} | {schema} | {scale}%")
                    print(f"{'─'*60}")
                    result = run_scaled_query(
                        query_id=qid,
                        schema=schema,
                        scale_pct=scale,
                        cutoff=cutoff,
                        session=session,
                        cutoffs=cutoffs,
                        iterations=iterations,
                        results_dir=results_dir,
                    )
                    if result:
                        result["scale_pct"]   = scale
                        result["schema"]      = schema
                        result["cutoff_date"] = str(cutoff)
                        all_results.append(result)
                    else:
                        failed.append(f"{qid}_{schema}@{scale}%")
        finally:
            cluster.shutdown()

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'═'*60}")
    print(f"  SCALABILITY SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Query':<6} {'Schema':<12} {'Scale':>6} {'p50':>10} {'p95':>10} {'p99':>10}")
    print(f"  {'─'*6} {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for r in all_results:
        lms = r.get("latency_ms", {})
        print(
            f"  {r['query_id']:<6} {r.get('schema',''):<12} "
            f"{str(r.get('scale_pct','?'))+'%':>6} "
            f"{lms.get('p50', 0):>9.2f}ms "
            f"{lms.get('p95', 0):>9.2f}ms "
            f"{lms.get('p99', 0):>9.2f}ms"
        )

    # Save summary
    summary_naive_path = os.path.join(args.results_dir_naive, "cassandra_naive_scalability_summary.json")
    summary_opt_path   = os.path.join(args.results_dir_optimised, "cassandra_optimised_scalability_summary.json")
    summary_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db": "cassandra",
        "cutoffs": {k: str(v) for k, v in cutoffs.items()},
        "benchmarks": all_results,
    }
    for path in (summary_naive_path, summary_opt_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary_data, f, indent=2)
    info(f"Summaries saved → {summary_naive_path}")
    info(f"              → {summary_opt_path}")

    print(f"\n  Total wall time : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Completed : {len(all_results)} / {len(all_results) + len(failed)}")

    if failed:
        warn(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All Cassandra scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()