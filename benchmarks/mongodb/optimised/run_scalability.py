"""
benchmarks/mongodb/optimised/run_scalability.py — MongoDB Optimised Scalability
================================================================================
Re-runs Q1-Q7 at 10% and 50% data scale for the MongoDB optimised implementation.

Mirrors benchmarks/mongodb/naive/run_scalability.py exactly in methodology —
scale is defined by date range cutoff computed from the events collection,
Q3 uses row-based scaling — so curves are directly comparable between naive
and optimised, and both are comparable to the PostgreSQL baseline.

Optimised schema differences reflected here:
  Q1  — total_usd / monthly_price_usd are native floats (no float() conversion)
  Q2  — single find_one on invoices; lines are embedded (invoice_lines gone)
  Q3  — cart is a native BSON list (no json.loads)
  Q4  — single pipeline on orders with multikey index on items.product_id
         (order_items collection gone)
  Q5  — is_active is a bool; $text index has field weights
  Q6  — metadata is a native BSON subdocument (no json.loads)
  Q7  — fully server-side: $group → $densify → $setWindowFields pipeline
         (requires MongoDB 5.1+; fails loudly if not met)

Output:
    results/mongodb_optimised_Q{n}_scale10.json
    results/mongodb_optimised_Q{n}_scale50.json
    results/mongodb_optimised_scalability_summary.json

Usage:
    python run_scalability.py                    # both scales, Q1-Q7, 1000 iterations
    python run_scalability.py --scale 10         # 10% scale only
    python run_scalability.py --scale 50         # 50% scale only
    python run_scalability.py --iterations 100   # smoke test
    python run_scalability.py --only Q1 Q6       # specific queries only
    python run_scalability.py --dry-run          # 100 iterations, both scales
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from benchmarks.harness import run_benchmark
from benchmarks.mongodb.mongo_conn import get_db

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "results"
)

# ── colour helpers ────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── constants matching individual benchmark files ─────────────────────────────

WINDOW_DAYS_Q6     = 30
WINDOW_DAYS_Q7     = 183
CONFIRMED_STATUSES = ["confirmed", "shipped", "delivered"]

SEARCH_TERMS = [
    "brushes", "typography", "illustration", "photography", "animation",
    "branding", "mockup", "watercolour", "photoshop brushes", "video editing",
    "certificate course", "logo design", "colour palette", "font pack",
    "texture pack", "motion graphics", "social media", "icon set",
    "web design", "canva template", "vector illustration", "beginner design",
    "digital course", "design assets", "procreate brushes",
]

# ── date helpers ──────────────────────────────────────────────────────────────

def iso(d: date) -> str:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat()

def parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00")).date()

def parse_dt(s: str) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))

# ── cutoff computation ────────────────────────────────────────────────────────

def compute_cutoffs(db) -> dict:
    """
    Identical to naive run_scalability.py — anchored on events.occurred_at
    so the cutoff dates are the same across both phases and PostgreSQL.
    """
    result = db["events"].aggregate([
        {"$group": {
            "_id":      None,
            "min_date": {"$min": "$occurred_at"},
            "max_date": {"$max": "$occurred_at"},
        }}
    ])
    row = next(result, None)
    if not row:
        raise RuntimeError("No events found — run the optimised loader first.")

    min_date   = parse_iso_date(row["min_date"])
    max_date   = parse_iso_date(row["max_date"])
    range_days = (max_date - min_date).days

    return {
        "min_date":     min_date,
        "max_date":     max_date,
        "range_days":   range_days,
        "cutoff_10pct": min_date + timedelta(days=int(range_days * 0.10)),
        "cutoff_50pct": min_date + timedelta(days=int(range_days * 0.50)),
    }

# ── one-time reference data loaders (Q1 / Q7) ────────────────────────────────

def load_subs_by_id(db) -> dict:
    return {
        d["_id"]: d
        for d in db["subscriptions"].find(
            {}, {"_id": 1, "tier_id": 1, "user_id": 1, "started_at": 1}
        )
    }

def load_subs_by_user(db) -> dict:
    by_user = defaultdict(list)
    for d in db["subscriptions"].find(
        {}, {"_id": 1, "tier_id": 1, "user_id": 1, "started_at": 1}
    ):
        by_user[d["user_id"]].append(d)
    for uid in by_user:
        by_user[uid].sort(key=lambda s: s["started_at"], reverse=True)
    return dict(by_user)

def load_pricing(db) -> list[dict]:
    """
    Optimised: monthly_price_usd is a native float — no float() conversion
    needed in resolve_price().
    """
    return list(db["subscription_tier_pricing"].find(
        {}, {"_id": 0, "tier_id": 1, "valid_from": 1,
             "valid_to": 1, "monthly_price_usd": 1}
    ))

def load_tiers(db) -> dict:
    return {
        d["_id"]: d["name"]
        for d in db["subscription_tiers"].find({}, {"_id": 1, "name": 1})
    }

def resolve_price(tier_id: str, invoice_dt: datetime,
                  pricing_rows: list[dict]) -> float | None:
    for row in pricing_rows:
        if row["tier_id"] != tier_id:
            continue
        valid_from = parse_dt(row["valid_from"])
        valid_to   = parse_dt(row["valid_to"]) if row.get("valid_to") else None
        if valid_from <= invoice_dt and (valid_to is None or valid_to > invoice_dt):
            return row["monthly_price_usd"]   # already a float in optimised schema
    return None

# ── ID pool helpers ───────────────────────────────────────────────────────────

def fetch_invoice_id_pool(db, iso_cutoff: str, pool_size: int = 1000) -> list[str]:
    docs = list(db["invoices"].aggregate([
        {"$match": {"created_at": {"$lt": iso_cutoff}}},
        {"$sample": {"size": pool_size}},
        {"$project": {"_id": 1}},
    ]))
    if not docs:
        raise RuntimeError(f"No invoices before {iso_cutoff}")
    return [d["_id"] for d in docs]

def fetch_user_id_pool_sessions(db, scale_pct: int, pool_size: int = 1000) -> list[str]:
    """
    Row-based scaling for Q3 — identical rationale to naive and PostgreSQL.
    Sessions were generated in a narrow 2025 window; date-range scaling
    would return near-zero rows at 10% scale.
    """
    all_users = [
        d["_id"]
        for d in db["sessions"].aggregate([{"$group": {"_id": "$user_id"}}])
    ]
    if not all_users:
        raise RuntimeError("No sessions found.")
    limit  = max(1, int(len(all_users) * scale_pct / 100))
    subset = random.sample(all_users, min(limit, len(all_users)))
    return random.choices(subset, k=min(pool_size, len(subset)))

def fetch_product_id_pool(db, iso_cutoff: str, pool_size: int = 1000) -> list[str]:
    """
    Optimised: order_items is eliminated — products are drawn from the
    embedded items array inside orders.
    """
    docs = list(db["orders"].aggregate([
        {"$match": {
            "status":     {"$in": CONFIRMED_STATUSES},
            "created_at": {"$lt": iso_cutoff},
        }},
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.product_id"}},
        {"$sample": {"size": pool_size}},
    ]))
    if not docs:
        raise RuntimeError(f"No products in confirmed orders before {iso_cutoff}")
    return [d["_id"] for d in docs]

def fetch_user_anchor_pool_events(db, iso_cutoff: str, pool_size: int = 1000) -> list[tuple[str, str]]:
    """
    Sample (user_id, occurred_at) pairs from events before iso_cutoff.
    The window is centred on the anchor event so every iteration is
    guaranteed to return at least one row — identical fix to standalone q6.
    """
    docs = list(db["events"].aggregate([
        {"$match": {"occurred_at": {"$lt": iso_cutoff}}},
        {"$sample": {"size": pool_size}},
        {"$project": {"_id": 0, "user_id": 1, "occurred_at": 1}},
    ]))
    if not docs:
        raise RuntimeError(f"No events before {iso_cutoff}")
    return [(d["user_id"], d["occurred_at"]) for d in docs]

# ── query function factories ──────────────────────────────────────────────────

def make_q1_fn(db, iso_cutoff: str, subs_by_id, subs_by_user, pricing, tiers):
    """
    Mirrors q1_revenue.py with created_at < iso_cutoff filter.
    Optimised: total_usd and monthly_price_usd are native floats.
    """
    def _run():
        invoices = list(db["invoices"].find(
            {"status": "paid", "created_at": {"$lt": iso_cutoff}},
            {"_id": 1, "user_id": 1, "invoice_type": 1,
             "total_usd": 1, "created_at": 1, "subscription_id": 1},
        ))
        groups: dict = defaultdict(lambda: {"count": 0, "revenue": 0.0})
        for inv in invoices:
            inv_dt   = parse_dt(inv["created_at"])
            inv_type = inv.get("invoice_type", "")
            tier_id  = None
            if inv_type == "subscription":
                sub = subs_by_id.get(inv.get("subscription_id", ""))
                if sub:
                    tier_id = sub["tier_id"]
            elif inv_type == "marketplace":
                for sub in subs_by_user.get(inv.get("user_id", ""), []):
                    if parse_dt(sub["started_at"]) <= inv_dt:
                        tier_id = sub["tier_id"]
                        break
            if tier_id is None:
                continue
            price = resolve_price(tier_id, inv_dt, pricing)
            if price is None:
                continue
            month_key = inv_dt.strftime("%Y-%m")
            key = (month_key, tier_id, price)
            groups[key]["count"]   += 1
            groups[key]["revenue"] += inv.get("total_usd", 0)  # native float
        _ = [
            {"month": k[0], "tier_name": tiers.get(k[1], k[1]),
             "price": k[2], "revenue": round(v["revenue"], 2)}
            for k, v in groups.items()
        ]
    return _run


def make_q2_fn(db, invoice_ids: list[str]):
    """
    Optimised: single find_one — lines are embedded in the invoice document.
    No invoice_lines collection query needed.
    """
    def _run():
        inv_id = random.choice(invoice_ids)
        # One query: invoice + embedded lines + embedded customer snapshot
        db["invoices"].find_one({"_id": inv_id})
    return _run


def make_q3_fn(db, user_ids: list[str]):
    """
    Optimised: cart is a native BSON list — no json.loads().
    """
    def _run():
        uid = random.choice(user_ids)
        doc = db["sessions"].find_one(
            {"user_id": uid},
            sort=[("last_active_at", -1)],
        )
        if doc:
            _ = doc.get("cart", [])  # already a list; no deserialisation
    return _run


def make_q4_fn(db, product_ids: list[str], iso_cutoff: str):
    """
    Optimised: single $match → $unwind → $group pipeline on orders.
    Uses multikey index on items.product_id. order_items collection gone.
    """
    def _run():
        seed = random.choice(product_ids)
        pipeline = [
            {"$match": {
                "items.product_id": seed,
                "status":           {"$in": CONFIRMED_STATUSES},
                "created_at":       {"$lt": iso_cutoff},
            }},
            {"$unwind": "$items"},
            {"$match": {"items.product_id": {"$ne": seed}}},
            {"$group": {
                "_id":           "$items.product_id",
                "co_buy_count":  {"$sum": 1},
                "product_name":  {"$first": "$items.product_name"},
                "product_type":  {"$first": "$items.product_type"},
            }},
            {"$sort":  {"co_buy_count": -1}},
            {"$limit": 10},
        ]
        list(db["orders"].aggregate(pipeline))
    return _run


def make_q5_fn(db, iso_cutoff: str):
    """
    Optimised: is_active is a bool; text index has field weights.
    """
    def _run():
        term = random.choice(SEARCH_TERMS)
        list(db["products"].find(
            {
                "$text":      {"$search": term},
                "is_active":  True,           # bool, not string "True"
                "created_at": {"$lt": iso_cutoff},
            },
            {
                "score":        {"$meta": "textScore"},
                "name":         1,
                "product_type": 1,
                "price_usd":    1,
                "attributes":   1,            # native BSON dict
            }
        ).sort([("score", {"$meta": "textScore"})]).limit(20))
    return _run


def make_q6_fn(db, pairs: list[tuple[str, str]]):
    """
    Optimised: metadata returned as native BSON subdocument (no json.loads).
    Window centred on a real event — guaranteed non-empty result.
    """
    def _run():
        user_id, anchor_iso = random.choice(pairs)
        anchor = datetime.fromisoformat(anchor_iso.strip().replace("Z", "+00:00"))
        start  = (anchor - timedelta(days=15)).isoformat()
        end    = (anchor + timedelta(days=15)).isoformat()
        list(db["events"].find(
            {"user_id":    user_id,
             "occurred_at": {"$gte": start, "$lt": end}},
            {"event_type": 1, "occurred_at": 1, "product_id": 1,
             "session_id": 1, "metadata": 1},
        ).sort("occurred_at", -1))
    return _run


def make_q7_fn(db, iso_cutoff: str, cutoff_date: date, data_min: date):
    """
    Optimised: fully server-side pipeline using $densify + $setWindowFields.
    Requires MongoDB 5.1+. Fails loudly if the server does not support it.

    Window is a random WINDOW_DAYS_Q7 range within [data_min, cutoff_date],
    matching the standalone q7_rolling_revenue.py fix — no hardcoded anchor.
    """
    def _run():
        delta     = (cutoff_date - data_min).days
        max_start = max(0, delta - WINDOW_DAYS_Q7)
        start     = data_min + timedelta(days=random.randint(0, max_start))
        end       = min(start + timedelta(days=WINDOW_DAYS_Q7 - 1), cutoff_date)
        iso_start = datetime(start.year, start.month, start.day,
                             tzinfo=timezone.utc).isoformat()
        iso_end   = datetime(end.year, end.month, end.day, 23, 59, 59,
                             tzinfo=timezone.utc).isoformat()

        pipeline = [
            {"$match": {
                "status":     "paid",
                "created_at": {"$gte": iso_start, "$lte": iso_end,
                               "$lt": iso_cutoff},
            }},
            {"$addFields": {
                "day_date": {
                    "$dateTrunc": {
                        "date": {"$dateFromString": {"dateString": "$created_at"}},
                        "unit": "day",
                    }
                },
            }},
            {"$group": {
                "_id": {
                    "day":             "$day_date",
                    "subscription_id": "$subscription_id",
                },
                "daily_revenue": {"$sum": "$total_usd"},  # native float
            }},
            {"$lookup": {
                "from":         "subscriptions",
                "localField":   "_id.subscription_id",
                "foreignField": "_id",
                "as":           "sub_docs",
            }},
            {"$addFields": {
                "tier_id": {"$arrayElemAt": ["$sub_docs.tier_id", 0]},
            }},
            {"$project": {"sub_docs": 0}},
            {"$lookup": {
                "from":         "subscription_tiers",
                "localField":   "tier_id",
                "foreignField": "_id",
                "as":           "tier_docs",
            }},
            {"$addFields": {
                "tier_name": {"$arrayElemAt": ["$tier_docs.name", 0]},
            }},
            {"$project": {"tier_docs": 0, "tier_id": 0}},
            {"$group": {
                "_id": {
                    "day":       "$_id.day",
                    "tier_name": "$tier_name",
                },
                "daily_revenue": {"$sum": "$daily_revenue"},
            }},
            {"$densify": {
                "field": "_id.day",
                "partitionByFields": ["_id.tier_name"],
                "range": {"step": 1, "unit": "day", "bounds": "partition"},
            }},
            {"$fill": {
                "sortBy":      {"_id.day": 1},
                "partitionBy": "$_id.tier_name",
                "output": {"daily_revenue": {"value": 0}},
            }},
            {"$setWindowFields": {
                "partitionBy": "$_id.tier_name",
                "sortBy":      {"_id.day": 1},
                "output": {
                    "rolling_7d_avg": {
                        "$avg": "$daily_revenue",
                        "window": {"documents": [-6, 0]},
                    },
                },
            }},
            {"$project": {
                "_id":                0,
                "day":                {"$dateToString": {"format": "%Y-%m-%d", "date": "$_id.day"}},
                "tier_name":          "$_id.tier_name",
                "daily_revenue_usd":  {"$round": ["$daily_revenue", 2]},
                "rolling_7d_avg_usd": {"$round": ["$rolling_7d_avg", 2]},
            }},
            {"$sort": {"day": 1, "tier_name": 1}},
        ]

        list(db["invoices"].aggregate(pipeline))
    return _run

# ── scaled query runner ───────────────────────────────────────────────────────

def run_scaled_query(
    query_id:    str,
    scale_pct:   int,
    cutoff_date: date,
    db,
    iterations:  int,
    results_dir: str,
    ref_data:    dict,
) -> dict | None:

    iso_cutoff  = iso(cutoff_date)
    suffix      = f"scale{scale_pct}"
    output_path = os.path.join(
        results_dir, f"mongodb_optimised_{query_id}_{suffix}.json"
    )

    try:
        if query_id == "Q1":
            query_fn    = make_q1_fn(
                db, iso_cutoff,
                ref_data["subs_by_id"], ref_data["subs_by_user"],
                ref_data["pricing"],    ref_data["tiers"],
            )
            concurrency = 1
            label = (
                f"Q1 at {scale_pct}% scale — invoice fetch cutoff: {iso_cutoff}. "
                "Optimised: native float total_usd, compound subscription index."
            )

        elif query_id == "Q2":
            pool        = fetch_invoice_id_pool(db, iso_cutoff)
            query_fn    = make_q2_fn(db, pool)
            concurrency = 1
            label = (
                f"Q2 at {scale_pct}% scale — invoice pool before {iso_cutoff}. "
                "Optimised: single find_one — lines embedded in invoice document."
            )

        elif query_id == "Q3":
            pool        = fetch_user_id_pool_sessions(db, scale_pct)
            query_fn    = make_q3_fn(db, pool)
            concurrency = 50
            label = (
                f"Q3 at {scale_pct}% scale — row-based: user pool from "
                f"{scale_pct}% of session users. "
                "Optimised: cart is native BSON list, no json.loads."
            )

        elif query_id == "Q4":
            pool        = fetch_product_id_pool(db, iso_cutoff)
            query_fn    = make_q4_fn(db, pool, iso_cutoff)
            concurrency = 1
            label = (
                f"Q4 at {scale_pct}% scale — orders cutoff: {iso_cutoff}. "
                "Optimised: single pipeline on orders with multikey index "
                "on items.product_id. order_items collection eliminated."
            )

        elif query_id == "Q5":
            query_fn    = make_q5_fn(db, iso_cutoff)
            concurrency = 1
            label = (
                f"Q5 at {scale_pct}% scale — product corpus cutoff: {iso_cutoff}. "
                "Optimised: $text with field weights; is_active as bool."
            )

        elif query_id == "Q6":
            pool        = fetch_user_anchor_pool_events(db, iso_cutoff)
            query_fn    = make_q6_fn(db, pool)
            concurrency = 1
            label = (
                f"Q6 at {scale_pct}% scale — events cutoff: {iso_cutoff}. "
                "Optimised: metadata as native BSON dict, no json.loads. "
                "Window anchored on sampled real event — guaranteed non-empty."
            )

        elif query_id == "Q7":
            query_fn    = make_q7_fn(db, iso_cutoff, cutoff_date, ref_data['data_min'])
            concurrency = 1
            label = (
                f"Q7 at {scale_pct}% scale — invoice window cutoff: {iso_cutoff}. "
                "Optimised: fully server-side $densify + $setWindowFields pipeline. "
                "Requires MongoDB 5.1+."
            )

        else:
            warn(f"Unknown query {query_id}, skipping.")
            return None

        result = run_benchmark(
            query_fn=query_fn,
            db="mongodb_optimised",
            query_id=query_id,
            label=label,
            iterations=iterations,
            concurrency=concurrency,
            output_path=output_path,
        )
        ok(f"{query_id} @ {scale_pct}% → {output_path}")
        return result

    except Exception as e:
        fail(f"{query_id} @ {scale_pct}% failed: {e}")
        import traceback; traceback.print_exc()
        return None

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Optimised scalability baselines (Q1-Q7)"
    )
    parser.add_argument(
        "--scale", type=int, choices=[10, 50], default=None,
        help="Run only one scale (10 or 50). Default: both.",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Iterations per query per scale (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="100 iterations, both scales — quick smoke test",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="Q",
        help="Run only specific queries e.g. --only Q1 Q6",
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR, dest="results_dir",
        help=f"Directory to save results (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    iterations = 100 if args.dry_run else args.iterations
    scales     = [10, 50] if args.scale is None else [args.scale]
    queries    = [f"Q{i}" for i in range(1, 8)]
    if args.only:
        queries = [q.upper() for q in args.only]

    print("\n" + "═" * 60)
    print("  MongoDB Optimised — Scalability Baselines")
    print("═" * 60)
    print(f"  Scales      : {scales}")
    print(f"  Queries     : {queries}")
    print(f"  Iterations  : {iterations} {'(dry-run)' if args.dry_run else ''}")
    print(f"  Results dir : {args.results_dir}")
    if "Q7" in queries:
        print(f"  Note        : Q7 requires MongoDB 5.1+ ($densify + $setWindowFields)")

    os.makedirs(args.results_dir, exist_ok=True)

    db = get_db(schema="optimised")

    # ── compute cutoffs ───────────────────────────────────────────────────────
    info("Computing date range cutoffs from events collection...")
    cutoffs = compute_cutoffs(db)
    print(f"  Dataset range : {cutoffs['min_date']} → {cutoffs['max_date']} "
          f"({cutoffs['range_days']} days)")
    print(f"  10% cutoff    : {cutoffs['cutoff_10pct']}")
    print(f"  50% cutoff    : {cutoffs['cutoff_50pct']}")

    # ── pre-load reference data
    # data_min is used by Q7 to anchor random windows within real data.
    needs_ref = "Q1" in queries
    ref_data  = {"data_min": cutoffs["min_date"]}  # always set for Q7
    if needs_ref:
        info("Pre-loading subscriptions, tiers, and pricing for Q1...")
        ref_data["subs_by_id"]   = load_subs_by_id(db)
        ref_data["subs_by_user"] = load_subs_by_user(db)
        ref_data["pricing"]      = load_pricing(db)
        ref_data["tiers"]        = load_tiers(db)
        print(f"  Loaded {len(ref_data['subs_by_id']):,} subscriptions, "
              f"{len(ref_data['pricing'])} pricing rows, "
              f"{len(ref_data['tiers'])} tiers.")
    elif "Q7" in queries:
        ref_data["tiers"] = load_tiers(db)

    # ── run benchmarks ────────────────────────────────────────────────────────
    all_results = []
    failed      = []
    total_start = time.perf_counter()

    for scale in scales:
        cutoff_date = cutoffs[f"cutoff_{scale}pct"]
        print(f"\n{'═'*60}")
        print(f"  Running at {scale}% scale (cutoff: {cutoff_date})")
        print(f"{'═'*60}")

        for qid in queries:
            print(f"\n{'─'*60}")
            print(f"  {qid} @ {scale}%")
            print(f"{'─'*60}")
            result = run_scaled_query(
                query_id=qid,
                scale_pct=scale,
                cutoff_date=cutoff_date,
                db=db,
                iterations=iterations,
                results_dir=args.results_dir,
                ref_data=ref_data,
            )
            if result:
                result["scale_pct"]   = scale
                result["cutoff_date"] = str(cutoff_date)
                all_results.append(result)
            else:
                failed.append(f"{qid}@{scale}%")

    total_elapsed = time.perf_counter() - total_start

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SCALABILITY RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Query':<8} {'Scale':>6} {'p50':>10} {'p95':>10} {'p99':>10}")
    print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for r in all_results:
        lms = r.get("latency_ms", {})
        print(
            f"  {r['query_id']:<8} {str(r.get('scale_pct','?'))+'%':>6} "
            f"{lms.get('p50', 0):>9.2f}ms "
            f"{lms.get('p95', 0):>9.2f}ms "
            f"{lms.get('p99', 0):>9.2f}ms"
        )

    summary_path = os.path.join(
        args.results_dir, "mongodb_optimised_scalability_summary.json"
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db":           "mongodb_optimised",
        "cutoffs": {
            "min_date":     str(cutoffs["min_date"]),
            "max_date":     str(cutoffs["max_date"]),
            "range_days":   cutoffs["range_days"],
            "cutoff_10pct": str(cutoffs["cutoff_10pct"]),
            "cutoff_50pct": str(cutoffs["cutoff_50pct"]),
        },
        "benchmarks": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    info(f"Combined summary saved → {summary_path}")

    print(f"\n  Total wall time : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Completed       : {len(all_results)} / "
          f"{len(all_results) + len(failed)}")

    if failed:
        warn(f"Failed: {', '.join(failed)}")
        print(f"\n  Re-run failures with:")
        unique_qs = list(dict.fromkeys(q.split("@")[0] for q in failed))
        print(f"    python run_scalability.py --only {' '.join(unique_qs)}\n")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}All scalability baselines complete.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()