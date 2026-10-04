"""
benchmarks/elasticsearch/es_revenue.py — Elasticsearch Q1 and Q7 (naive + optimised)
====================================================================================
Full PostgreSQL-equivalent implementations of the two revenue queries:

  Q1  Monthly revenue by subscription tier, last 12 months, with the price
      in effect (temporal pricing) — rows: month, tier_name,
      price_in_effect_usd, invoice_count, total_revenue_usd.

  Q7  7-day rolling average of daily revenue per tier over a 183-day window,
      gap-filled — rows: day, tier_name, daily_revenue_usd,
      rolling_7day_avg_usd  (n_days x 3 tiers).

Tier attribution rule (identical to PostgreSQL Q1/Q7):
  - subscription invoice → tier of invoices.subscription_id
  - marketplace invoice  → tier of the user's most recent subscription with
                           started_at <= invoice.created_at
  - invoices with no resolvable tier are dropped (PostgreSQL inner JOIN /
    JOIN LATERAL semantics).

NAIVE (naive_* indices, flat port of the relational schema)
───────────────────────────────────────────────────────────
tier_id does not exist on invoice documents, so every call (all timed):
  1. reads naive_subscription_tiers and naive_subscription_tier_pricing,
  2. scans naive_subscriptions (PIT + search_after),
  3. scans the paid invoices of the window (PIT + search_after),
  4. attributes tier + temporal price per invoice in Python,
  5. Q1: aggregates by (month, tier, price);
     Q7: aggregates per (tier, day), gap-fills, rolling 7-day average.
This is the client-side join an application would have to do on this schema,
and the same pattern as the Cassandra naive implementation.

OPTIMISED (optimised_* indices)
───────────────────────────────
tier_id, tier_name and price_in_effect_usd are denormalised onto each
invoice document at load time (loaders/es_tier_attribution.py), the same
pre-resolution the Cassandra optimised schema does (invoices_by_month_tier).
Each call is then ONE search request, all computed server-side:
  Q1: date_histogram(month) → terms(tier_name) → terms(price) → sum
  Q7: filters(tier) → date_histogram(day, min_doc_count=0, extended_bounds)
      → sum + moving_fn(window=7, shift=1, unweightedAvg)
The Python side only reshapes the buckets into rows (timed, negligible).

Q1 reference time
─────────────────
PostgreSQL Q1 filters created_at >= NOW() - 12 months. The data ends on
2026-01-01, so the window depends on when the benchmark is run. To keep the
re-run comparable with the PostgreSQL baseline, Q1 is anchored to the
timestamp of the PostgreSQL Q1 baseline run (Q1_AS_OF) unless overridden.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from elasticsearch import Elasticsearch

# Timestamp of benchmarks/postgres/results/postgres_q1_baseline.json
Q1_AS_OF = datetime(2026, 3, 4, 14, 51, 58, tzinfo=timezone.utc)

PAGE_SIZE   = 10_000
PIT_KEEP    = "2m"
ROLLING     = 7


# ── shared helpers ────────────────────────────────────────────────────────────

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def minus_12_months(dt: datetime) -> datetime:
    """Calendar 12 months back (PostgreSQL INTERVAL '12 months')."""
    try:
        return dt.replace(year=dt.year - 1)
    except ValueError:                      # 29 Feb
        return dt.replace(year=dt.year - 1, day=28)


def _norm_ts(v) -> str:
    """
    Normalise an ES date value from _source to 'YYYY-MM-DDTHH:MM:SS+00:00'
    so that lexicographic comparison == chronological comparison.
    The loaders store the CSV strings unchanged (already in this format);
    this only guards against a 'Z' suffix or missing seconds.
    """
    s = str(v)
    if len(s) == 25 and s[10] == "T" and s.endswith("+00:00"):
        return s
    return _iso(datetime.fromisoformat(s.replace("Z", "+00:00")))


def scan(client: Elasticsearch, index: str, query: dict, fields: list[str]):
    """Point-in-time + search_after scan. Yields _source dicts."""
    pit = client.open_point_in_time(index=index, keep_alive=PIT_KEEP)["id"]
    try:
        search_after = None
        while True:
            kwargs = dict(
                pit={"id": pit, "keep_alive": PIT_KEEP},
                query=query,
                _source=fields,
                size=PAGE_SIZE,
                sort=["_shard_doc"],
                track_total_hits=False,
            )
            if search_after is not None:
                kwargs["search_after"] = search_after
            resp = client.search(**kwargs)
            hits = resp["hits"]["hits"]
            if not hits:
                break
            pit = resp.get("pit_id", pit)
            for h in hits:
                yield h["_source"]
            search_after = hits[-1]["sort"]
            if len(hits) < PAGE_SIZE:
                break
    finally:
        try:
            client.close_point_in_time(id=pit)
        except Exception:
            pass


def _read_tiers(client: Elasticsearch, prefix: str):
    """tier_id(int) → name, and pricing rows (tier_id, from, to, price)."""
    tiers = client.search(index=f"{prefix}subscription_tiers", size=100,
                          query={"match_all": {}})
    tier_names = {int(h["_source"]["id"]): h["_source"]["name"]
                  for h in tiers["hits"]["hits"]}
    pr = client.search(index=f"{prefix}subscription_tier_pricing", size=100,
                       query={"match_all": {}})
    pricing = []
    for h in pr["hits"]["hits"]:
        s = h["_source"]
        pricing.append((int(s["tier_id"]), _norm_ts(s["valid_from"]),
                        _norm_ts(s["valid_to"]) if s.get("valid_to") else None,
                        float(s["monthly_price_usd"])))
    return tier_names, pricing


def _price_at(pricing, tier_id: int, ts: str):
    for t, vf, vt, price in pricing:
        if t == tier_id and vf <= ts and (vt is None or vt > ts):
            return price
    return None


def _subscription_index(client: Elasticsearch, prefix: str):
    """sub_id → tier_id, and user_id → (sorted started_at list, tier list)."""
    sub_tier = {}
    per_user = defaultdict(list)
    for s in scan(client, f"{prefix}subscriptions", {"match_all": {}},
                  ["id", "user_id", "tier_id", "started_at"]):
        if s.get("tier_id") is None:
            continue
        tid = int(s["tier_id"])
        sub_tier[s["id"]] = tid
        if s.get("user_id") and s.get("started_at"):
            per_user[s["user_id"]].append((_norm_ts(s["started_at"]), tid))
    user_subs = {}
    for uid, lst in per_user.items():
        lst.sort(key=lambda x: x[0])
        user_subs[uid] = ([x[0] for x in lst], [x[1] for x in lst])
    return sub_tier, user_subs


def _attribute(inv: dict, ts: str, sub_tier: dict, user_subs: dict):
    """PostgreSQL attribution rule. Returns tier_id or None."""
    if inv.get("invoice_type") == "subscription":
        sid = inv.get("subscription_id")
        return sub_tier.get(sid) if sid else None
    if inv.get("invoice_type") == "marketplace":
        entry = user_subs.get(inv.get("user_id"))
        if not entry:
            return None
        starts, tiers = entry
        k = bisect.bisect_right(starts, ts)       # started_at <= created_at
        return tiers[k - 1] if k else None
    return None


def _paid_invoices(client, index, gte: str, lt: str | None = None, lte: str | None = None):
    rng = {"gte": gte}
    if lt:
        rng["lt"] = lt
    if lte:
        rng["lte"] = lte
    return scan(
        client, index,
        {"bool": {"filter": [{"term": {"status": "paid"}},
                             {"range": {"created_at": rng}}]}},
        ["user_id", "invoice_type", "subscription_id", "created_at", "total_usd"],
    )


def gap_fill_rolling(daily: dict, tier_names: dict, start: date, n_days: int) -> list[dict]:
    """daily[tier_id][offset] = revenue → PG-shaped Q7 rows (ordered day, tier)."""
    rows = []
    series = {}
    for tid in tier_names:
        d = daily.get(tid, {})
        series[tid] = [d.get(o, 0.0) for o in range(n_days)]
    order = sorted(tier_names, key=lambda t: tier_names[t])
    for o in range(n_days):
        day = (start + timedelta(days=o)).isoformat()
        for tid in order:
            s = series[tid]
            win = s[max(0, o - ROLLING + 1): o + 1]
            rows.append({
                "day":                  day,
                "tier_name":            tier_names[tid],
                "daily_revenue_usd":    round(s[o], 2),
                "rolling_7day_avg_usd": round(sum(win) / len(win), 2),
            })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# NAIVE
# ═════════════════════════════════════════════════════════════════════════════

def naive_q1(client: Elasticsearch, as_of: datetime = Q1_AS_OF) -> list[dict]:
    tier_names, pricing = _read_tiers(client, "naive_")
    sub_tier, user_subs = _subscription_index(client, "naive_")

    agg = defaultdict(lambda: [0, 0.0])            # (month, tier, price) → [count, sum]
    for inv in _paid_invoices(client, "naive_invoices",
                              gte=_iso(minus_12_months(as_of)), lte=_iso(as_of)):
        ts = _norm_ts(inv["created_at"])
        tid = _attribute(inv, ts, sub_tier, user_subs)
        if tid is None or tid not in tier_names:
            continue
        price = _price_at(pricing, tid, ts)
        if price is None:
            continue
        a = agg[(ts[:7], tier_names[tid], price)]
        a[0] += 1
        a[1] += float(inv.get("total_usd") or 0.0)

    return [
        {"month": m, "tier_name": t, "price_in_effect_usd": p,
         "invoice_count": c, "total_revenue_usd": round(s, 2)}
        for (m, t, p), (c, s) in sorted(agg.items())
    ]


def naive_q7(client: Elasticsearch, start: date, n_days: int) -> list[dict]:
    tier_names, _ = _read_tiers(client, "naive_")
    sub_tier, user_subs = _subscription_index(client, "naive_")

    end_excl = start + timedelta(days=n_days)
    daily = defaultdict(lambda: defaultdict(float))
    for inv in _paid_invoices(client, "naive_invoices",
                              gte=f"{start.isoformat()}T00:00:00+00:00",
                              lt=f"{end_excl.isoformat()}T00:00:00+00:00"):
        ts = _norm_ts(inv["created_at"])
        tid = _attribute(inv, ts, sub_tier, user_subs)
        if tid is None or tid not in tier_names:
            continue
        o = (date.fromisoformat(ts[:10]) - start).days
        daily[tid][o] += float(inv.get("total_usd") or 0.0)

    return gap_fill_rolling(daily, tier_names, start, n_days)


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMISED
# ═════════════════════════════════════════════════════════════════════════════

def optimised_tier_names(client: Elasticsearch) -> list[str]:
    """Startup (untimed): tier names, for the Q7 filters aggregation."""
    tier_names, _ = _read_tiers(client, "optimised_")
    return [tier_names[t] for t in sorted(tier_names, key=lambda t: tier_names[t])]


def optimised_q1(client: Elasticsearch, as_of: datetime = Q1_AS_OF) -> list[dict]:
    resp = client.search(
        index="optimised_invoices",
        size=0,
        track_total_hits=False,
        query={"bool": {"filter": [
            {"term":   {"status": "paid"}},
            {"range":  {"created_at": {"gte": _iso(minus_12_months(as_of)),
                                       "lte": _iso(as_of)}}},
            {"exists": {"field": "tier_id"}},
        ]}},
        aggs={"by_month": {
            "date_histogram": {"field": "created_at", "calendar_interval": "month",
                               "format": "yyyy-MM", "time_zone": "UTC"},
            "aggs": {"by_tier": {
                "terms": {"field": "tier_name", "size": 10},
                "aggs": {"by_price": {
                    "terms": {"field": "price_in_effect_usd", "size": 10},
                    "aggs": {"revenue": {"sum": {"field": "total_usd"}}},
                }},
            }},
        }},
    )
    rows = []
    for m in resp["aggregations"]["by_month"]["buckets"]:
        for t in m["by_tier"]["buckets"]:
            for p in t["by_price"]["buckets"]:
                rows.append({
                    "month":               m["key_as_string"],
                    "tier_name":           t["key"],
                    "price_in_effect_usd": round(float(p["key"]), 2),
                    "invoice_count":       p["doc_count"],
                    "total_revenue_usd":   round(p["revenue"]["value"], 2),
                })
    rows.sort(key=lambda r: (r["month"], r["tier_name"]))
    return rows


def optimised_q7(client: Elasticsearch, start: date, n_days: int,
                 tier_names: list[str]) -> list[dict]:
    end_incl = start + timedelta(days=n_days - 1)
    end_excl = start + timedelta(days=n_days)
    resp = client.search(
        index="optimised_invoices",
        size=0,
        track_total_hits=False,
        query={"bool": {"filter": [
            {"term":  {"status": "paid"}},
            {"range": {"created_at": {"gte": f"{start.isoformat()}T00:00:00+00:00",
                                      "lt":  f"{end_excl.isoformat()}T00:00:00+00:00"}}},
        ]}},
        aggs={"by_tier": {
            # filters (not terms) so a tier with no revenue in the window still
            # gets a full zero series — PostgreSQL's tier_days CROSS JOIN.
            "filters": {"filters": {n: {"term": {"tier_name": n}} for n in tier_names}},
            "aggs": {"by_day": {
                "date_histogram": {
                    "field": "created_at", "calendar_interval": "day",
                    "format": "yyyy-MM-dd", "time_zone": "UTC",
                    "min_doc_count": 0,
                    "extended_bounds": {"min": start.isoformat(),
                                        "max": end_incl.isoformat()},
                },
                "aggs": {
                    "revenue": {"sum": {"field": "total_usd"}},
                    # shift=1 → window = this bucket + 6 preceding
                    "rolling": {"moving_fn": {
                        "buckets_path": "revenue", "window": ROLLING, "shift": 1,
                        # zero-revenue (gap-filled) days must count as 0 in the
                        # window, as in PostgreSQL; the default policy (skip)
                        # would drop them and omit the value on those days.
                        "gap_policy": "insert_zeros",
                        "script": "MovingFunctions.unweightedAvg(values)",
                    }},
                },
            }},
        }},
    )
    rows = []
    for tier, tb in resp["aggregations"]["by_tier"]["buckets"].items():
        for b in tb["by_day"]["buckets"]:
            rows.append({
                "day":                  b["key_as_string"],
                "tier_name":            tier,
                "daily_revenue_usd":    round(b["revenue"]["value"], 2),
                "rolling_7day_avg_usd": round(b["rolling"]["value"] or 0.0, 2),
            })
    rows.sort(key=lambda r: (r["day"], r["tier_name"]))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# dry-run printers
# ═════════════════════════════════════════════════════════════════════════════

def print_q1(rows: list[dict]):
    if not rows:
        print("  ⚠  No rows — is data loaded (and, for optimised, tier-enriched)?")
        return
    print(f"  {'Month':<9} {'Tier':<10} {'Price':>8} {'Invoices':>9} {'Revenue (USD)':>16}")
    print(f"  {'─'*9} {'─'*10} {'─'*8} {'─'*9} {'─'*16}")
    for r in rows:
        print(f"  {r['month']:<9} {r['tier_name']:<10} {r['price_in_effect_usd']:>8.2f} "
              f"{r['invoice_count']:>9,} {r['total_revenue_usd']:>16,.2f}")
    print(f"\n  {len(rows)} rows.")


def print_q7(rows: list[dict], n_days: int):
    if not rows:
        print("  ⚠  No rows returned.")
        return
    zeros = sum(1 for r in rows if r["daily_revenue_usd"] == 0)
    print(f"  {len(rows)} rows (expected {n_days} days x 3 tiers = {n_days * 3}); "
          f"{zeros} gap-filled zero-revenue rows\n")
    print(f"  {'Day':<12} {'Tier':<10} {'Daily (USD)':>14} {'7-day avg':>12}")
    print(f"  {'─'*12} {'─'*10} {'─'*14} {'─'*12}")
    for r in rows[:12]:
        print(f"  {r['day']:<12} {r['tier_name']:<10} "
              f"{r['daily_revenue_usd']:>14,.2f} {r['rolling_7day_avg_usd']:>12,.2f}")
    if len(rows) > 12:
        print(f"  ... {len(rows) - 12} more rows")
