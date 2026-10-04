"""
loaders/es_tier_attribution.py — load-time tier attribution for optimised_invoices
=================================================================================
Optimised-schema denormalisation for Elasticsearch Q1 / Q7: every invoice
document gets the tier it is attributed to and the price in effect at the
time it was created, so the queries become single server-side aggregations.

Attribution rule (identical to PostgreSQL Q1 / Q7):
  - subscription invoice → tier of invoices.subscription_id
  - marketplace invoice  → tier of the user's most recent subscription with
                           started_at <= invoice.created_at
  - price_in_effect_usd  → subscription_tier_pricing row with
                           valid_from <= created_at < valid_to (or open-ended)
Invoices with no resolvable tier get no tier fields (they are excluded from
Q1/Q7, as with PostgreSQL's inner JOIN / JOIN LATERAL).

This is the same pre-resolution the Cassandra optimised loader performs for
invoices_by_month_tier / invoices_by_tier.

Used in two ways:
  1. imported by elasticsearch_optimised_loader.py (fresh loads), and
  2. run directly to enrich an ALREADY LOADED optimised_invoices index in
     place (partial updates, no reload):

        python loaders/es_tier_attribution.py
"""

from __future__ import annotations

import bisect
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
INDEX        = "optimised_invoices"

TIER_NAMES = {1: "Free", 2: "Pro", 3: "Business"}

# Same rows as SUBSCRIPTION_TIER_PRICING in the loaders / schema.sql
PRICING = [
    (1, "2023-01-01T00:00:00+00:00", None,                        0.00),
    (2, "2023-01-01T00:00:00+00:00", "2024-06-01T00:00:00+00:00", 14.99),
    (2, "2024-06-01T00:00:00+00:00", None,                        19.99),
    (3, "2023-01-01T00:00:00+00:00", "2024-06-01T00:00:00+00:00", 39.99),
    (3, "2024-06-01T00:00:00+00:00", None,                        49.99),
]

# Extra mapping fields added to optimised_invoices
TIER_FIELDS_MAPPING = {
    "tier_id":             {"type": "integer"},
    "tier_name":           {"type": "keyword"},
    "price_in_effect_usd": {"type": "double"},
}


def _price_at(tier_id: int, ts: str):
    for t, vf, vt, price in PRICING:
        if t == tier_id and vf <= ts and (vt is None or vt > ts):
            return price
    return None


class TierResolver:
    """Built once from subscriptions.csv; resolve(invoice_row) → fields dict."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.sub_tier: dict[str, int] = {}
        per_user = defaultdict(list)
        with open(data_dir / "subscriptions.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("tier_id"):
                    continue
                tid = int(r["tier_id"])
                self.sub_tier[r["id"]] = tid
                if r.get("user_id") and r.get("started_at"):
                    per_user[r["user_id"]].append((r["started_at"], tid))
        self.user_subs = {}
        for uid, lst in per_user.items():
            lst.sort(key=lambda x: x[0])
            self.user_subs[uid] = ([x[0] for x in lst], [x[1] for x in lst])

    def tier_for(self, row: dict):
        ts = row["created_at"]
        if row.get("invoice_type") == "subscription":
            sid = row.get("subscription_id")
            return self.sub_tier.get(sid) if sid else None
        if row.get("invoice_type") == "marketplace":
            entry = self.user_subs.get(row.get("user_id"))
            if not entry:
                return None
            starts, tiers = entry
            k = bisect.bisect_right(starts, ts)   # started_at <= created_at
            return tiers[k - 1] if k else None
        return None

    def resolve(self, row: dict) -> dict:
        tid = self.tier_for(row)
        if tid is None:
            return {}
        price = _price_at(tid, row["created_at"])
        if price is None:
            return {}
        return {"tier_id": tid, "tier_name": TIER_NAMES[tid], "price_in_effect_usd": price}


# ── standalone: enrich an existing optimised_invoices index in place ─────────

def enrich_existing_index(es, data_dir: Path = DATA_DIR, chunk_size: int = 2000):
    from elasticsearch import helpers

    es.indices.put_mapping(index=INDEX, properties=TIER_FIELDS_MAPPING)
    print(f"  Mapping updated on {INDEX}: {', '.join(TIER_FIELDS_MAPPING)}")

    resolver = TierResolver(data_dir)
    print(f"  Subscriptions indexed: {len(resolver.sub_tier):,} "
          f"({len(resolver.user_subs):,} users)")

    stats = {"attributed": 0, "unattributed": 0}

    def _actions():
        for name in ("marketplace_invoices.csv", "subscription_invoices.csv"):
            with open(data_dir / name, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    fields = resolver.resolve(r)
                    if not fields:
                        stats["unattributed"] += 1
                        continue
                    stats["attributed"] += 1
                    yield {"_op_type": "update", "_index": INDEX,
                           "_id": r["id"], "doc": fields}

    ok, errors = helpers.bulk(es, _actions(), chunk_size=chunk_size,
                              raise_on_error=False, stats_only=True,
                              request_timeout=120)
    es.indices.refresh(index=INDEX)
    print(f"  Updated {ok:,} invoice docs ({errors:,} errors). "
          f"Attributed: {stats['attributed']:,}, no tier: {stats['unattributed']:,}")

    check = es.count(index=INDEX, query={"exists": {"field": "tier_id"}})["count"]
    print(f"  Docs with tier_id now: {check:,}")
    return errors


def main():
    from dotenv import load_dotenv
    from elasticsearch import Elasticsearch
    load_dotenv()
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
                       request_timeout=120)
    print("\n  Elasticsearch optimised — tier attribution enrichment")
    print(f"  ES {es.info()['version']['number']}  |  index {INDEX}")
    errors = enrich_existing_index(es)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
