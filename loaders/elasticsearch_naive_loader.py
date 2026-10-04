"""
loaders/elasticsearch_naive_loader.py — Elasticsearch Naive Schema Loader
==========================================================================
Loads all 12 StreamCart tables into Elasticsearch using the naive schema:
  - One index per table, prefixed "naive_"
  - Explicit mappings with correct types, but no custom analysers, no boosting
  - JSONB columns (metadata, cart, attributes, preferences, features) stored
    as raw JSON strings — a direct port of PostgreSQL's opaque JSONB columns
  - Boolean CSV strings ("True" / "False") parsed to Python booleans
  - Nullable fields: empty CSV strings converted to JSON null

Naive design rationale
──────────────────────
Storing JSONB columns as keyword strings is the "engine effect" baseline:
it forces Elasticsearch to treat structured data as opaque blobs, exactly
as a developer would if porting a PostgreSQL schema with minimal ES knowledge.
The optimised loader corrects this by parsing these fields into native objects,
which is the "schema effect" the dissertation measures.

Index naming:
  naive_users, naive_seller_profiles, naive_subscription_tiers,
  naive_subscription_tier_pricing, naive_subscriptions, naive_products,
  naive_invoices, naive_invoice_lines, naive_orders, naive_order_items,
  naive_sessions, naive_events

Data sources:
  All CSV files are read from <project_root>/data/
  subscription_tiers and subscription_tier_pricing have no CSV — they are
    hardcoded here from the values in schema.sql
  invoices:      marketplace_invoices.csv + subscription_invoices.csv merged
  invoice_lines: marketplace_invoice_lines.csv + subscription_invoice_lines.csv merged
  events:        events.csv  (NOT events_q8.csv — that is for Q8 write-only)

Usage:
  python loaders/elasticsearch_naive_loader.py
  python loaders/elasticsearch_naive_loader.py --drop       # drop + recreate indices
  python loaders/elasticsearch_naive_loader.py --dry-run    # first 500 rows per index
  python loaders/elasticsearch_naive_loader.py --index naive_products  # single index
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────────
# loaders/ sits one level below the project root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ES_URL       = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
CHUNK_SIZE   = 500       # documents per bulk request
PREFIX       = "naive_"

# ── colour helpers (match project style) ──────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── type conversion helpers ────────────────────────────────────────────────────
# Empty string → None ensures ES stores JSON null for nullable columns

def _str(v):
    s = v.strip() if v else ""
    return s if s else None

def _bool(v):
    if not v or not v.strip():
        return None
    return v.strip().lower() == "true"

def _float(v):
    if not v or not v.strip():
        return None
    return float(v.strip())

def _int(v):
    if not v or not v.strip():
        return None
    return int(v.strip())

def _date(v):
    """ISO timestamp — pass through; ES date mapping parses it natively."""
    if not v or not v.strip():
        return None
    return v.strip()

def _jsonstr(v):
    """JSONB column stored as raw JSON string in naive schema.
    csv.DictReader already unescapes doubled quotes, so the value is
    a valid JSON string ready to store as-is."""
    if not v or not v.strip():
        return None
    return v.strip()

# ── CSV reader ─────────────────────────────────────────────────────────────────

def read_csv(path: Path, limit=None):
    """Yield rows as dicts. limit=N stops after N rows (dry-run mode)."""
    if not path.exists():
        warn(f"CSV not found: {path}")
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield row

# ── index management ───────────────────────────────────────────────────────────

def index_exists(es: Elasticsearch, name: str) -> bool:
    return es.indices.exists(index=name).body

def create_index(es: Elasticsearch, name: str, mappings: dict,
                 settings: dict | None = None, drop: bool = False):
    if index_exists(es, name):
        if drop:
            es.indices.delete(index=name)
            info(f"Dropped: {name}")
        else:
            warn(f"Already exists (skipping create): {name}  — use --drop to reset")
            return
    kwargs = {"mappings": mappings}
    if settings:
        kwargs["settings"] = settings
    es.indices.create(index=name, **kwargs)
    info(f"Created: {name}")

# ── bulk loader ────────────────────────────────────────────────────────────────

def load_docs(es: Elasticsearch, index: str, docs_iter, id_field: str = "id"):
    """
    Bulk-index an iterator of document dicts.
    Uses the document's id_field value as the ES _id.
    Returns (success_count, error_count).
    """
    def _actions():
        for doc in docs_iter:
            yield {
                "_index": index,
                "_id":    str(doc.get(id_field, "")),
                "_source": doc,
            }

    ok_count, err_count = helpers.bulk(
        es,
        _actions(),
        chunk_size=CHUNK_SIZE,
        raise_on_error=False,
        stats_only=True,
    )
    return ok_count, err_count

# ── static data (no CSV for these tables) ─────────────────────────────────────
# Values taken directly from schema.sql INSERT statements.

SUBSCRIPTION_TIERS = [
    {
        "id": "1",
        "name": "Free",
        "description": "Basic software access, up to 5 marketplace purchases/year",
        "features": (
            '{"seats": 1, "api_access": false, "priority_support": false, '
            '"marketplace_purchases_per_year": 5, '
            '"apps": {"CanvasEditor": "free", "VideoSuite": null}}'
        ),
    },
    {
        "id": "2",
        "name": "Pro",
        "description": "Full software, unlimited purchases, early access",
        "features": (
            '{"seats": 1, "api_access": false, "priority_support": false, '
            '"marketplace_purchases_per_year": -1, '
            '"apps": {"CanvasEditor": "premium", "VideoSuite": "standard"}}'
        ),
    },
    {
        "id": "3",
        "name": "Business",
        "description": "Everything in Pro plus team seats, API access, priority support",
        "features": (
            '{"seats": 10, "api_access": true, "priority_support": true, '
            '"marketplace_purchases_per_year": -1, '
            '"apps": {"CanvasEditor": "premium", "VideoSuite": "premium"}}'
        ),
    },
]

SUBSCRIPTION_TIER_PRICING = [
    {"id": "1_2023-01-01", "tier_id": 1, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": None,                       "monthly_price_usd": 0.00,  "is_active": True},
    {"id": "2_2023-01-01", "tier_id": 2, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00","monthly_price_usd": 14.99, "is_active": False},
    {"id": "2_2024-06-01", "tier_id": 2, "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None,                       "monthly_price_usd": 19.99, "is_active": True},
    {"id": "3_2023-01-01", "tier_id": 3, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00","monthly_price_usd": 39.99, "is_active": False},
    {"id": "3_2024-06-01", "tier_id": 3, "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None,                       "monthly_price_usd": 49.99, "is_active": True},
]

# ── mappings ───────────────────────────────────────────────────────────────────
# Convention:
#   UUIDs / token IDs → keyword
#   Short enumerations → keyword
#   Human-readable text → text + .keyword sub-field
#   JSONB blobs → keyword, index:false  (stored, not searchable)
#   Timestamps → date
#   Booleans → boolean
#   Amounts / quantities → float / integer

_TEXT = lambda: {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
_KW   = lambda: {"type": "keyword"}
_BLOB = lambda: {"type": "keyword", "index": False}   # raw JSON string, not indexed
_DATE = lambda: {"type": "date"}
_BOOL = lambda: {"type": "boolean"}
_FLT  = lambda: {"type": "float"}
_INT  = lambda: {"type": "integer"}

MAPPINGS = {

    f"{PREFIX}users": {
        "properties": {
            "id":            _KW(),
            "email":         _KW(),
            "full_name":     _TEXT(),
            "country_code":  _KW(),
            "city":          _TEXT(),
            "created_at":    _DATE(),
            "last_login_at": _DATE(),
            "is_active":     _BOOL(),
            "preferences":   _BLOB(),  # raw JSONB string
        }
    },

    f"{PREFIX}seller_profiles": {
        "properties": {
            "user_id":      _KW(),
            "display_name": _TEXT(),
            "legal_name":   _TEXT(),
            "tax_id":       _KW(),
            "payout_email": _KW(),
            "country_code": _KW(),
            "is_verified":  _BOOL(),
            "bio":          _TEXT(),
            "created_at":   _DATE(),
            "updated_at":   _DATE(),
        }
    },

    f"{PREFIX}subscription_tiers": {
        "properties": {
            "id":          _KW(),
            "name":        _KW(),
            "description": _TEXT(),
            "features":    _BLOB(),  # raw JSONB string
        }
    },

    f"{PREFIX}subscription_tier_pricing": {
        "properties": {
            "id":                _KW(),   # synthetic: "{tier_id}_{valid_from_date}"
            "tier_id":           _INT(),
            "valid_from":        _DATE(),
            "valid_to":          _DATE(),
            "monthly_price_usd": _FLT(),
            "is_active":         _BOOL(),
        }
    },

    f"{PREFIX}subscriptions": {
        "properties": {
            "id":                   _KW(),
            "user_id":              _KW(),
            "tier_id":              _INT(),
            "status":               _KW(),
            "started_at":           _DATE(),
            "current_period_start": _DATE(),
            "current_period_end":   _DATE(),
            "cancelled_at":         _DATE(),
            "cancel_reason":        _TEXT(),
            "billing_cycle":        _KW(),
            "created_at":           _DATE(),
            "updated_at":           _DATE(),
        }
    },

    f"{PREFIX}products": {
        "properties": {
            "id":           _KW(),
            "name":         _TEXT(),
            "slug":         _KW(),
            "product_type": _KW(),
            "description":  _TEXT(),
            "price_usd":    _FLT(),
            "currency":     _KW(),
            "is_active":    _BOOL(),
            "seller_id":    _KW(),
            "attributes":   _BLOB(),  # raw JSONB string
            "created_at":   _DATE(),
            "updated_at":   _DATE(),
        }
    },

    f"{PREFIX}invoices": {
        "properties": {
            "id":                   _KW(),
            "user_id":              _KW(),
            "invoice_type":         _KW(),
            "status":               _KW(),
            "subtotal_usd":         _FLT(),
            "tax_usd":              _FLT(),
            "discount_usd":         _FLT(),
            "total_usd":            _FLT(),
            "subscription_id":      _KW(),
            "billing_period_start": _DATE(),
            "billing_period_end":   _DATE(),
            "paid_at":              _DATE(),
            "due_at":               _DATE(),
            "created_at":           _DATE(),
        }
    },

    f"{PREFIX}invoice_lines": {
        "properties": {
            "id":             _KW(),
            "invoice_id":     _KW(),
            "product_id":     _KW(),
            "description":    _TEXT(),
            "quantity":       _INT(),
            "unit_price_usd": _FLT(),
            "line_total_usd": _FLT(),
            "created_at":     _DATE(),
        }
    },

    f"{PREFIX}orders": {
        "properties": {
            "id":               _KW(),
            "user_id":          _KW(),
            "invoice_id":       _KW(),
            "status":           _KW(),
            "shipping_name":    _TEXT(),
            "shipping_address": _TEXT(),
            "shipping_city":    _TEXT(),
            "shipping_country": _KW(),
            "shipping_postal":  _KW(),
            "created_at":       _DATE(),
            "updated_at":       _DATE(),
        }
    },

    f"{PREFIX}order_items": {
        "properties": {
            "id":               _KW(),
            "order_id":         _KW(),
            "product_id":       _KW(),
            "quantity":         _INT(),
            "unit_price_usd":   _FLT(),
            "line_total_usd":   _FLT(),
            "fulfilment_status":_KW(),
            "created_at":       _DATE(),
        }
    },

    f"{PREFIX}sessions": {
        "properties": {
            "id":            _KW(),
            "user_id":       _KW(),
            "cart":          _BLOB(),  # raw JSON string — naive port of JSONB
            "ip_address":    _KW(),
            "user_agent":    _TEXT(),
            "created_at":    _DATE(),
            "last_active_at":_DATE(),
            "expires_at":    _DATE(),
        }
    },

    f"{PREFIX}events": {
        "properties": {
            "id":          _KW(),
            "user_id":     _KW(),
            "event_type":  _KW(),
            "product_id":  _KW(),
            "session_id":  _KW(),
            "metadata":    _BLOB(),  # raw JSON string — naive port of JSONB
            "occurred_at": _DATE(),
        }
    },
}

# ── row transformers ───────────────────────────────────────────────────────────

def _user(row):
    return {
        "id":            row["id"],
        "email":         row["email"],
        "full_name":     row["full_name"],
        "country_code":  row["country_code"],
        "city":          _str(row.get("city")),
        "created_at":    _date(row["created_at"]),
        "last_login_at": _date(row.get("last_login_at")),
        "is_active":     _bool(row["is_active"]),
        "preferences":   _jsonstr(row.get("preferences")),
    }

def _seller_profile(row):
    return {
        "user_id":      row["user_id"],
        "display_name": row["display_name"],
        "legal_name":   _str(row.get("legal_name")),
        "tax_id":       _str(row.get("tax_id")),
        "payout_email": _str(row.get("payout_email")),
        "country_code": _str(row.get("country_code")),
        "is_verified":  _bool(row["is_verified"]),
        "bio":          _str(row.get("bio")),
        "created_at":   _date(row["created_at"]),
        "updated_at":   _date(row["updated_at"]),
    }

def _subscription(row):
    return {
        "id":                   row["id"],
        "user_id":              row["user_id"],
        "tier_id":              _int(row["tier_id"]),
        "status":               row["status"],
        "started_at":           _date(row["started_at"]),
        "current_period_start": _date(row["current_period_start"]),
        "current_period_end":   _date(row["current_period_end"]),
        "cancelled_at":         _date(row.get("cancelled_at")),
        "cancel_reason":        _str(row.get("cancel_reason")),
        "billing_cycle":        row["billing_cycle"],
        "created_at":           _date(row["created_at"]),
        "updated_at":           _date(row["updated_at"]),
    }

def _product(row):
    return {
        "id":           row["id"],
        "name":         row["name"],
        "slug":         row["slug"],
        "product_type": row["product_type"],
        "description":  row["description"],
        "price_usd":    _float(row["price_usd"]),
        "currency":     row["currency"],
        "is_active":    _bool(row["is_active"]),
        "seller_id":    row["seller_id"],
        "attributes":   _jsonstr(row.get("attributes")),
        "created_at":   _date(row["created_at"]),
        "updated_at":   _date(row["updated_at"]),
    }

def _invoice(row):
    return {
        "id":                   row["id"],
        "user_id":              row["user_id"],
        "invoice_type":         row["invoice_type"],
        "status":               row["status"],
        "subtotal_usd":         _float(row["subtotal_usd"]),
        "tax_usd":              _float(row["tax_usd"]),
        "discount_usd":         _float(row["discount_usd"]),
        "total_usd":            _float(row["total_usd"]),
        "subscription_id":      _str(row.get("subscription_id")),
        "billing_period_start": _date(row.get("billing_period_start")),
        "billing_period_end":   _date(row.get("billing_period_end")),
        "paid_at":              _date(row.get("paid_at")),
        "due_at":               _date(row["due_at"]),
        "created_at":           _date(row["created_at"]),
    }

def _invoice_line(row):
    return {
        "id":             row["id"],
        "invoice_id":     row["invoice_id"],
        "product_id":     _str(row.get("product_id")),
        "description":    row["description"],
        "quantity":       _int(row["quantity"]),
        "unit_price_usd": _float(row["unit_price_usd"]),
        "line_total_usd": _float(row["line_total_usd"]),
        "created_at":     _date(row["created_at"]),
    }

def _order(row):
    return {
        "id":               row["id"],
        "user_id":          row["user_id"],
        "invoice_id":       row["invoice_id"],
        "status":           row["status"],
        "shipping_name":    _str(row.get("shipping_name")),
        "shipping_address": _str(row.get("shipping_address")),
        "shipping_city":    _str(row.get("shipping_city")),
        "shipping_country": _str(row.get("shipping_country")),
        "shipping_postal":  _str(row.get("shipping_postal")),
        "created_at":       _date(row["created_at"]),
        "updated_at":       _date(row["updated_at"]),
    }

def _order_item(row):
    return {
        "id":                row["id"],
        "order_id":          row["order_id"],
        "product_id":        row["product_id"],
        "quantity":          _int(row["quantity"]),
        "unit_price_usd":    _float(row["unit_price_usd"]),
        "line_total_usd":    _float(row["line_total_usd"]),
        "fulfilment_status": row["fulfilment_status"],
        "created_at":        _date(row["created_at"]),
    }

def _session(row):
    return {
        "id":             row["id"],
        "user_id":        row["user_id"],
        "cart":           _jsonstr(row.get("cart")),  # raw JSON string in naive
        "ip_address":     _str(row.get("ip_address")),
        "user_agent":     _str(row.get("user_agent")),
        "created_at":     _date(row["created_at"]),
        "last_active_at": _date(row["last_active_at"]),
        "expires_at":     _date(row["expires_at"]),
    }

def _event(row):
    return {
        "id":          row["id"],
        "user_id":     row["user_id"],
        "event_type":  row["event_type"],
        "product_id":  _str(row.get("product_id")),
        "session_id":  _str(row.get("session_id")),
        "metadata":    _jsonstr(row.get("metadata")),  # raw JSON string in naive
        "occurred_at": _date(row["occurred_at"]),
    }

# ── per-index load logic ───────────────────────────────────────────────────────

def _load_one(es, index, docs_iter, id_field="id", label=""):
    n_ok, n_err = load_docs(es, index, docs_iter, id_field=id_field)
    tag = label or index
    if n_err == 0:
        ok(f"{tag:<45} {n_ok:>8,} docs")
    else:
        fail(f"{tag:<45} {n_ok:>8,} ok  {n_err:>6,} errors")
    return n_ok, n_err


def load_all(es: Elasticsearch, drop: bool = False, limit: int | None = None,
             only: str | None = None):
    """
    Create and populate all 12 naive indices.

    Parameters
    ----------
    es    : Elasticsearch client
    drop  : delete existing indices before recreating
    limit : max rows per CSV (None = all rows); used for --dry-run
    only  : if set, only load this index name (e.g. "naive_products")
    """
    def _should(name):
        return only is None or name == only

    totals = {"ok": 0, "err": 0}

    def _run(name, docs_iter, id_field="id"):
        if not _should(name):
            return
        create_index(es, name, MAPPINGS[name], drop=drop)
        n_ok, n_err = _load_one(es, name, docs_iter, id_field=id_field)
        totals["ok"]  += n_ok
        totals["err"] += n_err

    # ── 1. users ──────────────────────────────────────────────────────────────
    _run(f"{PREFIX}users",
         (_user(r) for r in read_csv(DATA_DIR / "users.csv", limit)))

    # ── 2. seller_profiles ────────────────────────────────────────────────────
    _run(f"{PREFIX}seller_profiles",
         (_seller_profile(r) for r in read_csv(DATA_DIR / "seller_profiles.csv", limit)),
         id_field="user_id")

    # ── 3. subscription_tiers (static data — no CSV) ──────────────────────────
    _run(f"{PREFIX}subscription_tiers",
         iter(SUBSCRIPTION_TIERS[:limit] if limit else SUBSCRIPTION_TIERS))

    # ── 4. subscription_tier_pricing (static data — no CSV) ───────────────────
    _run(f"{PREFIX}subscription_tier_pricing",
         iter(SUBSCRIPTION_TIER_PRICING[:limit] if limit else SUBSCRIPTION_TIER_PRICING))

    # ── 5. subscriptions ──────────────────────────────────────────────────────
    _run(f"{PREFIX}subscriptions",
         (_subscription(r) for r in read_csv(DATA_DIR / "subscriptions.csv", limit)))

    # ── 6. products ───────────────────────────────────────────────────────────
    _run(f"{PREFIX}products",
         (_product(r) for r in read_csv(DATA_DIR / "products.csv", limit)))

    # ── 7. invoices (marketplace + subscription CSVs merged) ──────────────────
    def _invoice_docs():
        for path in ("marketplace_invoices.csv", "subscription_invoices.csv"):
            for r in read_csv(DATA_DIR / path, limit):
                yield _invoice(r)

    _run(f"{PREFIX}invoices", _invoice_docs())

    # ── 8. invoice_lines (marketplace + subscription CSVs merged) ─────────────
    def _invoice_line_docs():
        for path in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
            for r in read_csv(DATA_DIR / path, limit):
                yield _invoice_line(r)

    _run(f"{PREFIX}invoice_lines", _invoice_line_docs())

    # ── 9. orders ─────────────────────────────────────────────────────────────
    _run(f"{PREFIX}orders",
         (_order(r) for r in read_csv(DATA_DIR / "orders.csv", limit)))

    # ── 10. order_items ───────────────────────────────────────────────────────
    _run(f"{PREFIX}order_items",
         (_order_item(r) for r in read_csv(DATA_DIR / "order_items.csv", limit)))

    # ── 11. sessions ──────────────────────────────────────────────────────────
    _run(f"{PREFIX}sessions",
         (_session(r) for r in read_csv(DATA_DIR / "sessions.csv", limit)))

    # ── 12. events (main dataset only — events_q8.csv is for Q8 writes) ───────
    _run(f"{PREFIX}events",
         (_event(r) for r in read_csv(DATA_DIR / "events.csv", limit)))

    return totals

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load StreamCart data into Elasticsearch naive indices"
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Delete existing indices before loading (default: skip if exists)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Load only the first 500 rows per index for sanity checking",
    )
    parser.add_argument(
        "--index", type=str, default=None, metavar="INDEX",
        help="Only load a single index (e.g. naive_products)",
    )
    args = parser.parse_args()

    limit = 500 if args.dry_run else None

    print("\n" + "=" * 60)
    print("  Elasticsearch — Naive Schema Loader")
    if args.dry_run:
        print("  DRY RUN — first 500 rows per index only")
    print("=" * 60)
    print(f"  ES endpoint : {ES_URL}")
    print(f"  Data dir    : {DATA_DIR}")
    print()

    es = Elasticsearch(ES_URL, request_timeout=60)

    try:
        info_resp = es.info()
        info(f"Connected — ES {info_resp['version']['number']}")
    except Exception as exc:
        fail(f"Cannot connect to Elasticsearch at {ES_URL}: {exc}")
        sys.exit(1)

    print()
    totals = load_all(es, drop=args.drop, limit=limit, only=args.index)
    print()
    print("=" * 60)
    ok(f"Total loaded: {totals['ok']:,} docs  |  errors: {totals['err']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()