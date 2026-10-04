"""
loaders/elasticsearch_optimised_loader.py — Elasticsearch Optimised Schema Loader
===================================================================================
Loads all 12 StreamCart tables into Elasticsearch using the optimised schema.
All structural changes are consequences of schema redesign for ES's strengths —
not independent query-level optimisations — preserving the engine-vs-schema
decomposition.

Schema changes vs naive
───────────────────────
  products index
    - Custom analyser (streamcart_analyzer): standard tokeniser → lowercase
      → English stop words → synonym expansion → English stemmer
    - Inline synonyms for the digital-marketplace domain (no file needed)
    - Keyword sub-fields on product_type (.keyword) and price_usd (.keyword)
      for faceted aggregation and exact-match filtering
    - norms=true on name (default; length-normalised BM25 scoring)
    - norms=false on product_type, currency, slug, seller_id (low-signal fields
      — length normalisation adds noise, not signal)
    - attributes stored as native JSON object (not string)

  invoices index — Q2 optimisation
    - invoice_lines embedded as nested objects (type: nested)
    - Lines are pre-loaded into memory and merged at load time
    - Q2 optimised fetches a complete invoice in a single GET

  events index — Q6 optimisation
    - metadata stored as native ES object (dynamic sub-fields)
    - Enables term/range queries on metadata sub-fields without scripting
    - In naive, metadata is an opaque string requiring script-based access

  sessions index
    - cart stored as native nested array (not JSON string)
    - Each cart item is a nested document with typed sub-fields

  All indices
    - All timestamp fields use format: strict_date_optional_time
      (explicit format avoids first-document type inference ambiguity)

Index naming: optimised_users, optimised_seller_profiles, ... (12 total)

Data sources: same CSVs as naive loader; see naive loader for full notes.

Usage:
  python loaders/elasticsearch_optimised_loader.py
  python loaders/elasticsearch_optimised_loader.py --drop
  python loaders/elasticsearch_optimised_loader.py --dry-run
  python loaders/elasticsearch_optimised_loader.py --index optimised_products
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from es_tier_attribution import TierResolver   # Q1/Q7 tier denormalisation

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ES_URL       = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
CHUNK_SIZE   = 500
PREFIX       = "optimised_"

# ── colour helpers ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── type helpers ───────────────────────────────────────────────────────────────

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
    if not v or not v.strip():
        return None
    return v.strip()

def _parse_json(v, fallback=None):
    """Parse a JSON string to a Python object. Returns fallback on error."""
    if not v or not v.strip():
        return fallback if fallback is not None else {}
    try:
        return json.loads(v.strip())
    except (json.JSONDecodeError, ValueError):
        return fallback if fallback is not None else {}

# ── CSV reader ─────────────────────────────────────────────────────────────────

def read_csv(path: Path, limit=None):
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

def index_exists(es, name):
    return es.indices.exists(index=name).body

def create_index(es, name, mappings, settings=None, drop=False):
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

def load_docs(es, index, docs_iter, id_field="id"):
    def _actions():
        for doc in docs_iter:
            yield {
                "_index": index,
                "_id":    str(doc.get(id_field, "")),
                "_source": doc,
            }
    ok_count, err_count = helpers.bulk(
        es, _actions(), chunk_size=CHUNK_SIZE,
        raise_on_error=False, stats_only=True,
    )
    return ok_count, err_count

# ── analyser settings for the products index ───────────────────────────────────
#
# streamcart_analyzer pipeline:
#   standard tokeniser  → unicode-aware word splitting, strips punctuation
#   lowercase           → case-insensitive matching
#   english_stop        → removes English stop words (a, the, for, …)
#   streamcart_synonyms → domain synonym expansion (applied before stemming
#                         so both the original and synonym forms are stemmed)
#   english_stemmer     → reduces inflected forms (brushes→brush, courses→cours)
#
# Synonym filter design
# ─────────────────────
# All synonyms are single-token equivalents — no multi-word phrases.
# Multi-word synonyms (e.g. "digital asset, digital download") require
# token-graph-aware synonym filters, which add complexity without
# meaningful gain for a dissertation benchmark. Single-token equivalents
# are safe, well-supported, and produce the intended recall improvement.
#
# Synonyms are inline (no file) for Docker portability — no volume mount
# or ES filesystem configuration needed.

PRODUCTS_SETTINGS = {
    "analysis": {
        "filter": {
            "english_stop": {
                "type": "stop",
                "stopwords": "_english_",
            },
            "english_stemmer": {
                "type": "stemmer",
                "language": "english",
            },
            "streamcart_synonyms": {
                "type": "synonym",
                "synonyms": [
                    # Tool / software
                    "photoshop, ps",
                    "procreate, procreate_app",
                    "illustrator, ai",
                    # Product formats
                    "brushes, brush",
                    "template, templates",
                    "mockup, mockups",
                    "icon, icons",
                    "font, fonts",
                    "texture, textures",
                    "vector, svg",
                    "preset, presets",
                    # Product categories
                    "course, tutorial, lesson, class",
                    "merch, merchandise",
                    # Styles
                    "watercolour, watercolor",
                    "typography, typeface",
                    "branding, logo",
                    "animation, motion",
                    # Skill levels
                    "beginner, intro, starter",
                    "advanced, expert, professional",
                ],
            },
        },
        "analyzer": {
            "streamcart_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                    "english_stop",
                    "streamcart_synonyms",
                    "english_stemmer",
                ],
            },
        },
    },
}

# ── static data ────────────────────────────────────────────────────────────────
# In optimised schema, features is stored as a native JSON object.

SUBSCRIPTION_TIERS = [
    {
        "id": "1",
        "name": "Free",
        "description": "Basic software access, up to 5 marketplace purchases/year",
        "features": {
            "seats": 1, "api_access": False, "priority_support": False,
            "marketplace_purchases_per_year": 5,
            "apps": {"CanvasEditor": "free", "VideoSuite": None},
        },
    },
    {
        "id": "2",
        "name": "Pro",
        "description": "Full software, unlimited purchases, early access",
        "features": {
            "seats": 1, "api_access": False, "priority_support": False,
            "marketplace_purchases_per_year": -1,
            "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"},
        },
    },
    {
        "id": "3",
        "name": "Business",
        "description": "Everything in Pro plus team seats, API access, priority support",
        "features": {
            "seats": 10, "api_access": True, "priority_support": True,
            "marketplace_purchases_per_year": -1,
            "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"},
        },
    },
]

SUBSCRIPTION_TIER_PRICING = [
    {"id": "1_2023-01-01", "tier_id": 1, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": None,                        "monthly_price_usd": 0.00,  "is_active": True},
    {"id": "2_2023-01-01", "tier_id": 2, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 14.99, "is_active": False},
    {"id": "2_2024-06-01", "tier_id": 2, "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None,                        "monthly_price_usd": 19.99, "is_active": True},
    {"id": "3_2023-01-01", "tier_id": 3, "valid_from": "2023-01-01T00:00:00+00:00",
     "valid_to": "2024-06-01T00:00:00+00:00", "monthly_price_usd": 39.99, "is_active": False},
    {"id": "3_2024-06-01", "tier_id": 3, "valid_from": "2024-06-01T00:00:00+00:00",
     "valid_to": None,                        "monthly_price_usd": 49.99, "is_active": True},
]

# ── mapping helpers ────────────────────────────────────────────────────────────

_DATE_OPT = lambda: {"type": "date", "format": "strict_date_optional_time"}
_KW       = lambda: {"type": "keyword"}
_TEXT_STD = lambda: {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
_TEXT_ES  = lambda: {                              # uses streamcart_analyzer
    "type": "text",
    "analyzer": "streamcart_analyzer",
    "fields": {"keyword": {"type": "keyword"}},
}
_BOOL = lambda: {"type": "boolean"}
_FLT  = lambda: {"type": "float"}
_INT  = lambda: {"type": "integer"}

# ── mappings ───────────────────────────────────────────────────────────────────

MAPPINGS = {

    f"{PREFIX}users": {
        "properties": {
            "id":            _KW(),
            "email":         _KW(),
            "full_name":     _TEXT_STD(),
            "country_code":  _KW(),
            "city":          _TEXT_STD(),
            "created_at":    _DATE_OPT(),
            "last_login_at": _DATE_OPT(),
            "is_active":     _BOOL(),
            # preferences not used in any of Q1-Q8; stored as keyword blob
            "preferences":   {"type": "keyword", "index": False},
        }
    },

    f"{PREFIX}seller_profiles": {
        "properties": {
            "user_id":      _KW(),
            "display_name": _TEXT_STD(),
            "legal_name":   _TEXT_STD(),
            "tax_id":       _KW(),
            "payout_email": _KW(),
            "country_code": _KW(),
            "is_verified":  _BOOL(),
            "bio":          _TEXT_STD(),
            "created_at":   _DATE_OPT(),
            "updated_at":   _DATE_OPT(),
        }
    },

    f"{PREFIX}subscription_tiers": {
        "properties": {
            "id":          _KW(),
            "name":        _KW(),
            "description": _TEXT_STD(),
            # features stored as native object — dynamic sub-field mapping
            "features":    {"type": "object", "dynamic": True},
        }
    },

    f"{PREFIX}subscription_tier_pricing": {
        "properties": {
            "id":                _KW(),
            "tier_id":           _INT(),
            "valid_from":        _DATE_OPT(),
            "valid_to":          _DATE_OPT(),
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
            "started_at":           _DATE_OPT(),
            "current_period_start": _DATE_OPT(),
            "current_period_end":   _DATE_OPT(),
            "cancelled_at":         _DATE_OPT(),
            "cancel_reason":        _TEXT_STD(),
            "billing_cycle":        _KW(),
            "created_at":           _DATE_OPT(),
            "updated_at":           _DATE_OPT(),
        }
    },

    f"{PREFIX}products": {
        # ── Q5 killer index ──────────────────────────────────────────────────
        # name       : norms=true (default) — length-normalised BM25 scoring.
        #              Long product names don't unfairly dominate short ones.
        # description: norms=false — descriptions vary wildly in length (1 line
        #              to 5 paragraphs); length normalisation would penalise
        #              detailed products, so we disable it and rely on TF/IDF.
        # product_type: norms=false, low cardinality — only 3 values
        #              (course, digital_asset, merch). IDF is essentially
        #              constant for these terms; norms add disk cost with no
        #              scoring benefit. .keyword sub-field for facet filter.
        # price_usd  : float for range queries, .keyword for exact faceting.
        # slug        : identifier — keyword only, norms=false irrelevant.
        # seller_id   : UUID — keyword only.
        # attributes  : native JSON object for sub-field access (not used in Q5
        #              but correct for the optimised schema).
        "properties": {
            "id":           _KW(),
            "name":         {
                "type":     "text",
                "analyzer": "streamcart_analyzer",
                "norms":    True,
                "fields":   {"keyword": {"type": "keyword"}},
            },
            "slug":         {"type": "keyword", "norms": False},
            "product_type": {
                "type":     "text",
                "analyzer": "streamcart_analyzer",
                "norms":    False,
                "fields":   {"keyword": {"type": "keyword"}},
            },
            "description":  {
                "type":     "text",
                "analyzer": "streamcart_analyzer",
                "norms":    False,
            },
            "price_usd":    {
                "type":   "float",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "currency":     {"type": "keyword"},
            "is_active":    _BOOL(),
            "seller_id":    {"type": "keyword"},
            "attributes":   {"type": "object", "dynamic": True},
            "created_at":   _DATE_OPT(),
            "updated_at":   _DATE_OPT(),
        }
    },

    f"{PREFIX}invoices": {
        # ── Q2 killer index ──────────────────────────────────────────────────
        # invoice_lines are embedded as a nested type.
        # type, nested creates a hidden child document per line item, allowing
        # independent querying of lines without cross-line false matches
        # (which plain object type would produce for multi-line invoices).
        # Q2 optimised performs a single GET by invoice _id and receives the
        # complete invoice + all lines in one response with no secondary query.
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
            "billing_period_start": _DATE_OPT(),
            "billing_period_end":   _DATE_OPT(),
            "paid_at":              _DATE_OPT(),
            "due_at":               _DATE_OPT(),
            "created_at":           _DATE_OPT(),
            # Denormalised at load time for Q1/Q7 (see es_tier_attribution.py):
            # tier the invoice is attributed to + price in effect at created_at.
            "tier_id":              _INT(),
            "tier_name":            _KW(),
            "price_in_effect_usd":  {"type": "double"},
            "lines": {
                "type": "nested",
                "properties": {
                    "id":             _KW(),
                    "product_id":     _KW(),
                    "description":    _TEXT_STD(),
                    "quantity":       _INT(),
                    "unit_price_usd": _FLT(),
                    "line_total_usd": _FLT(),
                    "created_at":     _DATE_OPT(),
                },
            },
        }
    },

    f"{PREFIX}invoice_lines": {
        # Standalone invoice_lines index — structurally identical to naive.
        # Not used by Q2 optimised (which uses nested lines in invoices),
        # but kept to maintain 12-index symmetry with the naive schema.
        "properties": {
            "id":             _KW(),
            "invoice_id":     _KW(),
            "product_id":     _KW(),
            "description":    _TEXT_STD(),
            "quantity":       _INT(),
            "unit_price_usd": _FLT(),
            "line_total_usd": _FLT(),
            "created_at":     _DATE_OPT(),
        }
    },

    f"{PREFIX}orders": {
        "properties": {
            "id":               _KW(),
            "user_id":          _KW(),
            "invoice_id":       _KW(),
            "status":           _KW(),
            "shipping_name":    _TEXT_STD(),
            "shipping_address": _TEXT_STD(),
            "shipping_city":    _TEXT_STD(),
            "shipping_country": _KW(),
            "shipping_postal":  _KW(),
            "created_at":       _DATE_OPT(),
            "updated_at":       _DATE_OPT(),
        }
    },

    f"{PREFIX}order_items": {
        "properties": {
            "id":                _KW(),
            "order_id":          _KW(),
            "product_id":        _KW(),
            "quantity":          _INT(),
            "unit_price_usd":    _FLT(),
            "line_total_usd":    _FLT(),
            "fulfilment_status": _KW(),
            "created_at":        _DATE_OPT(),
        }
    },

    f"{PREFIX}sessions": {
        # cart stored as nested array — each item is a typed sub-document.
        # Q3 naive deserialises the JSON string in Python; Q3 optimised
        # receives a fully typed nested array directly from ES.
        "properties": {
            "id":            _KW(),
            "user_id":       _KW(),
            "ip_address":    _KW(),
            "user_agent":    _TEXT_STD(),
            "created_at":    _DATE_OPT(),
            "last_active_at":_DATE_OPT(),
            "expires_at":    _DATE_OPT(),
            "cart": {
                "type": "nested",
                "properties": {
                    "product_id":   _KW(),
                    "product_name": _TEXT_STD(),
                    "product_type": _KW(),
                    "quantity":     _INT(),
                    "price_usd":    _FLT(),
                },
            },
        }
    },

    f"{PREFIX}events": {
        # ── Q6 optimisation ──────────────────────────────────────────────────
        # metadata stored as native ES object (dynamic sub-fields).
        # Q6 naive must use script-based access to filter on metadata fields
        # because the value is an opaque string. Q6 optimised can use
        # structured term/range queries directly on metadata.* sub-fields.
        # occurred_at with explicit date format is the primary range query field.
        "properties": {
            "id":          _KW(),
            "user_id":     _KW(),
            "event_type":  _KW(),
            "product_id":  _KW(),
            "session_id":  _KW(),
            "occurred_at": _DATE_OPT(),
            "metadata":    {"type": "object", "dynamic": True},
        }
    },
}

# ── row transformers ───────────────────────────────────────────────────────────
# Only fields that differ from naive are documented; the rest follow the same
# pattern as the naive loader.

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
        # preferences: keep as keyword blob — not used in any benchmark query
        "preferences":   row.get("preferences") or None,
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
        # attributes: parsed to native object in optimised schema
        "attributes":   _parse_json(row.get("attributes"), fallback={}),
        "created_at":   _date(row["created_at"]),
        "updated_at":   _date(row["updated_at"]),
    }

def _invoice(row, embedded_lines=None):
    doc = {
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
        "lines":                embedded_lines if embedded_lines is not None else [],
    }
    return doc

def _invoice_line_doc(row):
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

def _invoice_line_nested(row):
    """Invoice line shaped for embedding inside an invoice document.
    Omits invoice_id (redundant when nested inside the parent invoice)."""
    return {
        "id":             row["id"],
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
    # cart: parse JSON string to native Python list for ES nested type
    cart_raw = row.get("cart") or "[]"
    cart = _parse_json(cart_raw, fallback=[])
    # Ensure cart is a list (edge case: bare {} instead of [])
    if not isinstance(cart, list):
        cart = []
    return {
        "id":             row["id"],
        "user_id":        row["user_id"],
        "cart":           cart,
        "ip_address":     _str(row.get("ip_address")),
        "user_agent":     _str(row.get("user_agent")),
        "created_at":     _date(row["created_at"]),
        "last_active_at": _date(row["last_active_at"]),
        "expires_at":     _date(row["expires_at"]),
    }

def _event(row):
    # metadata: parse JSON string to native dict for ES object type
    metadata = _parse_json(row.get("metadata"), fallback={})
    return {
        "id":          row["id"],
        "user_id":     row["user_id"],
        "event_type":  row["event_type"],
        "product_id":  _str(row.get("product_id")),
        "session_id":  _str(row.get("session_id")),
        "metadata":    metadata,
        "occurred_at": _date(row["occurred_at"]),
    }

# ── invoice lines pre-loader ───────────────────────────────────────────────────

def _preload_invoice_lines(data_dir: Path) -> dict:
    """
    Read ALL invoice lines into memory keyed by invoice_id.

    This is necessary for embedding lines inside invoice documents at load
    time. Since lines must be fully loaded before invoices, we cannot
    stream both CSVs simultaneously. Memory usage is proportional to the
    total invoice_lines row count — acceptable for the dataset sizes used
    in this dissertation (~500K lines at 100% scale).

    Returns: {invoice_id: [line_doc, ...]}
    """
    info("Pre-loading invoice lines into memory (needed for embedding)...")
    lines_by_invoice = defaultdict(list)
    total = 0
    for path in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
        for row in read_csv(data_dir / path):     # no row limit — need all lines
            lines_by_invoice[row["invoice_id"]].append(_invoice_line_nested(row))
            total += 1
    info(f"  Loaded {total:,} lines across {len(lines_by_invoice):,} invoices")
    return lines_by_invoice

# ── per-index load logic ───────────────────────────────────────────────────────

def _load_one(es, index, docs_iter, id_field="id"):
    n_ok, n_err = load_docs(es, index, docs_iter, id_field=id_field)
    if n_err == 0:
        ok(f"{index:<48} {n_ok:>8,} docs")
    else:
        fail(f"{index:<48} {n_ok:>8,} ok  {n_err:>6,} errors")
    return n_ok, n_err


def load_all(es: Elasticsearch, drop: bool = False, limit: int | None = None,
             only: str | None = None):
    def _should(name):
        return only is None or name == only

    totals = {"ok": 0, "err": 0}

    def _run(name, docs_iter, id_field="id", settings=None):
        if not _should(name):
            return
        create_index(es, name, MAPPINGS[name], settings=settings, drop=drop)
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

    # ── 3. subscription_tiers (static data, features as native object) ────────
    _run(f"{PREFIX}subscription_tiers",
         iter(SUBSCRIPTION_TIERS[:limit] if limit else SUBSCRIPTION_TIERS))

    # ── 4. subscription_tier_pricing (static data) ────────────────────────────
    _run(f"{PREFIX}subscription_tier_pricing",
         iter(SUBSCRIPTION_TIER_PRICING[:limit] if limit else SUBSCRIPTION_TIER_PRICING))

    # ── 5. subscriptions ──────────────────────────────────────────────────────
    _run(f"{PREFIX}subscriptions",
         (_subscription(r) for r in read_csv(DATA_DIR / "subscriptions.csv", limit)))

    # ── 6. products (custom analyser settings applied here only) ──────────────
    _run(f"{PREFIX}products",
         (_product(r) for r in read_csv(DATA_DIR / "products.csv", limit)),
         settings=PRODUCTS_SETTINGS)

    # ── 7. invoices with embedded lines ───────────────────────────────────────
    # Pre-load invoice lines into memory (ignores limit — all lines needed
    # even if we only load a subset of invoices in dry-run mode).
    lines_needed = _should(f"{PREFIX}invoices")
    lines_by_invoice = _preload_invoice_lines(DATA_DIR) if lines_needed else {}

    tier_resolver = TierResolver(DATA_DIR) if lines_needed else None

    def _invoice_docs():
        for path in ("marketplace_invoices.csv", "subscription_invoices.csv"):
            for r in read_csv(DATA_DIR / path, limit):
                doc = _invoice(r, embedded_lines=lines_by_invoice.get(r["id"], []))
                doc.update(tier_resolver.resolve(r))
                yield doc

    _run(f"{PREFIX}invoices", _invoice_docs())

    # ── 8. invoice_lines (standalone — kept for 12-index symmetry) ────────────
    # Q2 optimised uses nested lines in invoices above, not this index.
    def _invoice_line_docs():
        for path in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
            for r in read_csv(DATA_DIR / path, limit):
                yield _invoice_line_doc(r)

    _run(f"{PREFIX}invoice_lines", _invoice_line_docs())

    # ── 9. orders ─────────────────────────────────────────────────────────────
    _run(f"{PREFIX}orders",
         (_order(r) for r in read_csv(DATA_DIR / "orders.csv", limit)))

    # ── 10. order_items ───────────────────────────────────────────────────────
    _run(f"{PREFIX}order_items",
         (_order_item(r) for r in read_csv(DATA_DIR / "order_items.csv", limit)))

    # ── 11. sessions (cart as native nested array) ────────────────────────────
    _run(f"{PREFIX}sessions",
         (_session(r) for r in read_csv(DATA_DIR / "sessions.csv", limit)))

    # ── 12. events (metadata as native object) ────────────────────────────────
    _run(f"{PREFIX}events",
         (_event(r) for r in read_csv(DATA_DIR / "events.csv", limit)))

    return totals

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load StreamCart data into Elasticsearch optimised indices"
    )
    parser.add_argument("--drop", action="store_true",
        help="Delete existing indices before loading")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
        help="Load only the first 500 rows per index")
    parser.add_argument("--index", type=str, default=None, metavar="INDEX",
        help="Only load a single index (e.g. optimised_products)")
    args = parser.parse_args()

    limit = 500 if args.dry_run else None

    print("\n" + "=" * 60)
    print("  Elasticsearch — Optimised Schema Loader")
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