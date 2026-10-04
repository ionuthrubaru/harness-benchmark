"""
loaders/cassandra_naive_loader.py — Cassandra Naive Schema Loader
=================================================================
Phase 3 — Cassandra, naive schema.

Naive schema design philosophy
────────────────────────────────
The naive schema mirrors the PostgreSQL relational layout as literally as
Cassandra permits. Each entity gets one table; the UUID (or text hash for
sessions) used as the primary key in PostgreSQL is used as the sole partition
key in Cassandra. No query-driven denormalisation is applied. This deliberately
violates Cassandra's core design rule ("model for your queries") in order to
isolate the engine effect — the performance difference attributable purely to
switching the database engine, holding schema design constant.

Consequences deliberately accepted in the naive schema
──────────────────────────────────────────────────────
1. ALLOW FILTERING required on most queries
   Any filter on a non-partition-key column (user_id, occurred_at, status,
   invoice_type, etc.) requires Cassandra to scan every partition on every
   node — the Cassandra equivalent of a PostgreSQL sequential scan. Benchmark
   scripts will include ALLOW FILTERING explicitly and document it. This is
   data, not a defect: it quantifies the cost of ignoring Cassandra's access
   model.

2. Python-side joins
   Cassandra has no JOIN. Queries that require related data (Q1: invoices +
   subscriptions + tier pricing; Q2: invoices + invoice_lines + products; etc.)
   must issue multiple CQL statements and merge results in Python. The join
   overhead is part of the naive latency measurement and is explicitly noted in
   the methodology.

3. No secondary indexes created
   SAI (Storage-Attached Index) and SASI indexes are not created in the naive
   schema. They would begin to approximate the optimised schema by introducing
   query-specific access paths. The naive schema has no query-optimising
   indexes beyond the mandatory partition key index.

Cassandra-specific deviations from the PostgreSQL schema
─────────────────────────────────────────────────────────
• No foreign key constraints
  Cassandra does not support referential integrity. This is an engine
  limitation, not a schema choice. Documented in the methodology chapter.

• JSON columns stored as TEXT
  PostgreSQL JSONB columns (preferences, cart, metadata, attributes, features)
  have no equivalent in Cassandra. They are stored as raw JSON TEXT strings.
  Content is byte-for-byte identical; the engine simply cannot query inside
  the structure without a SAI index (not created in the naive schema).

• products.search_vector omitted
  The search_vector column is a PostgreSQL-generated TSVECTOR — it is computed
  by a trigger from name + description + product_type and has no counterpart in
  Cassandra or in the source CSV. It is omitted from the Cassandra table. Q5
  (full-text search) in the naive schema uses ALLOW FILTERING + LIKE on the
  name column, which is the most literal Cassandra translation of a text search
  without a dedicated index.

• NUMERIC(p,s) → DECIMAL
  PostgreSQL's fixed-precision NUMERIC maps to Cassandra's arbitrary-precision
  DECIMAL. Semantically equivalent for all values in the dataset.

• SMALLINT PK for subscription_tiers → INT
  The tier IDs (1, 2, 3) are stored as INT. No functional difference at these
  values; Cassandra does not have a 2-byte integer type.

• subscription_tier_pricing composite PK
  PostgreSQL PRIMARY KEY (tier_id, valid_from). Cassandra PRIMARY KEY
  (tier_id, valid_from) makes tier_id the partition key and valid_from the
  clustering column. All pricing rows for a given tier live in the same
  partition, sorted by valid_from ascending. This is the closest Cassandra
  translation of the PostgreSQL composite PK and happens to be query-efficient
  for this table even in the naive schema — accepted as a structural inevitability
  rather than a query-driven optimisation.

• seller_profiles.total_sales
  Present in the PostgreSQL schema as a denormalised counter but absent from
  seller_profiles.csv (it is maintained by application logic). The column is
  included in the Cassandra table for schema fidelity; all loaded rows insert 0.

• invoices table — two source CSVs
  marketplace_invoices.csv and subscription_invoices.csv share identical column
  structure and are both loaded into the single invoices table. The
  invoice_type column ('marketplace' vs 'subscription') distinguishes them.
  Same applies to invoice_lines ← marketplace_invoice_lines.csv +
  subscription_invoice_lines.csv.

Keyspace
─────────
cassandra_naive — SimpleStrategy, replication_factor=1.
SimpleStrategy is correct for single-datacenter deployments. Replication
factor 1 means one copy per partition, the only viable option on a single-node
Docker instance. Documented in the methodology as the server configuration for
this experiment.

Usage
──────
    cd loaders
    python cassandra_naive_loader.py            # full load (drops + recreates keyspace)
    python cassandra_naive_loader.py --dry-run  # connect, create schema, skip CSV load
"""

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import ConsistencyLevel
from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── constants ─────────────────────────────────────────────────────────────────

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_NAIVE", "cassandra_naive")

# Number of parallel in-flight INSERT requests when loading large tables.
# execute_concurrent_with_args default is 100; 200 gives better throughput on
# a local single-node instance without overwhelming the JVM heap.
INSERT_CONCURRENCY = 200

# Rows processed per progress-reporting chunk. Purely cosmetic — does not
# affect correctness or performance of the inserts.
CHUNK_SIZE = 10_000

# ── DDL ───────────────────────────────────────────────────────────────────────
#
# Each CREATE TABLE mirrors its PostgreSQL counterpart as closely as Cassandra
# allows. See module docstring for per-column deviation notes.

CREATE_KEYSPACE = f"""
CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
    WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
    AND durable_writes = true
"""

# users — sole partition key = id (UUID), mirroring PostgreSQL PK.
CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id              uuid        PRIMARY KEY,
    email           text,
    full_name       text,
    country_code    text,
    city            text,
    created_at      timestamp,
    last_login_at   timestamp,
    is_active       boolean,
    preferences     text
)
"""

# seller_profiles — sole partition key = user_id (UUID), mirroring PostgreSQL PK.
# total_sales included for schema fidelity; defaults to 0 during load (absent from CSV).
CREATE_SELLER_PROFILES = """
CREATE TABLE IF NOT EXISTS seller_profiles (
    user_id         uuid        PRIMARY KEY,
    display_name    text,
    legal_name      text,
    tax_id          text,
    payout_email    text,
    country_code    text,
    is_verified     boolean,
    bio             text,
    total_sales     int,
    created_at      timestamp,
    updated_at      timestamp
)
"""

# subscription_tiers — sole partition key = id (INT).
# PostgreSQL uses SMALLINT; Cassandra does not have a 2-byte integer type → INT.
CREATE_SUBSCRIPTION_TIERS = """
CREATE TABLE IF NOT EXISTS subscription_tiers (
    id          int         PRIMARY KEY,
    name        text,
    description text,
    features    text
)
"""

# subscription_tier_pricing — composite PK: tier_id (partition) + valid_from (clustering).
# This mirrors the PostgreSQL composite PK (tier_id, valid_from) and groups all
# pricing rows for a tier in a single partition, sorted by valid_from ascending.
# valid_to is NULL for the currently active price row; Cassandra stores this as null.
CREATE_SUBSCRIPTION_TIER_PRICING = """
CREATE TABLE IF NOT EXISTS subscription_tier_pricing (
    tier_id             int,
    valid_from          timestamp,
    valid_to            timestamp,
    monthly_price_usd   decimal,
    is_active           boolean,
    PRIMARY KEY (tier_id, valid_from)
)
"""

# subscriptions — sole partition key = id (UUID).
CREATE_SUBSCRIPTIONS = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id                      uuid        PRIMARY KEY,
    user_id                 uuid,
    tier_id                 int,
    status                  text,
    started_at              timestamp,
    current_period_start    timestamp,
    current_period_end      timestamp,
    cancelled_at            timestamp,
    cancel_reason           text,
    billing_cycle           text,
    created_at              timestamp,
    updated_at              timestamp
)
"""

# products — sole partition key = id (UUID).
# search_vector (PostgreSQL tsvector) omitted — see module docstring.
CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    id              uuid        PRIMARY KEY,
    name            text,
    slug            text,
    product_type    text,
    description     text,
    price_usd       decimal,
    currency        text,
    is_active       boolean,
    seller_id       uuid,
    attributes      text,
    created_at      timestamp,
    updated_at      timestamp
)
"""

# invoices — sole partition key = id (UUID).
# Loaded from two source CSVs: marketplace_invoices.csv + subscription_invoices.csv.
# subscription_id, billing_period_start, billing_period_end, paid_at are nullable
# (absent / empty for marketplace invoices).
CREATE_INVOICES = """
CREATE TABLE IF NOT EXISTS invoices (
    id                      uuid        PRIMARY KEY,
    user_id                 uuid,
    invoice_type            text,
    status                  text,
    subtotal_usd            decimal,
    tax_usd                 decimal,
    discount_usd            decimal,
    total_usd               decimal,
    subscription_id         uuid,
    billing_period_start    timestamp,
    billing_period_end      timestamp,
    paid_at                 timestamp,
    due_at                  timestamp,
    created_at              timestamp
)
"""

# invoice_lines — sole partition key = id (UUID).
# Loaded from marketplace_invoice_lines.csv + subscription_invoice_lines.csv.
# product_id is nullable (NULL for subscription renewal lines which charge a tier,
# not a specific product).
CREATE_INVOICE_LINES = """
CREATE TABLE IF NOT EXISTS invoice_lines (
    id              uuid        PRIMARY KEY,
    invoice_id      uuid,
    product_id      uuid,
    description     text,
    quantity        int,
    unit_price_usd  decimal,
    line_total_usd  decimal,
    created_at      timestamp
)
"""

# orders — sole partition key = id (UUID).
CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    id                  uuid        PRIMARY KEY,
    user_id             uuid,
    invoice_id          uuid,
    status              text,
    shipping_name       text,
    shipping_address    text,
    shipping_city       text,
    shipping_country    text,
    shipping_postal     text,
    created_at          timestamp,
    updated_at          timestamp
)
"""

# order_items — sole partition key = id (UUID).
CREATE_ORDER_ITEMS = """
CREATE TABLE IF NOT EXISTS order_items (
    id                  uuid        PRIMARY KEY,
    order_id            uuid,
    product_id          uuid,
    quantity            int,
    unit_price_usd      decimal,
    line_total_usd      decimal,
    fulfilment_status   text,
    created_at          timestamp
)
"""

# sessions — sole partition key = id (TEXT).
# PostgreSQL uses VARCHAR(64) for the session token (a hex hash string, not a UUID).
# Cassandra uses text; no change in semantics.
# cart is stored as raw JSON text (PostgreSQL JSONB → Cassandra text).
# ip_address is stored as text (PostgreSQL INET → Cassandra has no INET type).
CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id              text        PRIMARY KEY,
    user_id         uuid,
    cart            text,
    ip_address      text,
    user_agent      text,
    created_at      timestamp,
    last_active_at  timestamp,
    expires_at      timestamp
)
"""

# events — sole partition key = id (UUID).
# Naive design: id as sole PK means any query filtering on user_id or occurred_at
# requires ALLOW FILTERING — the canonical anti-pattern for Cassandra event storage.
# The optimised schema will replace this with a (user_id, month) partition key and
# occurred_at clustering column. That difference is what the schema effect measures.
# product_id and session_id are nullable.
CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id              uuid        PRIMARY KEY,
    user_id         uuid,
    event_type      text,
    product_id      uuid,
    session_id      text,
    metadata        text,
    occurred_at     timestamp
)
"""

ALL_DDL = [
    ("users",                    CREATE_USERS),
    ("seller_profiles",          CREATE_SELLER_PROFILES),
    ("subscription_tiers",       CREATE_SUBSCRIPTION_TIERS),
    ("subscription_tier_pricing",CREATE_SUBSCRIPTION_TIER_PRICING),
    ("subscriptions",            CREATE_SUBSCRIPTIONS),
    ("products",                 CREATE_PRODUCTS),
    ("invoices",                 CREATE_INVOICES),
    ("invoice_lines",            CREATE_INVOICE_LINES),
    ("orders",                   CREATE_ORDERS),
    ("order_items",              CREATE_ORDER_ITEMS),
    ("sessions",                 CREATE_SESSIONS),
    ("events",                   CREATE_EVENTS),
]

# ── type-coercion helpers ─────────────────────────────────────────────────────
#
# CSV values are always strings. These helpers convert to the types expected by
# the Cassandra Python driver for each column type.

def _uuid(v: str) -> uuid.UUID | None:
    """Parse a UUID string → uuid.UUID, or None if empty."""
    return uuid.UUID(v) if v and v.strip() else None


def _dt(v: str) -> datetime | None:
    """
    Parse an ISO 8601 timestamp string → timezone-aware datetime, or None.
    The Cassandra driver stores datetimes as UTC internally; passing a
    timezone-aware datetime is the safest approach.
    Handles both '+00:00' suffix (CSV format) and 'Z' suffix.
    """
    if not v or not v.strip():
        return None
    # Replace 'Z' with '+00:00' for fromisoformat compatibility (Python < 3.11)
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


def _dec(v: str) -> Decimal | None:
    """Parse a decimal string → Decimal, or None if empty."""
    return Decimal(v) if v and v.strip() else None


def _bool(v: str) -> bool | None:
    """Parse 'True'/'False' string → Python bool, or None if empty."""
    if not v or not v.strip():
        return None
    return v.strip().lower() == "true"


def _int(v: str) -> int | None:
    """Parse an integer string → int, or None if empty."""
    return int(v.strip()) if v and v.strip() else None


def _text(v: str) -> str | None:
    """Return the string as-is, or None if empty."""
    return v if v and v.strip() else None


# ── connection (inline, no keyspace yet) ──────────────────────────────────────

def _connect_no_keyspace():
    """
    Open a cluster-level connection without selecting a keyspace.
    Used at the start of the loader to issue CREATE KEYSPACE before
    the keyspace exists. Returns (cluster, session).
    """
    auth = PlainTextAuthProvider(
        username=os.getenv("CASSANDRA_USER", "cassandra"),
        password=os.getenv("CASSANDRA_PASSWORD", "cassandra"),
    )
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        consistency_level=ConsistencyLevel.LOCAL_ONE,
        request_timeout=60.0,   # generous for DDL (CREATE TABLE can be slow on cold start)
    )
    cluster = Cluster(
        contact_points=["localhost"],
        port=int(os.getenv("CASSANDRA_PORT", "9042")),
        auth_provider=auth,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    )
    session = cluster.connect()
    return cluster, session


# ── schema management ─────────────────────────────────────────────────────────

def drop_keyspace(session):
    print(f"  Dropping keyspace '{KEYSPACE}' if it exists...")
    session.execute(f"DROP KEYSPACE IF EXISTS {KEYSPACE}")
    print(f"  ✔ Keyspace dropped.")


def create_schema(session):
    print(f"\n  Creating keyspace '{KEYSPACE}'...")
    session.execute(CREATE_KEYSPACE)
    session.set_keyspace(KEYSPACE)
    print(f"  ✔ Keyspace created. Creating tables...")

    for table_name, ddl in ALL_DDL:
        session.execute(ddl)
        print(f"    ✔ {table_name}")

    print(f"  ✔ All {len(ALL_DDL)} tables created.")


# ── insert helper ─────────────────────────────────────────────────────────────

def _bulk_insert(session, prepared, all_params: list, table_name: str):
    """
    Insert all_params into the given prepared statement using
    execute_concurrent_with_args, reporting progress every CHUNK_SIZE rows.

    Parameters
    ----------
    session    : active Cassandra session (keyspace already set)
    prepared   : PreparedStatement returned by session.prepare()
    all_params : list of tuples, one per row
    table_name : used only for progress output
    """
    total = len(all_params)
    if total == 0:
        print(f"  {table_name}: 0 rows — skipping.")
        return

    inserted = 0
    for i in range(0, total, CHUNK_SIZE):
        chunk = all_params[i : i + CHUNK_SIZE]
        execute_concurrent_with_args(
            session,
            prepared,
            chunk,
            concurrency=INSERT_CONCURRENCY,
            raise_on_first_error=True,
        )
        inserted += len(chunk)
        print(f"\r  {table_name}: {inserted:,}/{total:,}", end="", flush=True)

    print(f"\r  ✔ {table_name}: {total:,} rows loaded.            ")


# ── hardcoded data ─────────────────────────────────────────────────────────────

def load_subscription_tiers(session):
    """
    Insert the three subscription tiers.
    These are hardcoded (not from a CSV) in both PostgreSQL and Cassandra.
    features is stored as a JSON text string, matching the TEXT column type.
    """
    prepared = session.prepare("""
        INSERT INTO subscription_tiers (id, name, description, features)
        VALUES (?, ?, ?, ?)
    """)

    rows = [
        (
            1, "Free",
            "Basic software access, up to 5 marketplace purchases/year",
            json.dumps({
                "seats": 1, "api_access": False, "priority_support": False,
                "marketplace_purchases_per_year": 5,
                "apps": {"CanvasEditor": "free", "VideoSuite": None},
            }),
        ),
        (
            2, "Pro",
            "Full software, unlimited purchases, early access",
            json.dumps({
                "seats": 1, "api_access": False, "priority_support": False,
                "marketplace_purchases_per_year": -1,
                "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"},
            }),
        ),
        (
            3, "Business",
            "Everything in Pro plus team seats, API access, priority support",
            json.dumps({
                "seats": 10, "api_access": True, "priority_support": True,
                "marketplace_purchases_per_year": -1,
                "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"},
            }),
        ),
    ]
    for row in rows:
        session.execute(prepared, row)
    print(f"  ✔ subscription_tiers: 3 rows loaded.")


def load_subscription_tier_pricing(session):
    """
    Insert the five pricing rows.
    Hardcoded as in PostgreSQL. valid_to is None for the currently active price.
    The composite PK (tier_id, valid_from) is preserved — tier_id as partition
    key, valid_from as clustering column (see CREATE TABLE comment above).
    """
    prepared = session.prepare("""
        INSERT INTO subscription_tier_pricing
            (tier_id, valid_from, valid_to, monthly_price_usd, is_active)
        VALUES (?, ?, ?, ?, ?)
    """)

    rows = [
        (1, datetime(2023, 1, 1, tzinfo=timezone.utc), None,
         Decimal("0.00"), True),
        (2, datetime(2023, 1, 1, tzinfo=timezone.utc),
         datetime(2024, 6, 1, tzinfo=timezone.utc),
         Decimal("14.99"), False),
        (2, datetime(2024, 6, 1, tzinfo=timezone.utc), None,
         Decimal("19.99"), True),
        (3, datetime(2023, 1, 1, tzinfo=timezone.utc),
         datetime(2024, 6, 1, tzinfo=timezone.utc),
         Decimal("39.99"), False),
        (3, datetime(2024, 6, 1, tzinfo=timezone.utc), None,
         Decimal("49.99"), True),
    ]
    for row in rows:
        session.execute(prepared, row)
    print(f"  ✔ subscription_tier_pricing: 5 rows loaded.")


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_users(session):
    prepared = session.prepare("""
        INSERT INTO users
            (id, email, full_name, country_code, city,
             created_at, last_login_at, is_active, preferences)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "users.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _text(row["email"]),
                _text(row["full_name"]),
                _text(row["country_code"]),
                _text(row["city"]),
                _dt(row["created_at"]),
                _dt(row["last_login_at"]),
                _bool(row["is_active"]),
                _text(row["preferences"]),   # raw JSON string
            ))
    _bulk_insert(session, prepared, params, "users")


def load_seller_profiles(session):
    prepared = session.prepare("""
        INSERT INTO seller_profiles
            (user_id, display_name, legal_name, tax_id, payout_email,
             country_code, is_verified, bio, total_sales, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "seller_profiles.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["user_id"]),
                _text(row["display_name"]),
                _text(row["legal_name"]),
                _text(row["tax_id"]),
                _text(row["payout_email"]),
                _text(row["country_code"]),
                _bool(row["is_verified"]),
                _text(row["bio"]),
                0,                           # total_sales: not in CSV, default to 0
                _dt(row["created_at"]),
                _dt(row["updated_at"]),
            ))
    _bulk_insert(session, prepared, params, "seller_profiles")


def load_subscriptions(session):
    prepared = session.prepare("""
        INSERT INTO subscriptions
            (id, user_id, tier_id, status, started_at,
             current_period_start, current_period_end,
             cancelled_at, cancel_reason, billing_cycle,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "subscriptions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _uuid(row["user_id"]),
                _int(row["tier_id"]),
                _text(row["status"]),
                _dt(row["started_at"]),
                _dt(row["current_period_start"]),
                _dt(row["current_period_end"]),
                _dt(row["cancelled_at"]),    # nullable
                _text(row["cancel_reason"]), # nullable
                _text(row["billing_cycle"]),
                _dt(row["created_at"]),
                _dt(row["updated_at"]),
            ))
    _bulk_insert(session, prepared, params, "subscriptions")


def load_products(session):
    prepared = session.prepare("""
        INSERT INTO products
            (id, name, slug, product_type, description,
             price_usd, currency, is_active, seller_id,
             attributes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "products.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _text(row["name"]),
                _text(row["slug"]),
                _text(row["product_type"]),
                _text(row["description"]),
                _dec(row["price_usd"]),
                _text(row["currency"]),
                _bool(row["is_active"]),
                _uuid(row["seller_id"]),
                _text(row["attributes"]),    # raw JSON string
                _dt(row["created_at"]),
                _dt(row["updated_at"]),
            ))
    _bulk_insert(session, prepared, params, "products")


def load_invoices(session):
    """
    Loads both marketplace_invoices.csv and subscription_invoices.csv into the
    single invoices table. The invoice_type column distinguishes them.
    nullable fields: subscription_id, billing_period_start, billing_period_end,
    paid_at (empty for some marketplace invoices in the dataset).
    """
    prepared = session.prepare("""
        INSERT INTO invoices
            (id, user_id, invoice_type, status,
             subtotal_usd, tax_usd, discount_usd, total_usd,
             subscription_id, billing_period_start, billing_period_end,
             paid_at, due_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for filename in ("marketplace_invoices.csv", "subscription_invoices.csv"):
        with open(os.path.join(DATA_DIR, filename), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                params.append((
                    _uuid(row["id"]),
                    _uuid(row["user_id"]),
                    _text(row["invoice_type"]),
                    _text(row["status"]),
                    _dec(row["subtotal_usd"]),
                    _dec(row["tax_usd"]),
                    _dec(row["discount_usd"]),
                    _dec(row["total_usd"]),
                    _uuid(row["subscription_id"]),          # nullable
                    _dt(row["billing_period_start"]),       # nullable
                    _dt(row["billing_period_end"]),         # nullable
                    _dt(row["paid_at"]),                    # nullable
                    _dt(row["due_at"]),
                    _dt(row["created_at"]),
                ))
    _bulk_insert(session, prepared, params, "invoices")


def load_invoice_lines(session):
    """
    Loads marketplace_invoice_lines.csv and subscription_invoice_lines.csv into
    the single invoice_lines table.
    product_id is nullable for subscription renewal lines (no specific product,
    just a tier charge).
    """
    prepared = session.prepare("""
        INSERT INTO invoice_lines
            (id, invoice_id, product_id, description,
             quantity, unit_price_usd, line_total_usd, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for filename in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
        with open(os.path.join(DATA_DIR, filename), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                params.append((
                    _uuid(row["id"]),
                    _uuid(row["invoice_id"]),
                    _uuid(row["product_id"]),   # nullable for subscription lines
                    _text(row["description"]),
                    _int(row["quantity"]),
                    _dec(row["unit_price_usd"]),
                    _dec(row["line_total_usd"]),
                    _dt(row["created_at"]),
                ))
    _bulk_insert(session, prepared, params, "invoice_lines")


def load_orders(session):
    prepared = session.prepare("""
        INSERT INTO orders
            (id, user_id, invoice_id, status,
             shipping_name, shipping_address, shipping_city,
             shipping_country, shipping_postal,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "orders.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _uuid(row["user_id"]),
                _uuid(row["invoice_id"]),
                _text(row["status"]),
                _text(row["shipping_name"]),
                _text(row["shipping_address"]),
                _text(row["shipping_city"]),
                _text(row["shipping_country"]),
                _text(row["shipping_postal"]),
                _dt(row["created_at"]),
                _dt(row["updated_at"]),
            ))
    _bulk_insert(session, prepared, params, "orders")


def load_order_items(session):
    prepared = session.prepare("""
        INSERT INTO order_items
            (id, order_id, product_id, quantity,
             unit_price_usd, line_total_usd, fulfilment_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "order_items.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _uuid(row["order_id"]),
                _uuid(row["product_id"]),
                _int(row["quantity"]),
                _dec(row["unit_price_usd"]),
                _dec(row["line_total_usd"]),
                _text(row["fulfilment_status"]),
                _dt(row["created_at"]),
            ))
    _bulk_insert(session, prepared, params, "order_items")


def load_sessions(session):
    """
    Session IDs are 64-character hex strings (not UUIDs) — stored as text.
    cart is a JSON array stored as raw text.
    ip_address is stored as text (no Cassandra INET type).
    """
    prepared = session.prepare("""
        INSERT INTO sessions
            (id, user_id, cart, ip_address, user_agent,
             created_at, last_active_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "sessions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                row["id"],                      # text, not UUID
                _uuid(row["user_id"]),
                _text(row["cart"]),             # raw JSON text
                _text(row["ip_address"]),       # text (no INET type in Cassandra)
                _text(row["user_agent"]),
                _dt(row["created_at"]),
                _dt(row["last_active_at"]),
                _dt(row["expires_at"]),
            ))
    _bulk_insert(session, prepared, params, "sessions")


def load_events(session):
    """
    events is the largest table and the primary target of Q6 (Cassandra's killer query).
    In the naive schema, id is the sole partition key — any filter on user_id or
    occurred_at requires ALLOW FILTERING and a full table scan. This is the central
    performance anti-pattern the naive schema demonstrates.

    product_id and session_id are nullable (empty in the CSV for non-product events).
    metadata is stored as raw JSON text.
    """
    prepared = session.prepare("""
        INSERT INTO events
            (id, user_id, event_type, product_id, session_id, metadata, occurred_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    with open(os.path.join(DATA_DIR, "events.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params.append((
                _uuid(row["id"]),
                _uuid(row["user_id"]),
                _text(row["event_type"]),
                _uuid(row["product_id"]),     # nullable
                _text(row["session_id"]),     # nullable text (not UUID)
                _text(row["metadata"]),       # raw JSON text
                _dt(row["occurred_at"]),
            ))
    _bulk_insert(session, prepared, params, "events")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cassandra naive schema loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full load drops and recreates the keyspace from scratch.\n"
            "--dry-run creates the schema but skips all CSV inserts.\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Create schema only — do not insert any data.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra — Naive Schema Loader")
    print("=" * 60)
    print(f"  Keyspace : {KEYSPACE}")
    print(f"  Data dir : {DATA_DIR}")
    if args.dry_run:
        print("  Mode     : DRY RUN (schema only, no data)")
    print()

    cluster, session = _connect_no_keyspace()

    try:
        # Drop the keyspace (full rebuild — mirrors MongoDB loader behaviour)
        drop_keyspace(session)

        # Create keyspace + all tables
        create_schema(session)

        if args.dry_run:
            print("\n  Dry run complete — schema verified, no data inserted.")
            return

        # ── Load data ────────────────────────────────────────────────────────
        # Order matters only for readability; Cassandra has no FK constraints,
        # so any insertion order is valid.

        print("\n  Loading data...")
        print("  " + "─" * 56)

        load_subscription_tiers(session)
        load_subscription_tier_pricing(session)
        load_users(session)
        load_seller_profiles(session)
        load_subscriptions(session)
        load_products(session)
        load_invoices(session)
        load_invoice_lines(session)
        load_orders(session)
        load_order_items(session)
        load_sessions(session)
        load_events(session)          # largest table — loaded last for cleaner progress output

        print()
        print("  " + "─" * 56)
        print("  ✔ Cassandra naive schema fully loaded.")
        print(f"  Keyspace: {KEYSPACE}")
        print()

    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()