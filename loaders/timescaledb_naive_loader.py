"""
loaders/timescaledb_naive_loader.py — TimescaleDB Naive Schema Loader
======================================================================
Phase 3 — TimescaleDB, naive schema.

Naive schema design philosophy
────────────────────────────────
The naive schema mirrors the PostgreSQL relational schema as closely as
possible. Every table is created with the same columns, constraints, and
indexes as the PostgreSQL baseline. The only structural additions are:

  SELECT create_hypertable('events',   'occurred_at')
  SELECT create_hypertable('invoices', 'created_at')

These two calls convert the time-series tables into TimescaleDB hypertables
— the minimum possible change to run TimescaleDB. No continuous aggregates,
no compression, no custom chunk intervals. This answers the engine effect
question: does switching to TimescaleDB (with hypertables) give any
performance benefit over plain PostgreSQL, before any time-series-specific
optimisation is applied?

Why events and invoices are hypertables in the naive schema
────────────────────────────────────────────────────────────
Hypertables are TimescaleDB's fundamental storage unit — they are what
makes TimescaleDB a time-series database rather than just PostgreSQL with
extensions. Creating hypertables on the two time-anchored tables (events
by occurred_at, invoices by created_at) is the minimum meaningful use of
TimescaleDB and is what any practitioner would do as a first step. Not
creating any hypertables would make the naive schema identical to the
PostgreSQL baseline and produce zero engine effect — not a useful data point.

The other tables (users, products, subscriptions, etc.) are not hypertables
because they are not time-series data — they are reference/entity tables
that grow slowly and are queried by ID, not by time range.

Hypertable chunk interval
──────────────────────────
Default chunk interval = 7 days for both hypertables.
TimescaleDB's default is 7 days, which is appropriate for the dataset's
date range (~2 years of events). The optimised schema will reduce the
invoice chunk interval to 1 month (aligning with monthly billing cycles)
and add compression after 30 days. The naive schema uses the default
so that any performance difference between naive and optimised is
attributable to those deliberate optimisations, not to incidental
chunk size differences.

Schema fidelity
────────────────
The schema is reproduced verbatim from schema.sql with the following
deliberate exceptions:
  • uuid-ossp extension: re-created if not present (same as PostgreSQL)
  • search_vector trigger: included — TimescaleDB supports PostgreSQL
    triggers and functions natively, so this is directly reproducible
  • All indexes from schema.sql are created identically

Two-source invoice loading
───────────────────────────
marketplace_invoices.csv and subscription_invoices.csv are both loaded
into the single invoices table. Same approach as all other loaders in
this project.

Usage:
    cd loaders
    python timescaledb_naive_loader.py            # full load
    python timescaledb_naive_loader.py --dry-run  # schema only, no data
    python timescaledb_naive_loader.py --verify   # check row counts only
"""

import argparse
import csv
import json
import os
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg): print(f"  {RED}✘ {msg}{RESET}")
def info(msg): print(f"  {BLUE}> {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}! {msg}{RESET}")

# ── connection ─────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"),
        connect_timeout=10,
    )

# ── schema DDL ─────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- users
CREATE TABLE IF NOT EXISTS users (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255)    NOT NULL UNIQUE,
    full_name       VARCHAR(255)    NOT NULL,
    country_code    CHAR(2)         NOT NULL,
    city            VARCHAR(100),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    preferences     JSONB           NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_users_email      ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_country    ON users (country_code);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at);

-- seller_profiles
CREATE TABLE IF NOT EXISTS seller_profiles (
    user_id         UUID            PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    display_name    VARCHAR(255)    NOT NULL,
    legal_name      VARCHAR(255),
    tax_id          VARCHAR(100),
    payout_email    VARCHAR(255),
    country_code    CHAR(2),
    is_verified     BOOLEAN         NOT NULL DEFAULT FALSE,
    bio             TEXT,
    total_sales     INTEGER         NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_seller_profiles_verified ON seller_profiles (is_verified);
CREATE INDEX IF NOT EXISTS idx_seller_profiles_country  ON seller_profiles (country_code);

-- subscription_tiers
CREATE TABLE IF NOT EXISTS subscription_tiers (
    id          SMALLINT        PRIMARY KEY,
    name        VARCHAR(50)     NOT NULL UNIQUE,
    description TEXT,
    features    JSONB           NOT NULL DEFAULT '{}'
);

-- subscription_tier_pricing
CREATE TABLE IF NOT EXISTS subscription_tier_pricing (
    tier_id             SMALLINT        NOT NULL REFERENCES subscription_tiers (id),
    valid_from          TIMESTAMPTZ     NOT NULL,
    valid_to            TIMESTAMPTZ,
    monthly_price_usd   NUMERIC(8,2)    NOT NULL,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    PRIMARY KEY (tier_id, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_tier_pricing_tier_id    ON subscription_tier_pricing (tier_id);
CREATE INDEX IF NOT EXISTS idx_tier_pricing_valid_from ON subscription_tier_pricing (valid_from);

-- subscriptions
CREATE TABLE IF NOT EXISTS subscriptions (
    id                      UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id                 UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    tier_id                 SMALLINT        NOT NULL REFERENCES subscription_tiers (id),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'active',
    started_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    current_period_start    TIMESTAMPTZ     NOT NULL,
    current_period_end      TIMESTAMPTZ     NOT NULL,
    cancelled_at            TIMESTAMPTZ,
    cancel_reason           TEXT,
    billing_cycle           VARCHAR(10)     NOT NULL DEFAULT 'monthly',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id    ON subscriptions (user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_tier_id    ON subscriptions (tier_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status     ON subscriptions (status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_period_end ON subscriptions (current_period_end);

-- products
CREATE TABLE IF NOT EXISTS products (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255)    NOT NULL,
    slug            VARCHAR(255)    NOT NULL UNIQUE,
    product_type    VARCHAR(20)     NOT NULL,
    description     TEXT            NOT NULL,
    price_usd       NUMERIC(8,2)    NOT NULL,
    currency        CHAR(3)         NOT NULL DEFAULT 'USD',
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    seller_id       UUID            NOT NULL REFERENCES users (id),
    attributes      JSONB           NOT NULL DEFAULT '{}',
    search_vector   TSVECTOR,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_products_type       ON products (product_type);
CREATE INDEX IF NOT EXISTS idx_products_seller_id  ON products (seller_id);
CREATE INDEX IF NOT EXISTS idx_products_price      ON products (price_usd);
CREATE INDEX IF NOT EXISTS idx_products_active     ON products (is_active);
CREATE INDEX IF NOT EXISTS idx_products_attributes ON products USING GIN (attributes);
CREATE INDEX IF NOT EXISTS idx_products_search     ON products USING GIN (search_vector);

CREATE OR REPLACE FUNCTION update_product_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.name, '')),        'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.product_type, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_products_search_vector ON products;
CREATE TRIGGER trg_products_search_vector
    BEFORE INSERT OR UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_product_search_vector();

-- invoices — will become a hypertable on created_at.
-- TimescaleDB requirement: ALL unique constraints (including PKs) must
-- include the partition column. Therefore:
--   PK becomes (id, created_at) — not just (id).
--   Unique index on id alone is not permitted on a hypertable.
--   FK references from invoice_lines and orders to invoices(id) are
--   dropped because the FK target would need to be (id, created_at),
--   which would require adding created_at to every child table.
-- These are documented as TimescaleDB engine constraints in the methodology.
CREATE TABLE IF NOT EXISTS invoices (
    id                      UUID            NOT NULL DEFAULT uuid_generate_v4(),
    user_id                 UUID            NOT NULL REFERENCES users (id),
    invoice_type            VARCHAR(20)     NOT NULL,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
    subtotal_usd            NUMERIC(10,2)   NOT NULL,
    tax_usd                 NUMERIC(10,2)   NOT NULL DEFAULT 0.00,
    discount_usd            NUMERIC(10,2)   NOT NULL DEFAULT 0.00,
    total_usd               NUMERIC(10,2)   NOT NULL,
    subscription_id         UUID            REFERENCES subscriptions (id),
    billing_period_start    TIMESTAMPTZ,
    billing_period_end      TIMESTAMPTZ,
    paid_at                 TIMESTAMPTZ,
    due_at                  TIMESTAMPTZ     NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);
CREATE INDEX IF NOT EXISTS idx_invoices_user_id         ON invoices (user_id);
CREATE INDEX IF NOT EXISTS idx_invoices_subscription_id ON invoices (subscription_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status          ON invoices (status);
CREATE INDEX IF NOT EXISTS idx_invoices_created_at      ON invoices (created_at);
CREATE INDEX IF NOT EXISTS idx_invoices_type            ON invoices (invoice_type);

-- invoice_lines
CREATE TABLE IF NOT EXISTS invoice_lines (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- FK to invoices removed: TimescaleDB unique constraints must include
    -- the partition column (created_at). Adding created_at to this FK would
    -- require changing every child table. Data integrity is maintained by
    -- the load process. Documented as a TimescaleDB engine constraint.
    invoice_id      UUID            NOT NULL,
    product_id      UUID            REFERENCES products (id),
    description     VARCHAR(500)    NOT NULL,
    quantity        SMALLINT        NOT NULL DEFAULT 1,
    unit_price_usd  NUMERIC(8,2)    NOT NULL,
    line_total_usd  NUMERIC(10,2)   NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_invoice_lines_invoice_id ON invoice_lines (invoice_id);
CREATE INDEX IF NOT EXISTS idx_invoice_lines_product_id ON invoice_lines (product_id);

-- orders
CREATE TABLE IF NOT EXISTS orders (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id),
    invoice_id          UUID            NOT NULL,  -- FK removed: see invoice_lines note above
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
    shipping_name       VARCHAR(255),
    shipping_address    VARCHAR(500),
    shipping_city       VARCHAR(100),
    shipping_country    CHAR(2),
    shipping_postal     VARCHAR(20),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orders_user_id    ON orders (user_id);
CREATE INDEX IF NOT EXISTS idx_orders_invoice_id ON orders (invoice_id);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders (created_at);

-- order_items
CREATE TABLE IF NOT EXISTS order_items (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id            UUID            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    product_id          UUID            NOT NULL REFERENCES products (id),
    quantity            SMALLINT        NOT NULL DEFAULT 1,
    unit_price_usd      NUMERIC(8,2)    NOT NULL,
    line_total_usd      NUMERIC(10,2)   NOT NULL,
    fulfilment_status   VARCHAR(20)     NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id   ON order_items (order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items (product_id);

-- sessions
CREATE TABLE IF NOT EXISTS sessions (
    id              VARCHAR(64)     PRIMARY KEY,
    user_id         UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    cart            JSONB           NOT NULL DEFAULT '[]',
    ip_address      INET,
    user_agent      VARCHAR(500),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id     ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at  ON sessions (expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions (last_active_at);

-- events — will become a hypertable on occurred_at.
-- TimescaleDB requires the partition column (occurred_at) to be part of
-- the primary key. The PK becomes composite (id, occurred_at) instead of
-- just (id). This is a documented deviation from the PostgreSQL schema —
-- the unique constraint on id is preserved via a separate unique index.
CREATE TABLE IF NOT EXISTS events (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL REFERENCES users (id),
    event_type      VARCHAR(50)     NOT NULL,
    product_id      UUID            REFERENCES products (id),
    session_id      VARCHAR(64)     REFERENCES sessions (id),
    metadata        JSONB           NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, occurred_at)
);
CREATE INDEX IF NOT EXISTS idx_events_user_time   ON events (user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type        ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_product_id  ON events (product_id);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events (occurred_at DESC);
"""

# ── hypertable setup ───────────────────────────────────────────────────────────

HYPERTABLE_SQL = """
-- Convert events → hypertable partitioned by occurred_at.
-- Default chunk interval: 7 days (TimescaleDB default).
-- chunk_time_interval left at default so the naive schema uses TimescaleDB's
-- out-of-the-box behaviour. The optimised schema will tune this per-table.
SELECT create_hypertable(
    'events',
    'occurred_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

-- Convert invoices → hypertable partitioned by created_at.
SELECT create_hypertable(
    'invoices',
    'created_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);
"""

# ── hardcoded seed data ────────────────────────────────────────────────────────

SUBSCRIPTION_TIERS = [
    (1, 'Free',     'Basic software access, up to 5 marketplace purchases/year',
     json.dumps({"seats": 1, "api_access": False, "priority_support": False,
                 "marketplace_purchases_per_year": 5,
                 "apps": {"CanvasEditor": "free", "VideoSuite": None}})),
    (2, 'Pro',      'Full software, unlimited purchases, early access',
     json.dumps({"seats": 1, "api_access": False, "priority_support": False,
                 "marketplace_purchases_per_year": -1,
                 "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"}})),
    (3, 'Business', 'Everything in Pro plus team seats, API access, priority support',
     json.dumps({"seats": 10, "api_access": True, "priority_support": True,
                 "marketplace_purchases_per_year": -1,
                 "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"}})),
]

SUBSCRIPTION_TIER_PRICING = [
    (1, '2023-01-01 00:00:00+00', None,                      0.00,  True),
    (2, '2023-01-01 00:00:00+00', '2024-06-01 00:00:00+00', 14.99, False),
    (2, '2024-06-01 00:00:00+00', None,                      19.99, True),
    (3, '2023-01-01 00:00:00+00', '2024-06-01 00:00:00+00', 39.99, False),
    (3, '2024-06-01 00:00:00+00', None,                      49.99, True),
]

# ── CSV loading helpers ────────────────────────────────────────────────────────

def _csv_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def _copy_from_csv(cur, table: str, csv_path: str, columns: list[str]):
    """
    Use COPY for fast bulk loading — identical approach to the PostgreSQL loader.
    COPY is the standard PostgreSQL/TimescaleDB bulk load mechanism and is
    compatible with hypertables (TimescaleDB intercepts the COPY and routes
    rows to the correct chunk automatically).
    """
    col_list = ", ".join(columns)
    sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    with open(csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)


# ── schema creation ────────────────────────────────────────────────────────────

def create_schema(conn):
    info("Creating tables and indexes...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    ok("Schema created.")


def create_hypertables(conn):
    """
    Convert events and invoices to hypertables.
    Must be called BEFORE loading data — TimescaleDB can migrate existing
    data (migrate_data=TRUE) but it is faster to create the hypertable on an
    empty table and let COPY route rows to chunks directly.
    """
    info("Creating hypertables (events → occurred_at, invoices → created_at)...")
    with conn.cursor() as cur:
        cur.execute(HYPERTABLE_SQL)
    conn.commit()
    ok("Hypertables created.")


def drop_and_recreate(conn):
    """Drop all tables and recreate from scratch. Handles FK ordering."""
    info("Dropping existing tables (if any)...")
    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS
                events, sessions, order_items, orders,
                invoice_lines, invoices, products,
                subscriptions, subscription_tier_pricing,
                subscription_tiers, seller_profiles, users
            CASCADE;
        """)
    conn.commit()
    ok("Tables dropped.")

# ── seed data ──────────────────────────────────────────────────────────────────

def load_subscription_tiers(conn):
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO subscription_tiers (id, name, description, features) VALUES %s",
            SUBSCRIPTION_TIERS,
        )
    conn.commit()
    ok(f"subscription_tiers: {len(SUBSCRIPTION_TIERS)} rows")


def load_subscription_tier_pricing(conn):
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO subscription_tier_pricing
               (tier_id, valid_from, valid_to, monthly_price_usd, is_active)
               VALUES %s""",
            SUBSCRIPTION_TIER_PRICING,
        )
    conn.commit()
    ok(f"subscription_tier_pricing: {len(SUBSCRIPTION_TIER_PRICING)} rows")

# ── CSV loaders ────────────────────────────────────────────────────────────────

def load_users(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "users", _csv_path("users.csv"),
                       ["id", "email", "full_name", "country_code", "city",
                        "created_at", "last_login_at", "is_active", "preferences"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        ok(f"users: {cur.fetchone()[0]:,} rows")


def load_seller_profiles(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "seller_profiles", _csv_path("seller_profiles.csv"),
                       ["user_id", "display_name", "legal_name", "tax_id",
                        "payout_email", "country_code", "is_verified", "bio",
                        "created_at", "updated_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM seller_profiles")
        ok(f"seller_profiles: {cur.fetchone()[0]:,} rows")


def load_subscriptions(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "subscriptions", _csv_path("subscriptions.csv"),
                       ["id", "user_id", "tier_id", "status", "started_at",
                        "current_period_start", "current_period_end",
                        "cancelled_at", "cancel_reason", "billing_cycle",
                        "created_at", "updated_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM subscriptions")
        ok(f"subscriptions: {cur.fetchone()[0]:,} rows")


def load_products(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "products", _csv_path("products.csv"),
                       ["id", "name", "slug", "product_type", "description",
                        "price_usd", "currency", "is_active", "seller_id",
                        "attributes", "created_at", "updated_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM products")
        ok(f"products: {cur.fetchone()[0]:,} rows")


def load_invoices(conn):
    """
    Load from both marketplace_invoices.csv and subscription_invoices.csv.
    TimescaleDB's COPY routing is transparent — rows go to the correct
    time chunk automatically based on created_at.
    """
    total = 0
    for filename in ("marketplace_invoices.csv", "subscription_invoices.csv"):
        with conn.cursor() as cur:
            _copy_from_csv(cur, "invoices", _csv_path(filename),
                           ["id", "user_id", "invoice_type", "status",
                            "subtotal_usd", "tax_usd", "discount_usd", "total_usd",
                            "subscription_id", "billing_period_start",
                            "billing_period_end", "paid_at", "due_at", "created_at"])
        conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM invoices")
        total = cur.fetchone()[0]
    ok(f"invoices: {total:,} rows (marketplace + subscription)")


def load_invoice_lines(conn):
    for filename in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
        with conn.cursor() as cur:
            _copy_from_csv(cur, "invoice_lines", _csv_path(filename),
                           ["id", "invoice_id", "product_id", "description",
                            "quantity", "unit_price_usd", "line_total_usd", "created_at"])
        conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM invoice_lines")
        ok(f"invoice_lines: {cur.fetchone()[0]:,} rows")


def load_orders(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "orders", _csv_path("orders.csv"),
                       ["id", "user_id", "invoice_id", "status",
                        "shipping_name", "shipping_address", "shipping_city",
                        "shipping_country", "shipping_postal",
                        "created_at", "updated_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM orders")
        ok(f"orders: {cur.fetchone()[0]:,} rows")


def load_order_items(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "order_items", _csv_path("order_items.csv"),
                       ["id", "order_id", "product_id", "quantity",
                        "unit_price_usd", "line_total_usd",
                        "fulfilment_status", "created_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM order_items")
        ok(f"order_items: {cur.fetchone()[0]:,} rows")


def load_sessions(conn):
    with conn.cursor() as cur:
        _copy_from_csv(cur, "sessions", _csv_path("sessions.csv"),
                       ["id", "user_id", "cart", "ip_address", "user_agent",
                        "created_at", "last_active_at", "expires_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sessions")
        ok(f"sessions: {cur.fetchone()[0]:,} rows")


def load_events(conn):
    """
    Largest table — loaded last.
    TimescaleDB COPY routes each row to the correct 7-day chunk automatically.
    """
    with conn.cursor() as cur:
        _copy_from_csv(cur, "events", _csv_path("events.csv"),
                       ["id", "user_id", "event_type", "product_id",
                        "session_id", "metadata", "occurred_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        ok(f"events: {cur.fetchone()[0]:,} rows")


def verify(conn):
    tables = [
        "users", "seller_profiles", "subscription_tiers",
        "subscription_tier_pricing", "subscriptions", "products",
        "invoices", "invoice_lines", "orders", "order_items",
        "sessions", "events",
    ]
    print("\n  Row counts:")
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            status = ok if count > 0 else warn
            status(f"  {table:<35} {count:>12,}")

    # Confirm hypertables exist
    print("\n  Hypertables:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hypertable_name, num_chunks
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name;
        """)
        rows = cur.fetchall()
    if rows:
        for name, chunks in rows:
            ok(f"  {name:<35} {chunks:>4} chunks")
    else:
        warn("No hypertables found — create_hypertables() may not have run.")

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TimescaleDB naive schema loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full load: drops all tables, recreates schema, "
            "creates hypertables, loads all CSVs.\n"
            "--dry-run: creates schema + hypertables, skips CSV loading.\n"
            "--verify: prints row counts and hypertable info only.\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Create schema only — skip CSV loading.")
    parser.add_argument("--verify", action="store_true",
                        help="Print row counts and hypertable info then exit.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB — Naive Schema Loader")
    print("=" * 60)
    print(f"  Host     : localhost:5433")
    print(f"  Database : {os.getenv('TIMESCALE_DB')}")
    print(f"  Data dir : {DATA_DIR}")
    if args.dry_run:
        print("  Mode     : DRY RUN (schema only)")
    elif args.verify:
        print("  Mode     : VERIFY")
    print()

    conn = get_connection()
    conn.autocommit = False

    try:
        if args.verify:
            verify(conn)
            return

        # Full rebuild
        drop_and_recreate(conn)
        create_schema(conn)
        create_hypertables(conn)

        if args.dry_run:
            print("\n  Dry run complete — schema and hypertables verified.")
            verify(conn)
            return

        print("\n  Loading data...")
        print("  " + "─" * 56)

        load_subscription_tiers(conn)
        load_subscription_tier_pricing(conn)
        load_users(conn)
        load_seller_profiles(conn)
        load_subscriptions(conn)
        load_products(conn)
        load_invoices(conn)
        load_invoice_lines(conn)
        load_orders(conn)
        load_order_items(conn)
        load_sessions(conn)
        load_events(conn)

        print()
        print("  " + "─" * 56)
        verify(conn)
        print()
        ok("TimescaleDB naive schema fully loaded.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()