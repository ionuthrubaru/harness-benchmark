"""
loaders/timescaledb_optimised_loader.py — TimescaleDB Optimised Schema Loader
==============================================================================
Phase 3 — TimescaleDB, optimised schema.

Optimised schema additions over naive
───────────────────────────────────────
The optimised schema starts from the same base tables as the naive schema
and adds four time-series-specific features:

1. Tuned chunk intervals (instead of the default 7 days)
2. Continuous aggregate: daily_revenue_by_tier
3. Compression policies on events and invoices
4. Refresh of the continuous aggregate after data load

These are the features that make TimescaleDB more than "PostgreSQL with
hypertables" — they demonstrate the full value of the specialised engine.

Feature 1 — Tuned chunk intervals
───────────────────────────────────
  events:   1 month  (was: 7 days)
  invoices: 1 month  (was: 7 days)

TimescaleDB's default 7-day chunk interval is designed for high-frequency
IoT workloads where data arrives every second. For the events dataset
(~1M rows over 2 years) and invoices (~several hundred thousand rows over
2 years), 7-day chunks produce ~104 chunks per table — more than needed
and more chunk-metadata overhead than warranted.

1-month chunks produce ~24 chunks per table, which:
  • Reduces the per-query chunk pruning overhead (fewer chunk metadata
    entries to scan when deciding which chunks to read)
  • Aligns with natural query patterns: Q6 queries 30-day windows
    (1-2 chunks), Q7 and Q1 query multi-month ranges (N chunks)
  • Aligns with invoices' monthly billing cycle — a month of invoices
    fits in one chunk, keeping related data physically co-located

Schema effect for Q6: naive (7-day chunks, ~104 chunks) vs optimised
(1-month chunks, ~24 chunks). Both benefit from chunk pruning over
PostgreSQL's sequential scan — the difference is in pruning overhead.

Feature 2 — Continuous aggregate: daily_revenue_by_tier
─────────────────────────────────────────────────────────
  CREATE MATERIALIZED VIEW daily_revenue_by_tier
  WITH (timescaledb.continuous) AS
  SELECT
      time_bucket('1 day', i.created_at) AS day,
      s.tier_id,
      st.name                             AS tier_name,
      COUNT(*)                            AS invoice_count,
      SUM(i.total_usd)                    AS total_usd
  FROM invoices i
  JOIN subscriptions s   ON s.id  = i.subscription_id
  JOIN subscription_tiers st ON st.id = s.tier_id
  WHERE i.status = 'paid' AND i.invoice_type = 'subscription'
  GROUP BY 1, 2, 3;

Scope: subscription invoices only. Marketplace invoices require a LATERAL
temporal join (find the most recent subscription for each user at purchase
time) which cannot be expressed in a continuous aggregate (no correlated
subqueries or LATERAL in aggregate SELECT lists). Marketplace revenue is
handled in Q1 via a lightweight raw query — subscription invoices are the
majority of revenue and the primary time-series signal.

This aggregate powers:
  Q7 optimised: time_bucket_gapfill('1 day', day) on this aggregate,
                using localfill/interpolation for gap-filling and a
                7-day rolling average window function. This is the
                TimescaleDB killer feature — gap-filling and rolling
                aggregates over a pre-materialised daily summary.

  Q1 optimised: time_bucket('1 month', day) on this aggregate for the
                subscription revenue component, plus a small raw scan
                for marketplace attribution.

Continuous aggregates are incrementally refreshed — only new data is
processed, not the entire historical dataset. After the initial full
refresh (at load time), ongoing refreshes are O(new data), not O(all data).
The query reads from the materialised view, not raw invoices — this is
the pre-computation that gives TimescaleDB its time-series advantage.

Feature 3 — Compression
─────────────────────────
  events:   compress chunks older than 7 days
             segmentby=user_id, orderby=occurred_at
  invoices: compress chunks older than 30 days
             segmentby=invoice_type, orderby=created_at

TimescaleDB compression converts columnar data to a compressed columnar
format per chunk. Compression is applied via a scheduled policy (which
runs the background compression job) and also triggered manually here
via compress_chunk() on all existing chunks after loading.

Benefits measured in Q6 and Q7:
  • Compressed chunks read less I/O per time-range scan (columnar
    format skips irrelevant columns entirely)
  • segmentby=user_id on events means all events for one user in a
    chunk are stored contiguously — Q6's (user_id, occurred_at) scan
    benefits directly from this physical co-location

Note: compressed chunks cannot be directly modified (INSERTs go to a
new uncompressed chunk; UPDATEs/DELETEs require decompression). This
is acceptable for the dissertation dataset which is static after loading.

Feature 4 — Refresh policy
────────────────────────────
After data loading, the continuous aggregate is manually refreshed over
the full data date range. In production, a scheduled refresh policy would
keep it incrementally up to date. For this benchmark, a one-time full
refresh at load time is sufficient since no new data will be added.

Schema effect summary
──────────────────────
  Q6: chunk pruning on 1-month chunks + compressed columnar storage → fast range scans
  Q7: time_bucket_gapfill() on pre-materialised daily_revenue_by_tier → instant rolling avg
  Q1: time_bucket() on continuous aggregate for subscription revenue
  Q2–Q5: unchanged from naive (TimescaleDB adds nothing for non-time-series queries)
  Q8: events hypertable write path (same as naive — Q8 has no optimised variant)

Usage:
    cd loaders
    python timescaledb_optimised_loader.py            # full load
    python timescaledb_optimised_loader.py --dry-run  # schema + hypertables + aggregate, no data
    python timescaledb_optimised_loader.py --verify   # check row counts, chunks, aggregate
"""

import argparse
import csv
import json
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94d"
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

# ── base schema — identical to naive ──────────────────────────────────────────
# Copied verbatim from the naive loader. The optimised schema starts from the
# same relational foundation — all optimisations are additive on top of it.

BASE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

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

CREATE TABLE IF NOT EXISTS subscription_tiers (
    id          SMALLINT        PRIMARY KEY,
    name        VARCHAR(50)     NOT NULL UNIQUE,
    description TEXT,
    features    JSONB           NOT NULL DEFAULT '{}'
);

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

CREATE TABLE IF NOT EXISTS invoice_lines (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
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

CREATE TABLE IF NOT EXISTS orders (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id),
    invoice_id          UUID            NOT NULL,
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

# ── hypertables with tuned chunk intervals ────────────────────────────────────

HYPERTABLE_SQL = """
-- events hypertable: 1-month chunks (vs naive's 7-day default).
-- Rationale: Q6 queries 30-day windows → 1-2 month-chunks per query.
-- Fewer chunks = less chunk-metadata overhead during pruning.
SELECT create_hypertable(
    'events',
    'occurred_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);

-- invoices hypertable: 1-month chunks (vs naive's 7-day default).
-- Rationale: aligns with monthly billing cycles. Q1 and Q7 query
-- multi-month ranges — month-aligned chunks minimise partial-chunk reads.
SELECT create_hypertable(
    'invoices',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);
"""

# ── continuous aggregate ───────────────────────────────────────────────────────

CONTINUOUS_AGGREGATE_SQL = """
-- daily_revenue_by_tier: pre-materialised daily subscription revenue per tier.
--
-- Scope: subscription invoices only (status='paid', invoice_type='subscription').
-- Marketplace invoices are excluded because their tier attribution requires
-- a LATERAL temporal join (most recent subscription for user at purchase time)
-- which is not supported inside continuous aggregate SELECT lists.
-- Marketplace revenue is handled separately in Q1 via a small raw query.
--
-- Powers:
--   Q7 optimised: time_bucket_gapfill('1 day', day) on this view — the
--                 TimescaleDB killer feature. Gap-fills days with zero
--                 revenue and computes the 7-day rolling average using
--                 avg() OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
--                 entirely within the pre-materialised aggregate.
--   Q1 optimised: time_bucket('1 month', day) on this view to aggregate
--                 monthly subscription revenue per tier, combined with a
--                 small raw LATERAL query for marketplace attribution.
--
-- WITH NO DATA: the aggregate is populated by the explicit CALL
-- refresh_continuous_aggregate() below, which runs after all data is loaded.
-- This is more efficient than letting TimescaleDB background-refresh during
-- the load — we want the refresh to cover the exact data range.
DROP MATERIALIZED VIEW IF EXISTS daily_revenue_by_tier CASCADE;
CREATE MATERIALIZED VIEW daily_revenue_by_tier
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', i.created_at)  AS day,
    s.tier_id,
    st.name                              AS tier_name,
    COUNT(*)                             AS invoice_count,
    SUM(i.total_usd)                     AS total_usd
FROM invoices i
JOIN subscriptions s
    ON s.id  = i.subscription_id
JOIN subscription_tiers st
    ON st.id = s.tier_id
WHERE i.status       = 'paid'
  AND i.invoice_type = 'subscription'
GROUP BY 1, 2, 3
WITH NO DATA;
"""

# ── compression policies ──────────────────────────────────────────────────────

COMPRESSION_SQL = """
-- events compression: compress chunks older than 7 days.
--
-- segmentby=user_id: all events for one user within a chunk are stored
--   contiguously in the compressed format. Q6 scans (user_id, occurred_at)
--   — segmenting by user_id makes this a single contiguous read within
--   the relevant chunks.
-- orderby=occurred_at DESC: rows within each user segment are ordered by
--   time descending, matching Q6's ORDER BY occurred_at DESC — no sort needed
--   after decompression.
ALTER TABLE events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id',
    timescaledb.compress_orderby   = 'occurred_at DESC'
);
SELECT add_compression_policy('events',   INTERVAL '7 days',  if_not_exists => TRUE);

-- invoices compression: compress chunks older than 30 days.
--
-- segmentby=invoice_type: groups subscription and marketplace invoices
--   separately within each chunk. Q1 and Q7 filter by invoice_type='subscription'
--   — segmenting by it allows the decompressor to skip marketplace segments.
-- orderby=created_at DESC: matches the typical query access pattern
--   (recent invoices first).
ALTER TABLE invoices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'invoice_type',
    timescaledb.compress_orderby   = 'created_at DESC'
);
SELECT add_compression_policy('invoices', INTERVAL '30 days', if_not_exists => TRUE);
"""

# ── seed data — identical to naive ────────────────────────────────────────────

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

# ── helpers ────────────────────────────────────────────────────────────────────

def _csv_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def _copy_from_csv(cur, table: str, csv_path: str, columns: list[str]):
    col_list = ", ".join(columns)
    sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    with open(csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)

# ── schema steps ───────────────────────────────────────────────────────────────

def drop_and_recreate(conn):
    info("Dropping existing tables (if any)...")
    with conn.cursor() as cur:
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS daily_revenue_by_tier CASCADE;")
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


def create_base_schema(conn):
    info("Creating base tables and indexes...")
    with conn.cursor() as cur:
        cur.execute(BASE_SCHEMA_SQL)
    conn.commit()
    ok("Base schema created.")


def create_hypertables(conn):
    info("Creating hypertables (1-month chunks)...")
    with conn.cursor() as cur:
        cur.execute(HYPERTABLE_SQL)
    conn.commit()
    ok("Hypertables created (events: 1 month, invoices: 1 month).")


def create_continuous_aggregate(conn):
    """
    Create the continuous aggregate WITH NO DATA.
    Populated after data load via refresh_continuous_aggregate().
    """
    info("Creating continuous aggregate: daily_revenue_by_tier...")
    with conn.cursor() as cur:
        cur.execute(CONTINUOUS_AGGREGATE_SQL)
    conn.commit()
    ok("Continuous aggregate created (empty — will refresh after data load).")


def enable_compression(conn):
    info("Enabling compression policies...")
    with conn.cursor() as cur:
        cur.execute(COMPRESSION_SQL)
    conn.commit()
    ok("Compression policies enabled (events: 7d, invoices: 30d).")


def compress_all_chunks(conn):
    """
    Manually compress all existing chunks that meet the policy threshold.
    After bulk loading, the scheduled compression job has not run yet.
    Triggering it manually ensures Q6 and Q7 benchmarks run against
    compressed storage, which is what they would encounter in production.
    """
    info("Compressing all eligible chunks (this may take a minute)...")
    # compress_chunk() also cannot run inside a transaction block
    conn.commit()
    conn.autocommit = True
    events_compressed   = 0
    invoices_compressed = 0
    try:
        with conn.cursor() as cur:
            # Column names: chunk_schema + chunk_name (TimescaleDB 2.x)
            cur.execute("""
                SELECT chunk_schema, chunk_name
                FROM timescaledb_information.chunks
                WHERE hypertable_name = 'events'
                  AND range_end < NOW() - INTERVAL '7 days'
                  AND NOT is_compressed
                ORDER BY range_start;
            """)
            events_chunks = cur.fetchall()

        for schema, name in events_chunks:
            with conn.cursor() as cur:
                cur.execute("SELECT compress_chunk(%s);", (f"{schema}.{name}",))
            events_compressed += 1

        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_schema, chunk_name
                FROM timescaledb_information.chunks
                WHERE hypertable_name = 'invoices'
                  AND range_end < NOW() - INTERVAL '30 days'
                  AND NOT is_compressed
                ORDER BY range_start;
            """)
            invoices_chunks = cur.fetchall()

        for schema, name in invoices_chunks:
            with conn.cursor() as cur:
                cur.execute("SELECT compress_chunk(%s);", (f"{schema}.{name}",))
            invoices_compressed += 1

    finally:
        conn.autocommit = False
    ok(f"Compressed {events_compressed} events chunk(s), "
       f"{invoices_compressed} invoices chunk(s).")


def refresh_continuous_aggregate(conn):
    """
    Populate daily_revenue_by_tier over the full data date range.
    Uses CALL refresh_continuous_aggregate(view, start, end) with NULL bounds
    to refresh the entire range.

    Must run outside a transaction block — autocommit is temporarily enabled.
    CALL refresh_continuous_aggregate() is a procedural call that manages its
    own internal transactions and cannot be nested inside an explicit transaction.
    """
    info("Refreshing continuous aggregate (full date range)...")
    conn.commit()          # close any open transaction before switching autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CALL refresh_continuous_aggregate(
                    'daily_revenue_by_tier',
                    NULL,
                    NULL
                );
            """)
    finally:
        conn.autocommit = False

    # Verify row count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM daily_revenue_by_tier")
        count = cur.fetchone()[0]
    ok(f"Continuous aggregate refreshed: {count:,} rows in daily_revenue_by_tier.")

# ── data loaders — identical to naive ─────────────────────────────────────────

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
        ok(f"invoices: {cur.fetchone()[0]:,} rows (marketplace + subscription)")


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
    with conn.cursor() as cur:
        _copy_from_csv(cur, "events", _csv_path("events.csv"),
                       ["id", "user_id", "event_type", "product_id",
                        "session_id", "metadata", "occurred_at"])
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        ok(f"events: {cur.fetchone()[0]:,} rows")

# ── verify ─────────────────────────────────────────────────────────────────────

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
            (ok if count > 0 else warn)(f"  {table:<35} {count:>12,}")

    print("\n  Hypertables:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hypertable_name, num_chunks,
                   compression_enabled
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name;
        """)
        for name, chunks, compressed in cur.fetchall():
            ok(f"  {name:<20} {chunks:>4} chunks  "
               f"compression={'ON' if compressed else 'OFF'}")

    print("\n  Continuous aggregate:")
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(*) FROM daily_revenue_by_tier")
            count = cur.fetchone()[0]
            ok(f"  daily_revenue_by_tier: {count:,} rows")
        except Exception as e:
            warn(f"  daily_revenue_by_tier not found: {e}")

    print("\n  Compressed chunks:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hypertable_name,
                   SUM(CASE WHEN is_compressed THEN 1 ELSE 0 END) AS compressed,
                   COUNT(*) AS total
            FROM timescaledb_information.chunks
            WHERE hypertable_name IN ('events', 'invoices')
            GROUP BY hypertable_name
            ORDER BY hypertable_name;
        """)
        chunk_rows = cur.fetchall()
    for name, comp, total in chunk_rows:
        ok(f"  {name:<20} {int(comp)}/{total} chunks compressed")

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TimescaleDB optimised schema loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Create schema, hypertables, aggregate — skip data load.")
    parser.add_argument("--verify", action="store_true",
                        help="Print row counts, chunk info, aggregate stats then exit.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TimescaleDB — Optimised Schema Loader")
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

        drop_and_recreate(conn)
        create_base_schema(conn)
        create_hypertables(conn)
        create_continuous_aggregate(conn)
        enable_compression(conn)

        if args.dry_run:
            print("\n  Dry run complete — schema verified.")
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
        info("Post-load: refreshing continuous aggregate...")
        refresh_continuous_aggregate(conn)

        info("Post-load: compressing eligible chunks...")
        compress_all_chunks(conn)

        print()
        print("  " + "─" * 56)
        verify(conn)
        print()
        ok("TimescaleDB optimised schema fully loaded.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()