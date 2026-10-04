-- ============================================================
-- PostgreSQL — Unified Relational Schema
-- ============================================================
-- EXTENSIONS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- UUID generation

-- ============================================================
-- SECTION 1 — CORE ENTITIES
-- ============================================================

-- ------------------------------------------------------------
-- users
-- ------------------------------------------------------------
CREATE TABLE users (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    email               VARCHAR(255)    NOT NULL UNIQUE,
    full_name           VARCHAR(255)    NOT NULL,
    country_code        CHAR(2)         NOT NULL,                  -- ISO 3166-1 alpha-2
    city                VARCHAR(100),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_login_at       TIMESTAMPTZ,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    preferences         JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_users_email        ON users (email);
CREATE INDEX idx_users_country      ON users (country_code);
CREATE INDEX idx_users_created_at   ON users (created_at);

-- ------------------------------------------------------------
-- seller_profiles
-- Optional extension of users — only exists if the user sells.
-- Keeps legal/payout info separate from the core users table.
-- ------------------------------------------------------------
CREATE TABLE seller_profiles (
    user_id             UUID            PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    display_name        VARCHAR(255)    NOT NULL,
    legal_name          VARCHAR(255),
    tax_id              VARCHAR(100),
    payout_email        VARCHAR(255),
    country_code        CHAR(2),
    is_verified         BOOLEAN         NOT NULL DEFAULT FALSE,
    bio                 TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_seller_profiles_verified ON seller_profiles (is_verified);
CREATE INDEX idx_seller_profiles_country  ON seller_profiles (country_code);

-- ------------------------------------------------------------
-- subscription_tiers
-- Lookup table for the three service tiers.
-- Pricing is NOT stored here — see subscription_tier_pricing.
-- ------------------------------------------------------------
CREATE TABLE subscription_tiers (
    id                  SMALLINT        PRIMARY KEY,
    name                VARCHAR(50)     NOT NULL UNIQUE,
    description         TEXT,
    features            JSONB           NOT NULL DEFAULT '{}'
);

INSERT INTO subscription_tiers (id, name, description, features) VALUES
(1, 'Free',     'Basic software access, up to 5 marketplace purchases/year',
    '{"seats": 1, "api_access": false, "priority_support": false,
      "marketplace_purchases_per_year": 5,
      "apps": {"CanvasEditor": "free", "VideoSuite": null}}'),
(2, 'Pro',      'Full software, unlimited purchases, early access',
    '{"seats": 1, "api_access": false, "priority_support": false,
      "marketplace_purchases_per_year": -1,
      "apps": {"CanvasEditor": "premium", "VideoSuite": "standard"}}'),
(3, 'Business', 'Everything in Pro plus team seats, API access, priority support',
    '{"seats": 10, "api_access": true, "priority_support": true,
      "marketplace_purchases_per_year": -1,
      "apps": {"CanvasEditor": "premium", "VideoSuite": "premium"}}');


-- ------------------------------------------------------------
-- subscription_tier_pricing
-- Full price history for each tier.
-- Current price = WHERE is_active = TRUE (one per tier only).
-- Composite PK (tier_id, valid_from) — a tier cannot have two
-- prices starting at the exact same moment.
-- Relevant to: Q1, Q7
-- ------------------------------------------------------------
CREATE TABLE subscription_tier_pricing (
    tier_id             SMALLINT        NOT NULL REFERENCES subscription_tiers (id),
    valid_from          TIMESTAMPTZ     NOT NULL,
    valid_to            TIMESTAMPTZ,                              -- NULL = currently active
    monthly_price_usd   NUMERIC(8,2)    NOT NULL,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    PRIMARY KEY (tier_id, valid_from)
);

-- Enforce only one active price per tier at any time
CREATE UNIQUE INDEX idx_tier_pricing_one_active_per_tier
    ON subscription_tier_pricing (tier_id)
    WHERE is_active = TRUE;

CREATE INDEX idx_tier_pricing_tier_id    ON subscription_tier_pricing (tier_id);
CREATE INDEX idx_tier_pricing_valid_from ON subscription_tier_pricing (valid_from);

-- Price history: Pro and Business had an increase in June 2024.
-- invoices before June 2024 report the old price, after report the new.
INSERT INTO subscription_tier_pricing (tier_id, valid_from, valid_to, monthly_price_usd, is_active) VALUES
(1, '2023-01-01 00:00:00+00', NULL,                      0.00,  TRUE),
(2, '2023-01-01 00:00:00+00', '2024-06-01 00:00:00+00', 14.99, FALSE),
(2, '2024-06-01 00:00:00+00', NULL,                      19.99, TRUE),
(3, '2023-01-01 00:00:00+00', '2024-06-01 00:00:00+00', 39.99, FALSE),
(3, '2024-06-01 00:00:00+00', NULL,                      49.99, TRUE);


-- ------------------------------------------------------------
-- subscriptions
-- One active subscription per user at a time (enforced via
-- partial unique index on user_id WHERE status = 'active').
-- ------------------------------------------------------------
CREATE TABLE subscriptions (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    tier_id             SMALLINT        NOT NULL REFERENCES subscription_tiers (id),
    status              VARCHAR(20)     NOT NULL DEFAULT 'active',
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    current_period_start TIMESTAMPTZ   NOT NULL,
    current_period_end   TIMESTAMPTZ   NOT NULL,
    cancelled_at        TIMESTAMPTZ,
    cancel_reason       TEXT,
    billing_cycle       VARCHAR(10)     NOT NULL DEFAULT 'monthly',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_subscriptions_one_active_per_user
    ON subscriptions (user_id)
    WHERE status = 'active';

CREATE INDEX idx_subscriptions_user_id    ON subscriptions (user_id);
CREATE INDEX idx_subscriptions_tier_id    ON subscriptions (tier_id);
CREATE INDEX idx_subscriptions_status     ON subscriptions (status);
CREATE INDEX idx_subscriptions_period_end ON subscriptions (current_period_end);

-- ------------------------------------------------------------
-- products
-- All marketplace items — Courses, Digital Assets, and Merch.
-- The product_type column determines which attributes are
-- stored in the attributes JSONB column.
--
-- search_vector is a pre-computed tsvector used by PostgreSQL's
-- full-text search (Q5 baseline). Elasticsearch will index the
-- name + description + attributes fields natively and outperform
-- this at scale with BM25 ranking and custom analysers.
-- ------------------------------------------------------------
CREATE TABLE products (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                VARCHAR(255)    NOT NULL,
    slug                VARCHAR(255)    NOT NULL UNIQUE,
    product_type        VARCHAR(20)     NOT NULL,
    description         TEXT            NOT NULL,
    price_usd           NUMERIC(8,2)    NOT NULL,
    currency            CHAR(3)         NOT NULL DEFAULT 'USD',
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    seller_id           UUID            NOT NULL REFERENCES users (id),
    -- Attributes shape varies by product_type:
    --
    -- course:        {"duration_hours": 12, "level": "beginner",
    --                 "instructor": "Jane Smith", "language": "en",
    --                 "certificate": true, "topics": ["design", "typography"]}
    --
    -- digital_asset: {"file_format": ["ABR", "PNG"], "software_compatibility":
    --                 ["Photoshop", "Procreate"], "asset_count": 50,
    --                 "resolution": "4K", "asset_type": "brush_pack"}
    --
    -- merch:         {"sizes_available": ["S","M","L","XL"],
    --                 "colours": ["black","white"], "material": "100% cotton",
    --                 "weight_kg": 0.3, "requires_shipping": true}
    attributes          JSONB           NOT NULL DEFAULT '{}',

    -- Pre-computed full-text search vector (PostgreSQL Q5 baseline).
    -- Weighted: name (A) > description (B) > product_type (C).
    -- Elasticsearch will replicate this across name, description,
    -- and attributes with BM25 scoring — this column is the baseline.
    search_vector       TSVECTOR,

    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_products_type         ON products (product_type);
CREATE INDEX idx_products_seller_id    ON products (seller_id);
CREATE INDEX idx_products_price        ON products (price_usd);
CREATE INDEX idx_products_active       ON products (is_active);
CREATE INDEX idx_products_attributes   ON products USING GIN (attributes);
-- GIN index on tsvector enables fast full-text search in PostgreSQL (Q5 baseline)
CREATE INDEX idx_products_search       ON products USING GIN (search_vector);

-- Trigger to keep search_vector up to date whenever a product is inserted/updated
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

CREATE TRIGGER trg_products_search_vector
    BEFORE INSERT OR UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_product_search_vector();


-- ============================================================
-- SECTION 2 — TRANSACTIONAL ENTITIES
-- ============================================================

-- ------------------------------------------------------------
-- invoices
-- Generated on subscription renewals AND marketplace orders.
-- One invoice covers one renewal OR one order — never mixed.
-- ------------------------------------------------------------
CREATE TABLE invoices (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id),
    invoice_type        VARCHAR(20)     NOT NULL,
    -- invoice_type values: 'subscription', 'marketplace'
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
    -- status values: 'pending', 'paid', 'failed', 'refunded', 'void'
    subtotal_usd        NUMERIC(10,2)   NOT NULL,
    tax_usd             NUMERIC(10,2)   NOT NULL DEFAULT 0.00,
    discount_usd        NUMERIC(10,2)   NOT NULL DEFAULT 0.00,
    total_usd           NUMERIC(10,2)   NOT NULL,
    -- total = subtotal + tax - discount
    subscription_id     UUID            REFERENCES subscriptions (id),
    billing_period_start TIMESTAMPTZ,                             -- subscription invoices only
    billing_period_end   TIMESTAMPTZ,                             -- subscription invoices only
    paid_at             TIMESTAMPTZ,
    due_at              TIMESTAMPTZ     NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_invoices_user_id         ON invoices (user_id);
CREATE INDEX idx_invoices_subscription_id ON invoices (subscription_id);
CREATE INDEX idx_invoices_status          ON invoices (status);
CREATE INDEX idx_invoices_created_at      ON invoices (created_at);
CREATE INDEX idx_invoices_type            ON invoices (invoice_type);

-- ------------------------------------------------------------
-- invoice_lines
-- One row per item on the invoice.
-- unit_price_usd is PRE-TAX — tax is applied at invoice level.
-- ------------------------------------------------------------
CREATE TABLE invoice_lines (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    invoice_id          UUID            NOT NULL REFERENCES invoices (id) ON DELETE CASCADE,
    product_id          UUID            REFERENCES products (id),
    description         VARCHAR(500)    NOT NULL,
    quantity            SMALLINT        NOT NULL DEFAULT 1,
    unit_price_usd      NUMERIC(8,2)    NOT NULL,
    line_total_usd      NUMERIC(10,2)   NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_invoice_lines_invoice_id ON invoice_lines (invoice_id);
CREATE INDEX idx_invoice_lines_product_id ON invoice_lines (product_id);

-- ------------------------------------------------------------
-- orders
-- Marketplace purchases only (not subscriptions).
-- Each order maps to exactly one marketplace invoice.
-- ------------------------------------------------------------
CREATE TABLE orders (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id),
    invoice_id          UUID            NOT NULL REFERENCES invoices (id),
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
    -- status values: 'pending', 'confirmed', 'shipped', 'delivered',
    --                'cancelled', 'refunded'
    -- Shipping address — only populated for merch orders
    shipping_name       VARCHAR(255),
    shipping_address    VARCHAR(500),
    shipping_city       VARCHAR(100),
    shipping_country    CHAR(2),
    shipping_postal     VARCHAR(20),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_user_id    ON orders (user_id);
CREATE INDEX idx_orders_invoice_id ON orders (invoice_id);
CREATE INDEX idx_orders_status     ON orders (status);
CREATE INDEX idx_orders_created_at ON orders (created_at);


-- ------------------------------------------------------------
-- order_items
-- One row per product in a marketplace order.
-- Tracks fulfilment state per item (separate from invoice_lines
-- which is the immutable financial record).
-- This table drives the co-purchase graph in Neo4j (Q4).
-- ------------------------------------------------------------
CREATE TABLE order_items (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id            UUID            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    product_id          UUID            NOT NULL REFERENCES products (id),
    quantity            SMALLINT        NOT NULL DEFAULT 1,
    unit_price_usd      NUMERIC(8,2)    NOT NULL,
    line_total_usd      NUMERIC(10,2)   NOT NULL,
    fulfilment_status   VARCHAR(20)     NOT NULL DEFAULT 'pending',
    -- values: 'pending', 'delivered', 'shipped', 'failed'
    -- digital_asset → delivered immediately; merch → shipped/delivered
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_order_items_order_id   ON order_items (order_id);
CREATE INDEX idx_order_items_product_id ON order_items (product_id);


-- ============================================================
-- SECTION 3 — BEHAVIOURAL ENTITIES
-- ============================================================

-- ------------------------------------------------------------
-- sessions
-- Active user sessions including cart state.
-- This is the Redis use case (Q3): instant cart retrieval
-- under high concurrency.
-- ------------------------------------------------------------
CREATE TABLE sessions (
    id                  VARCHAR(64)     PRIMARY KEY,
    user_id             UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    cart                JSONB           NOT NULL DEFAULT '[]',
    ip_address          INET,
    user_agent          VARCHAR(500),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_active_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ     NOT NULL
);

CREATE INDEX idx_sessions_user_id     ON sessions (user_id);
CREATE INDEX idx_sessions_expires_at  ON sessions (expires_at);
CREATE INDEX idx_sessions_last_active ON sessions (last_active_at);

-- ------------------------------------------------------------
-- events
-- Append-only clickstream log — every user action on the platform.
-- This is the Cassandra use case (Q6): scanning millions of
-- time-ordered events per user in a date range.
-- Events are immutable once written — no updated_at column.
-- ------------------------------------------------------------
CREATE TABLE events (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID            NOT NULL REFERENCES users (id),
    event_type          VARCHAR(50)     NOT NULL,
    -- event_type values:
    --   'page_view', 'product_view', 'add_to_cart', 'remove_from_cart',
    --   'checkout_start', 'purchase', 'subscription_upgrade',
    --   'subscription_cancel', 'search', 'download', 'login', 'logout'
    product_id          UUID            REFERENCES products (id),
    session_id          VARCHAR(64)     REFERENCES sessions (id),
    metadata            JSONB           NOT NULL DEFAULT '{}',
    occurred_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Composite index optimised for the Cassandra query pattern:
-- WHERE user_id = ? AND occurred_at BETWEEN ? AND ?
CREATE INDEX idx_events_user_time   ON events (user_id, occurred_at DESC);
CREATE INDEX idx_events_type        ON events (event_type);
CREATE INDEX idx_events_product_id  ON events (product_id);
CREATE INDEX idx_events_occurred_at ON events (occurred_at DESC);

CREATE INDEX idx_invoices_type_status_created ON invoices (invoice_type, status, created_at);