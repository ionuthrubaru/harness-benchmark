"""
loaders/cassandra_optimised_loader.py — Cassandra Optimised Schema Loader
=========================================================================
Phase 3 — Cassandra, optimised schema.

Optimised schema design philosophy
─────────────────────────────────────
Cassandra's fundamental design rule is: model your tables for your queries,
not for your entities. Every table in this schema exists to serve exactly one
query. There are no entity tables — no flat `users`, `orders`, or `events`
table that queries can filter however they like. Instead, all joining, tier
attribution, price resolution, and co-purchase aggregation is done in Python
at load time, and the results are written into pre-shaped query tables.

This is the core of the schema effect. The performance difference between the
naive schema (one entity table per entity, ALLOW FILTERING everywhere, Python-
side joins at query time) and the optimised schema (one query table per query,
partition keys matching access patterns, zero ALLOW FILTERING) is entirely
attributable to schema design, not engine tuning.

Query-to-table mapping
───────────────────────
  Q1  →  invoices_by_month_tier   (tier attribution + price resolution pre-computed)
  Q2  →  invoices_full            (invoice + customer + lines + products denormalised)
  Q3  →  sessions_by_user         (partitioned by user_id, clustered by recency)
  Q4  →  also_bought              (co-purchase counts pre-aggregated at load time)
  Q5  →  products_search          (name_lower with SASI CONTAINS index)
  Q6  →  events_by_user_month     (partitioned by (user_id, year_month), clustered by time)
  Q7  →  invoices_by_tier         (partitioned by tier_id, clustered by date)
  Q8  →  uses the naive keyspace events table (Q8 has no optimised variant — see dissertation_todo.md)

Load-time pre-computations
───────────────────────────
The following joins and aggregations are performed in Python against the source
CSV files (not against a live Cassandra instance) before any data is written:

  invoices_by_month_tier
      For every paid invoice in the dataset:
        • Subscription invoices: tier_id comes directly from the linked subscription.
        • Marketplace invoices: tier_id is resolved by finding the subscription with
          the most recent started_at <= invoice created_at for that user. This is the
          same temporal attribution logic as PostgreSQL Q1's LATERAL JOIN, performed
          in Python using a sorted lookup.
        • monthly_price_usd_at_time: resolved by finding the pricing row whose
          valid_from <= invoice created_at AND (valid_to IS NULL OR valid_to >
          invoice created_at). Same logic as PostgreSQL's temporal JOIN predicate.
        The result is one row per invoice with tier_id, tier_name, and the price
        in effect at invoice time pre-embedded. Q1 reads up to 36 partitions
        (12 months × 3 tier IDs) and aggregates the pre-resolved amounts in Python.

  invoices_full
      All four source tables (invoices, users, invoice_lines, products) are joined
      in Python. One Cassandra row is written per invoice_line, embedding invoice-
      level fields and customer snapshot fields on every row. Product details are
      embedded for marketplace lines; subscription renewal lines (product_id = NULL)
      embed empty product fields. Q2 reads one partition (all rows for one invoice_id)
      and reconstructs the full document in Python — analogous to MongoDB's single
      document read.

  also_bought
      Orders with status in ('confirmed', 'shipped', 'delivered') are loaded from
      CSV. For each qualifying order, every pair of products in the order is a co-
      purchase. Counts are aggregated in a Python dict keyed by (product_a, product_b).
      Product details (name, type, price_usd, is_active) are embedded at insert time
      by looking up products.csv in memory. Only active products are included in the
      recommendations. Q4 reads one partition (all co-purchase rows for product_a)
      and returns the first 10 rows, ordered by co_purchase_count DESC by the
      clustering column — no sorting needed at query time.

  products_search
      name_lower is the product name lowercased at load time. SASI index with
      mode='CONTAINS' and a case-insensitive analyser enables `name_lower LIKE
      '%keyword%'` without ALLOW FILTERING. Schema effect for Q5: ALLOW FILTERING
      on naive → SASI LIKE on optimised.

  events_by_user_month
      year_month is TEXT computed from occurred_at at load time ('YYYY-MM' format).
      No aggregation. One row per event, written into the correct (user_id, year_month)
      partition. Q6 queries 1–2 partitions for a 30-day window, reading a time-range
      from each using the occurred_at clustering column — pure sequential partition
      scan, no ALLOW FILTERING.

  sessions_by_user
      No pre-computation beyond routing. Each session row is written with user_id as
      the partition key and last_active_at as the clustering column. Q3 issues a
      single-partition LIMIT 1 read ordered by last_active_at DESC.

  invoices_by_tier
      paid_date (DATE) is extracted from invoice created_at at load time. Tier
      attribution uses the same logic as invoices_by_month_tier. Only paid invoices
      are included. Q7 reads 3 partitions (one per tier_id), filters by date range,
      aggregates by day in Python, computes the 7-day rolling average in Python,
      and gap-fills days with zero revenue in Python.

SASI index for Q5
──────────────────
Cassandra 4.1 supports two secondary index types: SASI (SSTable-Attached Secondary
Index) and SAI (Storage-Attached Index). SASI is used here because it provides
mode='CONTAINS' with a case-insensitive analyser — the only mechanism in Cassandra
4.1 that enables substring text search without ALLOW FILTERING.

SASI caveats documented for the methodology chapter:
  • SASI is deprecated in Cassandra 5.0 in favour of SAI. It is fully supported
    and recommended for Cassandra 4.1, which is the version used in this experiment
    (image: cassandra:4.1 in docker-compose).
  • SAI in Cassandra 4.1 does not support CONTAINS mode or analyser configuration,
    making it unsuitable as a replacement here.
  • SASI is the idiomatic Cassandra 4.1 answer for substring text search. Using it
    is not an unusual or experimental choice for this version.
  • The schema effect for Q5 is the elimination of ALLOW FILTERING: naive Q5 scans
    every partition (full table scan) while optimised Q5 uses the SASI index. The
    quality of the text matching is intentionally not compared — that is Elasticsearch's
    domain (Q5's killer DB). Cassandra's Q5 implementation remains intentionally weaker
    than Elasticsearch in matching quality while being structurally correct for the schema
    effect measurement.

Keyspace
─────────
cassandra_optimised — SimpleStrategy, replication_factor=1.
Identical server configuration to cassandra_naive. Any performance difference
between the two keyspaces is attributable to schema design, not server configuration.

Usage
──────
    cd loaders
    python cassandra_optimised_loader.py            # full load
    python cassandra_optimised_loader.py --dry-run  # schema only, no data
"""

import argparse
import csv
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone, date
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

KEYSPACE = os.getenv("CASSANDRA_KEYSPACE_OPTIMISED", "cassandra_optimised")

INSERT_CONCURRENCY = 200
CHUNK_SIZE = 10_000

# Orders with these statuses are treated as completed purchases for Q4 co-purchase
# computation. Mirrors the PostgreSQL Q4 baseline filter.
COMPLETED_ORDER_STATUSES = {"confirmed", "shipped", "delivered"}

# ── type helpers ──────────────────────────────────────────────────────────────

def _uuid(v):
    return uuid.UUID(v) if v and v.strip() else None

def _dt(v):
    if not v or not v.strip():
        return None
    return datetime.fromisoformat(v.replace("Z", "+00:00"))

def _dec(v):
    return Decimal(v) if v and v.strip() else None

def _bool(v):
    if not v or not v.strip():
        return None
    return v.strip().lower() == "true"

def _int(v):
    return int(v.strip()) if v and v.strip() else None

def _text(v):
    return v if v and v.strip() else None

def _year_month(dt: datetime) -> str:
    """Extract 'YYYY-MM' string from a datetime. Used as part of partition keys."""
    return dt.strftime("%Y-%m")

def _paid_date(dt: datetime) -> date:
    """Extract the calendar date from a datetime for Q7's date clustering column."""
    return dt.date()

# ── DDL ───────────────────────────────────────────────────────────────────────

CREATE_KEYSPACE = f"""
CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
    WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
    AND durable_writes = true
"""

# ── Q6: events_by_user_month ─────────────────────────────────────────────────
# Partition key: (user_id, year_month)
#   • user_id groups all events for one user together.
#   • year_month (e.g. '2025-03') bounds partition size to one calendar month.
#     Without it, a high-volume user accumulates unbounded rows in one partition —
#     the "hot partition" anti-pattern. With it, each partition covers at most one
#     month's worth of events for one user.
#   • A 30-day window may straddle two calendar months (e.g. March 15 → April 14).
#     Q6 handles this by querying two partitions: (user_id, 'YYYY-M1') and
#     (user_id, 'YYYY-M2'). Two single-partition scans are faster than one full
#     table scan with ALLOW FILTERING.
# Clustering: occurred_at DESC, id ASC
#   • occurred_at DESC means rows come back newest-first from Cassandra without any
#     client-side sorting — the query result is already ordered.
#   • id (UUID) breaks ties when two events share an identical timestamp, which
#     can occur in synthetic data. Without a tie-breaker the clustering column
#     is not strictly unique and Cassandra may merge or mis-order rows.
CREATE_EVENTS_BY_USER_MONTH = """
CREATE TABLE IF NOT EXISTS events_by_user_month (
    user_id         uuid,
    year_month      text,
    occurred_at     timestamp,
    id              uuid,
    event_type      text,
    product_id      uuid,
    session_id      text,
    metadata        text,
    PRIMARY KEY ((user_id, year_month), occurred_at, id)
) WITH CLUSTERING ORDER BY (occurred_at DESC, id ASC)
"""

# ── Q3: sessions_by_user ─────────────────────────────────────────────────────
# Partition key: user_id
#   • All sessions for a user in one partition. Session volume per user is low
#     (typically O(10s) of sessions in the dataset) so unbounded growth is not
#     a concern here — the year_month bucketing used for events is unnecessary.
# Clustering: last_active_at DESC, id ASC
#   • Q3 asks for the most recently active session for a user.
#     LIMIT 1 ordered by last_active_at DESC returns it as the first row in
#     the partition scan — no filtering, no sorting at query time.
#   • id breaks ties as above.
CREATE_SESSIONS_BY_USER = """
CREATE TABLE IF NOT EXISTS sessions_by_user (
    user_id         uuid,
    last_active_at  timestamp,
    id              text,
    cart            text,
    ip_address      text,
    user_agent      text,
    created_at      timestamp,
    expires_at      timestamp,
    PRIMARY KEY ((user_id), last_active_at, id)
) WITH CLUSTERING ORDER BY (last_active_at DESC, id ASC)
"""

# ── Q4: also_bought ───────────────────────────────────────────────────────────
# Partition key: product_id
#   • All co-purchase recommendations for product A in one partition.
#     Q4 reads this partition and returns the first 10 rows — one partition
#     read, no computation at query time.
# Clustering: co_purchase_count DESC, co_product_id ASC
#   • co_purchase_count DESC means the most-recommended products appear first.
#     The benchmark's LIMIT 10 returns the top 10 without any Python-side sort.
#   • co_product_id ASC breaks ties among products with equal counts,
#     matching the PostgreSQL Q4 ORDER BY co_purchase_count DESC, p.name.
# Embedded columns: co_product_name, co_product_type, co_product_price_usd,
#     co_product_is_active — denormalised at load time so Q4 returns a fully
#     usable result from one partition read without a second lookup to products.
# Note: only active products (is_active = True) are included, consistent with
#     the PostgreSQL Q4 WHERE p.is_active = TRUE filter.
CREATE_ALSO_BOUGHT = """
CREATE TABLE IF NOT EXISTS also_bought (
    product_id              uuid,
    co_purchase_count       int,
    co_product_id           uuid,
    co_product_name         text,
    co_product_type         text,
    co_product_price_usd    decimal,
    co_product_is_active    boolean,
    confidence              decimal,
    PRIMARY KEY ((product_id), co_purchase_count, co_product_id)
) WITH CLUSTERING ORDER BY (co_purchase_count DESC, co_product_id ASC)
"""

# ── Q2: invoices_full ─────────────────────────────────────────────────────────
# Partition key: invoice_id
#   • One partition per invoice. All line items for invoice X are in partition X.
#     Q2 reads one partition and reconstructs the full invoice document in Python.
#     Analogous to MongoDB's single embedded document read: both return a complete
#     invoice in one storage operation.
# Clustering: line_id ASC
#   • Orders lines consistently within the partition. Any stable order works;
#     ASC on UUID gives deterministic ordering.
# Denormalised columns:
#   • Invoice-level fields repeated on every row (invoice_type, status, totals,
#     billing dates) — standard Cassandra write amplification accepted for read speed.
#   • Customer snapshot: customer_full_name, customer_email, customer_country_code
#     embedded at load time from users.csv. Avoids a separate partition read.
#   • Product details: product_name, product_type, product_price_usd embedded at
#     load time from products.csv. NULL for subscription renewal lines (no product).
CREATE_INVOICES_FULL = """
CREATE TABLE IF NOT EXISTS invoices_full (
    invoice_id              uuid,
    line_id                 uuid,
    invoice_type            text,
    invoice_status          text,
    subtotal_usd            decimal,
    tax_usd                 decimal,
    discount_usd            decimal,
    total_usd               decimal,
    subscription_id         uuid,
    billing_period_start    timestamp,
    billing_period_end      timestamp,
    paid_at                 timestamp,
    due_at                  timestamp,
    invoice_created_at      timestamp,
    customer_id             uuid,
    customer_full_name      text,
    customer_email          text,
    customer_country_code   text,
    line_description        text,
    line_quantity           int,
    line_unit_price_usd     decimal,
    line_total_usd          decimal,
    product_id              uuid,
    product_name            text,
    product_type            text,
    product_price_usd       decimal,
    PRIMARY KEY ((invoice_id), line_id)
) WITH CLUSTERING ORDER BY (line_id ASC)
"""

# ── Q1: invoices_by_month_tier ────────────────────────────────────────────────
# Partition key: (year_month, tier_id)
#   • year_month (e.g. '2025-03') and tier_id (1/2/3) together define a partition.
#     Q1 queries 12 months × 3 tiers = up to 36 partitions, counts invoices per
#     partition, and sums total_usd — then aggregates the 36 results in Python.
#     Each partition read is a small sequential scan of a month's worth of invoices
#     for one tier. No ALLOW FILTERING.
#   • Using year_month as part of the partition key rather than a date range filter
#     is what makes the 36-partition fan-out possible. Naive Q1 would need ALLOW
#     FILTERING on created_at across a single flat invoices table.
# Clustering: invoice_id ASC
#   • Unique ordering within partition. No business requirement on order here;
#     invoice_id just ensures uniqueness.
# Pre-computed columns:
#   • tier_name and monthly_price_usd_at_time are resolved at load time (see module
#     docstring). The benchmark script does not perform any temporal lookups —
#     it reads the pre-resolved values directly.
#   • invoice_type is included so Q1 can distinguish subscription vs marketplace
#     invoices in its output, matching the PostgreSQL Q1 result shape.
CREATE_INVOICES_BY_MONTH_TIER = """
CREATE TABLE IF NOT EXISTS invoices_by_month_tier (
    year_month                  text,
    tier_id                     int,
    invoice_id                  uuid,
    invoice_type                text,
    total_usd                   decimal,
    tier_name                   text,
    monthly_price_usd_at_time   decimal,
    created_at                  timestamp,
    PRIMARY KEY ((year_month, tier_id), invoice_id)
) WITH CLUSTERING ORDER BY (invoice_id ASC)
"""

# ── Q7: invoices_by_tier ──────────────────────────────────────────────────────
# Partition key: tier_id
#   • Three partitions total, one per tier. Each partition contains all paid
#     invoices for that tier across the full dataset date range.
#   • Q7 reads all three partitions for a date range (Python-filtered on paid_date),
#     aggregates by day in Python, computes the 7-day rolling average in Python,
#     and gap-fills missing days in Python. Small partitions (one per tier) make
#     the full date-range read cheap even without a date range push-down index.
# Clustering: paid_date ASC, invoice_id ASC
#   • paid_date (DATE — Cassandra's LocalDate type) allows Cassandra to serve rows
#     for a date range in sorted order. A WHERE paid_date >= ? AND paid_date <= ?
#     slice within a partition does not require ALLOW FILTERING because paid_date
#     is a clustering column (range predicates on clustering columns are native CQL).
#   • invoice_id breaks ties on the same date.
# Note: only paid invoices are loaded into this table. Tier attribution is
#     pre-computed using the same Python logic as invoices_by_month_tier.
CREATE_INVOICES_BY_TIER = """
CREATE TABLE IF NOT EXISTS invoices_by_tier (
    tier_id         int,
    paid_date       date,
    invoice_id      uuid,
    total_usd       decimal,
    tier_name       text,
    invoice_type    text,
    PRIMARY KEY ((tier_id), paid_date, invoice_id)
) WITH CLUSTERING ORDER BY (paid_date ASC, invoice_id ASC)
"""

# ── Q5: products_search ───────────────────────────────────────────────────────
# Partition key: id (UUID)
#   • Standard single-partition key. The SASI index (below) is what enables
#     text search across all partitions without ALLOW FILTERING.
# name_lower: product name stored in lowercase at load time.
#   • SASI requires the data to match the analyser's normalisation.
#     Storing name_lower at insert time avoids per-query LOWER() transformations
#     and ensures the SASI index is built on already-lowercased data.
# SASI index (created after the table):
#   • mode=CONTAINS enables substring matching: name_lower LIKE '%keyword%'.
#   • analyzed=true with CaseSensitiveAnalyzer on already-lowercased data
#     is equivalent to a case-insensitive contains search.
#   • SASI is the correct choice for Cassandra 4.1. It is deprecated in 5.0
#     in favour of SAI, but SAI in 4.1 does not support CONTAINS mode.
#     Full documentation: see module docstring section "SASI index for Q5".
# Other product fields are included for result completeness (Q5 should return
#     the same columns as the PostgreSQL Q5 baseline).
CREATE_PRODUCTS_SEARCH = """
CREATE TABLE IF NOT EXISTS products_search (
    id              uuid        PRIMARY KEY,
    name            text,
    name_lower      text,
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

# SASI index on name_lower.
# Created as a separate statement after the table — Cassandra DDL requirement.
CREATE_SASI_INDEX = f"""
CREATE CUSTOM INDEX IF NOT EXISTS products_search_name_lower_sasi
    ON {KEYSPACE}.products_search (name_lower)
    USING 'org.apache.cassandra.index.sasi.SASIIndex'
    WITH OPTIONS = {{
        'mode': 'CONTAINS',
        'analyzer_class': 'org.apache.cassandra.index.sasi.analyzer.StandardAnalyzer',
        'analyzed': 'true',
        'case_sensitive': 'false'
    }}
"""

ALL_DDL = [
    ("events_by_user_month",    CREATE_EVENTS_BY_USER_MONTH),
    ("sessions_by_user",        CREATE_SESSIONS_BY_USER),
    ("also_bought",             CREATE_ALSO_BOUGHT),
    ("invoices_full",           CREATE_INVOICES_FULL),
    ("invoices_by_month_tier",  CREATE_INVOICES_BY_MONTH_TIER),
    ("invoices_by_tier",        CREATE_INVOICES_BY_TIER),
    ("products_search",         CREATE_PRODUCTS_SEARCH),
]

# ── connection ─────────────────────────────────────────────────────────────────

def _connect_no_keyspace():
    auth = PlainTextAuthProvider(
        username=os.getenv("CASSANDRA_USER", "cassandra"),
        password=os.getenv("CASSANDRA_PASSWORD", "cassandra"),
    )
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        consistency_level=ConsistencyLevel.LOCAL_ONE,
        request_timeout=120.0,  # generous for DDL + SASI index creation
    )
    cluster = Cluster(
        contact_points=["localhost"],
        port=int(os.getenv("CASSANDRA_PORT", "9042")),
        auth_provider=auth,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    )
    session = cluster.connect()
    session.default_fetch_size = 10_000
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

    # SASI index must be created after the products_search table.
    # Index creation is asynchronous in Cassandra — it builds in the background.
    # For a single-node instance the index is ready by the time the loader
    # finishes inserting products, so no explicit wait is needed here.
    print(f"    Creating SASI index on products_search.name_lower...")
    session.execute(CREATE_SASI_INDEX)
    print(f"    ✔ SASI index created.")

    print(f"  ✔ All {len(ALL_DDL)} tables + SASI index created.")

# ── insert helper ─────────────────────────────────────────────────────────────

def _bulk_insert(session, prepared, all_params: list, table_name: str):
    total = len(all_params)
    if total == 0:
        print(f"  {table_name}: 0 rows — skipping.")
        return
    inserted = 0
    for i in range(0, total, CHUNK_SIZE):
        chunk = all_params[i : i + CHUNK_SIZE]
        execute_concurrent_with_args(
            session, prepared, chunk,
            concurrency=INSERT_CONCURRENCY,
            raise_on_first_error=True,
        )
        inserted += len(chunk)
        print(f"\r  {table_name}: {inserted:,}/{total:,}", end="", flush=True)
    print(f"\r  ✔ {table_name}: {total:,} rows loaded.            ")

# ── in-memory source data builders ────────────────────────────────────────────
#
# All joining and pre-computation happens here, in Python, against the source
# CSV files. Nothing is read from Cassandra during this phase — the loader
# is self-contained and reproducible from the raw data files alone.

def _load_csv(filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_users_index() -> dict:
    """uuid → {full_name, email, country_code} for embedding into invoices_full."""
    idx = {}
    for row in _load_csv("users.csv"):
        idx[row["id"]] = {
            "full_name":    _text(row["full_name"]),
            "email":        _text(row["email"]),
            "country_code": _text(row["country_code"]),
        }
    return idx


def build_products_index() -> dict:
    """uuid → {name, product_type, price_usd, is_active, ...} for Q4 embedding and Q2."""
    idx = {}
    for row in _load_csv("products.csv"):
        idx[row["id"]] = {
            "name":         _text(row["name"]),
            "product_type": _text(row["product_type"]),
            "price_usd":    _dec(row["price_usd"]),
            "is_active":    _bool(row["is_active"]),
            "description":  _text(row["description"]),
            "currency":     _text(row["currency"]),
            "seller_id":    _uuid(row["seller_id"]),
            "attributes":   _text(row["attributes"]),
            "created_at":   _dt(row["created_at"]),
            "updated_at":   _dt(row["updated_at"]),
            "slug":         _text(row["slug"]),
        }
    return idx


def build_subscriptions_index() -> dict:
    """
    user_id → sorted list of (started_at, tier_id) tuples, ascending by started_at.
    Used for tier attribution on marketplace invoices: find the subscription with
    the highest started_at that is still <= the invoice created_at (LATERAL JOIN
    equivalent).
    """
    from collections import defaultdict
    user_subs = defaultdict(list)
    for row in _load_csv("subscriptions.csv"):
        uid = row["user_id"]
        started = _dt(row["started_at"])
        tier = _int(row["tier_id"])
        if uid and started and tier is not None:
            user_subs[uid].append((started, tier))
    # Sort ascending so we can binary-search or scan for the latest <= invoice_date
    for uid in user_subs:
        user_subs[uid].sort(key=lambda x: x[0])
    return dict(user_subs)


def resolve_tier_for_marketplace_invoice(
    user_id: str,
    invoice_created_at: datetime,
    user_subs: dict,
) -> int | None:
    """
    Return the tier_id of the subscription most recently started before or at
    invoice_created_at, or None if no subscription exists for this user.
    Mirrors PostgreSQL Q1's LATERAL subquery:
        SELECT s.tier_id FROM subscriptions s
        WHERE s.user_id = i.user_id AND s.started_at <= i.created_at
        ORDER BY s.started_at DESC LIMIT 1
    """
    subs = user_subs.get(user_id, [])
    # subs is sorted ascending by started_at; scan from the end for the latest
    tier_id = None
    for started_at, tier in subs:
        if started_at <= invoice_created_at:
            tier_id = tier
        else:
            break
    return tier_id



def resolve_price_at_time(
    tier_id: int,
    invoice_created_at: datetime,
    pricing: list[tuple],
) -> Decimal | None:
    """
    Return the monthly_price_usd for tier_id at invoice_created_at.
    Finds the pricing row where:
        valid_from <= invoice_created_at AND (valid_to IS NULL OR valid_to > invoice_created_at)
    Mirrors PostgreSQL Q1's temporal JOIN predicate.
    """
    for t_id, valid_from, valid_to, price in pricing:
        if t_id != tier_id:
            continue
        if valid_from <= invoice_created_at:
            if valid_to is None or valid_to > invoice_created_at:
                return price
    return None


TIER_NAMES = {1: "Free", 2: "Pro", 3: "Business"}

# ── table loaders ─────────────────────────────────────────────────────────────

def load_events_by_user_month(session):
    prepared = session.prepare("""
        INSERT INTO events_by_user_month
            (user_id, year_month, occurred_at, id,
             event_type, product_id, session_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for row in _load_csv("events.csv"):
        occurred_at = _dt(row["occurred_at"])
        if occurred_at is None:
            continue
        params.append((
            _uuid(row["user_id"]),
            _year_month(occurred_at),   # computed partition component
            occurred_at,
            _uuid(row["id"]),
            _text(row["event_type"]),
            _uuid(row["product_id"]),   # nullable
            _text(row["session_id"]),   # nullable text
            _text(row["metadata"]),
        ))
    _bulk_insert(session, prepared, params, "events_by_user_month")


def load_sessions_by_user(session):
    prepared = session.prepare("""
        INSERT INTO sessions_by_user
            (user_id, last_active_at, id,
             cart, ip_address, user_agent,
             created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for row in _load_csv("sessions.csv"):
        last_active = _dt(row["last_active_at"])
        user_id = _uuid(row["user_id"])
        if last_active is None or user_id is None:
            continue
        params.append((
            user_id,
            last_active,
            row["id"],              # text, not UUID
            _text(row["cart"]),
            _text(row["ip_address"]),
            _text(row["user_agent"]),
            _dt(row["created_at"]),
            _dt(row["expires_at"]),
        ))
    _bulk_insert(session, prepared, params, "sessions_by_user")


def load_also_bought(session, products_index: dict):
    """
    Pre-compute co-purchase counts from order_items.csv and orders.csv.

    Algorithm:
      1. Load all orders; keep only those with completed statuses.
      2. Load all order_items; group product_ids by order_id.
      3. For each order, enumerate every unordered product pair (A, B).
         Increment co_purchase_count[A][B] and co_purchase_count[B][A].
      4. For each (product_a, product_b, count) triplet, compute confidence
         = count / total_orders_containing_product_a.
      5. Insert into also_bought, embedding product_b details from products_index.
         Skip pairs where product_b is not active.

    This mirrors the PostgreSQL Q4 co-purchase query executed once at load time
    instead of on every query invocation.
    """
    print("  also_bought: computing co-purchase counts from order_items...")

    # Step 1: qualifying order IDs
    qualifying_orders = set()
    for row in _load_csv("orders.csv"):
        if _text(row["status"]) in COMPLETED_ORDER_STATUSES:
            qualifying_orders.add(row["id"])

    # Step 2: product lists per order
    order_products = defaultdict(set)
    for row in _load_csv("order_items.csv"):
        oid = row["order_id"]
        pid = row["product_id"]
        if oid in qualifying_orders and pid:
            order_products[oid].add(pid)

    # Step 3: co-purchase counts and order counts per product
    co_counts = defaultdict(lambda: defaultdict(int))   # co_counts[A][B] = count
    order_counts = defaultdict(int)                     # how many orders contain A

    for oid, products in order_products.items():
        products = list(products)
        for i, pa in enumerate(products):
            order_counts[pa] += 1
            for pb in products:
                if pa != pb:
                    co_counts[pa][pb] += 1

    print(f"  also_bought: {len(co_counts):,} products with co-purchases found.")

    # Step 4 + 5: build insert params
    prepared = session.prepare("""
        INSERT INTO also_bought
            (product_id, co_purchase_count, co_product_id,
             co_product_name, co_product_type, co_product_price_usd,
             co_product_is_active, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for pa_str, co_dict in co_counts.items():
        pa = _uuid(pa_str)
        if pa is None:
            continue
        total_orders_a = order_counts[pa_str]
        for pb_str, count in co_dict.items():
            pb_info = products_index.get(pb_str)
            if pb_info is None or not pb_info["is_active"]:
                # Skip inactive or unknown products — mirrors PostgreSQL Q4 WHERE p.is_active = TRUE
                continue
            pb = _uuid(pb_str)
            confidence = (
                Decimal(count) / Decimal(total_orders_a)
                if total_orders_a > 0 else Decimal("0")
            )
            params.append((
                pa,
                count,
                pb,
                pb_info["name"],
                pb_info["product_type"],
                pb_info["price_usd"],
                pb_info["is_active"],
                confidence.quantize(Decimal("0.0001")),
            ))

    _bulk_insert(session, prepared, params, "also_bought")


def load_invoices_full(session, users_index: dict, products_index: dict):
    """
    Denormalise invoices + invoice_lines + users + products into invoices_full.
    One Cassandra row per invoice_line. Invoice-level and customer fields are
    repeated on every row of the same invoice partition.
    """
    # Load raw data
    invoices = {}
    for filename in ("marketplace_invoices.csv", "subscription_invoices.csv"):
        for row in _load_csv(filename):
            invoices[row["id"]] = row

    # Index invoice_lines by invoice_id
    lines_by_invoice = defaultdict(list)
    for filename in ("marketplace_invoice_lines.csv", "subscription_invoice_lines.csv"):
        for row in _load_csv(filename):
            lines_by_invoice[row["invoice_id"]].append(row)

    prepared = session.prepare("""
        INSERT INTO invoices_full (
            invoice_id, line_id,
            invoice_type, invoice_status,
            subtotal_usd, tax_usd, discount_usd, total_usd,
            subscription_id, billing_period_start, billing_period_end,
            paid_at, due_at, invoice_created_at,
            customer_id, customer_full_name, customer_email, customer_country_code,
            line_description, line_quantity, line_unit_price_usd, line_total_usd,
            product_id, product_name, product_type, product_price_usd
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """)

    params = []
    for inv_id, inv in invoices.items():
        inv_uuid       = _uuid(inv["id"])
        customer_id    = _uuid(inv["user_id"])
        customer_info  = users_index.get(inv["user_id"], {})

        for line in lines_by_invoice.get(inv_id, []):
            prod_id   = line["product_id"]
            prod_info = products_index.get(prod_id, {}) if prod_id and prod_id.strip() else {}

            params.append((
                inv_uuid,
                _uuid(line["id"]),
                _text(inv["invoice_type"]),
                _text(inv["status"]),
                _dec(inv["subtotal_usd"]),
                _dec(inv["tax_usd"]),
                _dec(inv["discount_usd"]),
                _dec(inv["total_usd"]),
                _uuid(inv["subscription_id"]),
                _dt(inv["billing_period_start"]),
                _dt(inv["billing_period_end"]),
                _dt(inv["paid_at"]),
                _dt(inv["due_at"]),
                _dt(inv["created_at"]),
                customer_id,
                customer_info.get("full_name"),
                customer_info.get("email"),
                customer_info.get("country_code"),
                _text(line["description"]),
                _int(line["quantity"]),
                _dec(line["unit_price_usd"]),
                _dec(line["line_total_usd"]),
                _uuid(prod_id) if prod_id and prod_id.strip() else None,
                prod_info.get("name"),
                prod_info.get("product_type"),
                prod_info.get("price_usd"),
            ))

    _bulk_insert(session, prepared, params, "invoices_full")


def load_invoices_by_month_tier(session, user_subs: dict, pricing: list[tuple]):
    """
    Pre-compute tier attribution and price resolution for every paid invoice.
    See module docstring for the full algorithm.
    """
    prepared = session.prepare("""
        INSERT INTO invoices_by_month_tier
            (year_month, tier_id, invoice_id,
             invoice_type, total_usd,
             tier_name, monthly_price_usd_at_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)

    # Load subscriptions index for marketplace invoice tier resolution
    # (already passed in as user_subs)

    # Load subscription_id → tier_id map for subscription invoices
    sub_tier = {}
    for row in _load_csv("subscriptions.csv"):
        sub_tier[row["id"]] = _int(row["tier_id"])

    params = []
    skipped = 0
    for filename in ("marketplace_invoices.csv", "subscription_invoices.csv"):
        for row in _load_csv(filename):
            if _text(row["status"]) != "paid":
                continue

            created_at   = _dt(row["created_at"])
            inv_type     = _text(row["invoice_type"])
            inv_id       = _uuid(row["id"])
            total        = _dec(row["total_usd"])

            if created_at is None or inv_id is None:
                skipped += 1
                continue

            # Resolve tier_id
            if inv_type == "subscription":
                tier_id = sub_tier.get(row["subscription_id"])
            else:
                tier_id = resolve_tier_for_marketplace_invoice(
                    row["user_id"], created_at, user_subs
                )

            if tier_id is None:
                # Marketplace invoice with no subscription history — skip.
                # Documented limitation: mirrors the PostgreSQL Q1 behaviour where
                # invoices with no matching subscription are excluded by the LATERAL JOIN.
                skipped += 1
                continue

            price_at_time = resolve_price_at_time(tier_id, created_at, pricing)
            if price_at_time is None:
                skipped += 1
                continue

            params.append((
                _year_month(created_at),
                tier_id,
                inv_id,
                inv_type,
                total,
                TIER_NAMES.get(tier_id, str(tier_id)),
                price_at_time,
                created_at,
            ))

    if skipped:
        print(f"  invoices_by_month_tier: {skipped} invoices skipped "
              f"(no subscription or pricing match — consistent with PostgreSQL Q1 LATERAL JOIN).")
    _bulk_insert(session, prepared, params, "invoices_by_month_tier")


def load_invoices_by_tier(session, user_subs: dict, pricing: list[tuple]):
    """
    Load paid invoices into invoices_by_tier for Q7.
    Uses the same tier attribution logic as invoices_by_month_tier.
    Partition key is tier_id (3 partitions); clustering is paid_date.
    """
    prepared = session.prepare("""
        INSERT INTO invoices_by_tier
            (tier_id, paid_date, invoice_id,
             total_usd, tier_name, invoice_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """)

    sub_tier = {}
    for row in _load_csv("subscriptions.csv"):
        sub_tier[row["id"]] = _int(row["tier_id"])

    params = []
    skipped = 0
    for filename in ("marketplace_invoices.csv", "subscription_invoices.csv"):
        for row in _load_csv(filename):
            if _text(row["status"]) != "paid":
                continue
            created_at = _dt(row["created_at"])
            inv_type   = _text(row["invoice_type"])
            inv_id     = _uuid(row["id"])
            total      = _dec(row["total_usd"])

            if created_at is None or inv_id is None:
                skipped += 1
                continue

            if inv_type == "subscription":
                tier_id = sub_tier.get(row["subscription_id"])
            else:
                tier_id = resolve_tier_for_marketplace_invoice(
                    row["user_id"], created_at, user_subs
                )

            if tier_id is None:
                skipped += 1
                continue

            params.append((
                tier_id,
                _paid_date(created_at),     # Cassandra LocalDate
                inv_id,
                total,
                TIER_NAMES.get(tier_id, str(tier_id)),
                inv_type,
            ))

    if skipped:
        print(f"  invoices_by_tier: {skipped} invoices skipped (no tier match).")
    _bulk_insert(session, prepared, params, "invoices_by_tier")


def load_products_search(session, products_index: dict):
    """
    Load products into products_search with name_lower pre-computed.
    The SASI index is built automatically by Cassandra after the inserts.
    """
    prepared = session.prepare("""
        INSERT INTO products_search
            (id, name, name_lower, product_type, description,
             price_usd, currency, is_active, seller_id,
             attributes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    params = []
    for pid_str, p in products_index.items():
        name = p["name"]
        params.append((
            _uuid(pid_str),
            name,
            name.lower() if name else None,     # pre-lowercased for SASI CONTAINS
            p["product_type"],
            p["description"],
            p["price_usd"],
            p["currency"],
            p["is_active"],
            p["seller_id"],
            p["attributes"],
            p["created_at"],
            p["updated_at"],
        ))
    _bulk_insert(session, prepared, params, "products_search")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cassandra optimised schema loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full load drops and recreates the keyspace from scratch.\n"
            "--dry-run creates the schema but skips all data inserts.\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Create schema only — do not insert any data.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Cassandra — Optimised Schema Loader")
    print("=" * 60)
    print(f"  Keyspace : {KEYSPACE}")
    print(f"  Data dir : {DATA_DIR}")
    if args.dry_run:
        print("  Mode     : DRY RUN (schema only, no data)")
    print()

    cluster, session = _connect_no_keyspace()

    try:
        drop_keyspace(session)
        create_schema(session)

        if args.dry_run:
            print("\n  Dry run complete — schema verified, no data inserted.")
            return

        # ── Build in-memory indexes from source CSVs ──────────────────────────
        # All joining and pre-computation is done here before any Cassandra writes.
        print("\n  Building in-memory indexes from source CSVs...")
        print("  " + "─" * 56)

        print("  Loading users index...")
        users_index    = build_users_index()
        print(f"  ✔ users: {len(users_index):,} entries")

        print("  Loading products index...")
        products_index = build_products_index()
        print(f"  ✔ products: {len(products_index):,} entries")

        print("  Loading subscriptions index (for tier attribution)...")
        user_subs      = build_subscriptions_index()
        print(f"  ✔ subscriptions: {len(user_subs):,} users with subscription history")

        # Hardcoded pricing data (same values as the naive loader's hardcoded inserts)
        pricing = [
            (1, datetime(2023, 1, 1, tzinfo=timezone.utc), None,                                       Decimal("0.00")),
            (2, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("14.99")),
            (2, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("19.99")),
            (3, datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc),  Decimal("39.99")),
            (3, datetime(2024, 6, 1, tzinfo=timezone.utc), None,                                       Decimal("49.99")),
        ]
        print("  ✔ Pricing history hardcoded (5 rows — matches naive loader).")

        # ── Load query-driven tables ──────────────────────────────────────────
        print("\n  Loading query-driven tables...")
        print("  " + "─" * 56)

        # Q6 — largest table, simplest transformation (just adds year_month column)
        load_events_by_user_month(session)

        # Q3 — route sessions by user_id partition key
        load_sessions_by_user(session)

        # Q4 — pre-aggregate co-purchase counts (most complex pre-computation)
        load_also_bought(session, products_index)

        # Q2 — denormalise invoices + lines + customer + product
        load_invoices_full(session, users_index, products_index)

        # Q1 — pre-compute tier attribution + price resolution per invoice
        load_invoices_by_month_tier(session, user_subs, pricing)

        # Q7 — route paid invoices by tier_id, clustering on paid_date
        load_invoices_by_tier(session, user_subs, pricing)

        # Q5 — products with name_lower + SASI index
        load_products_search(session, products_index)

        print()
        print("  " + "─" * 56)
        print("  ✔ Cassandra optimised schema fully loaded.")
        print(f"  Keyspace : {KEYSPACE}")
        print()
        print("  Note: The SASI index on products_search.name_lower builds")
        print("  asynchronously. If Q5 returns no results immediately after")
        print("  loading, wait ~30 seconds for index build to complete.")
        print()

    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()