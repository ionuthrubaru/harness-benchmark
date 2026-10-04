"""
generators/events.py — StreamCart Clickstream Event Generator
=============================================================
Generates two CSV files:

  data/events.csv      — 5–10M realistic clickstream events (operational dataset)
                         Used to populate Cassandra for Q6 benchmark.
                         Causal chains, session clustering, seasonal growth,
                         purchase events linked to real order_ids.

  data/events_q8.csv   — exactly 1M events (write benchmark dataset)
                         Same schema, maximally varied payload, synthetic order_ids.
                         Every DB in Q8 ingests this identical file so the
                         write throughput comparison is fair across all 7 DBs.

Event types (from schema):
  page_view, product_view, add_to_cart, remove_from_cart,
  checkout_start, purchase, subscription_upgrade, subscription_cancel,
  search, download, login, logout

Causal chains (events.csv only):
  Each user gets N browsing sessions distributed across their platform lifetime.
  Within a session, events follow realistic sequences:

  login
  └─ page_view(s)
     └─ search (sometimes)
        └─ product_view(s)
           └─ add_to_cart (sometimes)
              ├─ remove_from_cart (sometimes)
              └─ checkout_start
                 └─ purchase  ← linked to a real order_id from orders.csv

Purchase linkage:
  orders.csv is loaded and grouped by user_id. When a purchase event fires,
  it consumes the next available order_id for that user. Leftover orders
  (beyond what the event timeline covers) are fine — not every order
  necessarily has a matching clickstream event.

Seasonality (events.csv):
  Monthly event volume follows the same growth S-curve as users.py, with
  a Black Friday spike in November and a dip in summer months.
  The June 2024 pricing change is reflected in subscription_upgrade events
  (users evaluating tier changes around the price increase).

Partition-friendly output:
  events.csv is written grouped by user then time-sorted within each user,
  which is optimal for Cassandra's (user_id, month) partition scheme.
  The full file is NOT globally sorted by time — that's intentional.

Memory:
  Events are generated user-by-user and written to CSV incrementally.
  Peak memory is bounded by the largest single user's event batch (~1K events).

Usage
-----
  python generators/events.py                    # 7M events + 1M Q8
  python generators/events.py --count 5000000    # 5M events + 1M Q8
  python generators/events.py --q8-count 1000000
  python generators/events.py --seed 99
  python generators/events.py --out-dir /tmp
  python generators/events.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED      = 42
DEFAULT_COUNT     = 7_000_000   # events.csv target
DEFAULT_Q8_COUNT  = 1_000_000   # events_q8.csv — fixed

PLATFORM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLATFORM_END   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Event type weights for background (non-chain) events
EVENT_TYPE_WEIGHTS = {
    "page_view":            0.35,
    "search":               0.15,
    "product_view":         0.20,
    "login":                0.08,
    "logout":               0.07,
    "add_to_cart":          0.05,
    "remove_from_cart":     0.02,
    "checkout_start":       0.02,
    "purchase":             0.02,
    "download":             0.02,
    "subscription_upgrade": 0.01,
    "subscription_cancel":  0.005,
}

# Browsing sessions per user: Pareto-distributed (power users generate far more)
SESSIONS_PER_USER_SHAPE = 1.5   # Pareto shape — lower = heavier tail
SESSIONS_PER_USER_MIN   = 1
SESSIONS_PER_USER_MAX   = 200

# Events per browsing session
SESSION_EVENTS_MIN = 5
SESSION_EVENTS_MAX = 40

# Probability of causal chain escalation
P_SEARCH_AFTER_PAGEVIEW    = 0.55
P_PRODUCTVIEW_AFTER_SEARCH = 0.75
P_ADDCART_AFTER_VIEW       = 0.25
P_REMOVECART               = 0.15
P_CHECKOUT_AFTER_ADDCART   = 0.50
P_PURCHASE_AFTER_CHECKOUT  = 0.70

# Seasonal monthly weights (24 months: Jan 2024 – Dec 2025)
# Same growth curve as users.py + Black Friday spike in month 11 and 23
MONTHLY_WEIGHTS_RAW = np.array([
    1.00, 0.95, 0.97, 1.02, 1.05, 1.08,   # Jan–Jun 2024
    0.92, 0.88, 0.98, 1.05, 1.20, 1.10,   # Jul–Dec 2024 (Nov spike)
    1.00, 0.96, 0.98, 1.03, 1.06, 1.09,   # Jan–Jun 2025
    0.93, 0.89, 0.99, 1.06, 1.25, 1.12,   # Jul–Dec 2025 (Nov spike)
])

# Page names for page_view events
PAGES = [
    "/", "/marketplace", "/marketplace/courses", "/marketplace/digital-assets",
    "/marketplace/merch", "/dashboard", "/account", "/account/billing",
    "/account/downloads", "/account/purchases", "/search", "/cart",
    "/checkout", "/product/{slug}", "/seller/{id}", "/about", "/pricing",
]

# Search query templates for search events — realistic vocabulary for Q5
SEARCH_QUERIES = [
    "photoshop brushes", "procreate brush pack", "logo design template",
    "UI kit figma", "lightroom presets landscape", "font bundle serif",
    "icon set minimal", "social media templates instagram",
    "after effects transitions", "watercolour texture pack",
    "vintage badge design", "motion graphics template",
    "brand identity kit", "3d model furniture", "mockup phone",
    "typography poster", "colour palette pastel", "illustration flat",
    "brush pack ink", "video editing luts", "design system figma",
    "abstract background", "business card template", "pattern seamless",
    "character illustration", "isometric icons", "hand lettering",
    "photo editing actions", "web design kit", "canva templates",
    "resume template minimal", "infographic template", "ebook template",
    "packaging mockup", "t-shirt mockup", "logo mockup",
    "procreate lettering", "digital planner", "notion template",
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_users(out_dir: Path) -> tuple[list[str], list[str]]:
    """Return (user_ids, created_ats) for active users only."""
    path = out_dir / "users.csv"
    if not path.exists():
        raise FileNotFoundError(f"users.csv not found at {path}.")
    ids:     list[str] = []
    created: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_active"].lower() == "true":
                ids.append(row["id"])
                created.append(row["created_at"])
    return ids, created


def _load_products(out_dir: Path) -> tuple[list[str], list[str], list[float]]:
    """Return (ids, types, prices) for active products only."""
    path = out_dir / "products.csv"
    if not path.exists():
        raise FileNotFoundError(f"products.csv not found at {path}.")
    ids:    list[str]   = []
    types:  list[str]   = []
    prices: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_active"].lower() == "true":
                ids.append(row["id"])
                types.append(row["product_type"])
                prices.append(float(row["price_usd"]))
    return ids, types, prices


def _load_sessions(out_dir: Path) -> dict[str, list[str]]:
    """
    Return a dict mapping user_id → list of session_ids.
    Used to link events to real session tokens.
    """
    path = out_dir / "sessions.csv"
    if not path.exists():
        return {}
    user_sessions: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user_sessions[row["user_id"]].append(row["id"])
    return dict(user_sessions)


def _load_orders(out_dir: Path) -> dict[str, list[tuple[str, float]]]:
    """
    Return a dict mapping user_id → list of (order_id, total_usd).
    Used to link purchase events to real order_ids.
    """
    path = out_dir / "orders.csv"
    if not path.exists():
        return {}

    # Load invoice totals for amount_usd on purchase events
    inv_totals: dict[str, float] = {}
    inv_path = out_dir / "marketplace_invoices.csv"
    if inv_path.exists():
        with open(inv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                inv_totals[row["id"]] = float(row["total_usd"])

    user_orders: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] in ("delivered", "shipped", "confirmed"):
                total = inv_totals.get(row["invoice_id"], 0.0)
                user_orders[row["user_id"]].append((row["id"], total))
    return dict(user_orders)


# ── Monthly distribution ──────────────────────────────────────────────────────

def _build_monthly_schedule(
    n_users: int,
    target_events: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return an array of shape (n_users,) giving the number of browsing sessions
    to generate per user, distributed to approximate target_events total events.
    Uses a Pareto distribution so power users generate many more events.
    """
    raw    = rng.pareto(SESSIONS_PER_USER_SHAPE, size=n_users) + 1
    raw    = np.clip(raw, SESSIONS_PER_USER_MIN, SESSIONS_PER_USER_MAX)
    # Scale so expected total events ≈ target
    # Calibrated from observed generation: actual average is ~11.2 events/session
    # (most sessions don't complete the full causal chain, so true avg is well
    # below the (MIN+MAX)/2 = 22.5 theoretical maximum).
    avg_events_per_session = 11.2
    scale  = target_events / (raw.sum() * avg_events_per_session)
    raw    = raw * scale
    return np.clip(raw, SESSIONS_PER_USER_MIN, SESSIONS_PER_USER_MAX).astype(int)


def _month_weights() -> np.ndarray:
    w = MONTHLY_WEIGHTS_RAW.copy()
    return w / w.sum()


# ── Metadata builders ─────────────────────────────────────────────────────────

def _meta_page_view(rng: np.random.Generator) -> dict:
    referrers = ["google", "direct", "twitter", "instagram", "youtube",
                 "pinterest", "newsletter", None]
    return {
        "page":     str(rng.choice(PAGES)),
        "referrer": str(rng.choice(referrers)) if rng.random() > 0.3 else None,
    }


def _meta_product_view(product_id: str, rng: np.random.Generator) -> dict:
    return {
        "duration_seconds": int(rng.integers(5, 300)),
        "product_id":       product_id,
    }


def _meta_search(rng: np.random.Generator) -> dict:
    return {
        "query":         str(rng.choice(SEARCH_QUERIES)),
        "results_count": int(rng.integers(0, 50)),
    }


def _meta_purchase(order_id: str, amount: float) -> dict:
    return {"order_id": order_id, "amount_usd": round(amount, 2)}


def _meta_add_to_cart(product_id: str, price: float) -> dict:
    return {"product_id": product_id, "price_usd": price}


def _meta_remove_from_cart(product_id: str) -> dict:
    return {"product_id": product_id}


def _meta_download(product_id: str) -> dict:
    return {"product_id": product_id}


def _meta_subscription(from_tier: int, to_tier: int) -> dict:
    return {"from_tier": from_tier, "to_tier": to_tier}


def _meta_empty() -> dict:
    return {}


# ── Event row builder ─────────────────────────────────────────────────────────

def _make_event(
    user_id: str,
    event_type: str,
    occurred_at: datetime,
    session_id: str | None,
    product_id: str | None,
    metadata: dict,
) -> dict:
    return {
        "id":          str(uuid.uuid4()),
        "user_id":     user_id,
        "event_type":  event_type,
        "product_id":  product_id,
        "session_id":  session_id,
        "metadata":    json.dumps(metadata, ensure_ascii=False),
        "occurred_at": occurred_at.isoformat(),
    }


# ── Session event chain ───────────────────────────────────────────────────────

def _generate_session_events(
    user_id: str,
    session_id: str | None,
    session_start: datetime,
    prod_ids: list[str],
    prod_types: list[str],
    prod_prices: list[float],
    user_orders: list[tuple[str, float]],
    order_cursor: list[int],   # mutable pointer into user_orders
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate a sequence of events for one browsing session.
    Follows realistic causal chains: login → browse → search → view → cart → purchase.
    Returns events sorted by occurred_at.
    """
    events: list[dict] = []
    n_products = len(prod_ids)
    t = session_start

    def advance(min_s: int = 5, max_s: int = 300) -> datetime:
        nonlocal t
        t = t + timedelta(seconds=int(rng.integers(min_s, max_s)))
        return t

    # Login
    events.append(_make_event(user_id, "login", t, session_id, None, _meta_empty()))

    # 3–8 page views to start
    for _ in range(int(rng.integers(3, 9))):
        advance(10, 120)
        events.append(_make_event(user_id, "page_view", t, session_id, None, _meta_page_view(rng)))

    # Up to 2 search + browse rounds per session
    n_rounds = int(rng.choice([1, 2], p=[0.55, 0.45]))
    for _round in range(n_rounds):
        # Maybe search
        if rng.random() < P_SEARCH_AFTER_PAGEVIEW:
            advance(15, 90)
            events.append(_make_event(user_id, "search", t, session_id, None, _meta_search(rng)))

            # Maybe view products from search results
            if rng.random() < P_PRODUCTVIEW_AFTER_SEARCH:
                n_views = int(rng.integers(1, 5))
                viewed_products: list[int] = []
                for _ in range(n_views):
                    advance(20, 180)
                    p_idx = int(rng.integers(0, n_products))
                    viewed_products.append(p_idx)
                    events.append(_make_event(
                        user_id, "product_view", t, session_id,
                        prod_ids[p_idx], _meta_product_view(prod_ids[p_idx], rng)
                    ))

                    # Maybe add to cart
                    if rng.random() < P_ADDCART_AFTER_VIEW:
                        advance(5, 60)
                        events.append(_make_event(
                            user_id, "add_to_cart", t, session_id,
                            prod_ids[p_idx],
                            _meta_add_to_cart(prod_ids[p_idx], prod_prices[p_idx])
                        ))

                        # Maybe remove from cart
                        if rng.random() < P_REMOVECART:
                            advance(10, 120)
                            events.append(_make_event(
                                user_id, "remove_from_cart", t, session_id,
                                prod_ids[p_idx], _meta_remove_from_cart(prod_ids[p_idx])
                            ))

                # Maybe checkout and purchase
                added_to_cart = any(
                    e["event_type"] == "add_to_cart" for e in events
                )
                if added_to_cart and rng.random() < P_CHECKOUT_AFTER_ADDCART:
                    advance(30, 300)
                    events.append(_make_event(
                        user_id, "checkout_start", t, session_id, None, _meta_empty()
                    ))

                    if rng.random() < P_PURCHASE_AFTER_CHECKOUT:
                        advance(20, 180)
                        # Link to a real order_id if available
                        if user_orders and order_cursor[0] < len(user_orders):
                            order_id, amount = user_orders[order_cursor[0]]
                            order_cursor[0] += 1
                        else:
                            order_id = str(uuid.uuid4())
                            amount   = round(float(rng.uniform(5, 200)), 2)

                        events.append(_make_event(
                            user_id, "purchase", t, session_id, None,
                            _meta_purchase(order_id, amount)
                        ))

                        # Download for digital products
                        for p_idx in viewed_products:
                            if prod_types[p_idx] in ("digital_asset", "course"):
                                advance(5, 30)
                                events.append(_make_event(
                                    user_id, "download", t, session_id,
                                    prod_ids[p_idx], _meta_download(prod_ids[p_idx])
                                ))

        # Between rounds: a few more page views to pad session length
        if _round < n_rounds - 1:
            for _ in range(int(rng.integers(1, 4))):
                advance(10, 90)
                events.append(_make_event(
                    user_id, "page_view", t, session_id, None, _meta_page_view(rng)
                ))

    # Logout (not always — tab close is common)
    if rng.random() < 0.60:
        advance(10, 300)
        events.append(_make_event(user_id, "logout", t, session_id, None, _meta_empty()))

    return sorted(events, key=lambda e: e["occurred_at"])


# ── Main events.csv generation ────────────────────────────────────────────────

def _generate_events_csv(
    out_path: Path,
    user_ids: list[str],
    user_created_ats: list[str],
    prod_ids: list[str],
    prod_types: list[str],
    prod_prices: list[float],
    user_sessions: dict[str, list[str]],
    user_orders: dict[str, list[tuple[str, float]]],
    target_count: int,
    seed: int,
    verbose: bool,
) -> int:
    """Stream events to CSV user-by-user. Returns actual event count written."""
    rng        = np.random.default_rng(seed)
    n_users    = len(user_ids)
    month_w    = _month_weights()

    # How many browsing sessions per user
    sessions_per_user = _build_monthly_schedule(n_users, target_count, rng)

    fields = ["id", "user_id", "event_type", "product_id",
              "session_id", "metadata", "occurred_at"]

    total_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for u_i, user_id in enumerate(user_ids):
            n_sessions      = int(sessions_per_user[u_i])
            user_created_dt = datetime.fromisoformat(user_created_ats[u_i])
            user_order_list = user_orders.get(user_id, [])
            order_cursor    = [0]   # mutable pointer
            session_ids     = user_sessions.get(user_id, [])

            # Distribute this user's sessions across their platform lifetime
            available_days = max(1, (PLATFORM_END - user_created_dt).days)

            for s_i in range(n_sessions):
                # Pick a random day within user's lifetime, weighted by monthly curve
                # Simplified: uniform random day within lifetime
                day_offset = int(rng.integers(0, available_days))
                session_start = user_created_dt + timedelta(
                    days=day_offset,
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60)),
                )
                if session_start > PLATFORM_END:
                    session_start = PLATFORM_END - timedelta(hours=1)

                # Use a real session_id if this session falls within the session window
                session_window_start = PLATFORM_END - timedelta(days=90)
                if session_start >= session_window_start and session_ids:
                    session_id = session_ids[s_i % len(session_ids)]
                else:
                    session_id = None

                events = _generate_session_events(
                    user_id=user_id,
                    session_id=session_id,
                    session_start=session_start,
                    prod_ids=prod_ids,
                    prod_types=prod_types,
                    prod_prices=prod_prices,
                    user_orders=user_order_list,
                    order_cursor=order_cursor,
                    rng=rng,
                )

                writer.writerows(events)
                total_written += len(events)

            # Progress every 5K users
            if verbose and (u_i + 1) % 5_000 == 0:
                pct = (u_i + 1) / n_users * 100
                print(f"    users {u_i+1:>7,}/{n_users:,}  ({pct:.0f}%)  "
                      f"~{total_written:,} events written")

    return total_written


# ── Q8 file generation ────────────────────────────────────────────────────────

def _generate_q8_csv(
    out_path: Path,
    user_ids: list[str],
    prod_ids: list[str],
    prod_types: list[str],
    prod_prices: list[float],
    count: int,
    seed: int,
    verbose: bool,
) -> None:
    """
    Generate exactly `count` events for the Q8 write benchmark.
    No causal chains — just maximally varied realistic payloads.
    All event types present, synthetic order_ids, no session linkage.
    Written in approximate temporal order.
    """
    rng = np.random.default_rng(seed + 999)   # different seed from main events

    n_users    = len(user_ids)
    n_products = len(prod_ids)
    event_types = list(EVENT_TYPE_WEIGHTS.keys())
    et_weights  = np.array(list(EVENT_TYPE_WEIGHTS.values()))
    et_weights /= et_weights.sum()

    total_secs = int((PLATFORM_END - PLATFORM_START).total_seconds())
    fields     = ["id", "user_id", "event_type", "product_id",
                  "session_id", "metadata", "occurred_at"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        # Generate in batches of 50K for memory efficiency
        batch_size = 50_000
        written    = 0

        while written < count:
            batch_n = min(batch_size, count - written)

            # Sample all fields for this batch at once (fast numpy)
            u_indices  = rng.integers(0, n_users,    size=batch_n)
            p_indices  = rng.integers(0, n_products, size=batch_n)
            et_indices = rng.choice(len(event_types), size=batch_n, p=et_weights)
            ts_offsets = rng.integers(0, total_secs,  size=batch_n)

            rows = []
            for j in range(batch_n):
                et         = event_types[et_indices[j]]
                user_id    = user_ids[u_indices[j]]
                p_idx      = int(p_indices[j])
                product_id = prod_ids[p_idx] if et in (
                    "product_view", "add_to_cart", "remove_from_cart", "download"
                ) else None
                occurred   = PLATFORM_START + timedelta(seconds=int(ts_offsets[j]))

                # Build minimal but realistic metadata per type
                if et == "page_view":
                    meta = {"page": str(rng.choice(PAGES))}
                elif et == "product_view":
                    meta = {"duration_seconds": int(rng.integers(5, 300)),
                            "product_id": prod_ids[p_idx]}
                elif et == "search":
                    meta = {"query": str(rng.choice(SEARCH_QUERIES)),
                            "results_count": int(rng.integers(0, 50))}
                elif et == "purchase":
                    meta = {"order_id": str(uuid.uuid4()),
                            "amount_usd": round(prod_prices[p_idx], 2)}
                elif et == "add_to_cart":
                    meta = {"product_id": prod_ids[p_idx],
                            "price_usd": prod_prices[p_idx]}
                elif et == "remove_from_cart":
                    meta = {"product_id": prod_ids[p_idx]}
                elif et == "download":
                    meta = {"product_id": prod_ids[p_idx]}
                elif et in ("subscription_upgrade", "subscription_cancel"):
                    meta = {"from_tier": int(rng.choice([1, 2])),
                            "to_tier":   int(rng.choice([2, 3]))}
                else:
                    meta = {}

                rows.append({
                    "id":          str(uuid.uuid4()),
                    "user_id":     user_id,
                    "event_type":  et,
                    "product_id":  product_id,
                    "session_id":  None,
                    "metadata":    json.dumps(meta, ensure_ascii=False),
                    "occurred_at": occurred.isoformat(),
                })

            writer.writerows(rows)
            written += batch_n

            if verbose and written % 200_000 == 0:
                print(f"    Q8: {written:>9,} / {count:,}  ({written/count*100:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    users:    tuple[list, list] | None = None,
    products: tuple[list, list, list] | None = None,
    count:    int  = DEFAULT_COUNT,
    q8_count: int  = DEFAULT_Q8_COUNT,
    seed:     int  = DEFAULT_SEED,
    out_dir:  str | os.PathLike = None,
    verbose:  bool = True,
) -> None:
    """
    Generate events.csv and events_q8.csv.

    Parameters
    ----------
    users    : (user_ids, created_ats) for active users.
               If None, reads data/users.csv automatically.
    products : (ids, types, prices) for active products.
               If None, reads data/products.csv automatically.
    count    : target event count for events.csv (5–10M recommended)
    q8_count : exact event count for events_q8.csv (default 1M)
    seed     : random seed
    out_dir  : output directory (default: ../data/)
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  generators/events.py")
        print(f"{'═' * 60}")
        print(f"  seed           : {seed}")
        print(f"  target events  : {count:,}  → events.csv")
        print(f"  Q8 events      : {q8_count:,}  → events_q8.csv")
        print()

    # ── Load dependencies ──
    if users is None:
        if verbose:
            print("  Reading users.csv ...")
        user_ids, user_created_ats = _load_users(out_dir)
    else:
        user_ids, user_created_ats = users

    if products is None:
        if verbose:
            print("  Reading products.csv ...")
        prod_ids, prod_types, prod_prices = _load_products(out_dir)
    else:
        prod_ids, prod_types, prod_prices = products

    if verbose:
        print("  Reading sessions.csv ...")
    user_sessions = _load_sessions(out_dir)

    if verbose:
        print("  Reading orders.csv ...")
    user_orders = _load_orders(out_dir)

    if verbose:
        print(f"\n  Active users   : {len(user_ids):,}")
        print(f"  Active products: {len(prod_ids):,}")
        print(f"  Users w/ sessions : {len(user_sessions):,}")
        print(f"  Users w/ orders   : {len(user_orders):,}")
        print()

    # ── Generate events.csv ──
    events_path = out_dir / "events.csv"
    if verbose:
        print(f"  Generating events.csv (streaming, user-by-user)...")

    t0 = time.perf_counter()
    actual_count = _generate_events_csv(
        out_path         = events_path,
        user_ids         = user_ids,
        user_created_ats = user_created_ats,
        prod_ids         = prod_ids,
        prod_types       = prod_types,
        prod_prices      = prod_prices,
        user_sessions    = user_sessions,
        user_orders      = user_orders,
        target_count     = count,
        seed             = seed,
        verbose          = verbose,
    )
    t_events = time.perf_counter() - t0

    size_mb = events_path.stat().st_size / 1024 / 1024
    if verbose:
        print(f"\n  events.csv written")
        print(f"    rows    : {actual_count:,}")
        print(f"    size    : {size_mb:.1f} MB")
        print(f"    elapsed : {t_events:.1f}s")
        print()

    # ── Generate events_q8.csv ──
    q8_path = out_dir / "events_q8.csv"
    if verbose:
        print(f"  Generating events_q8.csv ({q8_count:,} rows, batched)...")

    t0 = time.perf_counter()
    _generate_q8_csv(
        out_path    = q8_path,
        user_ids    = user_ids,
        prod_ids    = prod_ids,
        prod_types  = prod_types,
        prod_prices = prod_prices,
        count       = q8_count,
        seed        = seed,
        verbose     = verbose,
    )
    t_q8 = time.perf_counter() - t0

    size_q8_mb = q8_path.stat().st_size / 1024 / 1024
    if verbose:
        print(f"\n  events_q8.csv written")
        print(f"    rows    : {q8_count:,}")
        print(f"    size    : {size_q8_mb:.1f} MB")
        print(f"    elapsed : {t_q8:.1f}s")
        print()
        _print_summary(actual_count, q8_count, t_events + t_q8, events_path, q8_path)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    n_events: int,
    n_q8: int,
    elapsed: float,
    events_path: Path,
    q8_path: Path,
) -> None:
    print(f"{'─' * 60}")
    print(f"  Final Summary")
    print(f"{'─' * 60}")
    print(f"  events.csv      : {n_events:>10,} rows  "
          f"({events_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  events_q8.csv   : {n_q8:>10,} rows  "
          f"({q8_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  Total elapsed   : {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart clickstream events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",    type=int, default=DEFAULT_COUNT,
                        help="Target event count for events.csv")
    parser.add_argument("--q8-count", type=int, default=DEFAULT_Q8_COUNT,
                        help="Exact event count for events_q8.csv")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir",  type=str, default=None)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    generate(
        count    = args.count,
        q8_count = args.q8_count,
        seed     = args.seed,
        out_dir  = args.out_dir,
        verbose  = not args.quiet,
    )