"""
generators/sessions.py Session & Cart Generator
=============================================================
Generates one CSV file:
  data/sessions.csv — 30–50K active user sessions with cart state

The cart column is a denormalised JSONB snapshot — product_id, quantity,
price_usd, product_name, and product_type are all embedded so Redis can
serve the full cart from a single key lookup with zero JOINs. This is
the entire point of Q3.

Session design
--------------
  * Only active users get sessions (is_active=True in users.csv)
  * Sessions are "active" — created_at and last_active_at are recent,
    within the last 90 days of PLATFORM_END (Oct–Dec 2025)
  * expires_at reflects realistic TTLs:
      - "remember me" sessions: 30 days from last_active_at
      - standard sessions:       24 hours from last_active_at
      - short sessions:           2 hours (mobile / incognito)
  * ~30% of sessions have an empty cart (user browsing, not buying)
  * ~70% have 1–10 items; cart size follows a right-skewed distribution
    (most carts have 1–3 items, a few have up to 10)

Cart snapshot
-------------
  Each cart item embeds:
    product_id    : UUID
    product_name  : VARCHAR(255) snapshot at time of add
    product_type  : 'course' | 'digital_asset' | 'merch'
    price_usd     : NUMERIC(8,2) snapshot at time of add
    quantity      : 1 for digital/course, 1–3 for merch

Session token
-------------
  VARCHAR(64) — generated as a 64-character hex string (32 bytes of entropy),
  matching the kind of token a real session middleware would produce.

Usage
-----
  python generators/sessions.py                  # reads data/users.csv + products.csv
  python generators/sessions.py --count 40000
  python generators/sessions.py --seed 99
  python generators/sessions.py --out-dir /tmp
  python generators/sessions.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED  = 42
DEFAULT_COUNT = 10_000

PLATFORM_END  = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Active sessions are created within the last 90 days of the platform window
SESSION_WINDOW_DAYS = 90

# Cart size distribution — P(0 items), P(1), P(2), ..., P(10)
# 30% empty, then right-skewed across 1–10
CART_SIZE_PROBS = np.array([
    0.30,                             # empty cart
    0.25, 0.18, 0.12, 0.07, 0.04,    # 1–5 items
    0.02, 0.01, 0.005, 0.004, 0.001, # 6–10 items
])
CART_SIZE_PROBS /= CART_SIZE_PROBS.sum()  # normalise
CART_SIZES = np.arange(0, 11)

# Session TTL distribution
TTL_TYPES = ["short", "standard", "remember_me"]
TTL_PROBS = [0.20, 0.55, 0.25]
TTL_HOURS = {
    "short":       2,
    "standard":    24,
    "remember_me": 24 * 30,
}

# Faker locale map (matches users.py)
COUNTRY_LOCALE = {
    "US": "en_US", "IN": "en_IN", "CN": "zh_CN", "BR": "pt_BR",
    "ID": "id_ID", "GB": "en_GB", "DE": "de_DE", "RU": "ru_RU",
    "MX": "es_MX", "JP": "ja_JP", "NG": "en_PH", "FR": "fr_FR",
    "PH": "en_PH", "TR": "tr_TR", "VN": "vi_VN", "PK": "en_IN",
    "EG": "fr_FR", "CA": "en_CA", "AU": "en_AU", "ZA": "en_GB",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_active_users(out_dir: Path) -> tuple[list[str], list[str]]:
    """Return (user_ids, country_codes) for active users only."""
    path = out_dir / "users.csv"
    if not path.exists():
        raise FileNotFoundError(f"users.csv not found at {path}.")

    ids:      list[str] = []
    countries: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_active"].lower() == "true":
                ids.append(row["id"])
                countries.append(row["country_code"])
    return ids, countries


def _load_active_products(out_dir: Path) -> tuple[list[str], list[str], list[str], list[float]]:
    """Return (ids, names, types, prices) for active products only."""
    path = out_dir / "products.csv"
    if not path.exists():
        raise FileNotFoundError(f"products.csv not found at {path}.")

    ids:    list[str]   = []
    names:  list[str]   = []
    types:  list[str]   = []
    prices: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_active"].lower() == "true":
                ids.append(row["id"])
                names.append(row["name"])
                types.append(row["product_type"])
                prices.append(float(row["price_usd"]))
    return ids, names, types, prices


# ── Session token ─────────────────────────────────────────────────────────────

def _session_token() -> str:
    """64-character hex string — 32 bytes of cryptographic entropy."""
    return secrets.token_hex(32)


# ── Cart builder ──────────────────────────────────────────────────────────────

def _build_cart(
    n_items: int,
    prod_ids: list[str],
    prod_names: list[str],
    prod_types: list[str],
    prod_prices: list[float],
    rng: np.random.Generator,
) -> list[dict]:
    """
    Build a cart as a list of denormalised item snapshots.
    Each item embeds product_id, product_name, product_type, price_usd, quantity.
    Quantity is 1 for digital/course, 1–3 for merch.
    """
    if n_items == 0:
        return []

    n_products = len(prod_ids)
    # Uniform random product selection for carts (not Zipf — users browse widely)
    chosen_indices = rng.choice(n_products, size=n_items, replace=False)

    cart = []
    for idx in chosen_indices:
        ptype    = prod_types[idx]
        quantity = 1 if ptype in ("digital_asset", "course") else int(rng.integers(1, 4))
        cart.append({
            "product_id":   prod_ids[idx],
            "product_name": prod_names[idx],
            "product_type": ptype,
            "price_usd":    prod_prices[idx],
            "quantity":     quantity,
        })
    return cart


# ── Faker pool ────────────────────────────────────────────────────────────────

def _build_faker_pool(seed: int) -> dict[str, Faker]:
    pool: dict[str, Faker] = {}
    for locale in set(COUNTRY_LOCALE.values()):
        Faker.seed(seed)
        pool[locale] = Faker(locale)
    return pool


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    users:    tuple[list[str], list[str]] | None = None,
    products: tuple[list, list, list, list] | None = None,
    count:    int  = DEFAULT_COUNT,
    seed:     int  = DEFAULT_SEED,
    out_dir:  str | os.PathLike = None,
    verbose:  bool = True,
) -> list[dict]:
    """
    Generate active sessions with cart state.

    Parameters
    ----------
    users    : (user_ids, country_codes) for active users.
               If None, reads data/users.csv automatically.
    products : (ids, names, types, prices) for active products.
               If None, reads data/products.csv automatically.
    count    : number of sessions to generate (30–50K recommended)
    seed     : random seed
    out_dir  : output directory (default: ../data/)

    Returns
    -------
    sessions : list of dicts
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    if users is None:
        if verbose:
            print("  Reading users.csv ...")
        user_ids, user_countries = _load_active_users(out_dir)
    else:
        user_ids, user_countries = users

    if products is None:
        if verbose:
            print("  Reading products.csv ...")
        prod_ids, prod_names, prod_types, prod_prices = _load_active_products(out_dir)
    else:
        prod_ids, prod_names, prod_types, prod_prices = products

    n_users    = len(user_ids)
    n_products = len(prod_ids)

    if verbose:
        print(f"\n{'═' * 55}")
        print(f"  generators/sessions.py")
        print(f"{'═' * 55}")
        print(f"  seed           : {seed}")
        print(f"  target         : {count:,} sessions")
        print(f"  active users   : {n_users:,}")
        print(f"  active products: {n_products:,}")
        print()

    t_start = time.perf_counter()

    faker_pool = _build_faker_pool(seed)

    # ── Pre-sample bulk arrays ──
    user_indices  = rng.choice(n_users,    size=count)
    cart_sizes    = rng.choice(CART_SIZES, size=count, p=CART_SIZE_PROBS)
    ttl_types     = rng.choice(TTL_TYPES,  size=count, p=TTL_PROBS)

    # Sessions are recent: created_at within the last SESSION_WINDOW_DAYS of PLATFORM_END
    window_start = PLATFORM_END - timedelta(days=SESSION_WINDOW_DAYS)
    window_secs  = int((PLATFORM_END - window_start).total_seconds())
    created_offsets = rng.integers(0, window_secs, size=count)

    # last_active_at: between created_at and min(created_at + TTL, PLATFORM_END)
    # expressed as a fraction of the remaining TTL window
    active_fractions = rng.random(size=count)

    if verbose:
        print(f"  Generating {count:,} sessions...")

    sessions: list[dict] = []

    for i in range(count):
        u_idx       = int(user_indices[i])
        user_id     = user_ids[u_idx]
        country     = user_countries[u_idx]
        locale      = COUNTRY_LOCALE.get(country, "en_US")
        fake        = faker_pool[locale]
        ttl_type    = str(ttl_types[i])
        ttl_hours   = TTL_HOURS[ttl_type]

        created_at    = window_start + timedelta(seconds=int(created_offsets[i]))
        expires_at    = created_at + timedelta(hours=ttl_hours)
        # last_active_at somewhere between created_at and expires_at
        active_window = (expires_at - created_at).total_seconds()
        last_active_at = created_at + timedelta(
            seconds=int(active_fractions[i] * active_window)
        )

        cart   = _build_cart(
            int(cart_sizes[i]),
            prod_ids, prod_names, prod_types, prod_prices, rng
        )

        sessions.append({
            "id":            _session_token(),
            "user_id":       user_id,
            "cart":          json.dumps(cart, ensure_ascii=False),
            "ip_address":    fake.ipv4_public(),
            "user_agent":    fake.user_agent(),
            "created_at":    created_at.isoformat(),
            "last_active_at": last_active_at.isoformat(),
            "expires_at":    expires_at.isoformat(),
        })

    # ── Write CSV ──
    fields = [
        "id", "user_id", "cart", "ip_address", "user_agent",
        "created_at", "last_active_at", "expires_at",
    ]

    out_path = out_dir / "sessions.csv"
    if verbose:
        print(f"  Writing sessions.csv ...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sessions)

    elapsed = time.perf_counter() - t_start

    if verbose:
        _print_summary(sessions, elapsed, out_path)

    return sessions


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(sessions: list[dict], elapsed: float, out_path: Path) -> None:
    import json as _json

    n          = len(sessions)
    carts      = [_json.loads(s["cart"]) for s in sessions]
    sizes      = [len(c) for c in carts]
    empty      = sum(1 for s in sizes if s == 0)
    nonempty   = n - empty
    avg_size   = sum(sizes) / n
    total_items = sum(sizes)

    ttl_dist: dict[str, int] = {}
    for s in sessions:
        created = datetime.fromisoformat(s["created_at"])
        expires = datetime.fromisoformat(s["expires_at"])
        hours   = (expires - created).total_seconds() / 3600
        if hours <= 2:
            label = "short (2h)"
        elif hours <= 25:
            label = "standard (24h)"
        else:
            label = "remember_me (30d)"
        ttl_dist[label] = ttl_dist.get(label, 0) + 1

    print()
    print(f"{'─' * 55}")
    print(f"  Summary")
    print(f"{'─' * 55}")
    print(f"  Total sessions     : {n:>8,}")
    print(f"  Empty carts        : {empty:>8,}  ({empty/n*100:.1f}%)")
    print(f"  Non-empty carts    : {nonempty:>8,}  ({nonempty/n*100:.1f}%)")
    print(f"  Total cart items   : {total_items:>8,}")
    print(f"  Avg cart size      : {avg_size:>8.2f}  (inc. empty)")
    print(f"  TTL distribution:")
    for label, cnt in sorted(ttl_dist.items()):
        print(f"    {label:<22} {cnt:>7,}  ({cnt/n*100:.1f}%)")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  sessions.csv       : {size_mb:.1f} MB")
    print(f"  Elapsed            : {elapsed:.2f}s")
    print(f"{'═' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart active sessions with cart state",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",   type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    generate(count=args.count, seed=args.seed, out_dir=args.out_dir, verbose=not args.quiet)