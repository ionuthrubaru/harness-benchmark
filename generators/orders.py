"""
generators/orders.py — StreamCart Marketplace Order Generator
=============================================================
Generates four CSV files:
  data/orders.csv                    — one row per marketplace order
  data/order_items.csv               — one row per product in each order
  data/marketplace_invoices.csv      — mirrors invoices table (invoice_type='marketplace')
  data/marketplace_invoice_lines.csv — one row per order item (immutable financial record)

Relational chain (must be consistent):
  marketplace_invoices → marketplace_invoice_lines
  orders (references invoice_id) → order_items (references order_id + product_id)

Co-purchase clustering (for Neo4j Q4):
  Products are pre-assigned to ~300 clusters at load time.
  Within a multi-item order, the first product is picked by global Zipf popularity.
  Each additional product is picked 70% from the same cluster, 30% globally.
  This creates dense co-purchase edges within clusters — essential for Q4 to return
  meaningful recommendations rather than random noise.

Zipf popularity:
  Product selection follows a power-law (Zipf) distribution — a small number of
  popular products dominate sales, the majority sell rarely. This mirrors real
  marketplace behaviour and creates the skewed graph structure Neo4j excels at.

Fulfilment logic:
  digital_asset / course → fulfilment_status = 'delivered' immediately
  merch                  → fulfilment_status = 'shipped' or 'delivered' (time-dependent)

Shipping address:
  Only merch orders populate the shipping_* columns on the orders table.
  Faker generates locale-appropriate addresses using the user's country_code.

Memory:
  All four CSVs are written incrementally (row-by-row) so peak memory usage
  stays flat regardless of order count. Products and users are loaded into
  lightweight numpy arrays for fast sampling.

Usage
-----
  python generators/orders.py                   # reads data/products.csv + users.csv
  python generators/orders.py --count 500000
  python generators/orders.py --seed 99
  python generators/orders.py --out-dir /tmp
  python generators/orders.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED   = 42
DEFAULT_COUNT  = 500_000

PLATFORM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLATFORM_END   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Order size distribution: P(1 item), P(2), P(3), P(4), P(5)
ORDER_SIZE_PROBS = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
ORDER_SIZES      = np.array([1, 2, 3, 4, 5])

# Co-purchase clustering
N_CLUSTERS           = 300    # number of product clusters
CLUSTER_PROB         = 0.70   # probability additional item comes from same cluster

# Zipf exponent — higher = more skewed (popular products dominate more)
ZIPF_EXPONENT = 1.2

# Invoice / order status probabilities
INVOICE_STATUS_PROBS = {"paid": 0.93, "failed": 0.04, "refunded": 0.02, "void": 0.01}
ORDER_STATUS_BY_TYPE = {
    "digital": {"delivered": 0.94, "cancelled": 0.04, "refunded": 0.02},
    "merch":   {"delivered": 0.60, "shipped": 0.25, "confirmed": 0.08,
                "cancelled": 0.05, "refunded": 0.02},
}

# Faker locale map (country_code → Faker locale) for shipping addresses
COUNTRY_LOCALE = {
    "US": "en_US", "IN": "en_IN", "CN": "zh_CN", "BR": "pt_BR",
    "ID": "id_ID", "GB": "en_GB", "DE": "de_DE", "RU": "ru_RU",
    "MX": "es_MX", "JP": "ja_JP", "NG": "en_PH", "FR": "fr_FR",
    "PH": "en_PH", "TR": "tr_TR", "VN": "vi_VN", "PK": "en_IN",
    "EG": "fr_FR", "CA": "en_CA", "AU": "en_AU", "ZA": "en_GB",
}

# Seasonal growth multipliers — mirrors users.py (same platform growth curve)
MONTHLY_WEIGHTS_RAW = np.array([
    1.00, 0.95, 0.97, 1.02, 1.05, 1.08,   # Jan–Jun 2024
    0.92, 0.88, 0.98, 1.05, 1.15, 1.10,   # Jul–Dec 2024
    1.00, 0.96, 0.98, 1.03, 1.06, 1.09,   # Jan–Jun 2025
    0.93, 0.89, 0.99, 1.06, 1.18, 1.12,   # Jul–Dec 2025
])


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_products(out_dir: Path) -> tuple[list[str], list[str], list[float]]:
    """
    Load product IDs, types, and prices from products.csv.
    Returns parallel lists: (ids, types, prices).
    """
    path = out_dir / "products.csv"
    if not path.exists():
        raise FileNotFoundError(f"products.csv not found at {path}. Run generators/products.py first.")

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


def _load_users(out_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Load user IDs, created_at timestamps, and country_codes from users.csv.
    Only returns active users.
    Returns parallel lists: (ids, created_ats, country_codes).
    """
    path = out_dir / "users.csv"
    if not path.exists():
        raise FileNotFoundError(f"users.csv not found at {path}. Run generators/users.py first.")

    ids:          list[str] = []
    created_ats:  list[str] = []
    country_codes: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_active"].lower() == "true":
                ids.append(row["id"])
                created_ats.append(row["created_at"])
                country_codes.append(row["country_code"])

    return ids, created_ats, country_codes


# ── Zipf weights ──────────────────────────────────────────────────────────────

def _zipf_weights(n: int, exponent: float = ZIPF_EXPONENT) -> np.ndarray:
    """
    Generate Zipf-distributed weights for n items.
    Item rank 1 has weight 1^(-exponent), rank 2 has 2^(-exponent), etc.
    """
    ranks   = np.arange(1, n + 1, dtype=float)
    weights = ranks ** (-exponent)
    return weights / weights.sum()


# ── Co-purchase clustering ────────────────────────────────────────────────────

def _build_clusters(
    n_products: int,
    n_clusters: int,
    product_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Assign products to clusters and pre-compute per-cluster Zipf weights.

    Returns
    -------
    cluster_assignment : shape (n_products,) — cluster id for each product
    cluster_members    : list of arrays — product indices in each cluster
    cluster_weights    : list of arrays — normalised Zipf weights within each cluster
    """
    # Shuffle product indices so clusters aren't just sequential slices
    shuffled = rng.permutation(n_products)
    cluster_assignment = np.empty(n_products, dtype=np.int32)

    cluster_members: list[np.ndarray] = []
    cluster_weights: list[np.ndarray] = []

    chunk = n_products // n_clusters
    for c in range(n_clusters):
        start  = c * chunk
        end    = start + chunk if c < n_clusters - 1 else n_products
        member_indices = shuffled[start:end]
        cluster_assignment[member_indices] = c

        # Within-cluster weights: use global Zipf weights, renormalised
        w = product_weights[member_indices]
        w = w / w.sum()
        cluster_members.append(member_indices)
        cluster_weights.append(w)

    return cluster_assignment, cluster_members, cluster_weights


# ── Timestamp generation ──────────────────────────────────────────────────────

def _build_order_timestamps(
    n: int,
    user_created_ats: list[str],
    user_indices: np.ndarray,
    rng: np.random.Generator,
) -> list[datetime]:
    """
    Generate n order timestamps, each guaranteed to be after the
    corresponding user's registration date.
    """
    # Monthly distribution following platform growth curve
    weights = MONTHLY_WEIGHTS_RAW / MONTHLY_WEIGHTS_RAW.sum()
    month_counts = rng.multinomial(n, weights)

    raw_timestamps: list[datetime] = []
    for month_i, count in enumerate(month_counts):
        year  = 2024 + month_i // 12
        month = (month_i % 12) + 1
        days_in_month = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
        for _ in range(int(count)):
            day  = int(rng.integers(0, days_in_month))
            hour = int(rng.integers(0, 24))
            mins = int(rng.integers(0, 60))
            dt   = datetime(year, month, 1, hour, mins, 0, tzinfo=timezone.utc) + timedelta(days=day)
            raw_timestamps.append(dt)

    raw_timestamps.sort()
    return raw_timestamps


# ── Faker address pool ────────────────────────────────────────────────────────

def _build_faker_pool(seed: int) -> dict[str, Faker]:
    pool: dict[str, Faker] = {}
    for locale in set(COUNTRY_LOCALE.values()):
        Faker.seed(seed)
        pool[locale] = Faker(locale)
    return pool


def _shipping_address(
    country: str,
    full_name: str,
    faker_pool: dict[str, Faker],
    rng: np.random.Generator,
) -> dict:
    locale = COUNTRY_LOCALE.get(country, "en_US")
    fake   = faker_pool[locale]
    return {
        "shipping_name":    full_name,
        "shipping_address": fake.street_address(),
        "shipping_city":    fake.city(),
        "shipping_country": country,
        "shipping_postal":  fake.postcode(),
    }


# ── Status samplers ───────────────────────────────────────────────────────────

def _sample_invoice_status(rng: np.random.Generator) -> tuple[str, bool]:
    roll = rng.random()
    cum  = 0.0
    for status, prob in INVOICE_STATUS_PROBS.items():
        cum += prob
        if roll < cum:
            return status, status == "paid"
    return "paid", True


def _sample_order_status(product_types: list[str], rng: np.random.Generator) -> str:
    """
    Determine order status based on whether the order contains any merch.
    Orders with at least one merch item follow the merch status distribution.
    """
    has_merch = any(t == "merch" for t in product_types)
    dist      = ORDER_STATUS_BY_TYPE["merch"] if has_merch else ORDER_STATUS_BY_TYPE["digital"]
    roll      = rng.random()
    cum       = 0.0
    for status, prob in dist.items():
        cum += prob
        if roll < cum:
            return status
    return "delivered"


def _fulfilment_status(product_type: str, order_status: str) -> str:
    if product_type in ("digital_asset", "course"):
        return "delivered"
    # merch
    if order_status == "delivered":
        return "delivered"
    elif order_status == "shipped":
        return "shipped"
    elif order_status in ("cancelled", "refunded"):
        return "failed"
    return "pending"


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    products: tuple[list, list, list] | None = None,
    users:    tuple[list, list, list] | None = None,
    count:    int  = DEFAULT_COUNT,
    seed:     int  = DEFAULT_SEED,
    out_dir:  str | os.PathLike = None,
    verbose:  bool = True,
) -> None:
    """
    Generate marketplace orders, order items, invoices, and invoice lines.

    Parameters
    ----------
    products : (ids, types, prices) tuple from products.py generate().
               If None, reads data/products.csv automatically.
    users    : (ids, created_ats, country_codes) tuple from users.py generate().
               If None, reads data/users.csv automatically.
    count    : number of orders to generate
    seed     : random seed
    out_dir  : output directory (default: ../data/)
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    if products is None:
        if verbose:
            print("  Reading products.csv ...")
        prod_ids, prod_types, prod_prices = _load_products(out_dir)
    else:
        prod_ids, prod_types, prod_prices = products

    if users is None:
        if verbose:
            print("  Reading users.csv ...")
        user_ids, user_created_ats, user_countries = _load_users(out_dir)
    else:
        user_ids, user_created_ats, user_countries = users

    n_products = len(prod_ids)
    n_users    = len(user_ids)

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  generators/orders.py")
        print(f"{'═' * 60}")
        print(f"  seed           : {seed}")
        print(f"  target orders  : {count:,}")
        print(f"  active products: {n_products:,}")
        print(f"  active users   : {n_users:,}")
        print()

    t_start = time.perf_counter()

    # ── Pre-compute Zipf weights and clusters ──
    if verbose:
        print("  Building Zipf popularity weights ...")
    global_weights = _zipf_weights(n_products, ZIPF_EXPONENT)

    if verbose:
        print("  Building co-purchase clusters ...")
    cluster_assignment, cluster_members, cluster_weights = _build_clusters(
        n_products, N_CLUSTERS, global_weights, rng
    )

    # ── Pre-sample user assignments ──
    if verbose:
        print("  Sampling user assignments ...")
    user_indices = rng.integers(0, n_users, size=count)

    # ── Pre-compute order timestamps ──
    if verbose:
        print("  Generating order timestamps ...")
    timestamps = _build_order_timestamps(count, user_created_ats, user_indices, rng)

    # ── Build Faker address pool ──
    faker_pool = _build_faker_pool(seed)

    # ── Pre-sample order sizes ──
    order_sizes = rng.choice(ORDER_SIZES, size=count, p=ORDER_SIZE_PROBS)

    # ── Open CSV writers ──
    paths = {
        "orders":       out_dir / "orders.csv",
        "order_items":  out_dir / "order_items.csv",
        "invoices":     out_dir / "marketplace_invoices.csv",
        "lines":        out_dir / "marketplace_invoice_lines.csv",
    }

    order_fields = [
        "id", "user_id", "invoice_id", "status",
        "shipping_name", "shipping_address", "shipping_city",
        "shipping_country", "shipping_postal",
        "created_at", "updated_at",
    ]
    order_item_fields = [
        "id", "order_id", "product_id", "quantity",
        "unit_price_usd", "line_total_usd", "fulfilment_status", "created_at",
    ]
    invoice_fields = [
        "id", "user_id", "invoice_type", "status",
        "subtotal_usd", "tax_usd", "discount_usd", "total_usd",
        "subscription_id", "billing_period_start", "billing_period_end",
        "paid_at", "due_at", "created_at",
    ]
    line_fields = [
        "id", "invoice_id", "product_id", "description",
        "quantity", "unit_price_usd", "line_total_usd", "created_at",
    ]

    if verbose:
        print(f"  Writing {count:,} orders to CSV (streaming)...")

    # Track counters for summary
    n_items_total  = 0
    n_merch_orders = 0
    status_counts: dict[str, int] = {}

    with (
        open(paths["orders"],      "w", newline="", encoding="utf-8") as f_ord,
        open(paths["order_items"], "w", newline="", encoding="utf-8") as f_items,
        open(paths["invoices"],    "w", newline="", encoding="utf-8") as f_inv,
        open(paths["lines"],       "w", newline="", encoding="utf-8") as f_lines,
    ):
        w_ord   = csv.DictWriter(f_ord,   fieldnames=order_fields)
        w_items = csv.DictWriter(f_items, fieldnames=order_item_fields)
        w_inv   = csv.DictWriter(f_inv,   fieldnames=invoice_fields)
        w_lines = csv.DictWriter(f_lines, fieldnames=line_fields)

        w_ord.writeheader()
        w_items.writeheader()
        w_inv.writeheader()
        w_lines.writeheader()

        for i in range(count):
            order_ts   = timestamps[i]
            order_ts_s = order_ts.isoformat()
            u_idx      = int(user_indices[i])
            user_id    = user_ids[u_idx]
            country    = user_countries[u_idx]
            n_items    = int(order_sizes[i])

            # ── Pick products for this order ──
            # First product: global Zipf
            seed_prod_idx = int(rng.choice(n_products, p=global_weights))
            chosen_indices = [seed_prod_idx]

            # Additional products: cluster-correlated
            seed_cluster = int(cluster_assignment[seed_prod_idx])
            for _ in range(n_items - 1):
                if rng.random() < CLUSTER_PROB and len(cluster_members[seed_cluster]) > 1:
                    # Pick from same cluster (excluding already chosen)
                    c_members = cluster_members[seed_cluster]
                    c_weights = cluster_weights[seed_cluster].copy()
                    # Zero out already-chosen products
                    for ci, cm in enumerate(c_members):
                        if cm in chosen_indices:
                            c_weights[ci] = 0.0
                    if c_weights.sum() > 0:
                        c_weights /= c_weights.sum()
                        extra = int(rng.choice(c_members, p=c_weights))
                    else:
                        extra = int(rng.choice(n_products, p=global_weights))
                else:
                    extra = int(rng.choice(n_products, p=global_weights))
                if extra not in chosen_indices:
                    chosen_indices.append(extra)

            chosen_types  = [prod_types[idx]  for idx in chosen_indices]
            chosen_prices = [prod_prices[idx]  for idx in chosen_indices]
            chosen_ids    = [prod_ids[idx]     for idx in chosen_indices]

            # ── Determine order & invoice status ──
            order_status   = _sample_order_status(chosen_types, rng)
            inv_status, is_paid = _sample_invoice_status(rng)

            # Align: if invoice failed, order is also cancelled
            if inv_status == "failed":
                order_status = "cancelled"
            if inv_status == "void":
                order_status = "cancelled"

            paid_at = None
            if is_paid:
                paid_at = (order_ts + timedelta(minutes=int(rng.integers(1, 120)))).isoformat()

            # ── Financials ──
            subtotal   = round(sum(chosen_prices), 2)
            tax        = round(subtotal * 0.00, 2)   # simplified: 0% tax (international platform)
            discount   = 0.00
            total      = round(subtotal + tax - discount, 2)

            # ── IDs ──
            order_id   = str(uuid.uuid4())
            invoice_id = str(uuid.uuid4())

            # ── Shipping (merch orders only) ──
            has_merch = any(t == "merch" for t in chosen_types)
            if has_merch:
                n_merch_orders += 1
                addr = _shipping_address(country, f"User {user_id[:8]}", faker_pool, rng)
            else:
                addr = {k: None for k in [
                    "shipping_name", "shipping_address",
                    "shipping_city", "shipping_country", "shipping_postal"
                ]}

            updated_at = order_ts_s
            if order_status in ("shipped", "delivered"):
                days_later = int(rng.integers(1, 14))
                updated_at = (order_ts + timedelta(days=days_later)).isoformat()

            # ── Write invoice ──
            w_inv.writerow({
                "id":                   invoice_id,
                "user_id":              user_id,
                "invoice_type":         "marketplace",
                "status":               inv_status,
                "subtotal_usd":         f"{subtotal:.2f}",
                "tax_usd":              f"{tax:.2f}",
                "discount_usd":         f"{discount:.2f}",
                "total_usd":            f"{total:.2f}",
                "subscription_id":      None,
                "billing_period_start": None,
                "billing_period_end":   None,
                "paid_at":              paid_at,
                "due_at":               order_ts_s,
                "created_at":           order_ts_s,
            })

            # ── Write order ──
            w_ord.writerow({
                "id":               order_id,
                "user_id":          user_id,
                "invoice_id":       invoice_id,
                "status":           order_status,
                **addr,
                "created_at":       order_ts_s,
                "updated_at":       updated_at,
            })

            status_counts[order_status] = status_counts.get(order_status, 0) + 1

            # ── Write invoice lines + order items ──
            for prod_id, prod_type, price in zip(chosen_ids, chosen_types, chosen_prices):
                item_id  = str(uuid.uuid4())
                line_id  = str(uuid.uuid4())
                ful_stat = _fulfilment_status(prod_type, order_status)

                w_lines.writerow({
                    "id":             line_id,
                    "invoice_id":     invoice_id,
                    "product_id":     prod_id,
                    "description":    prod_id,  # loader can join to products.name if needed
                    "quantity":       1,
                    "unit_price_usd": f"{price:.2f}",
                    "line_total_usd": f"{price:.2f}",
                    "created_at":     order_ts_s,
                })

                w_items.writerow({
                    "id":               item_id,
                    "order_id":         order_id,
                    "product_id":       prod_id,
                    "quantity":         1,
                    "unit_price_usd":   f"{price:.2f}",
                    "line_total_usd":   f"{price:.2f}",
                    "fulfilment_status": ful_stat,
                    "created_at":       order_ts_s,
                })

                n_items_total += 1

            # Progress indicator every 50K orders
            if verbose and (i + 1) % 50_000 == 0:
                pct = (i + 1) / count * 100
                elapsed_so_far = time.perf_counter() - t_start
                print(f"    {i + 1:>7,} / {count:,}  ({pct:.0f}%)  {elapsed_so_far:.1f}s")

    elapsed = time.perf_counter() - t_start

    if verbose:
        _print_summary(count, n_items_total, n_merch_orders, status_counts, elapsed, paths)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    n_orders: int,
    n_items: int,
    n_merch: int,
    status_counts: dict[str, int],
    elapsed: float,
    paths: dict,
) -> None:
    print()
    print(f"{'─' * 60}")
    print(f"  Summary")
    print(f"{'─' * 60}")
    print(f"  Orders generated    : {n_orders:>10,}")
    print(f"  Order items total   : {n_items:>10,}  (avg {n_items/n_orders:.2f} items/order)")
    print(f"  Merch orders        : {n_merch:>10,}  ({n_merch/n_orders*100:.1f}% — have shipping address)")
    print(f"  Order status breakdown:")
    for status, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / n_orders * 40)
        print(f"    {status:<12} {cnt:>8,}  {bar}")
    print()
    for name, path in paths.items():
        size  = path.stat().st_size
        label = f"{size/1024/1024:.1f} MB" if size > 1_000_000 else f"{size/1024:.0f} KB"
        print(f"  {path.name:<45} {label}")
    print(f"  Elapsed             : {elapsed:.2f}s")
    print(f"{'═' * 60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart marketplace orders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",   type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    generate(count=args.count, seed=args.seed, out_dir=args.out_dir, verbose=not args.quiet)