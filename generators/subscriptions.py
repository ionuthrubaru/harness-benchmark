"""
generators/subscriptions.py — StreamCart Subscription Generator
================================================================
Generates three CSV files:
  data/subscriptions.csv               — one row per subscription period
  data/subscription_invoices.csv       — mirrors invoices table (invoice_type='subscription')
  data/subscription_invoice_lines.csv  — one line per invoice

Only paid tiers (Pro=2, Business=3) get records. Free tier users are skipped.

Tier distribution among users who subscribe:
  Pro      ~70% of paid subscribers  (35% of all users)
  Business ~30% of paid subscribers  (15% of all users)

Pricing history (from subscription_tier_pricing):
  Pro:      $14.99/month before 2024-06-01, $19.99/month from 2024-06-01
  Business: $39.99/month before 2024-06-01, $49.99/month from 2024-06-01
  Annual:   10× monthly price (equivalent to 2 months free)

Churn & re-subscription:
  ~20% of paid subscribers cancel at some point.
  Of those, ~40% re-subscribe (possibly to a different tier) after a gap.
  This generates a full subscription history per user (cancelled row + new active row),
  which makes Q1's temporal JOIN more interesting.

The partial unique index (user_id WHERE status='active') is respected:
  at most one active subscription per user at any point in time.

Usage
-----
  python generators/subscriptions.py                  # reads data/users.csv automatically
  python generators/subscriptions.py --seed 99
  python generators/subscriptions.py --out-dir /tmp
  python generators/subscriptions.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED     = 42
PLATFORM_START   = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLATFORM_END     = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Share of ALL users who get a paid subscription
PAID_SUBSCRIBER_RATE = 0.50   # 35% Pro + 15% Business = 50% of users

# Among paid subscribers, split between tiers
PRO_SHARE        = 0.70
BUSINESS_SHARE   = 0.30

# Churn: probability a paid subscriber cancels at some point
CHURN_RATE       = 0.20
# Of churned users, probability they re-subscribe
RESUBSCRIBE_RATE = 0.40

# Billing cycle distribution
MONTHLY_RATE     = 0.70
ANNUAL_RATE      = 0.30

# Annual discount: charge 10 months for 12 (2 months free)
ANNUAL_MONTHS_BILLED = 10

# Pricing history — mirrors subscription_tier_pricing seed data
PRICE_HISTORY = {
    2: [  # Pro
        (datetime(2024, 1,  1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc), 14.99),
        (datetime(2024, 6,  1, tzinfo=timezone.utc), None,                                       19.99),
    ],
    3: [  # Business
        (datetime(2024, 1,  1, tzinfo=timezone.utc), datetime(2024, 6, 1, tzinfo=timezone.utc), 39.99),
        (datetime(2024, 6,  1, tzinfo=timezone.utc), None,                                       49.99),
    ],
}

CANCEL_REASONS = [
    "Too expensive",
    "Not using it enough",
    "Found a better alternative",
    "Temporary financial reasons",
    "Project finished",
    "Switching to annual plan",
    None,  # no reason given
]

# ── Pricing lookup ────────────────────────────────────────────────────────────

def _price_at(tier_id: int, when: datetime) -> float:
    """Return the monthly price for a tier at a given point in time."""
    for valid_from, valid_to, price in PRICE_HISTORY[tier_id]:
        if when >= valid_from and (valid_to is None or when < valid_to):
            return price
    # Fallback to latest price if somehow out of range
    return PRICE_HISTORY[tier_id][-1][2]


# ── Period helpers ────────────────────────────────────────────────────────────

def _next_month(dt: datetime) -> datetime:
    """Return the same day next month (clamped to month end)."""
    month = dt.month % 12 + 1
    year  = dt.year + (1 if dt.month == 12 else 0)
    day   = min(dt.day, [31,28,31,30,31,30,31,31,30,31,30,31][month - 1])
    return dt.replace(year=year, month=month, day=day)


def _add_months(dt: datetime, n: int) -> datetime:
    for _ in range(n):
        dt = _next_month(dt)
    return dt


def _ts(dt: datetime) -> str:
    return dt.isoformat()


# ── Invoice builders ──────────────────────────────────────────────────────────

def _make_monthly_invoices(
    sub_id: str,
    user_id: str,
    tier_id: int,
    period_start: datetime,
    period_end: datetime,   # exclusive — the subscription end date
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """
    Generate one invoice + one invoice_line per billing month between
    period_start and period_end (or PLATFORM_END, whichever is earlier).
    """
    invoices: list[dict]       = []
    invoice_lines: list[dict]  = []

    tier_name = "Pro" if tier_id == 2 else "Business"
    cutoff    = min(period_end, PLATFORM_END)
    cursor    = period_start

    while cursor < cutoff:
        month_end = _next_month(cursor)
        price     = _price_at(tier_id, cursor)

        # Most subscription invoices are paid; small failure/refund rate
        roll = rng.random()
        if roll < 0.92:
            status  = "paid"
            paid_at = cursor + timedelta(hours=int(rng.integers(0, 48)))
        elif roll < 0.96:
            status  = "failed"
            paid_at = None
        else:
            status  = "refunded"
            paid_at = cursor + timedelta(hours=int(rng.integers(0, 48)))

        inv_id = str(uuid.uuid4())
        month_label = cursor.strftime("%B %Y")

        inv = {
            "id":                   inv_id,
            "user_id":              user_id,
            "invoice_type":         "subscription",
            "status":               status,
            "subtotal_usd":         f"{price:.2f}",
            "tax_usd":              "0.00",
            "discount_usd":         "0.00",
            "total_usd":            f"{price:.2f}",
            "subscription_id":      sub_id,
            "billing_period_start": _ts(cursor),
            "billing_period_end":   _ts(month_end),
            "paid_at":              _ts(paid_at) if paid_at else None,
            "due_at":               _ts(cursor),
            "created_at":           _ts(cursor),
        }
        line = {
            "id":              str(uuid.uuid4()),
            "invoice_id":      inv_id,
            "product_id":      None,
            "description":     f"{tier_name} subscription — {month_label}",
            "quantity":        1,
            "unit_price_usd":  f"{price:.2f}",
            "line_total_usd":  f"{price:.2f}",
            "created_at":      _ts(cursor),
        }

        invoices.append(inv)
        invoice_lines.append(line)
        cursor = month_end

    return invoices, invoice_lines


def _make_annual_invoice(
    sub_id: str,
    user_id: str,
    tier_id: int,
    period_start: datetime,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """
    Generate one invoice + one invoice_line for an annual billing cycle.
    Charges 10 months (2 months free discount).
    """
    tier_name    = "Pro" if tier_id == 2 else "Business"
    monthly      = _price_at(tier_id, period_start)
    subtotal     = round(monthly * 12, 2)
    discount     = round(monthly * (12 - ANNUAL_MONTHS_BILLED), 2)
    total        = round(subtotal - discount, 2)
    year_label   = period_start.strftime("%Y")

    roll = rng.random()
    if roll < 0.94:
        status  = "paid"
        paid_at = period_start + timedelta(hours=int(rng.integers(0, 48)))
    elif roll < 0.97:
        status  = "failed"
        paid_at = None
    else:
        status  = "refunded"
        paid_at = period_start + timedelta(hours=int(rng.integers(0, 48)))

    inv_id = str(uuid.uuid4())

    inv = {
        "id":                   inv_id,
        "user_id":              user_id,
        "invoice_type":         "subscription",
        "status":               status,
        "subtotal_usd":         f"{subtotal:.2f}",
        "tax_usd":              "0.00",
        "discount_usd":         f"{discount:.2f}",
        "total_usd":            f"{total:.2f}",
        "subscription_id":      sub_id,
        "billing_period_start": _ts(period_start),
        "billing_period_end":   _ts(_add_months(period_start, 12)),
        "paid_at":              _ts(paid_at) if paid_at else None,
        "due_at":               _ts(period_start),
        "created_at":           _ts(period_start),
    }
    line = {
        "id":              str(uuid.uuid4()),
        "invoice_id":      inv_id,
        "product_id":      None,
        "description":     f"{tier_name} subscription — Annual {year_label} (2 months free)",
        "quantity":        ANNUAL_MONTHS_BILLED,
        "unit_price_usd":  f"{monthly:.2f}",
        "line_total_usd":  f"{total:.2f}",
        "created_at":      _ts(period_start),
    }

    return inv, line


# ── Subscription builder ──────────────────────────────────────────────────────

def _build_subscription_history(
    user_id: str,
    user_created_at: datetime,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build the full subscription history for one user.
    Returns (subscriptions, invoices, invoice_lines).
    """
    subs:          list[dict] = []
    all_invoices:  list[dict] = []
    all_lines:     list[dict] = []

    # Subscription starts between user registration and 30 days after
    delay_days   = int(rng.integers(0, 30))
    sub_start    = user_created_at + timedelta(days=delay_days)
    if sub_start > PLATFORM_END:
        return [], [], []

    tier_id      = 2 if rng.random() < PRO_SHARE else 3
    billing      = "monthly" if rng.random() < MONTHLY_RATE else "annual"
    will_churn   = rng.random() < CHURN_RATE

    # How long does this subscription last?
    max_days     = (PLATFORM_END - sub_start).days
    if will_churn:
        # Cancel somewhere between 1 month and 80% of possible lifetime
        cancel_after_days = int(rng.integers(30, max(31, int(max_days * 0.8))))
        cancel_at         = sub_start + timedelta(days=cancel_after_days)
        sub_end           = cancel_at
    else:
        cancel_at = None
        sub_end   = PLATFORM_END

    # Period boundaries
    if billing == "annual":
        period_end = _add_months(sub_start, 12)
    else:
        period_end = _next_month(sub_start)

    sub_id = str(uuid.uuid4())
    now    = _ts(sub_start)

    sub = {
        "id":                   sub_id,
        "user_id":              user_id,
        "tier_id":              tier_id,
        "status":               "cancelled" if will_churn else "active",
        "started_at":           _ts(sub_start),
        "current_period_start": _ts(sub_start),
        "current_period_end":   _ts(period_end),
        "cancelled_at":         _ts(cancel_at) if cancel_at else None,
        "cancel_reason":        str(rng.choice(CANCEL_REASONS)) if will_churn else None,
        "billing_cycle":        billing,
        "created_at":           now,
        "updated_at":           _ts(cancel_at) if cancel_at else now,
    }
    subs.append(sub)

    # Generate invoices for this subscription's active period
    if billing == "monthly":
        invs, lines = _make_monthly_invoices(
            sub_id, user_id, tier_id, sub_start, sub_end, rng
        )
        all_invoices.extend(invs)
        all_lines.extend(lines)
    else:
        # Annual: generate one invoice per year in range
        cursor = sub_start
        while cursor < sub_end and cursor < PLATFORM_END:
            inv, line = _make_annual_invoice(sub_id, user_id, tier_id, cursor, rng)
            all_invoices.append(inv)
            all_lines.append(line)
            cursor = _add_months(cursor, 12)

    # Re-subscription after churn
    if will_churn and rng.random() < RESUBSCRIBE_RATE:
        gap_days  = int(rng.integers(14, 180))   # 2 weeks to 6 months gap
        new_start = cancel_at + timedelta(days=gap_days)

        if new_start < PLATFORM_END:
            # Possibly switch tier
            if rng.random() < 0.30:
                new_tier = 3 if tier_id == 2 else 2   # flip tier
            else:
                new_tier = tier_id

            new_billing = "monthly" if rng.random() < MONTHLY_RATE else "annual"
            new_id      = str(uuid.uuid4())
            new_period  = (_next_month(new_start) if new_billing == "monthly"
                           else _add_months(new_start, 12))
            new_now     = _ts(new_start)

            new_sub = {
                "id":                   new_id,
                "user_id":              user_id,
                "tier_id":              new_tier,
                "status":               "active",
                "started_at":           _ts(new_start),
                "current_period_start": _ts(new_start),
                "current_period_end":   _ts(new_period),
                "cancelled_at":         None,
                "cancel_reason":        None,
                "billing_cycle":        new_billing,
                "created_at":           new_now,
                "updated_at":           new_now,
            }
            subs.append(new_sub)

            if new_billing == "monthly":
                invs, lines = _make_monthly_invoices(
                    new_id, user_id, new_tier, new_start, PLATFORM_END, rng
                )
                all_invoices.extend(invs)
                all_lines.extend(lines)
            else:
                cursor = new_start
                while cursor < PLATFORM_END:
                    inv, line = _make_annual_invoice(new_id, user_id, new_tier, cursor, rng)
                    all_invoices.append(inv)
                    all_lines.append(line)
                    cursor = _add_months(cursor, 12)

    return subs, all_invoices, all_lines


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    users: list[dict] | None = None,
    seed: int = DEFAULT_SEED,
    out_dir: str | os.PathLike = None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generate subscriptions, subscription invoices, and invoice lines.

    Parameters
    ----------
    users   : pre-loaded list of user dicts (from users.py).
              If None, reads data/users.csv automatically.
    seed    : random seed
    out_dir : output directory (default: ../data/)

    Returns
    -------
    subscriptions, invoices, invoice_lines
    """
    rng = np.random.default_rng(seed)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load users if not passed in
    if users is None:
        users_path = out_dir / "users.csv"
        if not users_path.exists():
            raise FileNotFoundError(
                f"users.csv not found at {users_path}. "
                "Run generators/users.py first, or pass users list directly."
            )
        if verbose:
            print(f"  Reading {users_path} ...")
        with open(users_path, newline="", encoding="utf-8") as f:
            users = list(csv.DictReader(f))

    if verbose:
        print(f"\n{'═' * 55}")
        print(f"  generators/subscriptions.py")
        print(f"{'═' * 55}")
        print(f"  seed          : {seed}")
        print(f"  users loaded  : {len(users):,}")
        print(f"  paid sub rate : {PAID_SUBSCRIBER_RATE:.0%}  (~{int(len(users) * PAID_SUBSCRIBER_RATE):,} users)")
        print()

    t_start = time.perf_counter()

    # Decide which users get paid subscriptions
    n_paid      = int(len(users) * PAID_SUBSCRIBER_RATE)
    paid_indices = rng.choice(len(users), size=n_paid, replace=False)
    paid_set     = set(paid_indices.tolist())

    all_subs:     list[dict] = []
    all_invoices: list[dict] = []
    all_lines:    list[dict] = []

    if verbose:
        print(f"  Generating subscription histories...")

    for i, user in enumerate(users):
        if i not in paid_set:
            continue

        created_at = datetime.fromisoformat(user["created_at"])
        subs, invs, lines = _build_subscription_history(
            user_id=user["id"],
            user_created_at=created_at,
            rng=rng,
        )
        all_subs.extend(subs)
        all_invoices.extend(invs)
        all_lines.extend(lines)

    # Write CSVs
    sub_fields = [
        "id", "user_id", "tier_id", "status", "started_at",
        "current_period_start", "current_period_end",
        "cancelled_at", "cancel_reason", "billing_cycle",
        "created_at", "updated_at",
    ]
    inv_fields = [
        "id", "user_id", "invoice_type", "status",
        "subtotal_usd", "tax_usd", "discount_usd", "total_usd",
        "subscription_id", "billing_period_start", "billing_period_end",
        "paid_at", "due_at", "created_at",
    ]
    line_fields = [
        "id", "invoice_id", "product_id", "description",
        "quantity", "unit_price_usd", "line_total_usd", "created_at",
    ]

    paths = {
        "subscriptions":              out_dir / "subscriptions.csv",
        "subscription_invoices":      out_dir / "subscription_invoices.csv",
        "subscription_invoice_lines": out_dir / "subscription_invoice_lines.csv",
    }

    if verbose:
        print(f"  Writing subscriptions.csv ...")
    with open(paths["subscriptions"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sub_fields)
        w.writeheader()
        w.writerows(all_subs)

    if verbose:
        print(f"  Writing subscription_invoices.csv ...")
    with open(paths["subscription_invoices"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields)
        w.writeheader()
        w.writerows(all_invoices)

    if verbose:
        print(f"  Writing subscription_invoice_lines.csv ...")
    with open(paths["subscription_invoice_lines"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=line_fields)
        w.writeheader()
        w.writerows(all_lines)

    elapsed = time.perf_counter() - t_start

    if verbose:
        _print_summary(all_subs, all_invoices, all_lines, elapsed, paths)

    return all_subs, all_invoices, all_lines


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    subs: list[dict],
    invoices: list[dict],
    lines: list[dict],
    elapsed: float,
    paths: dict,
) -> None:
    from collections import Counter

    statuses      = Counter(s["status"]       for s in subs)
    tiers         = Counter(s["tier_id"]      for s in subs)
    cycles        = Counter(s["billing_cycle"] for s in subs)
    inv_statuses  = Counter(i["status"]       for i in invoices)

    n_users_with_subs = len(set(s["user_id"] for s in subs))

    print()
    print(f"{'─' * 55}")
    print(f"  Summary")
    print(f"{'─' * 55}")
    print(f"  Users with subscriptions : {n_users_with_subs:>8,}")
    print(f"  Total subscription rows  : {len(subs):>8,}")
    print(f"    active                 : {statuses.get('active', 0):>8,}")
    print(f"    cancelled              : {statuses.get('cancelled', 0):>8,}")
    print(f"  Tier breakdown:")
    print(f"    Pro (2)                : {tiers.get('2', 0):>8,}")
    print(f"    Business (3)           : {tiers.get('3', 0):>8,}")
    print(f"  Billing cycle:")
    print(f"    monthly                : {cycles.get('monthly', 0):>8,}")
    print(f"    annual                 : {cycles.get('annual', 0):>8,}")
    print(f"  Invoices generated       : {len(invoices):>8,}")
    print(f"    paid                   : {inv_statuses.get('paid', 0):>8,}")
    print(f"    failed                 : {inv_statuses.get('failed', 0):>8,}")
    print(f"    refunded               : {inv_statuses.get('refunded', 0):>8,}")
    print(f"  Invoice lines            : {len(lines):>8,}")
    print()
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        label   = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        print(f"  {path.name:<40} {label}")
    print(f"  Elapsed                  : {elapsed:.2f}s")
    print(f"{'═' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart subscription data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    generate(seed=args.seed, out_dir=args.out_dir, verbose=not args.quiet)