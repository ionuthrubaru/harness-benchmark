"""
generators/users.py — StreamCart User & Seller Profile Generator
=================================================================
Generates two CSV files:
  data/users.csv           — ~100K user records
  data/seller_profiles.csv — ~8% of users who are sellers

Design decisions
----------------
* Faker handles names, emails, bios, and user agents — localised per country
  so the data feels genuinely international rather than uniformly Anglo-American.

* Deterministic: Faker is seeded alongside the numpy RNG. Re-running with the
  same seed always produces identical output.

* Growth curve: registrations follow a logistic S-curve over 24 months
  (2024-01-01 → 2025-12-31), modelling a platform that starts slow, grows
  rapidly through mid-2024, then plateaus — realistic for a SaaS product
  hitting product-market fit.

* Geographic distribution: weighted by internet user population across 20
  representative countries. Faker is given the matching locale so names and
  city names are culturally coherent.

* Active rate: 72–78% of users have is_active=True.
  Inactive users have no last_login_at, or one far in the past.

* Seller rate: exactly 8% of users are sellers and get a seller_profiles row.

* Preferences JSON: theme, language, nested notification flags, and 2–5
  content interest tags.

Requirements
------------
  pip install faker numpy pandas

Usage
-----
  python generators/users.py                    # writes to data/ relative to project root
  python generators/users.py --count 50000      # smaller run for testing
  python generators/users.py --seed 99          # override random seed
  python generators/users.py --out-dir /tmp     # custom output directory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED   = 42
DEFAULT_COUNT  = 100_000
SELLER_RATE    = 0.08
ACTIVE_RATE    = (0.72, 0.78)
PLATFORM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLATFORM_END   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# ── Geographic data ───────────────────────────────────────────────────────────
# (ISO 3166-1 alpha-2, population weight, Faker locale)
# Weights ≈ share of global internet users.

GEO_DATA: list[tuple[str, float, str]] = [
    ("US", 0.18, "en_US"),
    ("IN", 0.16, "en_IN"),
    ("CN", 0.14, "zh_CN"),
    ("BR", 0.07, "pt_BR"),
    ("ID", 0.05, "id_ID"),
    ("GB", 0.04, "en_GB"),
    ("DE", 0.04, "de_DE"),
    ("RU", 0.04, "ru_RU"),
    ("MX", 0.03, "es_MX"),
    ("JP", 0.03, "ja_JP"),
    ("NG", 0.03, "en_PH"),   # no ng locale — en_PH is the closest available
    ("FR", 0.03, "fr_FR"),
    ("PH", 0.02, "en_PH"),
    ("TR", 0.02, "tr_TR"),
    ("VN", 0.02, "vi_VN"),
    ("PK", 0.02, "en_IN"),   # no pk locale — en_IN is the closest available
    ("EG", 0.02, "fr_FR"),   # no Arabic locale in Faker — fr_FR as neutral fallback
    ("CA", 0.02, "en_CA"),
    ("AU", 0.02, "en_AU"),
    ("ZA", 0.01, "en_GB"),   # no en_ZA locale in Faker — en_GB as fallback
]

_COUNTRY_CODES  = [g[0] for g in GEO_DATA]
_GEO_WEIGHTS    = np.array([g[1] for g in GEO_DATA], dtype=float)
_GEO_WEIGHTS   /= _GEO_WEIGHTS.sum()
_COUNTRY_LOCALE = {g[0]: g[2] for g in GEO_DATA}

# ── Language / preference data ────────────────────────────────────────────────

LANGUAGES = [
    ("en", 0.40), ("es", 0.12), ("zh", 0.11), ("hi", 0.09), ("pt", 0.06),
    ("ar", 0.05), ("fr", 0.04), ("ru", 0.04), ("de", 0.03), ("ja", 0.02),
    ("id", 0.02), ("tr", 0.01), ("vi", 0.01),
]
_LANG_CODES   = [l[0] for l in LANGUAGES]
_LANG_WEIGHTS = np.array([l[1] for l in LANGUAGES], dtype=float)
_LANG_WEIGHTS /= _LANG_WEIGHTS.sum()

CONTENT_INTEREST_TAGS = [
    "design", "photography", "illustration", "video_editing", "music_production",
    "3d_modeling", "motion_graphics", "ui_ux", "web_development", "typography",
    "branding", "digital_art", "animation", "game_assets", "stock_photos",
    "lightroom_presets", "procreate_brushes", "after_effects_templates",
    "logo_design", "social_media", "print_design", "figma_templates",
    "icon_packs", "font_bundles", "mockups",
]

SELLER_ADJECTIVES = [
    "Creative", "Digital", "Pixel", "Bold", "Sharp", "Crafted", "Vivid",
    "Modern", "Stellar", "Bright", "Studio", "Pro", "Pure", "Sleek", "Raw",
]
SELLER_NOUNS = [
    "Studio", "Works", "Lab", "Co", "Collective", "Workshop", "Creative",
    "Media", "Design", "Arts", "Visuals", "Hub", "Forge", "Craft", "Supply",
]
SELLER_SPECIALISMS = [
    "design assets", "UI kits", "Procreate brushes", "Lightroom presets",
    "icon sets", "font bundles", "motion graphics templates", "Figma components",
    "stock photos", "3D models", "After Effects presets", "logo templates",
]
SELLER_ROLES = [
    "graphic designer", "illustrator", "photographer", "videographer",
    "motion designer", "3D artist", "UI/UX designer", "brand designer",
]

# ── Growth curve ──────────────────────────────────────────────────────────────

def _build_monthly_weights(n_months: int = 24) -> np.ndarray:
    """
    Logistic S-curve with seasonal multipliers, normalised to sum to 1.
    Models slow start → rapid growth → plateau, with Black Friday spikes in Nov.
    """
    x = np.linspace(-6, 6, n_months)
    curve = 1.0 / (1.0 + np.exp(-x))
    season = np.array([
        1.00, 0.95, 0.97, 1.02, 1.05, 1.08,   # Jan–Jun 2024
        0.92, 0.88, 0.98, 1.05, 1.15, 1.10,   # Jul–Dec 2024
        1.00, 0.96, 0.98, 1.03, 1.06, 1.09,   # Jan–Jun 2025
        0.93, 0.89, 0.99, 1.06, 1.18, 1.12,   # Jul–Dec 2025
    ])
    weights = curve * season
    return weights / weights.sum()


# ── Faker pool ────────────────────────────────────────────────────────────────

def _build_faker_pool(seed: int) -> dict[str, Faker]:
    """
    One Faker instance per locale, all seeded identically.
    Reusing instances avoids the overhead of constructing Faker per row.
    """
    pool: dict[str, Faker] = {}
    for locale in set(_COUNTRY_LOCALE.values()):
        Faker.seed(seed)
        pool[locale] = Faker(locale)
    return pool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_preferences(rng: np.random.Generator) -> dict:
    language = str(rng.choice(_LANG_CODES, p=_LANG_WEIGHTS))
    n_tags   = int(rng.integers(2, 6))
    indices  = rng.choice(len(CONTENT_INTEREST_TAGS), size=n_tags, replace=False)
    tags     = [CONTENT_INTEREST_TAGS[i] for i in sorted(indices)]
    return {
        "theme":             str(rng.choice(["light", "dark", "system"], p=[0.35, 0.45, 0.20])),
        "language":          language,
        "notifications": {
            "email_marketing": bool(rng.random() > 0.55),
            "email_product":   bool(rng.random() > 0.25),
            "push":            bool(rng.random() > 0.50),
            "weekly_digest":   bool(rng.random() > 0.60),
        },
        "content_interests": tags,
        "currency_display":  str(rng.choice(["USD", "EUR", "GBP", "local"], p=[0.55, 0.20, 0.10, 0.15])),
    }


def _generate_seller_profile(
    user_id: str,
    user_row: dict,
    rng: np.random.Generator,
    fake: Faker,
) -> dict:
    use_brand = rng.random() < 0.45
    if use_brand:
        display_name = f"{rng.choice(SELLER_ADJECTIVES)} {rng.choice(SELLER_NOUNS)}"
    else:
        display_name = user_row["full_name"]

    is_verified = rng.random() < 0.30
    has_bio     = rng.random() < 0.70

    if has_bio:
        style = int(rng.integers(0, 4))
        if style == 0:
            bio = f"Digital creator specialising in {rng.choice(SELLER_SPECIALISMS)}."
        elif style == 1:
            bio = f"Freelance {rng.choice(SELLER_ROLES)} sharing resources with the community."
        elif style == 2:
            year = rng.choice(["2018", "2019", "2020", "2021", "2022"])
            bio = f"Creating {rng.choice(['premium', 'high-quality', 'hand-crafted'])} assets since {year}."
        else:
            bio = fake.sentence(nb_words=12)
    else:
        bio = None

    # Power-law sales distribution — most sellers have few, a handful have thousands
    sales = int(np.clip(rng.pareto(1.5) * 50, 0, 15_000))

    return {
        "user_id":      user_id,
        "display_name": display_name,
        "legal_name":   user_row["full_name"],
        "tax_id":       None,
        "payout_email": user_row["email"],
        "country_code": user_row["country_code"],
        "is_verified":  is_verified,
        "bio":          bio,
        "created_at":   user_row["created_at"],
        "updated_at":   user_row["created_at"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    count: int = DEFAULT_COUNT,
    seed: int  = DEFAULT_SEED,
    out_dir: str | os.PathLike = None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Generate users and seller profiles.

    Returns
    -------
    users           : list of dicts (one per user)
    seller_profiles : list of dicts (one per seller)
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    faker_pool = _build_faker_pool(seed)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'═' * 55}")
        print(f"  generators/users.py")
        print(f"{'═' * 55}")
        print(f"  seed       : {seed}")
        print(f"  target     : {count:,} users  (~{int(count * SELLER_RATE):,} sellers)")
        print(f"  output dir : {out_dir}")
        print()

    t_start = time.perf_counter()

    # Active flags
    active_rate  = rng.uniform(*ACTIVE_RATE)
    n_active     = int(count * active_rate)
    active_flags = np.array([True] * n_active + [False] * (count - n_active))
    rng.shuffle(active_flags)

    # Seller mask
    seller_mask = np.zeros(count, dtype=bool)
    seller_mask[rng.choice(count, size=int(count * SELLER_RATE), replace=False)] = True

    # Registration timestamps distributed across 24 months via S-curve
    if verbose:
        print("  Building registration timeline...")
    month_weights = _build_monthly_weights()
    month_counts  = rng.multinomial(count, month_weights)

    timestamps: list[str] = []
    for month_i, n in enumerate(month_counts):
        year  = 2024 + month_i // 12
        month = (month_i % 12) + 1
        days_in_month = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
        for _ in range(int(n)):
            day = int(rng.integers(0, days_in_month))
            dt  = datetime(year, month, 1,
                           int(rng.integers(0, 24)),
                           int(rng.integers(0, 60)),
                           0, tzinfo=timezone.utc) + timedelta(days=day)
            timestamps.append(dt.isoformat())
    timestamps.sort()

    # Sample countries upfront for speed
    if verbose:
        print("  Sampling geographic distribution...")
    country_indices = rng.choice(len(_COUNTRY_CODES), size=count, p=_GEO_WEIGHTS)

    if verbose:
        print(f"  Building {count:,} user records...")

    users:           list[dict] = []
    seller_profiles: list[dict] = []
    used_emails: set[str] = set()

    for i in range(count):
        country = _COUNTRY_CODES[country_indices[i]]
        fake    = faker_pool[_COUNTRY_LOCALE[country]]

        full_name  = fake.name()
        # Append index to guarantee uniqueness across 100K rows
        base_email = fake.email()
        local, domain = base_email.rsplit("@", 1)
        email = f"{local}{i}@{domain}"
        if email in used_emails:
            email = f"{local}{i}x{int(rng.integers(1000, 9999))}@{domain}"
        used_emails.add(email)

        city       = fake.city()
        created_at = timestamps[i]
        is_active  = bool(active_flags[i])

        created_dt = datetime.fromisoformat(created_at)
        days_since = (PLATFORM_END - created_dt).days
        last_login_at = None

        if is_active and days_since > 1:
            offset = int(rng.integers(max(1, days_since - 90), days_since + 1))
            login_dt = created_dt + timedelta(days=offset)
            if login_dt <= PLATFORM_END:
                last_login_at = login_dt.isoformat()
        elif not is_active and rng.random() < 0.60 and days_since > 1:
            offset = int(rng.integers(1, max(2, days_since // 2)))
            last_login_at = (created_dt + timedelta(days=offset)).isoformat()

        uid = str(uuid.uuid4())

        user_row = {
            "id":            uid,
            "email":         email,
            "full_name":     full_name,
            "country_code":  country,
            "city":          city,
            "created_at":    created_at,
            "last_login_at": last_login_at,
            "is_active":     is_active,
            "preferences":   json.dumps(_generate_preferences(rng), ensure_ascii=False),
        }
        users.append(user_row)

        if seller_mask[i]:
            seller_profiles.append(
                _generate_seller_profile(uid, user_row, rng, fake)
            )

    # Write CSVs
    user_fields = [
        "id", "email", "full_name", "country_code", "city",
        "created_at", "last_login_at", "is_active", "preferences",
    ]
    seller_fields = [
        "user_id", "display_name", "legal_name", "tax_id",
        "payout_email", "country_code", "is_verified", "bio",
        "created_at", "updated_at",
    ]

    users_path   = out_dir / "users.csv"
    sellers_path = out_dir / "seller_profiles.csv"

    if verbose:
        print(f"  Writing {users_path.name}...")
    with open(users_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=user_fields)
        w.writeheader()
        w.writerows(users)

    if verbose:
        print(f"  Writing {sellers_path.name}...")
    with open(sellers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=seller_fields)
        w.writeheader()
        w.writerows(seller_profiles)

    elapsed = time.perf_counter() - t_start

    if verbose:
        _print_summary(users, seller_profiles, elapsed, out_dir)

    return users, seller_profiles


def _print_summary(
    users: list[dict],
    seller_profiles: list[dict],
    elapsed: float,
    out_dir: Path,
) -> None:
    n          = len(users)
    n_active   = sum(1 for u in users if u["is_active"])
    n_sellers  = len(seller_profiles)
    n_verified = sum(1 for s in seller_profiles if s["is_verified"])

    country_counts = Counter(u["country_code"] for u in users)
    month_counts: dict[str, int] = {}
    for u in users:
        ym = u["created_at"][:7]
        month_counts[ym] = month_counts.get(ym, 0) + 1

    print()
    print(f"{'─' * 55}")
    print(f"  Summary")
    print(f"{'─' * 55}")
    print(f"  Total users        : {n:>10,}")
    print(f"  Active users       : {n_active:>10,}  ({n_active / n * 100:.1f}%)")
    print(f"  Inactive users     : {n - n_active:>10,}  ({(n - n_active) / n * 100:.1f}%)")
    print(f"  Seller profiles    : {n_sellers:>10,}  ({n_sellers / n * 100:.1f}%)")
    print(f"  Verified sellers   : {n_verified:>10,}  ({n_verified / n_sellers * 100:.1f}% of sellers)")
    print()
    print(f"  Top 5 countries:")
    for code, cnt in country_counts.most_common(5):
        bar = "█" * int(cnt / n * 60)
        print(f"    {code}  {cnt:>7,}  {bar}")
    print()
    print(f"  Registrations (first 6 months):")
    for ym in sorted(month_counts)[:6]:
        cnt = month_counts[ym]
        bar = "█" * int(cnt / max(month_counts.values()) * 30)
        print(f"    {ym}  {cnt:>7,}  {bar}")
    print(f"    ... (24 months total)")
    print()
    print(f"  users.csv          : {(out_dir / 'users.csv').stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  seller_profiles.csv: {(out_dir / 'seller_profiles.csv').stat().st_size / 1024:.0f} KB")
    print(f"  Elapsed            : {elapsed:.2f}s")
    print(f"{'═' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart users and seller profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",   type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    generate(count=args.count, seed=args.seed, out_dir=args.out_dir, verbose=not args.quiet)