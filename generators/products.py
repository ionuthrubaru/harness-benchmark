"""
generators/products.py — StreamCart Product Generator
======================================================
Generates one CSV file:
  data/products.csv — ~50K product records

NOTE: search_vector is intentionally omitted from the CSV.
The PostgreSQL trigger (trg_products_search_vector) recomputes it
correctly on every INSERT, so there is no need to generate tsvector
strings in Python.

Distribution:
  40% digital_asset  (~20K)
  35% course         (~17.5K)
  25% merch          (~12.5K)

Price ranges:
  course        $20–$200
  digital_asset  $5–$80
  merch         $15–$120

Seller assignment:
  Each product is assigned to a seller from seller_profiles.csv.
  Assignment follows a power-law (popular sellers have more products),
  which also creates the co-purchase density needed for Neo4j (Q4).

Descriptions:
  Long-form, multi-sentence descriptions built from composable sentence
  banks per product type. This gives Elasticsearch's BM25 scorer and
  PostgreSQL's tsvector meaningful signal to rank against — short
  placeholder text would make Q5 results arbitrary.

90% of products are active (is_active=True).

Usage
-----
  python generators/products.py                  # reads data/seller_profiles.csv
  python generators/products.py --count 50000
  python generators/products.py --seed 99
  python generators/products.py --out-dir /tmp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED  = 42
DEFAULT_COUNT = 50_000
ACTIVE_RATE   = 0.90

PLATFORM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLATFORM_END   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

TYPE_DIST = {
    "digital_asset": 0.40,
    "course":        0.35,
    "merch":         0.25,
}

PRICE_RANGES = {
    "course":        (20.00,  200.00),
    "digital_asset": (5.00,   80.00),
    "merch":         (15.00,  120.00),
}

# ── Vocabulary banks ──────────────────────────────────────────────────────────
# These are used to build varied, keyword-rich product names and descriptions.
# The breadth of vocabulary directly affects how meaningful Q5 search results are.

# --- Digital Assets ---

DA_ASSET_TYPES = [
    "brush pack", "icon set", "UI kit", "font bundle", "texture pack",
    "mockup kit", "template pack", "preset collection", "pattern library",
    "illustration pack", "vector bundle", "photo pack", "LUT pack",
    "action set", "style guide", "colour palette", "graphic elements pack",
    "logo kit", "badge collection", "social media kit",
]

DA_STYLES = [
    "minimal", "bold", "retro", "modern", "hand-drawn", "geometric",
    "organic", "vintage", "flat", "isometric", "line art", "watercolour",
    "neon", "monochrome", "pastel", "grunge", "futuristic", "editorial",
    "Scandinavian", "brutalist",
]

DA_SUBJECTS = [
    "nature", "abstract", "typography", "business", "travel", "food",
    "architecture", "fashion", "technology", "lifestyle", "branding",
    "social media", "e-commerce", "podcast", "real estate", "fitness",
    "wellness", "finance", "education", "photography",
]

DA_SOFTWARE = [
    "Photoshop", "Illustrator", "Procreate", "Affinity Designer",
    "Figma", "Sketch", "Adobe XD", "InDesign", "After Effects",
    "Canva", "Lightroom", "Capture One",
]

DA_FORMATS = ["ABR", "PNG", "SVG", "AI", "EPS", "PSD", "TIFF", "PDF",
              "TTF", "OTF", "WOFF", "DNG", "CUBE", "ATN", "IDML"]

DA_RESOLUTIONS = ["2K", "4K", "8K", "300 DPI", "150 DPI", "72 DPI", "HD", "Full HD"]

DA_DESCRIPTION_SENTENCES = [
    "Crafted with obsessive attention to detail, every element in this pack has been hand-tuned for professional results.",
    "Whether you're working on brand identities, social content, or editorial layouts, these assets slot seamlessly into your workflow.",
    "Compatible with both raster and vector workflows, giving you flexibility across print and digital projects.",
    "Each file is fully layered and organised, so customising colours, sizes, and styles takes seconds rather than hours.",
    "Designed to save you time without sacrificing quality — everything is ready to use straight out of the archive.",
    "Built for creatives who refuse to compromise: every asset is production-ready and print-safe at any scale.",
    "The clean, organised folder structure means you'll find exactly what you need without hunting through hundreds of files.",
    "Includes a comprehensive style guide so you can maintain consistency across every deliverable.",
    "Tested across the most recent versions of the listed software to guarantee compatibility and stability.",
    "Perfect for freelancers, agencies, and in-house teams who need reliable, high-quality assets on tight deadlines.",
    "Inspired by current design trends while maintaining a timeless quality that won't date your work.",
    "Each brush, icon, or element has been refined through dozens of iterations to ensure it performs exactly as expected.",
    "The high-resolution exports are optimised for both Retina displays and large-format print without any quality loss.",
    "Lifetime access to all future updates means this pack keeps growing as new assets are added over time.",
    "Used by over a thousand designers worldwide to elevate client projects, personal work, and commercial campaigns.",
    "The variety within this pack means it can anchor an entire project's visual language from start to finish.",
    "Comes with a detailed PDF guide walking you through installation, best practices, and creative ideas to get you started.",
    "All assets are royalty-free for commercial use — no attribution required, no licensing headaches.",
    "The subtle details — tapered stroke endings, pixel-perfect alignment, balanced negative space — are what set this apart.",
    "An essential addition to any designer's toolkit, regardless of your specialty or experience level.",
]

# --- Courses ---

COURSE_TOPICS = [
    "graphic design", "UI/UX design", "typography", "logo design",
    "brand identity", "illustration", "digital painting", "motion graphics",
    "video editing", "colour theory", "photography", "photo editing",
    "web design", "product design", "3D modelling", "character design",
    "icon design", "layout design", "packaging design", "user research",
    "design systems", "Figma", "Adobe Illustrator", "Adobe Photoshop",
    "After Effects", "Procreate", "Blender", "Cinema 4D",
]

COURSE_LEVELS = ["beginner", "intermediate", "advanced", "all levels"]

COURSE_OUTCOMES = [
    "build a professional portfolio you can show to clients on day one",
    "land your first freelance client within weeks of completing the course",
    "design logos and brand identities from scratch with confidence",
    "create scroll-stopping social media content that grows your audience",
    "master the industry-standard tools used by top studios worldwide",
    "transition into a full-time design career from any background",
    "charge premium rates by understanding what separates good design from great",
    "develop a signature style that makes your work instantly recognisable",
    "design production-ready UI screens and hand them off to developers",
    "understand the principles that make design work, not just the tools",
    "build a complete brand identity system from strategy to final assets",
    "edit and retouch photos to a commercial standard",
    "create cinematic video content with professional colour grading",
    "design typefaces and custom lettering from concept to completion",
    "deliver motion graphics and animated content for digital platforms",
]

COURSE_FORMATS = [
    "Watch at your own pace across desktop, tablet, and mobile.",
    "Every lesson is structured around hands-on projects, not passive watching.",
    "Comes with downloadable project files, templates, and resource packs.",
    "Includes a private student community for feedback, critique, and networking.",
    "Lifetime access means you can revisit lessons whenever you need a refresher.",
    "Subtitles available in English, Spanish, Portuguese, and French.",
    "New lessons added regularly as the software and industry evolve.",
    "Each section ends with a practical assignment reviewed by the instructor.",
    "HD video lessons with detailed close-up recordings of every technique.",
    "A certificate of completion is available for your portfolio or LinkedIn profile.",
]

COURSE_DESCRIPTION_SENTENCES = [
    "This course was built because the instructor couldn't find one that actually covered the full process, from concept to client delivery.",
    "You'll learn by doing: every module is built around real-world briefs drawn from actual client projects.",
    "The curriculum has been refined over three years of teaching thousands of students across 60 countries.",
    "No prior experience is assumed — the course starts from first principles and builds methodically.",
    "By the final module, you'll have a body of work that demonstrates real, employable skill.",
    "The instructor brings fifteen years of agency experience to every lesson, sharing the shortcuts and hard lessons that most courses skip.",
    "Students consistently report landing paying clients, job offers, and freelance projects within months of enrolling.",
    "The course structure mirrors how professional projects actually run, so you're learning the workflow, not just isolated techniques.",
    "Questions are answered personally by the instructor, not outsourced to a forum moderator.",
    "The feedback system is built around actionable critique, not generic encouragement.",
    "Every technique is explained not just mechanically but conceptually, so you understand why, not just how.",
    "The software shortcuts, file organisation habits, and client communication templates alone are worth the enrolment price.",
    "Hundreds of five-star reviews from working designers, career changers, and students at major design schools.",
    "The companion workbook walks you through each project brief with space for your own notes and iterations.",
    "Designed to fit around a full-time job: most lessons are under twelve minutes, with longer project walkthroughs on weekends.",
]

INSTRUCTOR_FIRST = [
    "Alex", "Jamie", "Morgan", "Sam", "Taylor", "Jordan", "Casey", "Riley",
    "Dana", "Avery", "Quinn", "Reese", "Blake", "Cameron", "Drew",
    "Luca", "Sofia", "Marco", "Elena", "Mia",
]
INSTRUCTOR_LAST = [
    "Chen", "Park", "Rivera", "Müller", "Santos", "Okafor", "Kim",
    "Tanaka", "Rossi", "Patel", "Novak", "Andersen", "Kowalski",
    "Dubois", "Fischer", "Hernandez", "Yamamoto", "Singh", "Ferreira",
    "Johansson",
]

LANGUAGES = ["en", "es", "pt", "de", "fr", "it", "ja", "zh", "ko", "nl"]

# --- Merch ---

MERCH_PRODUCT_NAMES = [
    "T-Shirt", "Hoodie", "Sweatshirt", "Tote Bag", "Poster", "Art Print",
    "Sticker Sheet", "Enamel Pin", "Mug", "Notebook", "Desk Mat",
    "Phone Case", "Tote", "Canvas Print", "Zipper Pouch", "Cap",
    "Beanie", "Tray", "Coaster Set", "Pillow Cover",
]

MERCH_THEMES = [
    "typographic", "minimalist", "abstract art", "geometric", "botanical",
    "retro gaming", "space", "architecture", "vintage travel", "hand-lettered",
    "dark academia", "cottagecore", "Y2K", "street art", "fine art",
]

MERCH_MATERIALS = {
    "T-Shirt":      "100% organic cotton",
    "Hoodie":       "80% cotton, 20% polyester fleece",
    "Sweatshirt":   "80% cotton, 20% polyester",
    "Tote Bag":     "heavyweight canvas (12oz)",
    "Poster":       "200gsm matte paper",
    "Art Print":    "300gsm fine art paper, archival inks",
    "Sticker Sheet":"waterproof vinyl",
    "Enamel Pin":   "hard enamel, gold plating",
    "Mug":          "ceramic, dishwasher safe",
    "Notebook":     "lined, 120 pages, FSC-certified cover",
    "Desk Mat":     "microfibre surface, non-slip rubber base",
    "Phone Case":   "TPU + polycarbonate, raised edges",
    "Tote":         "heavyweight canvas (10oz)",
    "Canvas Print": "artist-grade canvas, solid pine stretcher bars",
    "Zipper Pouch": "vegan leather exterior, cotton lining",
    "Cap":          "100% cotton twill, adjustable strap",
    "Beanie":       "100% merino wool",
    "Tray":         "bamboo, food-safe lacquer",
    "Coaster Set":  "cork-backed ceramic, set of 4",
    "Pillow Cover": "100% linen, hidden zip",
}

MERCH_SIZES = {
    "T-Shirt":    ["XS", "S", "M", "L", "XL", "2XL"],
    "Hoodie":     ["XS", "S", "M", "L", "XL", "2XL"],
    "Sweatshirt": ["XS", "S", "M", "L", "XL", "2XL"],
    "Cap":        ["One Size"],
    "Beanie":     ["One Size"],
    "default":    [],
}

MERCH_COLOURS = [
    ["black", "white"], ["navy", "cream"], ["sage green", "white"],
    ["dusty pink", "charcoal"], ["terracotta", "natural"], ["black"],
    ["forest green", "black"], ["slate blue", "white"], ["burgundy", "cream"],
    ["mustard", "black", "white"],
]

MERCH_DESCRIPTION_SENTENCES = [
    "Every item is printed or manufactured to order, ensuring you receive a fresh product made specifically for you.",
    "The design is printed using industry-leading equipment to guarantee colour accuracy and long-lasting vibrancy.",
    "Ethically manufactured in a facility audited for fair wages, safe conditions, and sustainable practices.",
    "The fit has been tested across a wide range of body types to ensure the sizing chart is accurate and reliable.",
    "Ships worldwide in protective, minimal packaging designed to arrive in perfect condition.",
    "The material feels as good as it looks — no scratchy prints, no shrinking after the first wash.",
    "Ideal as a gift: arrives in clean, gift-ready packaging with an optional personalised card.",
    "The design is the work of an independent creator, so every purchase directly supports their studio.",
    "Designed for everyday use: durable enough for the commute, refined enough for the desk.",
    "Limited runs mean you're getting something genuinely scarce, not a mass-produced catalogue item.",
    "The colour palette has been carefully matched to Pantone references to ensure consistency across different item types.",
    "Machine washable at 30°C with colours guaranteed not to fade for at least fifty washes.",
    "Works as a standalone statement piece or layers perfectly with a minimalist wardrobe.",
    "The weighty material gives it a premium hand-feel that distinguishes it from cheaper alternatives immediately.",
    "Customers consistently note that the real item exceeds their expectations from the product photos.",
]

# ── Slug helpers ──────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def _unique_slug(base: str, seen: set[str], suffix: int) -> str:
    candidate = f"{base}-{suffix}"
    while candidate in seen:
        suffix += 1
        candidate = f"{base}-{suffix}"
    seen.add(candidate)
    return candidate


# ── Description builders ──────────────────────────────────────────────────────

def _build_description(
    product_type: str,
    name: str,
    attributes: dict,
    rng: np.random.Generator,
    n_sentences: int = 6,
) -> str:
    """
    Build a multi-sentence description tailored to product type.
    The first sentence is always a specific hook; the rest are drawn
    from the sentence bank, deduplicated per product.
    """
    parts: list[str] = []

    if product_type == "digital_asset":
        asset_type = attributes.get("asset_type", "pack")
        software   = attributes.get("software_compatibility", ["your favourite app"])
        sw_str     = " and ".join(software[:2]) if len(software) >= 2 else software[0]
        asset_count = attributes.get("asset_count", "")
        count_str  = f"{asset_count} " if asset_count else ""
        hook = (
            f"{name} is a {count_str}professionally designed {asset_type} "
            f"built for use in {sw_str}. "
        )
        parts.append(hook)
        pool = DA_DESCRIPTION_SENTENCES

    elif product_type == "course":
        topic   = attributes.get("topics", ["design"])[0]
        level   = attributes.get("level", "all levels")
        outcome = rng.choice(COURSE_OUTCOMES)
        fmt     = rng.choice(COURSE_FORMATS)
        hook = (
            f"A comprehensive {level} course on {topic} designed to help you {outcome}. "
            f"{fmt} "
        )
        parts.append(hook)
        pool = COURSE_DESCRIPTION_SENTENCES

    else:  # merch
        material = attributes.get("material", "premium material")
        theme    = rng.choice(MERCH_THEMES)
        hook = (
            f"{name} features an original {theme} design, "
            f"produced in {material}. "
        )
        parts.append(hook)
        pool = MERCH_DESCRIPTION_SENTENCES

    # Pick n_sentences - 1 more from the bank (no repeats)
    picks = int(min(n_sentences - 1, len(pool)))
    chosen = rng.choice(pool, size=picks, replace=False)
    parts.extend(chosen.tolist())

    return " ".join(parts)


# ── Attribute builders ────────────────────────────────────────────────────────

def _make_digital_asset_attributes(rng: np.random.Generator) -> dict:
    asset_type  = rng.choice(DA_ASSET_TYPES)
    n_software  = int(rng.integers(1, 4))
    software    = rng.choice(DA_SOFTWARE, size=n_software, replace=False).tolist()
    n_formats   = int(rng.integers(1, 4))
    formats     = rng.choice(DA_FORMATS, size=n_formats, replace=False).tolist()
    asset_count = int(rng.choice([10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500]))
    resolution  = rng.choice(DA_RESOLUTIONS)

    return {
        "asset_type":              asset_type,
        "file_format":             formats,
        "software_compatibility":  software,
        "asset_count":             asset_count,
        "resolution":              str(resolution),
    }


def _make_course_attributes(rng: np.random.Generator) -> dict:
    topic       = rng.choice(COURSE_TOPICS)
    # Pick 1–3 additional related topics
    n_extra     = int(rng.integers(0, 3))
    extra       = rng.choice(COURSE_TOPICS, size=n_extra, replace=False).tolist()
    topics      = list(dict.fromkeys([topic] + extra))   # deduplicate, preserve order

    instructor  = f"{rng.choice(INSTRUCTOR_FIRST)} {rng.choice(INSTRUCTOR_LAST)}"
    level       = rng.choice(COURSE_LEVELS)
    language    = rng.choice(LANGUAGES)
    certificate = bool(rng.random() > 0.30)

    # Duration: beginner courses shorter, advanced longer
    if level == "beginner":
        duration = int(rng.integers(3, 12))
    elif level == "intermediate":
        duration = int(rng.integers(8, 25))
    elif level == "advanced":
        duration = int(rng.integers(15, 50))
    else:
        duration = int(rng.integers(5, 35))

    return {
        "duration_hours": duration,
        "level":          level,
        "instructor":     instructor,
        "language":       str(language),
        "certificate":    certificate,
        "topics":         topics,
    }


def _make_merch_attributes(rng: np.random.Generator, product_name: str) -> dict:
    material   = MERCH_MATERIALS.get(product_name, "premium material")
    sizes      = MERCH_SIZES.get(product_name, MERCH_SIZES["default"])
    colours    = list(MERCH_COLOURS[int(rng.integers(0, len(MERCH_COLOURS)))])
    weight_kg  = round(float(rng.uniform(0.1, 1.2)), 2)
    shipping   = product_name not in ("Poster", "Art Print", "Sticker Sheet",
                                      "Enamel Pin", "Phone Case")

    attrs: dict = {
        "material":          material,
        "colours":           colours,
        "requires_shipping": shipping,
        "weight_kg":         weight_kg,
    }
    if sizes:
        attrs["sizes_available"] = sizes

    return attrs


# ── Name builders ─────────────────────────────────────────────────────────────

def _make_digital_asset_name(attrs: dict, rng: np.random.Generator) -> str:
    style     = rng.choice(DA_STYLES)
    subject   = rng.choice(DA_SUBJECTS)
    asset_type = attrs["asset_type"].title()
    templates = [
        f"{style.title()} {subject.title()} {asset_type}",
        f"The {style.title()} {asset_type}",
        f"{subject.title()} {asset_type} — {style.title()} Edition",
        f"{style.title()} {asset_type} Vol. {rng.integers(1, 6)}",
        f"{subject.title()} {asset_type} Collection",
    ]
    return str(rng.choice(templates))


def _make_course_name(attrs: dict, rng: np.random.Generator) -> str:
    topic = attrs["topics"][0].title()
    level = attrs["level"].title()
    templates = [
        f"Complete {topic} Masterclass",
        f"{topic}: From Beginner to Pro",
        f"The Ultimate {topic} Course",
        f"{topic} Fundamentals for {level}s",
        f"Professional {topic} — Full Workflow",
        f"Learn {topic}: A Project-Based Course",
        f"Modern {topic} with {attrs['instructor'].split()[0]}",
        f"{topic} Intensive",
    ]
    return str(rng.choice(templates))


def _make_merch_name(product_name: str, rng: np.random.Generator) -> str:
    theme = rng.choice(MERCH_THEMES)
    templates = [
        f"{theme.title()} {product_name}",
        f"The {theme.title()} {product_name}",
        f"{product_name} — {theme.title()} Series",
        f"Limited {theme.title()} {product_name}",
    ]
    return str(rng.choice(templates))


# ── Price sampler ─────────────────────────────────────────────────────────────

def _sample_price(product_type: str, rng: np.random.Generator) -> float:
    lo, hi = PRICE_RANGES[product_type]
    # Log-normal gives a realistic right-skewed price distribution
    # (most products cluster toward the lower end, a few premium items at the top)
    mu    = np.log((lo + hi) / 2)
    sigma = 0.5
    raw   = np.exp(rng.normal(mu, sigma))
    raw   = float(np.clip(raw, lo, hi))
    # Round to .99 pricing ~60% of the time, else round to nearest dollar
    if rng.random() < 0.60:
        return float(np.floor(raw)) + 0.99
    return float(round(raw, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(
    sellers: list[dict] | None = None,
    count: int = DEFAULT_COUNT,
    seed: int  = DEFAULT_SEED,
    out_dir: str | os.PathLike = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate products.

    Parameters
    ----------
    sellers  : pre-loaded list of seller_profile dicts.
               If None, reads data/seller_profiles.csv automatically.
    count    : number of products to generate
    seed     : random seed
    out_dir  : output directory (default: ../data/)

    Returns
    -------
    products : list of dicts
    """
    rng = np.random.default_rng(seed)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sellers if not provided
    if sellers is None:
        sellers_path = out_dir / "seller_profiles.csv"
        if not sellers_path.exists():
            raise FileNotFoundError(
                f"seller_profiles.csv not found at {sellers_path}. "
                "Run generators/users.py first."
            )
        if verbose:
            print(f"  Reading {sellers_path} ...")
        with open(sellers_path, newline="", encoding="utf-8") as f:
            sellers = list(csv.DictReader(f))

    verified_sellers = [s for s in sellers if str(s.get("is_verified", "")).lower() in ("true", "1", "yes")]
    if not verified_sellers:
        raise ValueError("No verified sellers found in seller_profiles. Check is_verified column.")
    seller_ids = [s["user_id"] for s in verified_sellers]

    if verbose:
        print(f"\n{'═' * 55}")
        print(f"  generators/products.py")
        print(f"{'═' * 55}")
        print(f"  seed           : {seed}")
        print(f"  target         : {count:,} products")
        print(f"  sellers pool   : {len(seller_ids):,} verified sellers")
        print()

    t_start = time.perf_counter()

    # ── Assign product type for every row upfront ──
    types = list(TYPE_DIST.keys())
    probs = np.array([TYPE_DIST[t] for t in types])
    type_indices = rng.choice(len(types), size=count, p=probs)

    # ── Seller assignment: power-law (popular sellers get more products) ──
    # Pareto-distributed weights so a minority of sellers dominate listings
    seller_raw_weights = rng.pareto(1.2, size=len(seller_ids)) + 1
    seller_weights     = seller_raw_weights / seller_raw_weights.sum()
    seller_assignments = rng.choice(len(seller_ids), size=count, p=seller_weights)

    # ── Active flags ──
    n_active     = int(count * ACTIVE_RATE)
    active_flags = np.array([True] * n_active + [False] * (count - n_active))
    rng.shuffle(active_flags)

    # ── Timestamps ──
    total_days  = (PLATFORM_END - PLATFORM_START).days
    day_offsets = rng.integers(0, total_days, size=count)
    hour_offsets = rng.integers(0, 24, size=count)

    if verbose:
        print(f"  Generating {count:,} products...")

    products:  list[dict] = []
    seen_slugs: set[str]  = set()

    for i in range(count):
        product_type = types[type_indices[i]]
        seller_id    = seller_ids[seller_assignments[i]]
        is_active    = bool(active_flags[i])

        created_dt   = PLATFORM_START + timedelta(
            days=int(day_offsets[i]), hours=int(hour_offsets[i])
        )
        # updated_at: either same as created, or slightly later
        if rng.random() < 0.35:
            update_offset = int(rng.integers(1, max(2, (PLATFORM_END - created_dt).days)))
            updated_dt    = created_dt + timedelta(days=update_offset)
        else:
            updated_dt = created_dt

        # Build attributes first (name depends on them)
        if product_type == "digital_asset":
            attrs = _make_digital_asset_attributes(rng)
            name  = _make_digital_asset_name(attrs, rng)
        elif product_type == "course":
            attrs = _make_course_attributes(rng)
            name  = _make_course_name(attrs, rng)
        else:  # merch
            merch_base = str(rng.choice(MERCH_PRODUCT_NAMES))
            attrs      = _make_merch_attributes(rng, merch_base)
            name       = _make_merch_name(merch_base, rng)

        price       = _sample_price(product_type, rng)
        description = _build_description(product_type, name, attrs, rng, n_sentences=6)
        slug_base   = _slugify(name)
        slug        = _unique_slug(slug_base, seen_slugs, i)

        products.append({
            "id":           str(uuid.uuid4()),
            "name":         name,
            "slug":         slug,
            "product_type": product_type,
            "description":  description,
            "price_usd":    f"{price:.2f}",
            "currency":     "USD",
            "is_active":    is_active,
            "seller_id":    seller_id,
            "attributes":   json.dumps(attrs, ensure_ascii=False),
            "created_at":   created_dt.isoformat(),
            "updated_at":   updated_dt.isoformat(),
            # search_vector intentionally omitted —
            # the PostgreSQL trigger computes it correctly on INSERT
        })

    # ── Write CSV ──
    fields = [
        "id", "name", "slug", "product_type", "description",
        "price_usd", "currency", "is_active", "seller_id",
        "attributes", "created_at", "updated_at",
    ]

    out_path = out_dir / "products.csv"
    if verbose:
        print(f"  Writing products.csv ...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(products)

    elapsed = time.perf_counter() - t_start

    if verbose:
        _print_summary(products, elapsed, out_path)

    return products


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(products: list[dict], elapsed: float, out_path: Path) -> None:
    from collections import Counter

    n            = len(products)
    types        = Counter(p["product_type"] for p in products)
    n_active     = sum(1 for p in products if p["is_active"])
    prices_by_type = {}
    for p in products:
        prices_by_type.setdefault(p["product_type"], []).append(float(p["price_usd"]))

    print()
    print(f"{'─' * 55}")
    print(f"  Summary")
    print(f"{'─' * 55}")
    print(f"  Total products   : {n:>8,}")
    print(f"  Active           : {n_active:>8,}  ({n_active / n * 100:.1f}%)")
    print(f"  Inactive         : {n - n_active:>8,}  ({(n - n_active) / n * 100:.1f}%)")
    print()
    print(f"  Type breakdown:")
    for pt, cnt in types.most_common():
        bar    = "█" * int(cnt / n * 40)
        prices = prices_by_type[pt]
        avg    = sum(prices) / len(prices)
        print(f"    {pt:<16} {cnt:>7,}  {bar}  avg ${avg:.2f}")
    print()
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  products.csv     : {size_mb:.1f} MB")
    print(f"  Elapsed          : {elapsed:.2f}s")
    print(f"{'═' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate StreamCart product catalogue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count",   type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    generate(count=args.count, seed=args.seed, out_dir=args.out_dir, verbose=not args.quiet)