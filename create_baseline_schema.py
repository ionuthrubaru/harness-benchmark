"""
create_baseline_schema.py — Phase 1: PostgreSQL Master Schema
=====================================================
1. Connects to the PostgreSQL Docker container
2. Creates all 12 tables from schema.sql
3. Verifies every table exists
4. Generates an ER diagram as PNG

Usage:
    python create_baseline_schema.py
    python create_baseline_schema.py --diagram-only   (skip schema, just regenerate diagram)
    python create_baseline_schema.py --verify-only    (just check tables exist)
"""

import os
import sys
import argparse
import psycopg2
from dotenv import load_dotenv
import graphviz
import html

load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}v {msg}{RESET}")
def fail(msg):  print(f"  {RED}x {msg}{RESET}")
def info(msg):  print(f"  {BLUE}> {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}! {msg}{RESET}")

EXPECTED_TABLES = [
    "users", "seller_profiles", "subscription_tiers",
    "subscription_tier_pricing", "subscriptions", "products",
    "invoices", "invoice_lines", "orders", "order_items",
    "sessions", "events",
]

def get_connection():
    return psycopg2.connect(
        host="localhost", port=5432,
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )

def create_schema():
    print(f"\n{'=' * 55}")
    print("  Step 1 - Creating PostgreSQL Master Schema")
    print(f"{'=' * 55}")

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        fail(f"schema.sql not found at {schema_path}")
        sys.exit(1)

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    info("Connecting to PostgreSQL...")
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()
    info("Executing schema.sql...")
    try:
        cur.execute(sql)
        ok("Schema executed successfully")
    except psycopg2.Error as e:
        if "already exists" in str(e):
            warn("Some tables already exist - run --verify-only to check")
        else:
            fail(f"SQL error: {e}")
            conn.close()
            sys.exit(1)
    cur.close()
    conn.close()

def verify_tables():
    print(f"\n{'=' * 55}")
    print("  Step 2 - Verifying Tables")
    print(f"{'=' * 55}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    existing = {row[0] for row in cur.fetchall()}

    all_ok = True
    for table in sorted(EXPECTED_TABLES):
        if table in existing:
            ok(f"{table}")
        else:
            fail(f"{table} - MISSING")
            all_ok = False

    print()
    info("Seeded table row counts:")
    for table in ["subscription_tiers", "subscription_tier_pricing"]:
        if table in existing:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            count = cur.fetchone()[0]
            print(f"    {table}: {count} rows")

    cur.close()
    conn.close()
    return all_ok

TABLES = {
    "users": ["id (PK)", "email", "full_name", "country_code", "city",
              "created_at", "last_login_at", "is_active", "preferences"],
    "seller_profiles": ["user_id (PK, FK->users)", "display_name", "legal_name",
                        "tax_id", "payout_email", "country_code", "is_verified",
                        "bio", "created_at", "updated_at"],
    "subscription_tiers": ["id (PK)", "name", "description", "features"],
    "subscription_tier_pricing": ["tier_id (PK, FK->subscription_tiers)",
                                   "valid_from (PK)", "valid_to",
                                   "monthly_price_usd", "is_active"],
    "subscriptions": ["id (PK)", "user_id (FK->users)",
                      "tier_id (FK->subscription_tiers)", "status",
                      "started_at", "current_period_start", "current_period_end",
                      "cancelled_at", "billing_cycle", "created_at", "updated_at"],
    "products": ["id (PK)", "name", "slug", "product_type", "description",
                 "price_usd", "currency", "is_active", "seller_id (FK->users)",
                 "attributes", "search_vector", "created_at", "updated_at"],
    "invoices": ["id (PK)", "user_id (FK->users)", "invoice_type", "status",
                 "subtotal_usd", "tax_usd", "discount_usd", "total_usd",
                 "subscription_id (FK->subscriptions)",
                 "billing_period_start", "billing_period_end",
                 "paid_at", "due_at", "created_at"],
    "invoice_lines": ["id (PK)", "invoice_id (FK->invoices)",
                      "product_id (FK->products)", "description",
                      "quantity", "unit_price_usd", "line_total_usd", "created_at"],
    "orders": ["id (PK)", "user_id (FK->users)", "invoice_id (FK->invoices)",
               "status", "shipping_name", "shipping_address",
               "shipping_city", "shipping_country", "shipping_postal",
               "created_at", "updated_at"],
    "order_items": ["id (PK)", "order_id (FK->orders)", "product_id (FK->products)",
                    "quantity", "unit_price_usd", "line_total_usd",
                    "fulfilment_status", "created_at"],
    "sessions": ["id (PK)", "user_id (FK->users)", "cart", "ip_address",
                 "user_agent", "created_at", "last_active_at", "expires_at"],
    "events": ["id (PK)", "user_id (FK->users)", "event_type",
               "product_id (FK->products)", "session_id (FK->sessions)",
               "metadata", "occurred_at"],
}

RELATIONSHIPS = [
    ("seller_profiles",           "users",              "user_id"),
    ("subscription_tier_pricing", "subscription_tiers", "tier_id"),
    ("subscriptions",             "users",              "user_id"),
    ("subscriptions",             "subscription_tiers", "tier_id"),
    ("products",                  "users",              "seller_id"),
    ("invoices",                  "users",              "user_id"),
    ("invoices",                  "subscriptions",      "subscription_id"),
    ("invoice_lines",             "invoices",           "invoice_id"),
    ("invoice_lines",             "products",           "product_id"),
    ("orders",                    "users",              "user_id"),
    ("orders",                    "invoices",           "invoice_id"),
    ("order_items",               "orders",             "order_id"),
    ("order_items",               "products",           "product_id"),
    ("sessions",                  "users",              "user_id"),
    ("events",                    "users",              "user_id"),
    ("events",                    "products",           "product_id"),
    ("events",                    "sessions",           "session_id"),
]

SECTION_COLOURS = {
    "users":                      "#2E86AB",
    "seller_profiles":            "#2E86AB",
    "subscription_tiers":         "#A23B72",
    "subscription_tier_pricing":  "#A23B72",
    "subscriptions":              "#A23B72",
    "products":                   "#F18F01",
    "invoices":                   "#C73E1D",
    "invoice_lines":              "#C73E1D",
    "orders":                     "#C73E1D",
    "order_items":                "#C73E1D",
    "sessions":                   "#3B1F2B",
    "events":                     "#3B1F2B",
}

def generate_er_diagram():
    print(f"\n{'=' * 55}")
    print("  Step 3 - Generating ER Diagram")
    print(f"{'=' * 55}")

    output_dir = os.path.join(os.path.dirname(__file__), "analysis", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    dot = graphviz.Digraph(name="Master_Schema", format="png")
    dot.attr(
        rankdir="LR", splines="ortho", nodesep="0.5", ranksep="1.5",
        fontname="Helvetica", bgcolor="white",
        label="StreamCart — Unified Relational Schema",
        labelloc="t", fontsize="22",
    )

    for table, columns in TABLES.items():
        colour = SECTION_COLOURS.get(table, "#555555")
        rows = ""
        for col in columns:
            is_pk = "PK" in col
            is_fk = "FK" in col
            font_colour = "#FFD700" if is_pk else ("#AADDFF" if is_fk else "white")
            bold_open  = "<B>" if is_pk else ""
            bold_close = "</B>" if is_pk else ""
            safe_port  = col.split(" ")[0].replace("(","").replace(")","")
            safe_col = html.escape(col)
            rows += (
                f'<TR><TD ALIGN="LEFT" PORT="{safe_port}" BGCOLOR="{colour}DD">'
                f'<FONT COLOR="{font_colour}">{bold_open}{safe_col}{bold_close}</FONT>'
                f'</TD></TR>'
            )
        label = (
            f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
            f'<TR><TD BGCOLOR="{colour}"><FONT COLOR="white"><B>{table.upper()}</B></FONT></TD></TR>'
            f'{rows}</TABLE>>'
        )
        dot.node(table, label=label, shape="none", margin="0", fontname="Helvetica")

    for from_table, to_table, label in RELATIONSHIPS:
        dot.edge(from_table, to_table, label=f" {label} ",
                 fontsize="9", fontname="Helvetica", color="#666666",
                 arrowhead="crow", arrowtail="none")

    legend = (
        '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="4" CELLPADDING="6" BGCOLOR="white">'
        '<TR><TD COLSPAN="2" ALIGN="CENTER"><B>Legend</B></TD></TR>'
        '<TR><TD BGCOLOR="#2E86AB" WIDTH="20"> </TD><TD ALIGN="LEFT">Core entities</TD></TR>'
        '<TR><TD BGCOLOR="#A23B72"> </TD><TD ALIGN="LEFT">Subscription</TD></TR>'
        '<TR><TD BGCOLOR="#F18F01"> </TD><TD ALIGN="LEFT">Marketplace</TD></TR>'
        '<TR><TD BGCOLOR="#C73E1D"> </TD><TD ALIGN="LEFT">Transactional</TD></TR>'
        '<TR><TD BGCOLOR="#3B1F2B"> </TD><TD ALIGN="LEFT">Behavioural</TD></TR>'
        '<TR><TD ALIGN="CENTER"><FONT COLOR="#FFD700"><B>|</B></FONT></TD><TD ALIGN="LEFT">Primary key</TD></TR>'
        '<TR><TD ALIGN="CENTER"><FONT COLOR="#AADDFF">|</FONT></TD><TD ALIGN="LEFT">Foreign key</TD></TR>'
        '</TABLE>>'
    )
    dot.node("legend", label=legend, shape="none", margin="0", fontname="Helvetica")

    output_path = os.path.join(output_dir, "schema_er")
    info(f"Rendering to {output_path}.png ...")
    try:
        dot.render(output_path, cleanup=True)
        ok(f"ER diagram saved -> analysis/diagrams/schema_er.png")
    except Exception as e:
        fail(f"Diagram generation failed: {e}")
        warn("Make sure Graphviz is installed and 'dot' is on your PATH")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PostgreSQL schema and generate ER diagram")
    parser.add_argument("--diagram-only", action="store_true",
                        help="Skip schema creation, just regenerate the ER diagram")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just verify tables exist, no schema or diagram")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  PostgreSQL Master Schema Setup")
    print("=" * 55)

    if args.verify_only:
        sys.exit(0 if verify_tables() else 1)

    if not args.diagram_only:
        create_schema()

    if not verify_tables():
        fail("Some tables are missing - fix errors above before generating diagram")
        sys.exit(1)

    generate_er_diagram()

    print(f"\n{'=' * 55}")
    print(f"  {GREEN}All done! PostgreSQL schema is ready.{RESET}")
    print(f"{'=' * 55}\n")