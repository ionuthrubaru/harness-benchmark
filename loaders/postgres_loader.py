"""
loaders/postgres_loader.py — PostgreSQL Bulk Loader
====================================================
Loads all generated CSV files into PostgreSQL using COPY FROM STDIN

Loading order (dependency-aware, respects all foreign keys):
  1.  users                     ← data/users.csv
  2.  seller_profiles           ← data/seller_profiles.csv
  3.  subscription_tiers        ← already seeded by schema.sql  (skipped)
  4.  subscription_tier_pricing ← already seeded by schema.sql  (skipped)
  5.  subscriptions             ← data/subscriptions.csv
  6.  products                  ← data/products.csv
  7.  invoices                  ← data/subscription_invoices.csv
                                   + data/marketplace_invoices.csv  (merged)
  8.  invoice_lines             ← data/subscription_invoice_lines.csv
                                   + data/marketplace_invoice_lines.csv (merged)
  9.  orders                    ← data/orders.csv
  10. order_items               ← data/order_items.csv
  11. sessions                  ← data/sessions.csv
  12. events                    ← data/events.csv

Idempotency:
  Before loading each table, the loader checks whether it already contains
  rows. If it does, that table is skipped entirely. Run with --force to
  truncate and reload everything from scratch.

COPY mechanics:
  psycopg2's copy_expert() streams each CSV file directly into PostgreSQL
  with no intermediate Python parsing — the DB engine does all the work.
  Empty strings in nullable columns are converted to NULL via a pre-processing
  step using a StringIO buffer for small files, or a server-side NULL
  substitution for large files (events.csv).

  The products table omits search_vector — PostgreSQL's trigger
  (trg_products_search_vector) recomputes it correctly on every COPY row.

Usage
-----
  python loaders/postgres_loader.py
  python loaders/postgres_loader.py --force          # truncate & reload all
  python loaders/postgres_loader.py --table events   # reload one table only
  python loaders/postgres_loader.py --verify-only    # just print row counts
  python loaders/postgres_loader.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── Colours ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def fail(msg): print(f"  {RED}✘{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {BLUE}›{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{msg}{RESET}")


# ── Connection ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host     = os.getenv("POSTGRES_HOST", "localhost"),
        port     = int(os.getenv("POSTGRES_PORT", 5432)),
        user     = os.getenv("POSTGRES_USER"),
        password = os.getenv("POSTGRES_PASSWORD"),
        dbname   = os.getenv("POSTGRES_DB"),
    )


# ── Row count helper ──────────────────────────────────────────────────────────

def row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return cur.fetchone()[0]


# ── COPY helpers ──────────────────────────────────────────────────────────────

def _csv_to_buf(path: Path) -> io.StringIO:
    """
    Read a CSV file into a StringIO buffer, converting empty fields to \\N
    (PostgreSQL NULL marker). Uses Python's csv module to correctly handle
    quoted fields containing commas (e.g. JSON preferences/attributes columns).
    """
    import csv as _csv

    out = io.StringIO()
    writer = _csv.writer(out, lineterminator="\n")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
            else:
                writer.writerow([r"\N" if field == "" else field for field in row])

    out.seek(0)
    return out

def copy_table(
    conn,
    table: str,
    columns: list[str],
    path: Path,
    large: bool = False,
) -> int:
    """
    COPY a CSV file into table using the given column list.
    Returns the number of rows loaded.

    For large files (events.csv) we stream directly from disk rather than
    reading the whole thing into a StringIO buffer first.
    """
    col_list = ", ".join(columns)
    sql = (
        f"COPY {table} ({col_list}) "
        f"FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N')"
    )

    t0 = time.perf_counter()

    if large:
        # Process large files in chunks to keep memory bounded while still
        # substituting empty fields with \N before PostgreSQL sees them.
        import csv as _csv
        CHUNK = 10_000
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = _csv.reader(f)
            header = next(reader)
            first_chunk = True
            while True:
                rows = []
                for _ in range(CHUNK):
                    try:
                        rows.append(next(reader))
                    except StopIteration:
                        break
                if not rows:
                    break
                buf = io.StringIO()
                writer = _csv.writer(buf, lineterminator="\n")
                if first_chunk:
                    writer.writerow(header)
                    first_chunk = False
                for row in rows:
                    writer.writerow([r"\N" if field == "" else field for field in row])
                buf.seek(0)
                with conn.cursor() as cur:
                    cur.copy_expert(sql, buf)
        conn.commit()
    else:
        buf = _csv_to_buf(path)
        with conn.cursor() as cur:
            cur.copy_expert(sql, buf)
        conn.commit()

    elapsed = time.perf_counter() - t0
    n = row_count(conn, table)
    ok(f"{table:<30} {n:>10,} rows   ({elapsed:.1f}s)")
    return n


def copy_table_multi(
    conn,
    table: str,
    columns: list[str],
    paths: list[Path],
) -> int:
    """
    COPY multiple CSV files into the same table sequentially.
    Used to merge subscription_invoices + marketplace_invoices → invoices,
    and the matching invoice_lines pair.
    """
    t0 = time.perf_counter()
    sql = (
        f"COPY {table} ({', '.join(columns)}) "
        f"FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N')"
    )
    for path in paths:
        buf = _csv_to_buf(path)
        with conn.cursor() as cur:
            cur.copy_expert(sql, buf)
    conn.commit()

    elapsed = time.perf_counter() - t0
    n = row_count(conn, table)
    labels = " + ".join(p.name for p in paths)
    ok(f"{table:<30} {n:>10,} rows   ({elapsed:.1f}s)  [{labels}]")
    return n


# ── Idempotency check ─────────────────────────────────────────────────────────

def should_load(conn, table: str, force: bool) -> bool:
    n = row_count(conn, table)
    if n > 0 and not force:
        warn(f"{table:<30} {n:>10,} rows already — skipping (use --force to reload)")
        return False
    if n > 0 and force:
        info(f"--force: truncating {table} ...")
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table} CASCADE;")
        conn.commit()
    return True


# ── Table definitions ─────────────────────────────────────────────────────────
# Each entry: (table_name, [columns], csv_path_or_paths, large_file)
# Columns must exactly match the CSV headers AND the PostgreSQL table columns.
# search_vector is intentionally absent from products — trigger handles it.

def build_load_plan(data_dir: Path) -> list[dict]:
    return [
        {
            "table":   "users",
            "columns": [
                "id", "email", "full_name", "country_code", "city",
                "created_at", "last_login_at", "is_active", "preferences",
            ],
            "paths":  [data_dir / "users.csv"],
            "large":  False,   # 41MB is fine to buffer
        },
        {
            "table":   "seller_profiles",
            "columns": [
                "user_id", "display_name", "legal_name", "tax_id", "payout_email",
                "country_code", "is_verified", "bio",
                "created_at", "updated_at",
            ],
            "paths":  [data_dir / "seller_profiles.csv"],
            "large":  False,
        },
        {
            "table":   "subscriptions",
            "columns": [
                "id", "user_id", "tier_id", "status",
                "started_at", "current_period_start", "current_period_end",
                "cancelled_at", "cancel_reason", "billing_cycle",
                "created_at", "updated_at",
            ],
            "paths":  [data_dir / "subscriptions.csv"],
            "large":  False,
        },
        {
            "table":   "products",
            "columns": [
                "id", "name", "slug", "product_type", "description",
                "price_usd", "currency", "is_active", "seller_id",
                "attributes", "created_at", "updated_at",
                # search_vector intentionally omitted — trigger recomputes it
            ],
            "paths":  [data_dir / "products.csv"],
            "large":  False,
        },
        {
            "table":   "invoices",
            "columns": [
                "id", "user_id", "invoice_type", "status",
                "subtotal_usd", "tax_usd", "discount_usd", "total_usd",
                "subscription_id", "billing_period_start", "billing_period_end",
                "paid_at", "due_at", "created_at",
            ],
            # Merge both invoice sources into the single invoices table
            "paths":  [
                data_dir / "subscription_invoices.csv",
                data_dir / "marketplace_invoices.csv",
            ],
            "large":  False,
        },
        {
            "table":   "invoice_lines",
            "columns": [
                "id", "invoice_id", "product_id", "description",
                "quantity", "unit_price_usd", "line_total_usd", "created_at",
            ],
            "paths":  [
                data_dir / "subscription_invoice_lines.csv",
                data_dir / "marketplace_invoice_lines.csv",
            ],
            "large":  False,
        },
        {
            "table":   "orders",
            "columns": [
                "id", "user_id", "invoice_id", "status",
                "shipping_name", "shipping_address", "shipping_city",
                "shipping_country", "shipping_postal",
                "created_at", "updated_at",
            ],
            "paths":  [data_dir / "orders.csv"],
            "large":  False,
        },
        {
            "table":   "order_items",
            "columns": [
                "id", "order_id", "product_id", "quantity",
                "unit_price_usd", "line_total_usd", "fulfilment_status", "created_at",
            ],
            "paths":  [data_dir / "order_items.csv"],
            "large":  False,
        },
        {
            "table":   "sessions",
            "columns": [
                "id", "user_id", "cart", "ip_address", "user_agent",
                "created_at", "last_active_at", "expires_at",
            ],
            "paths":  [data_dir / "sessions.csv"],
            "large":  False,
        },
        {
            "table":   "events",
            "columns": [
                "id", "user_id", "event_type", "product_id",
                "session_id", "metadata", "occurred_at",
            ],
            "paths":  [data_dir / "events.csv"],
            "large":  True,
        },
    ]


# ── Verify ────────────────────────────────────────────────────────────────────

def verify(conn) -> bool:
    hdr("  Row counts — all tables")
    print(f"  {'Table':<35} {'Rows':>12}")
    print(f"  {'─' * 35} {'─' * 12}")

    tables = [
        "users", "seller_profiles",
        "subscription_tiers", "subscription_tier_pricing",
        "subscriptions", "products",
        "invoices", "invoice_lines",
        "orders", "order_items",
        "sessions", "events",
    ]

    all_ok = True
    for table in tables:
        try:
            n = row_count(conn, table)
            status = f"{GREEN}✔{RESET}" if n > 0 else f"{YELLOW}⚠ EMPTY{RESET}"
            print(f"  {table:<35} {n:>12,}  {status}")
            if n == 0 and table not in ("subscription_tiers", "subscription_tier_pricing"):
                all_ok = False
        except Exception as e:
            print(f"  {table:<35} {RED}ERROR: {e}{RESET}")
            all_ok = False

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    force:      bool = False,
    table_only: str  = None,
    verify_only: bool = False,
    data_dir:   Path = None,
) -> None:

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    hdr("═" * 58)
    hdr("  PostgreSQL Bulk Loader")
    hdr("═" * 58)
    info(f"data dir : {data_dir}")
    info(f"force    : {force}")
    if table_only:
        info(f"table    : {table_only} (single table mode)")

    conn = get_conn()

    if verify_only:
        verify(conn)
        conn.close()
        return

    plan = build_load_plan(data_dir)

    hdr("  Loading tables")
    print(f"  {'Table':<30} {'Rows':>10}   Time   Source(s)")
    print(f"  {'─' * 30} {'─' * 10}   {'─' * 6}   {'─' * 30}")

    t_total = time.perf_counter()

    for step in plan:
        table   = step["table"]
        columns = step["columns"]
        paths   = step["paths"]
        large   = step.get("large", False)

        # Single-table filter
        if table_only and table != table_only:
            continue

        # Check all source files exist
        missing = [p for p in paths if not p.exists()]
        if missing:
            warn(f"{table:<30} MISSING files: {[p.name for p in missing]} — skipping")
            continue

        # Idempotency check (only when loading all tables or forced)
        if not force and not table_only:
            if not should_load(conn, table, force=False):
                continue
        elif force or table_only:
            should_load(conn, table, force=True)

        try:
            if len(paths) == 1:
                copy_table(conn, table, columns, paths[0], large=large)
            else:
                copy_table_multi(conn, table, columns, paths)
        except psycopg2.Error as e:
            conn.rollback()
            fail(f"COPY failed for {table}: {e}")

    elapsed = time.perf_counter() - t_total

    hdr("  Verification")
    all_ok = verify(conn)
    conn.close()

    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}All tables loaded successfully.{RESET}  "
              f"Total time: {elapsed:.1f}s\n")
    else:
        print(f"  {YELLOW}Some tables are empty — check warnings above.{RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-load all StreamCart CSVs into PostgreSQL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Truncate and reload tables that already have data",
    )
    parser.add_argument(
        "--table", metavar="TABLE",
        help="Load a single table only (e.g. --table events)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Print row counts for all tables, do not load",
    )
    parser.add_argument(
        "--data-dir", metavar="DIR",
        help="Path to data directory (default: ../data/ relative to this script)",
    )
    args = parser.parse_args()

    run(
        force       = args.force,
        table_only  = args.table,
        verify_only = args.verify_only,
        data_dir    = Path(args.data_dir) if args.data_dir else None,
    )