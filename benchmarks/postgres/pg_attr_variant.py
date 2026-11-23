#!/usr/bin/env python3
"""
pg_attr_variant.py, runda I (R2 pct. 3): PostgreSQL cu atribuirea tier-ului și a prețului făcută la load.

Varianta mută la încărcare exact munca pe care ES optimised o mută la încărcare (loaders/es_tier_attribution.py):
pentru fiecare factură subscription/marketplace se precalculează tier-ul (abonamentul facturii, respectiv
ultimul abonament început până la data facturii) și prețul în vigoare la data facturii. Logica e copiată din
Q1/Q7 baseline (benchmarks/postgres/q1_revenue.py, q7_rolling_revenue.py), deci rezultatul e identic prin
construcție; modul verify o confirmă rând cu rând. Query-urile variantei fac doar filtrul și agregarea.

Moduri (din rădăcina repo-ului, cu .env-ul lui):
  build   --out build.json                    invoice_attr (copia invoices + 3 coloane de atribuire), indecșii lui invoices,
                                              VACUUM ANALYZE; timpi, dimensiuni, corelația fizică, controale
  verify  --out equivalence.txt               Q1 (fereastra AS_OF și tot intervalul), Q7 (7 ferestre și tot intervalul),
                                              baseline vs variantă; exit 1 la diferență
  run     --query Q1|Q7 --iterations N --output f.json   prin harness, exact ca driverele baseline
  explain --query Q1|Q7 --out f.txt           EXPLAIN (ANALYZE, BUFFERS) pe o execuție
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import timedelta

REPO = os.getcwd()
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "postgres"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

from pg_conn import get_connection  # noqa: E402  (jit off, work_mem 64MB: aceleași setări de sesiune ca baseline)
from benchmarks.harness import run_benchmark  # noqa: E402

AS_OF_SQL = "TIMESTAMPTZ '2026-03-04 14:51:58+00'"
WINDOW_DAYS = 183  # identic cu benchmarks/postgres/q7_rolling_revenue.py

# ── construcția (la load) ──────────────────────────────────────────────────────
# invoice_attr = copia integrală a tabelului invoices (toate coloanele, aceeași ordine fizică, aceiași indecși)
# plus trei coloane precalculate: tier-ul, numele lui și prețul în vigoare la data facturii. E echivalentul
# relațional al câmpurilor pe care loaders/es_tier_attribution.py le adaugă documentelor ES optimised, deci
# singura diferență față de baseline e atribuirea făcută o dată, la load, în locul join-urilor de la fiecare query.
CREATE_SQL = """
CREATE TABLE invoice_attr AS
WITH tier AS (
    -- subscription: tier-ul abonamentului facturii (ca în Q1/Q7 baseline)
    SELECT i.id AS invoice_id, sub.tier_id
    FROM invoices i
    JOIN subscriptions sub ON sub.id = i.subscription_id
    WHERE i.invoice_type = 'subscription'
    UNION ALL
    -- marketplace: ultimul abonament început până la data facturii (ca în Q1/Q7 baseline)
    SELECT i.id, active_sub.tier_id
    FROM invoices i
    JOIN LATERAL (
        SELECT s.tier_id
        FROM   subscriptions s
        WHERE  s.user_id    = i.user_id
          AND  s.started_at <= i.created_at
        ORDER BY s.started_at DESC
        LIMIT 1
    ) active_sub ON TRUE
    WHERE i.invoice_type = 'marketplace'
)
SELECT i.*,
       t.tier_id AS attr_tier_id,
       st.name   AS attr_tier_name,
       (SELECT stp.monthly_price_usd
          FROM subscription_tier_pricing stp
         WHERE stp.tier_id    = t.tier_id
           AND stp.valid_from <= i.created_at
           AND (stp.valid_to IS NULL OR stp.valid_to > i.created_at)
         ORDER BY stp.valid_from DESC
         LIMIT 1) AS attr_price_usd
FROM invoices i
LEFT JOIN tier t ON t.invoice_id = i.id
LEFT JOIN subscription_tiers st ON st.id = t.tier_id
ORDER BY i.ctid
"""
# aceiași indecși ca pe invoices (schema.sql): planificatorul are aceleași căi de acces ca la baseline
INDEX_SQL = [
    ("idx_user_id",                 "CREATE INDEX invoice_attr_user_id_idx ON invoice_attr (user_id)"),
    ("idx_subscription_id",         "CREATE INDEX invoice_attr_subscription_id_idx ON invoice_attr (subscription_id)"),
    ("idx_status",                  "CREATE INDEX invoice_attr_status_idx ON invoice_attr (status)"),
    ("idx_created_at",              "CREATE INDEX invoice_attr_created_at_idx ON invoice_attr (created_at)"),
    ("idx_type",                    "CREATE INDEX invoice_attr_type_idx ON invoice_attr (invoice_type)"),
    ("idx_type_status_created",     "CREATE INDEX invoice_attr_type_status_created_idx ON invoice_attr (invoice_type, status, created_at)"),
]
CHECKS_SQL = {
    "rows": "SELECT count(*) FROM invoice_attr",
    "rows_source_invoices": "SELECT count(*) FROM invoices",
    "rows_paid": "SELECT count(*) FROM invoice_attr WHERE status = 'paid'",
    "rows_sub_or_mkt_tier_null": "SELECT count(*) FROM invoice_attr WHERE invoice_type IN ('subscription', 'marketplace') AND attr_tier_id IS NULL",
    "rows_price_null_with_tier": "SELECT count(*) FROM invoice_attr WHERE attr_tier_id IS NOT NULL AND attr_price_usd IS NULL",
    "invoices_with_overlapping_prices": """
        SELECT count(*) FROM (
            SELECT a.id
            FROM invoice_attr a
            JOIN subscription_tier_pricing stp
              ON  stp.tier_id    = a.attr_tier_id
              AND stp.valid_from <= a.created_at
              AND (stp.valid_to IS NULL OR stp.valid_to > a.created_at)
            GROUP BY a.id HAVING count(*) > 1) x""",
    # egalități de started_at la același user: alegerea LIMIT 1 din baseline e atunci arbitrară (și în baseline)
    "subscription_start_ties": """
        SELECT count(*) FROM (SELECT user_id, started_at FROM subscriptions GROUP BY 1, 2 HAVING count(*) > 1) x""",
    "table_bytes": "SELECT pg_relation_size('invoice_attr')",
    "index_bytes": "SELECT pg_indexes_size('invoice_attr')",
    "total_bytes": "SELECT pg_total_relation_size('invoice_attr')",
    "source_invoices_table_bytes": "SELECT pg_relation_size('invoices')",
    "source_invoices_index_bytes": "SELECT pg_indexes_size('invoices')",
    "database_bytes": "SELECT pg_database_size(current_database())",
}
# corelația created_at cu ordinea fizică (pg_stats), pe ambele tabele: trebuie să fie aproape egale
CORR_SQL = """SELECT tablename, correlation FROM pg_stats
              WHERE schemaname = 'public' AND tablename IN ('invoices', 'invoice_attr') AND attname = 'created_at'"""

# ── query-urile variantei: aceeași ieșire ca baseline, fără join-uri de atribuire ──
Q1_ATTR_SQL = f"""
SELECT
    TO_CHAR(month, 'YYYY-MM') AS month,
    tier_name,
    price_in_effect_usd,
    invoice_count,
    total_revenue_usd
FROM (
    SELECT
        DATE_TRUNC('month', created_at) AS month,
        attr_tier_name                  AS tier_name,
        attr_price_usd                  AS price_in_effect_usd,
        COUNT(*)                        AS invoice_count,
        ROUND(SUM(total_usd), 2)        AS total_revenue_usd
    FROM invoice_attr
    WHERE status = 'paid'
      AND invoice_type IN ('subscription', 'marketplace')
      AND attr_tier_id IS NOT NULL
      AND attr_price_usd IS NOT NULL
      AND created_at >= {AS_OF_SQL} - INTERVAL '12 months'
    GROUP BY 1, 2, 3
) m
ORDER BY 1, 2;
"""
Q7_ATTR_SQL = """
WITH date_spine AS (
    SELECT generate_series(%(s)s::date, %(e)s::date, '1 day'::interval)::date AS day
),
tier_days AS (
    SELECT ds.day, st.id AS tier_id, st.name AS tier_name
    FROM date_spine ds
    CROSS JOIN subscription_tiers st
),
daily_revenue AS (
    SELECT DATE_TRUNC('day', a.created_at)::date AS day, a.attr_tier_id AS tier_id, SUM(a.total_usd) AS revenue
    FROM invoice_attr a
    WHERE a.status = 'paid'
      AND a.invoice_type IN ('subscription', 'marketplace')
      AND a.attr_tier_id IS NOT NULL
      AND a.created_at >= %(s)s::date
      AND a.created_at <  (%(e)s::date + INTERVAL '1 day')
    GROUP BY 1, 2
),
filled AS (
    SELECT td.day, td.tier_id, td.tier_name, COALESCE(SUM(dr.revenue), 0.00) AS daily_total
    FROM tier_days td
    LEFT JOIN daily_revenue dr ON dr.day = td.day AND dr.tier_id = td.tier_id
    GROUP BY td.day, td.tier_id, td.tier_name
)
SELECT
    day,
    tier_name,
    ROUND(daily_total, 2) AS daily_revenue_usd,
    ROUND(AVG(daily_total) OVER (PARTITION BY tier_id ORDER BY day
                                 ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS rolling_7day_avg_usd
FROM filled
ORDER BY day, tier_name;
"""


def baseline_sql(rel_path, name):
    """SQL-ul baseline exact cum e în driverul de pe server; NOW() se ancorează la AS_OF, ca în campanie."""
    src = open(os.path.join(REPO, rel_path), encoding="utf-8").read()
    a = src.index(name + ' = """') + len(name + ' = """')
    return src[a:src.index('"""', a)].replace("NOW()", AS_OF_SQL)


def data_range(conn):
    with conn.cursor() as cur:  # identic cu load_data_date_range din q7_rolling_revenue.py
        cur.execute("SELECT MIN(created_at)::date, MAX(created_at)::date FROM invoices WHERE status = 'paid';")
        return cur.fetchone()


def random_window(data_min, data_max, rng=random):
    max_start = max(0, (data_max - data_min).days - WINDOW_DAYS)  # identic cu random_window din q7_rolling_revenue.py
    start = data_min + timedelta(days=rng.randint(0, max_start))
    return start, start + timedelta(days=WINDOW_DAYS - 1)


def fetch(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [tuple(r) for r in cur.fetchall()]


def cmd_build(args):
    conn = get_connection()
    conn.autocommit = True  # VACUUM nu rulează într-o tranzacție
    out = {"steps_s": {}}
    with conn.cursor() as cur:
        cur.execute("SELECT version()"); out["version"] = cur.fetchone()[0]
        cur.execute("DROP TABLE IF EXISTS invoice_attr")
        # Revizia 2026-11: fără ANALYZE invoices (ar re-eșantiona statisticile tabelului de bază între rulări) și VACUUM
        # serial (PARALLEL 0), ca să nu depindă de /dev/shm-ul de 64 MiB din Docker.
        steps = [("attribution_ctas", CREATE_SQL)] + INDEX_SQL + [("vacuum_analyze", "VACUUM (ANALYZE, PARALLEL 0) invoice_attr")]
        for name, sql in steps:
            t0 = time.perf_counter(); cur.execute(sql); out["steps_s"][name] = round(time.perf_counter() - t0, 3)
            print(f"  {name}: {out['steps_s'][name]} s", flush=True)
        out["attribution_s"] = out["steps_s"]["attribution_ctas"]
        out["indexes_s"] = round(sum(v for k, v in out["steps_s"].items() if k.startswith("idx_")), 3)
        for k, sql in CHECKS_SQL.items():
            cur.execute(sql); out[k] = int(cur.fetchone()[0])
        cur.execute(CORR_SQL); out["created_at_correlation"] = {t: round(float(c), 4) for t, c in cur.fetchall()}
    conn.close()
    out["ok"] = (out["invoices_with_overlapping_prices"] == 0 and out["rows"] > 0
                 and out["rows"] == out["rows_source_invoices"])
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        sys.exit("EROARE: copie incompletă, tabel gol sau prețuri suprapuse; varianta nu e echivalentă prin construcție")


def cmd_verify(args):
    conn = get_connection()
    q1_base = baseline_sql("benchmarks/postgres/q1_revenue.py", "Q1_SQL")
    q7_base = baseline_sql("benchmarks/postgres/q7_rolling_revenue.py", "Q7_SQL")
    lines, ok = [], True
    a, b = sorted(fetch(conn, q1_base)), sorted(fetch(conn, Q1_ATTR_SQL))
    same = a == b
    ok &= same and len(a) > 0
    lines.append(f"Q1 (12 luni până la AS_OF): baseline {len(a)} rânduri, variantă {len(b)} rânduri, {'IDENTICE' if same else 'DIFERITE'}")
    if not same:
        lines += [f"   doar baseline: {r}" for r in sorted(set(a) - set(b))[:10]]
        lines += [f"   doar variantă: {r}" for r in sorted(set(b) - set(a))[:10]]
    dmin, dmax = data_range(conn)
    rng = random.Random(42)
    max_start = max(0, (dmax - dmin).days - WINDOW_DAYS)
    starts = [dmin, dmin + timedelta(days=max_start)] + [random_window(dmin, dmax, rng)[0] for _ in range(5)]
    for s in starts:
        e = s + timedelta(days=WINDOW_DAYS - 1)
        a = sorted(fetch(conn, q7_base, (s, e, s, e, s, e)))
        b = sorted(fetch(conn, Q7_ATTR_SQL, {"s": s, "e": e}))
        same = a == b
        ok &= same and len(a) > 0
        lines.append(f"Q7 {s} → {e}: baseline {len(a)} rânduri, variantă {len(b)} rânduri, {'IDENTICE' if same else 'DIFERITE'}")
        if not same:
            lines += [f"   doar baseline: {r}" for r in sorted(set(a) - set(b))[:5]]
            lines += [f"   doar variantă: {r}" for r in sorted(set(b) - set(a))[:5]]
    # treceri pe tot intervalul: Q1 fără limita de 12 luni (acoperă și schimbarea de preț din 2024-06-01) și Q7 pe toate zilele
    win12 = AS_OF_SQL + " - INTERVAL '12 months'"
    if q1_base.count(win12) == 2 and Q1_ATTR_SQL.count(win12) == 1:
        a = sorted(fetch(conn, q1_base.replace(win12, "'-infinity'::timestamptz")))
        b = sorted(fetch(conn, Q1_ATTR_SQL.replace(win12, "'-infinity'::timestamptz")))
        same = a == b
        ok &= same and len(a) > 0
        lines.append(f"Q1 pe tot intervalul: baseline {len(a)} rânduri, variantă {len(b)} rânduri, {'IDENTICE' if same else 'DIFERITE'}")
        if not same:
            lines += [f"   doar baseline: {r}" for r in sorted(set(a) - set(b))[:10]]
            lines += [f"   doar variantă: {r}" for r in sorted(set(b) - set(a))[:10]]
    else:
        ok = False
        lines.append("Q1 pe tot intervalul: forma SQL-ului baseline nu e cea așteptată (fereastra de 12 luni); verificare imposibilă")
    a = sorted(fetch(conn, q7_base, (dmin, dmax, dmin, dmax, dmin, dmax)))
    b = sorted(fetch(conn, Q7_ATTR_SQL, {"s": dmin, "e": dmax}))
    same = a == b
    ok &= same and len(a) > 0
    lines.append(f"Q7 pe tot intervalul {dmin} → {dmax}: baseline {len(a)} rânduri, variantă {len(b)} rânduri, {'IDENTICE' if same else 'DIFERITE'}")
    conn.close()
    lines.append("REZULTAT: " + ("ALL EQUIVALENT" if ok else "NU SUNT ECHIVALENTE"))
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        sys.exit(1)


def make_fn(conn, query):
    if query == "Q1":
        def _run():
            with conn.cursor() as cur:
                cur.execute(Q1_ATTR_SQL)
                cur.fetchall()
        return _run
    dmin, dmax = data_range(conn)

    def _run7():
        s, e = random_window(dmin, dmax)
        with conn.cursor() as cur:
            cur.execute(Q7_ATTR_SQL, {"s": s, "e": e})
            cur.fetchall()
    return _run7


LABELS = {
    "Q1": ("Monthly revenue by subscription tier, 12 months to Q1_AS_OF, with price in effect. PostgreSQL with "
           "load-time attribution: invoice_attr is a full copy of invoices (same columns, physical order and indexes) "
           "plus attr_tier_id, attr_tier_name and attr_price_usd computed once with the baseline Q1/Q7 join logic, the "
           "relational counterpart of es_tier_attribution.py. Query: filter + group by, no attribution joins."),
    "Q7": ("7-day rolling average of daily revenue per tier over a random 183-day window, gap-filled with "
           "generate_series. PostgreSQL with load-time attribution (invoice_attr, as for Q1); attribution joins removed "
           "from the query, spine, gap-fill and window function unchanged."),
}


def cmd_run(args):
    conn = get_connection()
    try:
        run_benchmark(query_fn=make_fn(conn, args.query), db="postgres_attr", query_id=args.query,
                      label=LABELS[args.query], iterations=args.iterations, concurrency=1, output_path=args.output)
    finally:
        conn.close()


def cmd_explain(args):
    conn = get_connection()
    if args.query == "Q1":
        rows = fetch(conn, "EXPLAIN (ANALYZE, BUFFERS) " + Q1_ATTR_SQL)
    else:
        dmin, dmax = data_range(conn)
        s, e = random_window(dmin, dmax, random.Random(7))
        rows = fetch(conn, "EXPLAIN (ANALYZE, BUFFERS) " + Q7_ATTR_SQL, {"s": s, "e": e})
    conn.close()
    open(args.out, "w", encoding="utf-8").write("\n".join(r[0] for r in rows) + "\n")
    print("\n".join(r[0] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    v = sub.add_parser("verify"); v.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--query", choices=["Q1", "Q7"], required=True)
    r.add_argument("--iterations", type=int, default=1000); r.add_argument("--output", required=True)
    x = sub.add_parser("explain"); x.add_argument("--query", choices=["Q1", "Q7"], required=True); x.add_argument("--out", required=True)
    a = p.parse_args()
    {"build": cmd_build, "verify": cmd_verify, "run": cmd_run, "explain": cmd_explain}[a.cmd](a)
