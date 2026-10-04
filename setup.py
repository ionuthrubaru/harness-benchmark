"""
setup.py — Phase 0 Health Check
================================
Connects to all databases and prints OK / FAIL for each.
MongoDB is checked twice — once for the naive DB, once for optimised.
Neo4j is checked twice — once for the naive container, once for optimised.

Usage:
    python setup.py               # runs health check
    python setup.py --wait        # waits for all DBs to be ready (useful right after docker-compose up)

docker-compose up -d
python setup.py
--------------------
docker-compose down
"""

import sys
import time
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(name):
    print(f"  {GREEN}✔ {name:<32} OK{RESET}")

def fail(name, error):
    print(f"  {RED}✘ {name:<32} FAIL — {error}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠ {msg}{RESET}")

# ── individual checks ─────────────────────────────────────────────────────────

def check_postgres():
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
        connect_timeout=5,
    )
    conn.close()
    ok("PostgreSQL")
    return True


def _check_mongo_db(env_var: str, label: str) -> bool:
    """Shared logic for checking a named MongoDB database."""
    from pymongo import MongoClient
    user     = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    db_name  = os.getenv(env_var)
    if not db_name:
        raise RuntimeError(f"{env_var} not set in .env")
    client = MongoClient(
        f"mongodb://{user}:{password}@localhost:27017/",
        serverSelectionTimeoutMS=5000,
    )
    client.admin.command("ping")
    count = len(client[db_name].list_collection_names())
    client.close()
    ok(f"{label} ({db_name}, {count} collections)")
    return True


def check_mongodb_naive():
    return _check_mongo_db("MONGO_DB_NAIVE", "MongoDB naive")


def check_mongodb_optimised():
    return _check_mongo_db("MONGO_DB_OPTIMISED", "MongoDB optimised")


def _check_neo4j(port: int, label: str) -> bool:
    """Shared logic for checking a Neo4j container by Bolt port."""
    from neo4j import GraphDatabase
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    driver   = GraphDatabase.driver(
        f"bolt://localhost:{port}",
        auth=(user, password),
        connection_timeout=5,
    )
    driver.verify_connectivity()
    # Count nodes as a simple data presence check
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        count  = result.single()["cnt"]
    driver.close()
    ok(f"{label} (port {port}, {count:,} nodes)")
    return True


def check_neo4j_naive():
    return _check_neo4j(
        int(os.getenv("NEO4J_NAIVE_PORT", 7687)),
        "Neo4j naive",
    )


def check_neo4j_optimised():
    return _check_neo4j(
        int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)),
        "Neo4j optimised",
    )


def check_elasticsearch():
    import urllib.request
    import json
    url = "http://localhost:9200/_cluster/health"
    with urllib.request.urlopen(url, timeout=5) as resp:
        health = json.loads(resp.read())
    if health["status"] == "red":
        raise RuntimeError(f"Cluster health is RED: {health}")
    ok("Elasticsearch")
    return True


def check_cassandra():
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
    auth = PlainTextAuthProvider(
        username=os.getenv("CASSANDRA_USER"),
        password=os.getenv("CASSANDRA_PASSWORD"),
    )
    cluster = Cluster(
        ["127.0.0.1"],
        port=9042,
        auth_provider=auth,
        connect_timeout=10,
    )
    session = cluster.connect()
    session.execute("SELECT release_version FROM system.local")
    cluster.shutdown()
    ok("Cassandra")
    return True


def check_timescaledb():
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB"),
        connect_timeout=5,
    )
    cur = conn.cursor()
    cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb';")
    row = cur.fetchone()
    if not row:
        raise RuntimeError("timescaledb extension not available")
    conn.close()
    ok("TimescaleDB")
    return True


# ── runner ────────────────────────────────────────────────────────────────────

# optional=True — will legitimately be empty until the optimised loader has run.
CHECKS = [
    ("PostgreSQL",          check_postgres,           False),
    ("MongoDB naive",       check_mongodb_naive,      False),
    ("MongoDB optimised",   check_mongodb_optimised,  True),
    ("Neo4j naive",         check_neo4j_naive,        False),
    ("Neo4j optimised",     check_neo4j_optimised,    True),  # optional until loader runs
    ("Elasticsearch",       check_elasticsearch,      False),
    ("Cassandra",           check_cassandra,          False),
    ("TimescaleDB",         check_timescaledb,        False),
]


def run_checks(wait=False, max_wait_seconds=120):
    print("\n" + "═" * 55)
    print("  Dissertation — Database Health Check")
    print("═" * 55)

    results  = {}
    deadline = time.time() + max_wait_seconds

    for name, check_fn, optional in CHECKS:
        while True:
            try:
                check_fn()
                results[name] = True
                break
            except Exception as e:
                if wait and time.time() < deadline:
                    print(f"  {YELLOW}⟳ {name:<32} not ready yet, retrying in 5s...{RESET}")
                    time.sleep(5)
                else:
                    if optional:
                        print(f"  {YELLOW}○ {name:<32} not loaded yet (optional){RESET}")
                    else:
                        fail(name, str(e))
                    results[name] = optional
                    break

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    hard_failures = [
        name for (name, _, optional), result
        in zip(CHECKS, results.values())
        if not result and not optional
    ]

    print(f"  Result: {passed}/{total} checks passed\n")

    if not hard_failures:
        print(f"  {GREEN}All systems GO{RESET}\n")
        optional_skipped = [
            name for (name, _, optional) in CHECKS
            if optional and not results.get(name)
        ]
        if optional_skipped:
            for name in optional_skipped:
                warn(f"{name} not yet loaded — run the optimised loader when ready")
        return 0
    else:
        print(f"  {RED}Fix the following before proceeding: {', '.join(hard_failures)}{RESET}\n")
        if not wait:
            warn("Tip: run with --wait if containers are still starting up")
        return 1


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dissertation DB health check")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Keep retrying for up to 2 minutes (useful right after docker-compose up)",
    )
    args = parser.parse_args()
    sys.exit(run_checks(wait=args.wait))