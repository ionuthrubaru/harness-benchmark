"""
benchmarks/neo4j/neo4j_conn.py — Shared Neo4j connection helper
================================================================
All Neo4j benchmark scripts import get_driver() from here.

Two separate Neo4j containers are used (Community Edition does not
support multiple databases per instance):
  neo4j_naive      — bolt://localhost:7687  (flat schema, FK-equivalent indexes)
  neo4j_optimised  — bolt://localhost:7688  (ALSO_BOUGHT edges, full-text indexes)

Usage:
    from neo4j_conn import get_driver

    # naive benchmark scripts
    driver = get_driver(port=int(os.getenv("NEO4J_NAIVE_PORT", 7687)))

    # optimised benchmark scripts
    driver = get_driver(port=int(os.getenv("NEO4J_OPTIMISED_PORT", 7688)))

    # Q8 concurrent write — needs pool >= thread count
    driver = get_driver(port=..., max_connection_pool_size=110)

    # always close the driver when done
    driver.close()

Session configuration:
  - Connection pool size defaults to 10 (sufficient for single-threaded
    Q1–Q7 benchmarks). Q8 passes max_connection_pool_size=n_threads+10
    so each of the 100 worker threads can hold a session open for its
    entire slice without contention.
  - fetch_size=1000 — controls how many records are pulled per round-trip
    from the server. Default is 1000 in the Python driver; set explicitly
    for clarity and reproducibility.
"""

import os
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

load_dotenv()


def get_driver(port: int = 7687, max_connection_pool_size: int = 10) -> Driver:
    """
    Return a Neo4j Driver connected to the given Bolt port.

    Parameters
    ----------
    port : int
        Bolt port of the target container.
        Pass int(os.getenv("NEO4J_NAIVE_PORT", 7687))      for naive.
        Pass int(os.getenv("NEO4J_OPTIMISED_PORT", 7688))  for optimised.

    max_connection_pool_size : int
        Maximum number of connections in the driver pool.
        Default 10 is fine for single-threaded Q1–Q7.
        Q8 should pass n_threads + 10 so all worker threads can each
        hold one session open concurrently without pool exhaustion.
    """
    uri      = f"bolt://localhost:{port}"
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_pool_size=max_connection_pool_size,
        fetch_size=1000,
        connection_timeout=10,
    )
    return driver