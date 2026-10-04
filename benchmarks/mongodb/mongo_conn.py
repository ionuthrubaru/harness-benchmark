"""
benchmarks/mongodb/mongo_conn.py — MongoDB Connection Helper
=============================================================
Mirrors the pattern of benchmarks/postgres/pg_conn.py.

Provides get_db() for both naive and optimised schemas.
Both databases live on the same MongoDB instance — they are
separate logical databases, not separate containers.

    MONGO_DB_NAIVE      — flat collections mirroring PostgreSQL schema
    MONGO_DB_OPTIMISED  — embedded documents, idiomatic schema redesign

Usage:
    from mongo_conn import get_db

    db = get_db()                    # naive (default)
    db = get_db(schema="naive")      # naive (explicit)
    db = get_db(schema="optimised")  # optimised
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """
    Return a module-level singleton MongoClient.
    Safe to call repeatedly — only one connection is created.
    Thread-safe: pymongo's MongoClient manages its own internal
    connection pool and is designed to be shared across threads.
    """
    global _client
    if _client is None:
        user     = os.getenv("MONGO_USER")
        password = os.getenv("MONGO_PASSWORD")
        if not user or not password:
            raise RuntimeError("MONGO_USER and MONGO_PASSWORD must be set in .env")
        _client = MongoClient(
            f"mongodb://{user}:{password}@localhost:27017/",
            serverSelectionTimeoutMS=5_000,
        )
    return _client


def get_db(schema: str = "naive"):
    """
    Return the pymongo Database object for the requested schema.

    Parameters
    ----------
    schema : str
        "naive"     — MONGO_DB_NAIVE     (default, flat collections)
        "optimised" — MONGO_DB_OPTIMISED (embedded document schema)

    Raises
    ------
    ValueError   if schema is not "naive" or "optimised"
    RuntimeError if the required .env variable is not set
    """
    if schema == "naive":
        db_name = os.getenv("MONGO_DB_NAIVE")
        if not db_name:
            raise RuntimeError("MONGO_DB_NAIVE not set in .env")
    elif schema == "optimised":
        db_name = os.getenv("MONGO_DB_OPTIMISED")
        if not db_name:
            raise RuntimeError("MONGO_DB_OPTIMISED not set in .env")
    else:
        raise ValueError(f"Unknown schema '{schema}' — must be 'naive' or 'optimised'")

    return get_client()[db_name]