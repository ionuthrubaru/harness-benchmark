"""
benchmarks/cassandra/cassandra_conn.py — Shared Cassandra connection helper
============================================================================
All Cassandra benchmark scripts (Q1–Q8, run_scalability) import get_session()
from here instead of constructing Cluster/Session objects locally.

Returns both (cluster, session) — callers must call cluster.shutdown() when
done to release pooled connections cleanly. This mirrors the pattern of
pg_conn.py (which returns a single psycopg2 connection), adapted for the
Cassandra driver's two-object model.

Execution profile settings applied to every session
─────────────────────────────────────────────────────
  consistency_level = LOCAL_ONE
        Requires acknowledgement from exactly one local replica. On a
        single-node Docker instance there is only one replica, so LOCAL_ONE
        is simultaneously the weakest and the strongest consistency level
        available — it cannot be reduced further and is equivalent to the
        default READ COMMITTED isolation of PostgreSQL benchmarks.
        Using LOCAL_ONE rather than ONE ensures the load-balancing policy
        never attempts to contact a node outside the local datacenter.

  request_timeout = 30.0 s
        The driver default is 10 s. Extended for two reasons:
          1. Cold-start warmup iterations may hit an empty OS page cache and
             take longer than on subsequent runs. A 10 s timeout would cause
             spurious TimeoutException errors during the 10 warmup iterations
             that the harness discards anyway.
          2. The naive schema requires ALLOW FILTERING on most queries, forcing
             full table scans. On the events table (potentially millions of rows)
             a naive Q6 or Q1 scan can exceed 10 s on early iterations.
        This setting is documented in the methodology as benchmark configuration,
        not a production tuning choice.

  fetch_size = 10 000
        Number of rows fetched per page when a result set spans multiple pages
        (the driver default is 5 000). Doubling it halves the number of
        round-trips for Q6 (events in a 30-day window, potentially thousands
        of rows per user). All benchmark scripts materialise the full result
        via list(session.execute(...)), so fetch_size affects latency without
        affecting correctness. 10 000 was chosen as a round number that keeps
        individual network packets well under typical MTU constraints.

No analogues to PostgreSQL's jit=off or work_mem exist in Cassandra
─────────────────────────────────────────────────────────────────────
  • Cassandra does not use a JIT compiler for CQL execution; there is no
    per-session JIT toggle.

  • Memtable / compaction memory is governed by cassandra.yaml server-side
    (MAX_HEAP_SIZE=512M, HEAP_NEWSIZE=128M set in docker-compose). These are
    fixed at container startup and cannot be altered per-session. They are
    noted in the methodology chapter as the server configuration for this
    experiment.

  • There is no work_mem equivalent. Cassandra does not perform in-process
    sort operations that can spill to disk the way PostgreSQL does for
    ORDER BY over large datasets. Sort operations in Cassandra are either
    served by clustering column order (zero additional cost) or performed
    client-side in Python.

Usage
──────
    from cassandra_conn import get_session

    cluster, session = get_session(keyspace="cassandra_naive")
    try:
        rows = list(session.execute("SELECT * FROM events LIMIT 10"))
    finally:
        cluster.shutdown()
"""

import os

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import ConsistencyLevel
from dotenv import load_dotenv

load_dotenv()


def get_session(keyspace: str | None = None, request_timeout: float = 30.0):
    """
    Create a Cassandra Cluster, open a Session, and return both.

    Parameters
    ----------
    keyspace : str or None
        If provided, the session connects directly to this keyspace (equivalent
        to 'USE keyspace' at connect time). If None the session is connected at
        the cluster level; the caller must call session.set_keyspace() before
        issuing any table queries. Pass None when the keyspace may not exist yet
        (e.g. in the loaders, which create the keyspace before switching to it).

    request_timeout : float
        Per-request timeout in seconds, applied to the default ExecutionProfile.
        Default is 30.0s, sufficient for all optimised queries and most naive ones.
        Callers that issue full table scans (e.g. naive Q6 on ~1M event rows) should
        pass a higher value — e.g. get_session(keyspace=..., request_timeout=300.0).
        Cannot be set post-construction when ExecutionProfile is in use (the driver
        raises ValueError if you attempt session.default_timeout = x after the fact).

    Returns
    -------
    (cluster, session) : tuple
        Always call cluster.shutdown() in a finally block to return connections
        to the pool and avoid resource leaks.

    Example
    -------
        cluster, session = get_session(keyspace="cassandra_naive")
        try:
            ...
        finally:
            cluster.shutdown()

        # For naive Q6 (full events table scan):
        cluster, session = get_session(keyspace="cassandra_naive", request_timeout=300.0)
    """

    auth = PlainTextAuthProvider(
        username=os.getenv("CASSANDRA_USER", "cassandra"),
        password=os.getenv("CASSANDRA_PASSWORD", "cassandra"),
    )

    # ExecutionProfile bundles all per-request settings into a named profile.
    # EXEC_PROFILE_DEFAULT makes this the profile used for every session.execute()
    # call that does not explicitly name a different profile.
    # request_timeout is set here — it cannot be changed on the Session object
    # after construction when profiles are in use.
    profile = ExecutionProfile(
        # DCAwareRoundRobinPolicy restricts routing to the local datacenter.
        # 'datacenter1' matches CASSANDRA_DC in docker-compose. On a single-node
        # instance this has no practical effect, but using the correct policy
        # avoids driver warnings and is the correct production pattern.
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
        consistency_level=ConsistencyLevel.LOCAL_ONE,
        request_timeout=request_timeout,
    )

    cluster = Cluster(
        contact_points=["localhost"],
        port=int(os.getenv("CASSANDRA_PORT", "9042")),
        auth_provider=auth,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        # protocol_version is intentionally left unset (None) so the driver
        # negotiates the highest mutually supported version with the server.
        # Cassandra 4.1 supports protocol v4 and v5; the driver will select v5
        # if both sides agree. Forcing a version would require keeping this file
        # in sync with the server version — not worth the maintenance cost.
    )

    session = cluster.connect(keyspace)  # keyspace=None is valid (cluster-level connect)

    # Set default page size for multi-page result sets.
    # This applies to all statements that do not set fetch_size explicitly.
    session.default_fetch_size = 10_000

    return cluster, session