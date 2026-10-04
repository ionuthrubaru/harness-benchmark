"""
analysis/rebuild_scalability_summaries.py
=========================================
Rebuilds every <engine>_scalability_summary.json from the per-query
scale files (q1..q7 x 10% / 50%) that sit next to it.

Why: the per-query JSON files are the primary raw results. Some summaries
were stale or missing (Cassandra naive/optimised summaries held only the
last 4 entries of one run and were byte-identical copies; MongoDB naive and
Neo4j naive/optimised had no summary). This script regenerates them
deterministically from the per-query files without touching those files.

Usage (from the repo root):
    python analysis/rebuild_scalability_summaries.py        # the 5 stale/missing ones
    python analysis/rebuild_scalability_summaries.py --all  # every engine
"""

import glob
import json
import os
import re
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (results dir, per-query glob, summary filename, db name)
TARGETS = [
    ("benchmarks/postgres/results/scale",                 "postgres_q*_scale*.json",            "postgres_scalability_summary.json",            "postgres"),
    ("benchmarks/mongodb/naive/results/scale",            "mongodb_naive_Q*_scale*.json",       "mongodb_naive_scalability_summary.json",       "mongodb_naive"),
    ("benchmarks/mongodb/optimised/results/scalability",  "mongodb_optimised_Q*_scale*.json",   "mongodb_optimised_scalability_summary.json",   "mongodb_optimised"),
    ("benchmarks/neo4j/naive/results/scalability",        "neo4j_naive_q*_scale*.json",         "neo4j_naive_scalability_summary.json",         "neo4j_naive"),
    ("benchmarks/neo4j/optimised/result/scalability",     "neo4j_optimised_q*_scale*.json",     "neo4j_optimised_scalability_summary.json",     "neo4j_optimised"),
    ("benchmarks/cassandra/naive/results/scale",          "cassandra_naive_q*_scale*.json",     "cassandra_naive_scalability_summary.json",     "cassandra_naive"),
    ("benchmarks/cassandra/optimised/results/scalability","cassandra_optimised_q*_scale*.json", "cassandra_optimised_scalability_summary.json", "cassandra_optimised"),
    ("benchmarks/elasticsearch/naive/results/scale",      "elasticsearch_naive_q*_scale*.json", "elasticsearch_naive_scalability_summary.json", "elasticsearch_naive"),
    ("benchmarks/elasticsearch/optimised/results/scale",  "elasticsearch_optimised_q*_scale*.json", "elasticsearch_optimised_scalability_summary.json", "elasticsearch_optimised"),
    ("benchmarks/timescaleDB/naive/results/scale",        "timescaledb_naive_q*_scale*.json",   "timescaledb_naive_scalability_summary.json",   "timescaledb_naive"),
    ("benchmarks/timescaleDB/optimised/results/scale",    "timescaledb_optimised_q*_scale*.json", "timescaledb_optimised_scalability_summary.json", "timescaledb_optimised"),
]

KEY_RE = re.compile(r"_q(\d)_scale(\d+)\.json$", re.IGNORECASE)
CUTOFF_RE = re.compile(r"cutoff (\d{4}-\d{2}-\d{2})")


def rebuild(rel_dir, pattern, summary_name, db):
    d = os.path.join(ROOT, rel_dir)
    files = glob.glob(os.path.join(d, pattern))
    if not files:
        print(f"  skip  {rel_dir} (no per-query files)")
        return
    entries = []
    cutoffs = {}
    for f in files:
        m = KEY_RE.search(f)
        if not m:
            continue
        q, scale = int(m.group(1)), int(m.group(2))
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        r.setdefault("scale_pct", scale)
        r["source_file"] = os.path.basename(f)
        c = CUTOFF_RE.search(r.get("label", ""))
        if c:
            cutoffs[f"cutoff_{scale}pct"] = c.group(1)
        entries.append(((scale, q), r))
    entries.sort(key=lambda e: e[0])
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/rebuild_scalability_summaries.py",
        "db": db,
        "cutoffs": cutoffs,
        "n_benchmarks": len(entries),
        "benchmarks": [r for _, r in entries],
    }
    path = os.path.join(d, summary_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(entries)} entries, cutoffs={cutoffs})")


DEFAULT_DBS = {"mongodb_naive", "neo4j_naive", "neo4j_optimised",
               "cassandra_naive", "cassandra_optimised"}

if __name__ == "__main__":
    import sys
    rebuild_all = "--all" in sys.argv
    for t in TARGETS:
        if rebuild_all or t[3] in DEFAULT_DBS:
            rebuild(*t)
