# StreamCart benchmark harness — PostgreSQL vs. five specialised databases

Companion archive for the dissertation *Performance Gain of Using a Specialised Database*
(Ioana Feraru). It contains the data generators, loaders, benchmark drivers, the timing
harness, the Docker infrastructure and **all raw results** (every individual timing).

**Contents at a glance**

| | |
|---|---|
| Engines | PostgreSQL 16 (baseline), MongoDB 7.0, Neo4j 5.18 CE, Elasticsearch 8.13, Cassandra 4.1, TimescaleDB (pg16) |
| Levels | every specialised engine twice: **naive** (relational schema ported literally) and **optimised** (schema designed for the engine) |
| Workload | Q1–Q7 reads (1000 timed iterations each) · Q8 concurrent writes (1,000,000 events at 10 / 50 / 100 threads) · Q1–Q7 at 10 % and 50 % data scale |
| Raw results | 77 read files · 18 write files · 140 scalability files + 11 summaries, all with `raw_timings_ms` |

---

## 1. Where things are

```
docker-compose.yml          all 7 containers (2 × Neo4j: naive :7687, optimised :7688)
.env.example                copy to .env and set passwords
requirements-lock.txt       exact Python package versions used (Python 3.11)
schema.sql                  PostgreSQL baseline schema
generators/                 synthetic data generators (seed 42)       → data/*.csv
loaders/                    one loader per engine × level
benchmarks/harness.py       the timing harness (run_benchmark)
benchmarks/<engine>/        drivers, Q8, scalability, results
analysis/                   equivalence check + summary rebuild
```

Per engine (paths relative to `benchmarks/`):

| Engine | Q1–Q7 drivers | Q1–Q7 results | Q8 driver → result | Scalability results |
|---|---|---|---|---|
| PostgreSQL | `postgres/q1…q7_*.py` | `postgres/results/postgres_q*_baseline.json` | `postgres/q8_write.py` → `postgres/results/postgres_q8_write_baseline_{10,50,100}.json` | `postgres/results/scale/` |
| MongoDB | `mongodb/{naive,optimised}/q1…q7_*.py` | `mongodb/{naive,optimised}/results/` | `mongodb/q8_write.py` → `mongodb/mongodb_Q8_*.json` | `mongodb/naive/results/scale/`, `mongodb/optimised/results/scalability/` |
| Neo4j | `neo4j/{naive,optimised}/q1…q7_*.py` | `neo4j/naive/results/`, `neo4j/optimised/result/` | `neo4j/q8_write.py` → `neo4j/neo4j_q8_*.json` | `…/scalability/` under each |
| Elasticsearch | `elasticsearch/{naive,optimised}/run_benchmarks.py` (+ `es_revenue.py` for Q1/Q7) | `elasticsearch/{naive,optimised}/results/` | `elasticsearch/naive/q8_write.py` → `…/naive/results/elasticsearch_naive_q8_*.json` | `…/results/scale/` |
| Cassandra | `cassandra/{naive,optimised}/q1…q7_*.py` | `cassandra/{naive,optimised}/results/` | `cassandra/q8_write.py` → `cassandra/cassandra_Q8_*.json` | `cassandra/naive/results/scale/`, `cassandra/optimised/results/scalability/` |
| TimescaleDB | `timescaleDB/naive/q1…q7_*.py`, `timescaleDB/optimised/run_benchmarks.py` | `timescaleDB/{naive,optimised}/results/` | `timescaleDB/q8_write.py` → `timescaleDB/timescaledb_Q8_*.json` | `…/results/scale/` |

Q8 has a single level per engine (no naive/optimised split). Scalability scripts:
`run_scalability.py` in each engine folder (Cassandra and Neo4j: one script for both levels).

## 2. The harness and what is timed

`benchmarks/harness.py → run_benchmark(query_fn, db, query_id, iterations, concurrency, output_path)`

* 10 untimed warm-up calls, then `iterations` timed calls (`time.perf_counter()` around the
  **whole** `query_fn` call); for concurrency > 1, threads start together on a barrier.
* Writes JSON: `db`, `query_id`, `label` (what exactly was measured), `timestamp`,
  `iterations`, `warmup_runs`, `concurrency`, `wall_time_s`, `latency_ms`
  (p50/p95/p99/mean/std_dev/min/max, nearest-rank) and **`raw_timings_ms`** (every run).
* Q8 drivers use the same statistics plus `n_threads`, `total_events`,
  `throughput_events_per_sec`, `errors`, Docker `resource_stats`.

**Client-side post-processing is inside the timing.** Anything a driver does in Python inside
`query_fn` (joins, aggregation, gap-filling, rolling averages) is part of the measured
latency. Only one-off preparation before `run_benchmark` (date ranges, random ID pools) is
untimed.

### Q1 and Q7: every engine returns the full PostgreSQL result

* **Q1**: revenue per month × subscription tier × price in effect at invoice time
  (`invoice_count`, `total_revenue_usd`), last 12 months.
* **Q7**: daily revenue per tier over a 183-day window, **gap-filled** (every day × every
  tier) with a **7-day rolling average** (current day + 6 preceding).

| Engine / level | Q1 | Q7 |
|---|---|---|
| PostgreSQL | server: CTEs, `JOIN LATERAL`, temporal JOIN on pricing | server: `generate_series`, window function |
| MongoDB naive / opt. | client attribution + aggregation, timed | client gap-fill + rolling, timed |
| Cassandra naive | client join + aggregation, timed¹ | client attribution, gap-fill, rolling, timed² |
| Cassandra optimised | tier + price pre-resolved at load; sum in client, timed | partition per tier; gap-fill + rolling in client, timed |
| Neo4j naive / opt. | Cypher traversal; temporal price + aggregation in client, timed | **one Cypher query, server-side** (`range()` gap-fill, list-slice rolling avg) |
| Elasticsearch naive | client, timed: PIT scans of subscriptions + paid invoices, tier attribution, temporal price, aggregation | client, timed: same attribution, daily aggregation, gap-fill, rolling avg |
| Elasticsearch optimised | tier + price denormalised at load (`loaders/es_tier_attribution.py`); **one server-side aggregation** | **one server-side search**: `filters(tier)` → `date_histogram(min_doc_count:0, extended_bounds)` → `sum` + `moving_fn` |
| TimescaleDB naive / opt. | server (SQL) | server (`time_bucket_gapfill`, continuous aggregates in optimised) |

¹ Uses the price at the window start instead of per invoice; every invoice in the Q1 window
post-dates the only price change (2024-06-01), so the values are identical.
² The subscription lookup is built once at startup (untimed).

**Equivalence evidence.** `analysis/check_q1_q7_equivalence.py` runs Q1 and Q7 on PostgreSQL
and on Neo4j / Elasticsearch for the same windows and compares row by row. Result
(3 October 2026):

```
Q1  Elasticsearch naive      20 rows  max abs diff 0.0000  ✔
Q1  Elasticsearch optimised  20 rows  max abs diff 0.0100  ✔
Q7  (3 windows × 549 rows)   Neo4j naive, Neo4j optimised,
                             Elasticsearch naive, Elasticsearch optimised   max diff ≤ 0.0100  ✔
RESULT: ALL EQUIVALENT ✔
```
Differences of 0.01 are cent rounding (Elasticsearch stores `total_usd` as 32-bit `float`).
The load-time tier attribution was additionally checked against the PostgreSQL SQL on all
724,237 invoices: 0 mismatches.

## 3. Reproducing

### 3.1 Setup
```powershell
copy .env.example .env                     # then edit the passwords
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements-lock.txt
docker compose up -d                       # or start only what you need (see 3.4)
```

### 3.2 Data and loading
```powershell
python generators/users.py ; python generators/products.py ; python generators/subscriptions.py
python generators/orders.py ; python generators/sessions.py ; python generators/events.py
python create_baseline_schema.py ; python loaders/postgres_loader.py
python loaders/<engine>_naive_loader.py ; python loaders/<engine>_optimised_loader.py
```
`data/*.csv` (~2 GB) is not included. Generators are seeded (NumPy, Faker), but primary keys
use `uuid4()` and session tokens use `secrets`, so a regenerated dataset is statistically
identical, not byte-identical. The exact CSVs are available on request.

### 3.3 Running benchmarks (examples)
```powershell
python benchmarks/postgres/q1_revenue.py                          # one query
python benchmarks/postgres/q1_revenue.py --dry-run                # run once, print rows
python benchmarks/elasticsearch/optimised/run_benchmarks.py --only Q1 Q7
python benchmarks/neo4j/naive/q7_rolling_revenue.py --iterations 100
python benchmarks/neo4j/q8_write.py --threads 100                 # Q8
python benchmarks/neo4j/run_scalability.py --only Q7 --schema naive --results-dir benchmarks/neo4j/naive/results/scalability
python analysis/check_q1_q7_equivalence.py                        # needs postgres + neo4j + ES up
python analysis/rebuild_scalability_summaries.py --all
```

### 3.4 Practical notes (learned the hard way)
* **Run one engine at a time.** All containers together exceed a 12.5 GB Docker VM; two Neo4j
  containers alone need ~14 GB. Stop what you are not measuring: `docker compose stop <service>`.
* **Wait ~30 s after starting a container.** Neo4j in particular refuses connections
  ("incomplete handshake") while it is still starting.
* **Neo4j memory is set explicitly** (heap 3 GB, page cache 3 GB, in `docker-compose.yml`).
  Without it, Q8 at 100 threads made the Docker VM run out of memory and the Neo4j process was
  killed silently. A Q8 run with errors or missing events is saved as `*_FAILED.json`.
* **Elasticsearch optimised Q1/Q7 needs the tier fields.** After loading the optimised indices
  with an older loader, run `python loaders/es_tier_attribution.py` once (adds `tier_id`,
  `tier_name`, `price_in_effect_usd` to `optimised_invoices`; ~2 min).
* **Elasticsearch naive Q1/Q7 is slow by design** (~6.6 s / ~3.1 s per call → ~3 h for both at
  1000 iterations): every call scans hundreds of thousands of documents and joins in Python.

## 4. Known caveats

* **Q1 window depends on the wall clock** (`NOW() - 12 months`; data spans 2024-01-01 →
  2026-01-01). The original runs were made in March 2026; the re-run Elasticsearch Q1 is
  pinned to the PostgreSQL baseline run time (`es_revenue.Q1_AS_OF` = 2026-03-04T14:51:58Z).
  To reproduce exactly, pin `NOW()` the same way.
* **Q1 scalability points are not comparable across engines — do not use them.** In the
  PostgreSQL, TimescaleDB and Elasticsearch scalability scripts the window
  `created_at >= NOW() - 12 months AND created_at < cutoff` is empty (both cutoffs are earlier
  than `NOW() - 12 months`), which is why those Q1 points are flat at 1–50 ms; MongoDB,
  Neo4j and Cassandra used other windows. Q2–Q7 scalability points are unaffected.
* **Scalability cutoffs** differ slightly: PostgreSQL, Neo4j, Elasticsearch, TimescaleDB use
  2024-03-20 / 2025-01-04; Cassandra and MongoDB 2024-03-14 / 2024-12-31 (different min date).
* **Cassandra naive Q6 scalability** uses 30 iterations and 2 warm-ups (each call is a ~2 min
  full scan); every other point uses 1000 iterations and 10 warm-ups.
* **Re-runs on 3 October 2026** (everything else is from March 2026), same machine and data:
  - Neo4j Q8 (10/50/100): uniqueness constraint on `:EventQ8(id)` added and the label cleared
    before each run — earlier runs lacked the constraint (MERGE did a label scan per insert,
    ~15 events/s) and are superseded. The `staging` block in each JSON records this.
    The 100-thread file's staging count (883,926 rather than 1,000,000) reflects an
    interrupted 100-thread attempt between the 50- and 100-thread runs, stopped after
    clearing the label and inserting 883,926 events; the recorded run then started,
    as verified, from an empty label.
  - Neo4j Q7 and Elasticsearch Q1/Q7 (+ their Q7 scalability): earlier versions returned a
    partial result (no gap-fill / rolling average in Neo4j; no tier breakdown in
    Elasticsearch). Re-implemented to return the full PostgreSQL result (see §2).
  - Neo4j Q1–Q6 were measured with automatic memory sizing, which chose the same 3 GB heap
    that is now configured explicitly.
* **Floating-point amounts in Elasticsearch** (`float` mapping) → sums may differ from
  PostgreSQL `NUMERIC` by a cent.
* **TimescaleDB image** uses the floating tag `latest-pg16`; the extension version can be read
  with `SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';`.
