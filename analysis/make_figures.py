"""
analysis/make_figures.py — regenerate the article figures from the raw JSONs
============================================================================
Reads every benchmark result in benchmarks/ (main runs, Q8, scalability
summaries) and renders the figure set used in the article:

  F1  p50 latency heatmap, 11 configurations x Q1-Q7 (log10 colour)
  F2  Cliff's delta panels: A naive vs PostgreSQL, B optimized vs naive,
      C optimized vs PostgreSQL  (computed from raw_timings_ms)
  F3  engine-effect heatmap   (p50 naive / p50 PostgreSQL)
  F4  schema-effect heatmap   (p50 naive / p50 optimised)
  F5  Q8 write throughput vs concurrency (log y)
  F6  Q8 p99 insert latency heatmap (engines x threads)
  F7  scalability growth-factor heatmap, Q2-Q7, 10% -> 100%
      (Q1 is excluded: see README, "Known caveats")

Usage, from the repo root (no databases needed, JSONs only):
    python analysis/make_figures.py            # -> analysis/figures/
    python analysis/make_figures.py --out DIR
Outputs每 figure as 300-dpi PNG and vector PDF.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "benchmarks")

QUERIES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
ENGINE_LABEL = {
    "postgres": "PostgreSQL", "mongodb": "MongoDB", "cassandra": "Cassandra",
    "neo4j": "Neo4j", "elasticsearch": "Elasticsearch", "timescaledb": "TimescaleDB",
}
ROW_ORDER = [
    ("postgres", "base"), ("mongodb", "naive"), ("mongodb", "optimised"),
    ("cassandra", "naive"), ("cassandra", "optimised"),
    ("neo4j", "naive"), ("neo4j", "optimised"),
    ("elasticsearch", "naive"), ("elasticsearch", "optimised"),
    ("timescaledb", "naive"), ("timescaledb", "optimised"),
]
SPEC_ENGINES = ["mongodb", "cassandra", "neo4j", "elasticsearch", "timescaledb"]


def row_label(engine, level):
    base = ENGINE_LABEL[engine]
    return base if level == "base" else f"{base} ({'n' if level == 'naive' else 'o'})"


# ── discovery ─────────────────────────────────────────────────────────────────

def discover():
    """Return (reads, q8, summaries).
    reads[(engine, level, query)] = dict with 'p50' and 'raw' (np.array)."""
    reads, q8, summaries = {}, [], []
    for dirpath, _dirs, files in os.walk(BENCH):
        rel = os.path.relpath(dirpath, BENCH).replace("\\", "/")
        parts = rel.split("/")
        engine = parts[0].lower() if parts and parts[0] != "." else None
        level = "naive" if "naive" in parts else (
            "optimised" if "optimised" in parts else "base")
        in_scale = any(p.lower() in ("scale", "scalability") for p in parts)
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                d = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            if fn.endswith("scalability_summary.json"):
                summaries.append(d)
                continue
            if d.get("throughput_events_per_sec") is not None:
                q8.append(d)
                continue
            qid = str(d.get("query_id", "")).upper()
            if (qid in QUERIES
                    and "scale_pct" not in d and not in_scale and engine):
                key = (engine, level, qid)
                prev = reads.get(key)
                if prev is None or d.get("timestamp", "") > prev["ts"]:
                    reads[key] = {
                        "p50": d["latency_ms"]["p50"],
                        "p99": d["latency_ms"].get("p99"),
                        "raw": np.asarray(d["raw_timings_ms"], dtype=float),
                        "ts": d.get("timestamp", ""),
                    }
    return reads, q8, summaries


def cliff_delta(a, b):
    sb = np.sort(b)
    g = np.searchsorted(sb, a, "left").sum()
    l = (len(b) - np.searchsorted(sb, a, "right")).sum()
    return (g - l) / (len(a) * len(b))


def magn(d):
    d = abs(d)
    return "L" if d >= .474 else "M" if d >= .33 else "S" if d >= .147 else "N"


def fmt_ms(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if v < 10:
        return f"{v:.2f}"
    if v < 1000:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def annot(ax, M, text, fontsize=7):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = text[i][j]
            if t:
                ax.text(j, i, t, ha="center", va="center", fontsize=fontsize)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name + ".png/.pdf")


# ── figures ───────────────────────────────────────────────────────────────────

def f1_latency_heatmap(reads, out):
    M = np.full((len(ROW_ORDER), 7), np.nan)
    for i, (e, l) in enumerate(ROW_ORDER):
        for j, q in enumerate(QUERIES):
            r = reads.get((e, l, q))
            if r:
                M[i, j] = r["p50"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(np.log10(M), cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(7), QUERIES)
    ax.set_yticks(range(len(ROW_ORDER)), [row_label(*r) for r in ROW_ORDER])
    annot(ax, M, [[fmt_ms(M[i, j]) for j in range(7)] for i in range(len(ROW_ORDER))])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("log10 p50 latency (ms)")
    ax.set_title("p50 latency by configuration and query (ms, log colour scale)")
    save(fig, out, "f1_latency_heatmap")


def f2_cliff_panels(reads, out):
    panels = [
        ("A  naive vs PostgreSQL", lambda e, q: (reads.get((e, "naive", q)),
                                                 reads.get(("postgres", "base", q)))),
        ("B  optimized vs naive", lambda e, q: (reads.get((e, "optimised", q)),
                                                reads.get((e, "naive", q)))),
        ("C  optimized vs PostgreSQL", lambda e, q: (reads.get((e, "optimised", q)),
                                                     reads.get(("postgres", "base", q)))),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.6))
    for ax, (title, pick) in zip(axes, panels):
        D = np.full((len(SPEC_ENGINES), 7), np.nan)
        T = [["" for _ in range(7)] for _ in SPEC_ENGINES]
        for i, e in enumerate(SPEC_ENGINES):
            for j, q in enumerate(QUERIES):
                a, b = pick(e, q)
                if a and b:
                    d = cliff_delta(a["raw"], b["raw"])
                    D[i, j] = d
                    T[i][j] = f"{d:+.2f}\n{magn(d)}"
        im = ax.imshow(D, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(7), QUERIES)
        ax.set_yticks(range(len(SPEC_ENGINES)),
                      [ENGINE_LABEL[e] for e in SPEC_ENGINES])
        annot(ax, D, T, fontsize=6.5)
        ax.set_title(f"Cliff's delta, panel {title} "
                     "(positive = first distribution slower)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Cliff's delta")
    save(fig, out, "f2_cliffs_delta_panels")


def _ratio_heatmap(reads, out, name, title, denom_of):
    R = np.full((len(SPEC_ENGINES), 7), np.nan)
    for i, e in enumerate(SPEC_ENGINES):
        for j, q in enumerate(QUERIES):
            num = reads.get((e, "naive", q))
            den = denom_of(e, q)
            if num and den:
                R[i, j] = num["p50"] / den["p50"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    vmax = np.nanmax(R)
    im = ax.imshow(R, cmap="RdYlGn_r" if "schema" not in name else "RdYlGn",
                   norm=LogNorm(vmin=max(np.nanmin(R), 1e-2), vmax=vmax),
                   aspect="auto")
    ax.set_xticks(range(7), QUERIES)
    ax.set_yticks(range(len(SPEC_ENGINES)), [ENGINE_LABEL[e] for e in SPEC_ENGINES])
    annot(ax, R, [[("" if math.isnan(R[i, j]) else
                    (f"{R[i, j]:,.0f}x" if R[i, j] >= 100 else f"{R[i, j]:.1f}x"
                     if R[i, j] >= 1 else f"{R[i, j]:.2f}x"))
                   for j in range(7)] for i in range(len(SPEC_ENGINES))])
    fig.colorbar(im, ax=ax, shrink=0.85, label="ratio (log scale)")
    ax.set_title(title, fontsize=10)
    save(fig, out, name)


def f5_f6_q8(q8, out):
    def eng(db):
        return ENGINE_LABEL[db.split("_")[0].lower()]
    series = {}
    for d in q8:
        series.setdefault(eng(d["db"]), {})[int(d["n_threads"])] = d
    order = ["PostgreSQL", "TimescaleDB", "Cassandra", "MongoDB",
             "Neo4j", "Elasticsearch"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in order:
        pts = series.get(name, {})
        xs = sorted(pts)
        ys = [pts[x]["throughput_events_per_sec"] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
        ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), fontsize=7,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks([10, 50, 100])
    ax.set_xlabel("concurrent threads")
    ax.set_ylabel("events per second (log)")
    ax.set_title("Q8: write throughput vs concurrency (1,000,000 single-record inserts)")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=.3)
    save(fig, out, "f5_q8_throughput")

    P = np.full((len(order), 3), np.nan)
    for i, name in enumerate(order):
        for j, t in enumerate([10, 50, 100]):
            d = series.get(name, {}).get(t)
            if d:
                P[i, j] = d["latency_ms"]["p99"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(P, cmap="RdYlBu_r", norm=LogNorm(), aspect="auto")
    ax.set_xticks(range(3), ["10 thr", "50 thr", "100 thr"])
    ax.set_yticks(range(len(order)), order)
    annot(ax, P, [[fmt_ms(P[i, j]) for j in range(3)] for i in range(len(order))])
    fig.colorbar(im, ax=ax, shrink=0.85, label="p99 insert latency (ms, log)")
    ax.set_title("Q8: p99 insert latency")
    save(fig, out, "f6_q8_p99_heatmap")


def f7_growth(reads, summaries, out):
    qs = QUERIES[1:]  # Q1 excluded, see README
    p10 = {}
    for s in summaries:
        db = s["db"].lower()
        engine = db.split("_")[0]
        level = "naive" if "naive" in db else ("optimised" if "optimised" in db else "base")
        for e in s["benchmarks"]:
            if e.get("scale_pct") == 10 and str(e.get("query_id", "")).upper() in qs:
                p10[(engine, level, e["query_id"].upper())] = e["latency_ms"]["p50"]
    G = np.full((len(ROW_ORDER), len(qs)), np.nan)
    for i, (e, l) in enumerate(ROW_ORDER):
        for j, q in enumerate(qs):
            full = reads.get((e, l, q))
            small = p10.get((e, l, q))
            if full and small:
                G[i, j] = full["p50"] / small
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(np.log10(G), cmap="RdYlGn_r",
                   norm=TwoSlopeNorm(vcenter=0.0), aspect="auto")
    ax.set_xticks(range(len(qs)), qs)
    ax.set_yticks(range(len(ROW_ORDER)), [row_label(*r) for r in ROW_ORDER])
    annot(ax, G, [[("" if math.isnan(G[i, j]) else f"{G[i, j]:.1f}x")
                   for j in range(len(qs))] for i in range(len(ROW_ORDER))])
    fig.colorbar(im, ax=ax, shrink=0.85, label="log10 growth factor")
    ax.set_title("Scalability: p50 growth factor, 10% -> 100% dataset "
                 "(Q1 excluded, see Known caveats)")
    save(fig, out, "f7_scalability_growth_q2_q7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reads, q8, summaries = discover()
    print(f"discovered: {len(reads)} read cells, {len(q8)} Q8 files, "
          f"{len(summaries)} scalability summaries")
    missing = [(e, l, q) for (e, l) in ROW_ORDER for q in QUERIES
               if (e, l, q) not in reads]
    if missing:
        print("WARNING, missing read cells:", missing, file=sys.stderr)
    f1_latency_heatmap(reads, args.out)
    f2_cliff_panels(reads, args.out)
    _ratio_heatmap(reads, args.out, "f3_engine_effect",
                   "Engine effect: p50 naive / p50 PostgreSQL (>1 = PostgreSQL faster)",
                   lambda e, q: reads.get(("postgres", "base", q)))
    _ratio_heatmap(reads, args.out, "f4_schema_effect",
                   "Schema effect: p50 naive / p50 optimized (>1 = optimization faster)",
                   lambda e, q: reads.get((e, "optimised", q)))
    f5_f6_q8(q8, args.out)
    f7_growth(reads, summaries, args.out)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
