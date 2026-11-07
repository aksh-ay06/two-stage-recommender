"""Aggregate runs (seeds x protocols) into summary tables and figures.

    python scripts/make_report.py

Reads results/runs/*/results.json and per_user.pkl, writes:
    results/summary_test.csv          mean and std across seeds, per protocol x system
    results/segments.csv              NDCG@10 by user-activity quartile (seed-42 main run)
    results/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recsys.metrics import evaluate, paired_bootstrap  # noqa: E402

RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"
MAIN_RUN = "last_k_minr1_seed42"
TWO_STAGE = "Two-stage (selected)"
PREV = "GBM on two-tower candidates"   # the single-source two-stage system, kept as a variant

# Reference palette (light mode). Only the first three categorical slots (validated all-pairs)
# plus a neutral gray, with one color per entity, the same in every figure.
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8f8e88"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = {TWO_STAGE: BLUE, PREV: ORANGE, "EASE": AQUA, "Two-tower": MUTED}
LEGEND = {TWO_STAGE: "Two-stage, ranker chosen on validation (uses EASE signal)",
          PREV: "Two-tower candidates only + GBM", "EASE": "EASE (tuned)", "Two-tower": "Two-tower alone"}
SHORT = {TWO_STAGE: "Two-stage", PREV: "Two-tower cands + GBM", "EASE": "EASE", "Two-tower": "Two-tower"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False,
})


def load_runs():
    rows, metas = [], {}
    for d in sorted(RUNS.iterdir()):
        f = d / "results.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        metas[d.name] = m
        for system, metrics in m["test"].items():
            # every system keeps its own row; the validation-selected ranker is ALSO reported as TWO_STAGE
            names = [system] + ([TWO_STAGE] if system == m["selected_ranker"] else [])
            for name in names:
                for metric, v in metrics.items():
                    rows.append(dict(run=d.name, protocol=m["protocol"], min_rating=m["min_rating"], seed=m["seed"],
                                     system=name, raw_system=system, metric=metric, value=v))
    return pd.DataFrame(rows), metas


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["protocol", "min_rating", "system", "metric"]).value
    s = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_seeds": g.size()}).reset_index()
    return s


def fig_ndcg_curve(per_user, selected):
    ks = np.arange(1, 51)
    systems = [TWO_STAGE, PREV, "EASE", "Two-tower"] if selected != PREV else [TWO_STAGE, "EASE", "Two-tower"]
    raw = {TWO_STAGE: selected}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ends = []
    for s in systems:
        ranked = per_user["ranked"][raw.get(s, s)]
        m = evaluate(ranked, per_user["targets"], ks=tuple(ks))
        y = np.array([m[f"NDCG@{k}"].mean() for k in ks])
        ax.plot(ks, y, color=SERIES[s], lw=2.4 if s == TWO_STAGE else 2, label=LEGEND[s], solid_capstyle="round")
        ends.append((y[-1], s))
    # direct labels at line ends, nudged apart so they never collide
    ends.sort()
    placed = []
    for y, s in ends:
        y_lab = max([y] + [p + 0.0065 for p in placed[-1:]])
        placed.append(y_lab)
        ax.annotate(SHORT[s], (50, y), xytext=(52, y_lab), textcoords="data", va="center", fontsize=9, color=INK2,
                    arrowprops=None, annotation_clip=False)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, None)
    ax.set_xlabel("K (list length)")
    ax.set_ylabel("NDCG@K")
    ax.set_title("NDCG@K on test, leave-last-5 protocol (seed 42)", loc="left", pad=40)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=8.5, handlelength=1.4, columnspacing=1.0)
    ax.text(1.5, 0.004, "Dip at K < 5 is expected: ideal DCG keeps growing until K = 5 held-out items",
            fontsize=8, color=INK2, va="bottom")
    fig.subplots_adjust(right=0.76, top=0.80)
    fig.savefig(FIG / "ndcg_at_k.png", dpi=160)
    plt.close(fig)


def segments(per_user, selected, seed=0):
    q = pd.qcut(per_user["hist_count"], 4, labels=False)
    edges = np.quantile(per_user["hist_count"], [0, .25, .5, .75, 1]).astype(int)
    labels = [f"Q{i + 1}\n{edges[i]}-{edges[i + 1]} ratings" for i in range(4)]
    sys_map = {"Two-tower": "Two-tower", "EASE": "EASE", PREV: PREV, TWO_STAGE: selected}
    m = {s: evaluate(per_user["ranked"][r], per_user["targets"], ks=(10,))["NDCG@10"] for s, r in sys_map.items()}
    rows = []
    for i in range(4):
        mask = q == i
        row = {"quartile": labels[i].replace("\n", " "), "users": int(mask.sum()),
               **{f"NDCG@10 {s}": m[s][mask].mean() for s in sys_map}}
        for base, tag in [("Two-tower", "two_tower"), ("EASE", "EASE"), (PREV, "prev_two_stage")]:
            lift, ci = paired_bootstrap(m[base][mask], m[TWO_STAGE][mask], seed=seed)
            row.update({f"lift_vs_{tag}": lift, f"lift_vs_{tag}_ci_lo": ci[0], f"lift_vs_{tag}_ci_hi": ci[1]})
        rows.append(row)
    seg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(4)
    w = 0.2
    for j, s in enumerate(["Two-tower", "EASE", PREV, TWO_STAGE]):
        vals = seg[f"NDCG@10 {s}"].to_numpy()
        bars = ax.bar(x + (j - 1.5) * (w + 0.015), vals, width=w, color=SERIES[s], label=SHORT[s], zorder=3)
        if s == TWO_STAGE:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 by user activity (history length quartile)", loc="left")
    ax.legend(loc="upper right", fontsize=9, ncols=2)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, seg[[f"NDCG@10 {s}" for s in sys_map]].to_numpy().max() * 1.3)
    fig.tight_layout()
    fig.savefig(FIG / "ndcg_by_activity.png", dpi=160)
    plt.close(fig)
    return seg


def fig_retrieval_scaling():
    f = ROOT / "results" / "serving_benchmark.json"
    if not f.exists():
        return
    b = json.load(open(f))["part_b"]
    n = np.array([r["items"] for r in b])
    exact = np.array([r["exact"]["p50"] for r in b])
    h = [min(r["hnsw"], key=lambda e: abs(e["recall@200"] - 0.95) if e["recall@200"] >= 0.95 else 9 + e["efSearch"]) for r in b]
    hnsw = np.array([e["p50"] for e in h])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(n, exact, color=ORANGE, lw=2, marker="o", ms=8, label="Exact (brute force)")
    ax.plot(n, hnsw, color=BLUE, lw=2, marker="o", ms=8, label="FAISS HNSW (recall@200 >= 0.95)")
    for xi, yi, e in zip(n, hnsw, h):
        ax.annotate(f"R={e['recall@200']:.2f}", (xi, yi), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    for xi, yi in zip(n, exact):
        ax.annotate(f"{yi:.1f} ms", (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Catalog size (items)")
    ax.set_ylabel("p50 latency per query (ms)")
    ax.set_title("Top-200 retrieval latency, single thread", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval_scaling.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, metas = load_runs()
    summary = summarize(df)
    summary.to_csv(ROOT / "results" / "summary_test.csv", index=False)
    wide = summary[summary.metric.isin(["NDCG@10", "Recall@10", "Recall@50", "Recall@200"])].pivot_table(
        index=["protocol", "min_rating", "system"], columns="metric", values=["mean", "std"])
    print(wide.round(4).to_string())
    print("\nselected ranker per run:", {r: m["selected_ranker"] for r, m in metas.items()})

    per_user = pd.read_pickle(RUNS / MAIN_RUN / "per_user.pkl")
    selected = metas[MAIN_RUN]["selected_ranker"]
    fig_ndcg_curve(per_user, selected)
    seg = segments(per_user, selected)
    seg.to_csv(ROOT / "results" / "segments.csv", index=False)
    print("\n", seg.round(4).to_string())
    fig_retrieval_scaling()
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
