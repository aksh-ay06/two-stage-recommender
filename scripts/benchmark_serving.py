"""Serving benchmark: per-request latency of the full two-stage path, and retrieval at scale.

    python scripts/benchmark_serving.py --run last_k_minr1_seed42

Part A (real model, ML-1M): one request = one user, single-threaded, measured step by step:
    user tower -> retrieve top-200 (exact) -> feature transform -> GBM score -> sort
    Per-user / per-item statistics are precomputed offline (FeatureStore), as in a feature store.

Part B (retrieval at scale): catalogs of 3.7k (real), 100k and 1M items. The synthetic items
are real item vectors plus noise, so they keep realistic cluster structure. Exact brute force
is compared with FAISS HNSW on latency and on recall@200 against the exact top-200.
"""
from __future__ import annotations

import os

# Pin every math library to one thread BEFORE numpy/torch/faiss load, so "single thread" is true.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recsys.data import load_ml1m, make_splits  # noqa: E402
from recsys.features import FeatureStore  # noqa: E402
from recsys.two_tower import HistoryIndex, TwoTower, TwoTowerConfig, embed_all  # noqa: E402

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)


def pct(ts_ms):
    a = np.asarray(ts_ms)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def part_a(run: str, n_requests: int, k: int):
    meta = json.load(open(ROOT / "results" / "runs" / run / "results.json"))
    ds = load_ml1m(ROOT / "data")
    sp = make_splits(ds, meta["protocol"], min_rating=meta["min_rating"])
    cfg = TwoTowerConfig(**meta["two_tower_config"])
    model = TwoTower(ds, cfg)
    weights = ROOT / "cache" / f"{run}_e{cfg.epochs}_c{k}.tt_test.pt"
    model.load_state_dict(torch.load(weights))
    model.eval()
    booster = lgb.Booster(model_file=str(ROOT / "results" / "runs" / run / "ranker.txt"))

    # ---- offline batch jobs (timed once, not per request)
    t = time.perf_counter()
    hidx = HistoryIndex(sp.train_val, model.pad, cfg.hist_len)
    _, V = embed_all(model, hidx, ds.n_users, ds.n_items)
    store = FeatureStore(ds, sp.train_val, V)
    seen = sp.train_val.groupby("user").item.apply(np.array).to_dict()
    offline_s = time.perf_counter() - t

    users = np.random.default_rng(0).choice(np.sort(sp.test.user.unique()), size=n_requests)
    start_of = dict(zip(hidx.user_ids, hidx.user_first))
    end_of = dict(zip(hidx.user_ids, hidx.user_end))
    names = booster.feature_name()
    steps = {s: [] for s in ["user_tower", "retrieval", "features", "gbm", "total"]}
    with torch.no_grad():
        for i, u in enumerate(users):
            t0 = time.perf_counter()
            hist = hidx.gather(torch.tensor([end_of[u]]), torch.tensor([start_of[u]]))
            q = model.user_vec(torch.tensor([u]), hist).numpy()[0]
            t1 = time.perf_counter()
            s = V @ q
            s[seen[u]] = -np.inf
            top = np.argpartition(-s, k)[:k]
            t2 = time.perf_counter()
            feats = store.transform(np.array([u]), top[None, :], s[top][None, :])
            t3 = time.perf_counter()
            scores = booster.predict(feats[names].to_numpy(np.float64), num_threads=1)
            ranked = top[np.argsort(-scores)]  # noqa: F841
            t4 = time.perf_counter()
            if i >= 20:  # warm-up
                for name, dt in zip(steps, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0]):
                    steps[name].append(dt * 1000)
    return {"offline_batch_sec": offline_s, "n_trees": booster.num_trees(), "candidates": k,
            "latency_ms": {s: pct(v) for s, v in steps.items()}}, model, hidx, V, ds


def part_b(U: np.ndarray, V: np.ndarray, sizes, k: int, n_queries: int):
    rng = np.random.default_rng(0)
    Q = U[rng.choice(len(U), n_queries, replace=False)].astype(np.float32)
    out = []
    for n in sizes:
        if n <= len(V):
            X = V[:n].astype(np.float32)
        else:
            X = V[rng.integers(0, len(V), n)] + rng.normal(0, 0.15, (n, V.shape[1])).astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.ascontiguousarray(X, dtype=np.float32)

        exact_top, exact_ms = [], []
        for q in Q:
            t = time.perf_counter()
            s = X @ q
            exact_top.append(set(np.argpartition(-s, k)[:k].tolist()))
            exact_ms.append((time.perf_counter() - t) * 1000)
        row = {"items": n, "exact": pct(exact_ms), "hnsw": []}

        t = time.perf_counter()
        faiss.omp_set_num_threads(os.cpu_count() or 4)  # build is offline: use all cores
        index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(X)
        faiss.omp_set_num_threads(1)
        row["hnsw_build_sec"] = time.perf_counter() - t
        for ef in (256, 512, 1024):
            index.hnsw.efSearch = ef
            ms, rec = [], []
            for q, ex in zip(Q, exact_top):
                t = time.perf_counter()
                _, I = index.search(q[None, :], k)
                ms.append((time.perf_counter() - t) * 1000)
                rec.append(len(ex & set(I[0].tolist())) / k)
            row["hnsw"].append({"efSearch": ef, "recall@200": float(np.mean(rec)), **pct(ms)})
        print(f"  n={n:>9,}  exact p50={row['exact']['p50']:.2f}ms  build={row['hnsw_build_sec']:.0f}s  " +
              "  ".join(f"ef={h['efSearch']}: R={h['recall@200']:.3f} p50={h['p50']:.2f}ms" for h in row["hnsw"]))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="last_k_minr1_seed42")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--sizes", type=int, nargs="+", default=[3706, 100_000, 1_000_000])
    args = ap.parse_args()

    print("== Part A: per-request latency, single thread ==")
    a, model, hidx, V, ds = part_a(args.run, args.requests, 200)
    for s, v in a["latency_ms"].items():
        print(f"  {s:11s} p50={v['p50']:.2f}ms  p95={v['p95']:.2f}ms  p99={v['p99']:.2f}ms")
    print(f"  offline batch (history index + item embeddings + feature store): {a['offline_batch_sec']:.1f}s")

    print("\n== Part B: retrieval at scale (exact vs FAISS HNSW, top-200, single-thread queries) ==")
    U, _ = embed_all(model, hidx, ds.n_users, ds.n_items)
    U = U[np.abs(U).sum(1) > 0]
    b = part_b(U, V, args.sizes, 200, args.queries)

    out = ROOT / "results" / "serving_benchmark.json"
    json.dump({"run": args.run, "part_a": a, "part_b": b}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
