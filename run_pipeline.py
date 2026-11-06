"""End-to-end two-stage recommender on MovieLens-1M.

    python run_pipeline.py                                   # per-user leave-last-5, all ratings, seed 42
    python run_pipeline.py --protocol global                 # global time split
    python run_pipeline.py --min-rating 4                    # only ratings >= 4 count as positives
    python run_pipeline.py --seed 43 --reuse-stage1          # reuse cached two-tower outputs if present

Protocol
  1. VAL stage : fit two-tower + features on `train`, retrieve candidates for VAL users and label
                 them with `val`. Baseline hyperparameters are tuned here. VAL users are split
                 into ranker-train users and ranker-holdout users; every ranker variant is
                 compared on the holdout users and the winner is chosen there.
  2. TEST stage: refit two-tower, baselines and features on `train+val`, rerank with the
                 rankers from step 1, score on `test`. The test set is used for reporting only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.baselines import BASELINES, binary_matrix, top_k_unseen
from recsys.data import load_ml1m, make_splits
from recsys.features import RECENCY_FEATURES, TWO_TOWER_FEATURES, FeatureStore
from recsys.metrics import evaluate, paired_bootstrap
from recsys.ranker import rerank, rerank_by, split_ranker_users, train_ranker
from recsys.two_tower import TwoTowerConfig, embed_all, retrieve, train_two_tower

ROOT = Path(__file__).parent
KS = (10, 20, 50, 200)
RANKER_VARIANTS = {
    "GBM (all features)": (),
    "GBM w/o recency sims": RECENCY_FEATURES,
    "GBM w/o two-tower feats": TWO_TOWER_FEATURES,
}


def stage1(ds, hist_df, target_df, n_cand, cfg, min_rating):
    model, hidx = train_two_tower(ds, hist_df, cfg, min_rating)
    U, V = embed_all(model, hidx, ds.n_users, ds.n_items)
    users = np.sort(target_df.user.unique())
    items, scores = retrieve(U, V, hist_df, n_cand, users)
    feats = FeatureStore(ds, hist_df, V).transform(users, items, scores)
    tgt = pd.MultiIndex.from_frame(target_df[["user", "item"]])
    feats["label"] = pd.MultiIndex.from_arrays([feats.user, feats.item]).isin(tgt).astype(int)
    targets = target_df.groupby("user").item.apply(set).reindex(users).tolist()
    return dict(users=users, items=items, feats=feats, targets=targets), model


def run_baselines(ds, splits, val_users, val_targets, test_users, n_cand):
    """Tune each baseline on VAL (NDCG@10), refit on train+val, rank TEST users."""
    X_tr = binary_matrix(splits.train, ds.n_users, ds.n_items)
    X_tv = binary_matrix(splits.train_val, ds.n_users, ds.n_items)
    val_lists, test_lists, chosen = {}, {}, {}
    for name, (score_fn, grid) in BASELINES.items():
        best = (-1.0, None, None)
        for p in grid:
            ranked = top_k_unseen(score_fn(X_tr, p), X_tr, val_users, n_cand)
            ndcg = evaluate(ranked, val_targets, ks=(10,))["NDCG@10"].mean()
            if p is not None:
                print(f"  [{name}] param={p:<5} val NDCG@10={ndcg:.4f}")
            if ndcg > best[0]:
                best = (ndcg, p, ranked)
        chosen[name] = best[1]
        val_lists[name] = best[2]
        test_lists[name] = top_k_unseen(score_fn(X_tv, best[1]), X_tv, test_users, n_cand)
    return val_lists, test_lists, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["last_k", "global"], default="last_k")
    ap.add_argument("--min-rating", type=int, default=1, help="ratings >= this are positives (targets)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=TwoTowerConfig.epochs)
    ap.add_argument("--holdout", type=int, default=5, help="last_k: items held out per user for val and test")
    ap.add_argument("--time-frac", type=float, default=0.1, help="global: fraction of the log for val and for test")
    ap.add_argument("--reuse-stage1", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    name = f"{args.protocol}_minr{args.min_rating}_seed{args.seed}"
    out = args.out or ROOT / "results" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache" / f"{name}_e{args.epochs}_c{args.candidates}.pkl"
    cfg = TwoTowerConfig(epochs=args.epochs, seed=args.seed)
    t0 = time.time()
    print(f"### run {name}")

    ds = load_ml1m(ROOT / "data")
    sp = make_splits(ds, args.protocol, k=args.holdout, time_frac=args.time_frac, min_rating=args.min_rating)
    print(f"train={len(sp.train):,} val_targets={len(sp.val):,} ({sp.val.user.nunique():,} users) "
          f"train_val={len(sp.train_val):,} test_targets={len(sp.test):,} ({sp.test.user.nunique():,} users)")

    # ---------------- stage 1 (cached: it is the slow part)
    if args.reuse_stage1 and cache.exists():
        print(f"loading cached stage-1 outputs: {cache.name}")
        s1 = pd.read_pickle(cache)
    else:
        print("\n== VAL stage: two-tower on train ==")
        val, _ = stage1(ds, sp.train, sp.val, args.candidates, cfg, args.min_rating)
        print("\n== TEST stage: two-tower on train+val ==")
        test, tt_model = stage1(ds, sp.train_val, sp.test, args.candidates, cfg, args.min_rating)
        s1 = dict(val=val, test=test)
        cache.parent.mkdir(exist_ok=True)
        pd.to_pickle(s1, cache)
        torch.save(tt_model.state_dict(), cache.with_suffix(".tt_test.pt"))
    val, test = s1["val"], s1["test"]

    # ---------------- baselines (tuned on VAL)
    print("\n== Baselines (hyperparameters tuned on VAL) ==")
    base_val, base_test, chosen = run_baselines(ds, sp, val["users"], val["targets"], test["users"], args.candidates)

    # ---------------- rankers: train on VAL ranker-train users, select on VAL holdout users
    print("\n== Rankers ==")
    r_train, r_hold = split_ranker_users(val["users"], seed=args.seed)
    rankers = {n: train_ranker(val["feats"], r_train, r_hold, drop, args.seed) for n, drop in RANKER_VARIANTS.items()}

    hold_mask = np.isin(val["users"], r_hold)
    hold_feats = val["feats"][val["feats"].user.isin(r_hold)]
    hold_targets = [t for t, m in zip(val["targets"], hold_mask) if m]
    val_lists = {**{n: l[hold_mask] for n, l in base_val.items()},
                 "Two-tower": val["items"][hold_mask],
                 "Two-tower, ordered by sim_recent_max": rerank_by(hold_feats, "sim_recent_max", args.candidates)}
    val_lists.update({n: rerank(m, hold_feats, args.candidates) for n, m in rankers.items()})
    val_table = pd.DataFrame({n: {m: v.mean() for m, v in evaluate(l, hold_targets, KS).items()}
                              for n, l in val_lists.items()}).T
    selected = val_table.loc[list(RANKER_VARIANTS), "NDCG@10"].idxmax()
    print(f"\nVAL (ranker-holdout users, n={hold_mask.sum()}) -> selected ranker: {selected}")
    print(val_table[["NDCG@10", "Recall@10", "NDCG@20", "Recall@200"]].round(4).to_string())

    # ---------------- TEST: reporting only
    test_lists = {**base_test,
                  "Two-tower": test["items"],
                  "Two-tower, ordered by sim_recent_max": rerank_by(test["feats"], "sim_recent_max", args.candidates)}
    test_lists.update({n: rerank(m, test["feats"], args.candidates) for n, m in rankers.items()})
    per_user = {n: evaluate(l, test["targets"], KS) for n, l in test_lists.items()}
    table = pd.DataFrame({n: {m: v.mean() for m, v in r.items()} for n, r in per_user.items()}).T
    print(f"\nTEST ({len(test['users']):,} users) - reporting only:")
    print(table.round(4).to_string())

    comparisons = {}
    pairs = [(selected, "Two-tower"), (selected, "EASE"), (selected, "Item-kNN"),
             (selected, "Two-tower, ordered by sim_recent_max"), ("EASE", "Two-tower")]
    print()
    for a, b in pairs:
        comparisons[f"{a} vs {b}"] = {}
        for m in ["NDCG@10", "Recall@10", "NDCG@20", "Recall@50"]:
            point, ci = paired_bootstrap(per_user[b][m], per_user[a][m], seed=args.seed)
            comparisons[f"{a} vs {b}"][m] = {"lift": point, "ci95": list(ci)}
        c = comparisons[f"{a} vs {b}"]["NDCG@10"]
        print(f"  {a} vs {b}: NDCG@10 {c['lift']:+.1%}  95% CI [{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}]")

    ranker = rankers[selected]
    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=ranker.feature_names_in_)
    (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_csv(out / "feature_importance.csv")
    ranker.booster_.save_model(str(out / "ranker.txt"), num_iteration=ranker.best_iteration_)
    table.to_csv(out / "metrics_test.csv")
    val_table.to_csv(out / "metrics_val_holdout.csv")
    hist_cnt = sp.train_val.user.value_counts().reindex(test["users"]).to_numpy()
    pd.to_pickle({"users": test["users"], "targets": test["targets"], "hist_count": hist_cnt,
                  "item_pop": np.bincount(sp.train_val.item, minlength=ds.n_items),
                  "ranked": {n: l[:, :50] for n, l in test_lists.items()}}, out / "per_user.pkl")
    json.dump({
        "run": name, "protocol": args.protocol, "min_rating": args.min_rating, "seed": args.seed,
        "n_test_users": len(test["users"]), "n_val_users": len(val["users"]), "n_val_holdout_users": int(hold_mask.sum()),
        "baseline_params": chosen, "selected_ranker": selected,
        "ranker_best_iterations": {n: m.best_iteration_ for n, m in rankers.items()},
        "val_holdout": val_table.to_dict(orient="index"), "test": table.to_dict(orient="index"),
        "comparisons": comparisons, "two_tower_config": vars(cfg), "runtime_sec": time.time() - t0,
    }, open(out / "results.json", "w"), indent=2, default=float)
    print(f"\nSaved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
