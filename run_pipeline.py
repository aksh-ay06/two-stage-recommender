"""End-to-end two-stage recommender on MovieLens-1M.

    python run_pipeline.py                 # full run
    python run_pipeline.py --epochs 2      # faster smoke run

Protocol (per-user temporal leave-last-5):
  1. VAL stage : fit two-tower on `train`, retrieve candidates, label them with `val`,
                 train the LightGBM ranker on those (features from `train` only).
  2. TEST stage: refit two-tower on `train+val`, retrieve candidates, build features
                 from `train+val`, rerank with the ranker from step 1, score on `test`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_ml1m, make_splits
from recsys.features import build_features
from recsys.metrics import evaluate, paired_bootstrap
from recsys.ranker import feature_cols, rerank, train_ranker
from recsys.two_tower import TwoTowerConfig, embed_all, retrieve, train_two_tower

ROOT = Path(__file__).parent


def candidate_stage(ds, hist_df, target_df, n_cand, cfg):
    model, hidx = train_two_tower(ds, hist_df, cfg)
    U, V = embed_all(model, hidx, ds.n_users, ds.n_items)
    users = np.sort(target_df.user.unique())
    items, scores = retrieve(U, V, hist_df, n_cand, users)
    feats = build_features(ds, hist_df, users, items, scores, V)
    target_sets = target_df.groupby("user").item.apply(set).reindex(users).tolist()
    tgt = pd.MultiIndex.from_frame(target_df[["user", "item"]])
    feats["label"] = pd.MultiIndex.from_arrays([feats.user, feats.item]).isin(tgt).astype(int)
    return users, items, feats, target_sets


def popularity_baseline(hist_df, users, k):
    pop = hist_df.item.value_counts().index.to_numpy()
    seen = hist_df.groupby("user").item.apply(set)
    out = np.empty((len(users), k), dtype=np.int64)
    for r, u in enumerate(users):
        s = seen.get(u, set())
        out[r] = [i for i in pop[: k + len(s)] if i not in s][:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=TwoTowerConfig.epochs)
    ap.add_argument("--holdout", type=int, default=5, help="items held out per user for val and test")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--ablate", nargs="+", default=[],
                    help="also train a ranker WITHOUT these features and compare it to the full ranker")
    ap.add_argument("--reuse-stage1", action="store_true",
                    help="load cached stage-1 candidates/features instead of retraining the two-tower models")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    cfg = TwoTowerConfig(epochs=args.epochs)
    cache = ROOT / "cache" / f"stage1_c{args.candidates}_e{args.epochs}_h{args.holdout}.pkl"
    t_start = time.time()

    ds = load_ml1m(ROOT / "data")
    train, val, train_val, test = make_splits(ds, args.holdout)
    print(f"MovieLens-1M: {ds.n_users:,} users, {ds.n_items:,} items, {len(ds.interactions):,} interactions")
    print(f"split rows: train={len(train):,} val={len(val):,} test={len(test):,}")

    if args.reuse_stage1 and cache.exists():
        print(f"\nLoading cached stage-1 outputs from {cache}")
        s1 = pd.read_pickle(cache)
    else:
        print("\n== VAL stage: retrieval on train, ranker labels from val ==")
        _, val_items, val_feats, val_targets = candidate_stage(ds, train, val, args.candidates, cfg)
        print("\n== TEST stage: retrieval on train+val ==")
        users, cand_items, test_feats, test_targets = candidate_stage(ds, train_val, test, args.candidates, cfg)
        s1 = dict(val_items=val_items, val_feats=val_feats, val_targets=val_targets,
                  users=users, cand_items=cand_items, test_feats=test_feats, test_targets=test_targets)
        cache.parent.mkdir(exist_ok=True)
        pd.to_pickle(s1, cache)
    val_feats, test_feats, test_targets = s1["val_feats"], s1["test_feats"], s1["test_targets"]
    users, cand_items = s1["users"], s1["cand_items"]
    val_recall = evaluate(s1["val_items"], s1["val_targets"], ks=(args.candidates,))[f"Recall@{args.candidates}"].mean()
    print(f"  candidate Recall@{args.candidates} on val: {val_recall:.4f}")

    print("\n== Ranker: trained on val labels, evaluated on test ==")
    ranker = train_ranker(val_feats)
    reranked = rerank(ranker, test_feats, len(users), args.candidates)
    systems = {
        "Popularity": popularity_baseline(train_val, users, args.candidates),
        "Two-tower (stage 1 only)": cand_items,
        "Two-tower + LightGBM rerank": reranked,
    }
    ablation_name = None
    if args.ablate:
        ablation_name = f"... rerank without {', '.join(args.ablate)}"
        ablated = train_ranker(val_feats, drop=tuple(args.ablate))
        systems[ablation_name] = rerank(ablated, test_feats, len(users), args.candidates)
    per_user = {name: evaluate(ranked, test_targets, ks=(10, 20, 50, args.candidates)) for name, ranked in systems.items()}

    metric_names = list(per_user["Two-tower + LightGBM rerank"].keys())
    table = pd.DataFrame({name: {m: v.mean() for m, v in res.items()} for name, res in per_user.items()}).T[metric_names]
    print("\nTest metrics (mean over users):")
    print(table.round(4).to_string())

    lifts = {}
    for m in ["NDCG@10", "Recall@10", "NDCG@20", "Recall@20", "Recall@50"]:
        point, ci = paired_bootstrap(per_user["Two-tower (stage 1 only)"][m], per_user["Two-tower + LightGBM rerank"][m])
        lifts[m] = {"lift": point, "ci95": list(ci)}
        print(f"  rerank lift {m:10s}: {point:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")

    ablation = {}
    if ablation_name:
        print(f"\nAblation: '{ablation_name}' vs full ranker (paired, same candidates):")
        for m in ["NDCG@10", "Recall@10", "NDCG@20", "Recall@20", "Recall@50"]:
            point, ci = paired_bootstrap(per_user["Two-tower + LightGBM rerank"][m], per_user[ablation_name][m])
            ablation[m] = {"change": point, "ci95": list(ci)}
            print(f"  {m:10s}: {point:+.2%}  95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}]")

    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=feature_cols(val_feats)).sort_values(ascending=False)
    imp = (imp / imp.sum()).rename("gain_share")
    imp.to_csv(args.out / "feature_importance.csv")
    print("\nTop ranker features (gain share):")
    print(imp.head(10).round(3).to_string())

    table.to_csv(args.out / "metrics.csv")
    json.dump({
        "n_users_eval": len(users), "candidates": args.candidates, "holdout_per_user": args.holdout,
        "val_candidate_recall": val_recall, "metrics": table.to_dict(orient="index"), "rerank_lift_vs_two_tower": lifts,
        "ablation": {"dropped": args.ablate, "change_vs_full_ranker": ablation},
        "ranker_best_iteration": ranker.best_iteration_, "two_tower_config": vars(cfg),
        "runtime_sec": time.time() - t_start,
    }, open(args.out / "results.json", "w"), indent=2, default=float)
    print(f"\nSaved results to {args.out}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
