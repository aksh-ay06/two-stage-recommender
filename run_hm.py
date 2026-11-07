"""Two-stage recommender on H&M: two-tower (+ cheap sources) retrieval, LightGBM rerank.

    python run_hm.py                    # full run (see RESULTS.md, section 7)
    python run_hm.py --smoke            # small, fast end-to-end check

Protocol: the competition's. Test = the last week of the data (2020-09-16..22), validation
= the week before. Each stage sees only a history window of ``--weeks`` weeks strictly before
its target week. Evaluated users are customers who buy in the target week and have history
in the window. Metric: MAP@12 (the Kaggle metric), plus Recall@12 and NDCG@12.

Every modelling decision (EASE λ, which ranker variant, early stopping) uses the validation
week. The two-tower model is trained once, on the validation-stage history (everything before
the validation week), and reused for the test stage. At test time its user embeddings are
recomputed from histories that include the validation week, as they would be in production.
Neither target week is ever used as a training target for stage 1.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.hm_data import load_hm, make_stage
from recsys.hm_features import (CATEGORICAL, OTHER_SOURCE_FEATURES, TWO_TOWER_FEATURES, EASE, HMStore, candidates,
                                fill, label, transform)
from recsys.hm_model import HMTwoTower, item_vectors, user_vectors
from recsys.metrics import evaluate, paired_bootstrap
from recsys.ranker import rerank, split_ranker_users, train_ranker
from recsys.two_tower import HistoryIndex, TwoTowerConfig, train_two_tower

ROOT = Path(__file__).resolve().parent
K = 12
VARIANTS = {  # name -> (candidate pool, features dropped)
    "GBM on two-tower candidates": ("tt", OTHER_SOURCE_FEATURES),
    "Multi-source GBM (all features)": ("all", ()),
    "Multi-source GBM w/o two-tower feats": ("all", TWO_TOWER_FEATURES),
}


def metrics(ranked, targets):
    return evaluate(ranked, targets, ks=(K,), maps=(K,))


def summarize(m):
    return {k: float(v.mean()) for k, v in m.items()}


class StageRunner:
    """Chunked candidates + features for one stage (history window -> target week)."""

    def __init__(self, d, st, model, cfg, V, ease_items, lam, args):
        self.st, self.args, self.model = st, args, model
        self.hidx = HistoryIndex(st.hist, model.pad, cfg.hist_len)
        self.store = HMStore(d, st, V, rebuy_k=args.rebuy_k, pop_k=args.pop_k)
        self.ease = EASE(d, st, n=ease_items)
        self.ease.fit(lam) if lam is not None else None
        self.n_items = d.n_items
        self.tgt = dict(zip(st.users.tolist(), st.targets))

    def chunks(self, users, with_features=True):
        for b in range(0, len(users), self.args.chunk):
            us = users[b:b + self.args.chunk]
            U = user_vectors(self.model, self.hidx, us)
            cand, lists, Se = candidates(self.store, self.ease, us, U, self.args.tt_k, self.args.ease_k)
            targets = [self.tgt[u] for u in us.tolist()]
            if with_features:
                cand = transform(self.store, self.ease, cand, us, U, Se)
                cand["label"] = label(cand, us, targets, self.n_items)
            yield us, targets, cand, lists, Se


def baselines(store, lists, Se, ease, n):
    return {
        "Popularity (last week)": np.tile(store.pop[:K], (n, 1)),
        "Re-buy, then popular": fill(lists["rebuy"], store.pop, K),
        "EASE": ease.items[np.argsort(-Se, axis=1)[:, :K]],
        "Two-tower": lists["tt"][:, :K],
    }


def source_recall(cand, lists, targets, users, tt_k, ease_k):
    out = {}
    pools = {"two-tower": cand.tt_rank < tt_k, "EASE": cand.ease_rank < ease_k,
             "re-buy": cand.rebuy_rank < cand.rebuy_rank.max(), "popular": cand.pop_rank < cand.pop_rank.max(),
             "union": np.ones(len(cand), bool)}
    pos = cand.label.to_numpy() > 0
    n_rel = np.array([len(t) for t in targets], float)
    row = np.searchsorted(users, cand.user.to_numpy())
    for name, m in pools.items():
        m = np.asarray(m) & pos
        out[name] = np.bincount(row[m], minlength=len(users)) / n_rel
    out["candidates_per_user"] = np.bincount(row, minlength=len(users)).astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=12, help="history window per stage")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-users", type=int, default=12000, help="validation users sampled for ranker training")
    ap.add_argument("--test-users", type=int, default=0, help="0 = every evaluated test user")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--tt-k", type=int, default=100)
    ap.add_argument("--ease-k", type=int, default=50)
    ap.add_argument("--ease-items", type=int, default=8000)
    ap.add_argument("--rebuy-k", type=int, default=30)
    ap.add_argument("--pop-k", type=int, default=30)
    ap.add_argument("--no-logq", action="store_true", help="train the two-tower without the logQ correction")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/hm")
    args = ap.parse_args()
    if args.smoke:
        args.weeks, args.epochs, args.val_users, args.test_users, args.ease_items = 4, 1, 3000, 3000, 2000
        args.out = "results/hm_smoke" + ("_nologq" if args.no_logq else "")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- data
    from_day = 733 - 6 - 7 - 7 * args.weeks
    d = load_hm(from_day=from_day)
    test_start = d.last_day - 6
    val_start = test_start - 7
    val = make_stage(d, val_start, args.weeks, "val")
    test = make_stage(d, test_start, args.weeks, "test")
    d.tx = None
    gc.collect()
    for st in (val, test):
        print(f"{st.name}: history {len(st.hist):,} purchases, {st.hist.user.nunique():,} customers | "
              f"target week buyers {st.n_buyers:,}, evaluated (have history) {len(st.users):,} "
              f"({len(st.users) / st.n_buyers:.0%}), mean items bought {np.mean([len(t) for t in st.targets]):.1f}")

    # ---------------------------------------------------------------- stage 1: two-tower
    cfg = TwoTowerConfig(dim=64, hist_len=20, epochs=args.epochs, batch_size=4096, lr=2e-3, seed=args.seed,
                         logq=not args.no_logq)
    cache = ROOT / "cache" / f"hm_tt_w{args.weeks}_e{args.epochs}_s{args.seed}{'_nologq' if args.no_logq else ''}.pt"
    model = HMTwoTower(d, cfg)
    if cache.exists():
        model.load_state_dict(torch.load(cache))
        model.eval()
        print(f"loaded two-tower weights {cache.name}")
    else:
        print("== two-tower training on the validation-stage history ==")
        model, _ = train_two_tower(d, val.hist, cfg, model=model)
        cache.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), cache)
    V = item_vectors(model, d.n_items)

    # ---------------------------------------------------------------- VAL stage
    print("== VAL stage ==")
    vr = StageRunner(d, val, model, cfg, V, args.ease_items, None, args)
    val_users = np.sort(rng.choice(val.users, size=min(args.val_users, len(val.users)), replace=False))
    tune_users = val_users[: min(4000, len(val_users))]
    tune_tg = [vr.tgt[u] for u in tune_users.tolist()]
    ease_scores = {}
    for lam in (50, 200, 800, 3200):
        vr.ease.fit(lam)
        Se = vr.ease.scores(tune_users)
        ease_scores[lam] = float(metrics(vr.ease.items[np.argsort(-Se, 1)[:, :K]], tune_tg)[f"MAP@{K}"].mean())
    lam = max(ease_scores, key=ease_scores.get)
    vr.ease.fit(lam)
    print(f"  EASE lambda on val (MAP@12): {ease_scores} -> {lam}")

    parts, val_lists = [], []
    for us, tg, feats, lists, Se in vr.chunks(val_users):
        parts.append(feats)
        print(f"  val chunk: {len(us):,} users, {len(feats):,} candidate rows, pos rate {feats.label.mean():.4f}")
    feats = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    tr_users, hold_users = split_ranker_users(val_users, 0.2, args.seed)
    rankers, val_table = {}, {}
    hold = feats[feats.user.isin(hold_users)]
    hold_tg = [vr.tgt[u] for u in hold_users.tolist()]
    for name, (pool, drop) in VARIANTS.items():
        df = feats if pool == "all" else feats[feats.tt_rank < args.tt_k]
        rankers[name] = train_ranker(df, tr_users, hold_users, drop=drop, seed=args.seed,
                                     categorical=CATEGORICAL, eval_at=K)
        h = hold if pool == "all" else hold[hold.tt_rank < args.tt_k]
        val_table[name] = summarize(metrics(rerank(rankers[name], h, K), hold_tg))
        del df
        gc.collect()
    selected = max(val_table, key=lambda n: val_table[n][f"MAP@{K}"])
    print(pd.DataFrame(val_table).T.round(4).to_string())
    print(f"VAL (ranker-holdout users, n={len(hold_users):,}) -> selected ranker: {selected}")
    del feats, hold, vr
    gc.collect()

    # ---------------------------------------------------------------- TEST stage
    print("== TEST stage ==")
    tr = StageRunner(d, test, model, cfg, V, args.ease_items, lam, args)
    test_users = test.users
    if args.test_users:
        test_users = np.sort(rng.choice(test_users, size=min(args.test_users, len(test_users)), replace=False))
    ranked = {}
    rec_parts, all_targets = [], []
    for us, tg, feats, lists, Se in tr.chunks(test_users):
        all_targets += tg
        sys = baselines(tr.store, lists, Se, tr.ease, len(us))
        for name, (pool, _) in VARIANTS.items():
            df = feats if pool == "all" else feats[feats.tt_rank < args.tt_k]
            sys[name] = rerank(rankers[name], df, K)
        for name, r in sys.items():
            ranked.setdefault(name, []).append(r)
        rec_parts.append(source_recall(feats, lists, tg, us, args.tt_k, args.ease_k))
        print(f"  test chunk: {len(us):,} users, {len(feats):,} rows ({time.time() - t_start:.0f}s)")
        del feats
        gc.collect()
    ranked = {k: np.concatenate(v) for k, v in ranked.items()}
    per_user = {name: metrics(r, all_targets) for name, r in ranked.items()}
    cand_recall = {k: float(np.concatenate([p[k] for p in rec_parts]).mean()) for k in rec_parts[0]}

    table = pd.DataFrame({n: summarize(m) for n, m in per_user.items()}).T
    order = ["Popularity (last week)", "Re-buy, then popular", "EASE", "Two-tower", *VARIANTS]
    table = table.loc[order]
    print(f"\nTEST ({len(all_targets):,} users):")
    print(table.round(4).to_string())
    print("  candidate recall (share of a user's purchases found by each source):",
          {k: round(v, 4) for k, v in cand_recall.items()})

    comparisons = {}
    m = f"MAP@{K}"
    for base in ["Two-tower", "Re-buy, then popular", "EASE", "Popularity (last week)",
                 *[v for v in VARIANTS if v != selected]]:
        lift, ci = paired_bootstrap(per_user[base][m], per_user[selected][m], seed=args.seed)
        comparisons[f"{selected} vs {base}"] = {"lift": float(lift), "ci95": [float(c) for c in ci]}
        print(f"  {selected} vs {base}: {m} {lift:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    lift, ci = paired_bootstrap(per_user["Two-tower"][m], per_user["GBM on two-tower candidates"][m], seed=args.seed)
    comparisons["GBM on two-tower candidates vs Two-tower"] = {"lift": float(lift), "ci95": [float(c) for c in ci]}
    print(f"  GBM on two-tower candidates vs Two-tower: {m} {lift:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")

    # ---------------------------------------------------------------- save
    table.to_csv(out / "metrics_test.csv")
    pd.DataFrame(val_table).T.to_csv(out / "metrics_val_holdout.csv")
    best = rankers[selected]
    pd.Series(best.booster_.feature_importance("gain"), index=best.feature_names_in_).pipe(
        lambda s: s / s.sum()).sort_values(ascending=False).rename("gain_share").to_csv(out / "feature_importance.csv")
    best.booster_.save_model(str(out / "ranker.txt"), num_iteration=best.best_iteration_)
    json.dump({
        "protocol": "H&M weekly: test = 2020-09-16..22, val = week before",
        "args": vars(args), "two_tower_config": cfg.__dict__, "ease_lambda": lam, "ease_tuning_map12": ease_scores,
        "n_test_users": len(all_targets), "test_buyers_total": test.n_buyers, "n_val_users": int(len(val_users)),
        "n_val_holdout_users": int(len(hold_users)), "selected_ranker": selected, "val_holdout": val_table,
        "ranker_best_iterations": {n: int(r.best_iteration_) for n, r in rankers.items()},
        "test": table.to_dict(orient="index"), "candidate_recall": cand_recall, "comparisons": comparisons,
        "runtime_sec": time.time() - t_start,
    }, open(out / "results.json", "w"), indent=2)
    print(f"\nSaved {out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
