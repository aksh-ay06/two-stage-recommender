"""Stage 2: LightGBM LambdaRank reranker over two-tower candidates."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURE_EXCLUDE


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def _groups(users: np.ndarray) -> np.ndarray:
    # rows are contiguous per user (built row-major by user) -> sizes of each run
    first = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
    return np.diff(np.r_[first, len(users)])


def train_ranker(df: pd.DataFrame, holdout_frac: float = 0.1, seed: int = 42, drop=()) -> lgb.LGBMRanker:
    # Users with no positive among their candidates give LambdaRank zero gradient; drop them.
    has_pos = df.groupby("user").label.transform("max") > 0
    df = df[has_pos]
    users = df.user.unique()
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(users, size=int(len(users) * holdout_frac), replace=False))
    is_hold = df.user.isin(hold).to_numpy()
    tr, va = df[~is_hold], df[is_hold]
    cols = [c for c in feature_cols(df) if c not in drop]
    if drop:
        print(f"  [ranker] dropping features: {list(drop)}")
    print(f"  [ranker] train rows={len(tr):,} users={tr.user.nunique():,} pos_rate={tr.label.mean():.4f}")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        lambdarank_truncation_level=20,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        tr[cols], tr.label, group=_groups(tr.user.to_numpy()),
        eval_X=(va[cols],), eval_y=(va.label,), eval_group=[_groups(va.user.to_numpy())],
        eval_at=[10, 20], categorical_feature=[c for c in CATEGORICAL if c in cols],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  [ranker] best_iter={model.best_iteration_}  holdout NDCG@10={model.best_score_['valid_0']['ndcg@10']:.4f}")
    return model


def rerank(model: lgb.LGBMRanker, df: pd.DataFrame, n_users: int, k: int) -> np.ndarray:
    scores = model.predict(df[list(model.feature_names_in_)], num_iteration=model.best_iteration_)
    items = df.item.to_numpy().reshape(n_users, k)
    order = np.argsort(-scores.reshape(n_users, k), axis=1, kind="stable")
    return np.take_along_axis(items, order, 1)
