"""Stage 2: LightGBM LambdaRank reranker over two-tower candidates."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURE_EXCLUDE


def feature_cols(df: pd.DataFrame, drop=()) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE and c not in drop]


def _groups(users: np.ndarray) -> np.ndarray:
    # rows are contiguous per user (built row-major by user) -> sizes of each run
    first = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
    return np.diff(np.r_[first, len(users)])


def split_ranker_users(users: np.ndarray, holdout_frac: float = 0.2, seed: int = 42):
    """Partition VAL-stage users into ranker-train and ranker-holdout users.

    The holdout users are used for early stopping AND for choosing between ranker
    variants, so the test set is never used to make a modelling decision.
    """
    rng = np.random.default_rng(seed)
    hold = rng.choice(users, size=int(len(users) * holdout_frac), replace=False)
    return np.setdiff1d(users, hold), np.sort(hold)


def train_ranker(df: pd.DataFrame, train_users: np.ndarray, hold_users: np.ndarray,
                 drop=(), seed: int = 42) -> lgb.LGBMRanker:
    # Users with no positive among their candidates give LambdaRank zero gradient; drop them.
    df = df[df.groupby("user").label.transform("max") > 0]
    tr, va = df[df.user.isin(train_users)], df[df.user.isin(hold_users)]
    cols = feature_cols(df, drop)
    print(f"  [ranker] drop={list(drop) or '-'}  train rows={len(tr):,} users={tr.user.nunique():,} pos_rate={tr.label.mean():.4f}")

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
        eval_at=[10], categorical_feature=[c for c in CATEGORICAL if c in cols],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"  [ranker] best_iter={model.best_iteration_}")
    return model


def _order(df: pd.DataFrame, scores: np.ndarray, k: int) -> np.ndarray:
    n = len(df) // k
    items = df.item.to_numpy().reshape(n, k)
    order = np.argsort(-scores.reshape(n, k), axis=1, kind="stable")
    return np.take_along_axis(items, order, 1)


def rerank(model: lgb.LGBMRanker, df: pd.DataFrame, k: int) -> np.ndarray:
    return _order(df, model.predict(df[list(model.feature_names_in_)], num_iteration=model.best_iteration_), k)


def rerank_by(df: pd.DataFrame, column: str, k: int) -> np.ndarray:
    """Heuristic reranker: sort each user's candidates by a single feature."""
    return _order(df, df[column].to_numpy(), k)
