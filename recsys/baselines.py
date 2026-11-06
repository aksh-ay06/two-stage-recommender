"""Non-neural baselines that are known to be strong on MovieLens.

* Popularity : most-interacted items in the history log.
* Item-kNN   : cosine item-item similarity on the binary user-item matrix, keeping each
               item's top ``k`` neighbours; score = sum of similarities to the user's items.
* EASE       : Steck (2019), "Embarrassingly Shallow Autoencoders". Closed-form linear
               item-item model  B = I - P / diag(P),  P = (X^T X + lambda I)^-1,  diag(B) = 0.

Hyperparameters (k, lambda) are chosen on the VAL stage only and then refit on train+val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def binary_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    X = np.zeros((n_users, n_items), dtype=np.float32)
    X[df.user.to_numpy(), df.item.to_numpy()] = 1.0
    return X


def popularity_scores(X: np.ndarray) -> np.ndarray:
    return np.broadcast_to(X.sum(0, dtype=np.float32), X.shape).copy()


def itemknn_scores(X: np.ndarray, k: int) -> np.ndarray:
    norms = np.sqrt(X.sum(0)) + 1e-8
    S = (X.T @ X) / norms[:, None] / norms[None, :]
    np.fill_diagonal(S, 0.0)
    if k < S.shape[0]:
        # for each target item (column), keep only its k most similar items
        drop = np.argpartition(-S, k, axis=0)[k:]
        np.put_along_axis(S, drop, 0.0, axis=0)
    return X @ S


def ease_scores(X: np.ndarray, lam: float) -> np.ndarray:
    G = (X.T @ X).astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    return (X @ B.astype(np.float32))


def top_k_unseen(scores: np.ndarray, X: np.ndarray, users: np.ndarray, k: int) -> np.ndarray:
    s = scores[users].copy()
    s[X[users] > 0] = -np.inf
    top = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, top, 1), axis=1, kind="stable")
    return np.take_along_axis(top, order, 1)


BASELINES = {
    "Popularity": (lambda X, _: popularity_scores(X), [None]),
    "Item-kNN": (itemknn_scores, [20, 50, 100, 200, 400, 800]),
    "EASE": (ease_scores, [100, 250, 500, 1000, 2000, 4000, 8000]),
}
