"""Offline ranking metrics (binary relevance) and a paired bootstrap for lifts."""
from __future__ import annotations

import numpy as np


def hits_matrix(ranked: np.ndarray, targets: list[set]) -> np.ndarray:
    """[n_users, K] boolean: is ranked[u, j] in user u's held-out set."""
    return np.array([[it in t for it in row] for row, t in zip(ranked, targets)], dtype=bool)


def recall_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].sum(1) / n_rel


def ndcg_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits[:, :k] * disc).sum(1)
    idcg = np.array([disc[: min(int(r), k)].sum() for r in n_rel])
    return dcg / idcg


def evaluate(ranked: np.ndarray, targets: list[set], ks=(10, 20, 50)) -> dict[str, np.ndarray]:
    """Per-user metric arrays (keep them per-user so we can bootstrap)."""
    hits = hits_matrix(ranked, targets)
    n_rel = np.array([len(t) for t in targets], dtype=float)
    out = {}
    for k in ks:
        if k <= ranked.shape[1]:
            out[f"Recall@{k}"] = recall_at_k(hits, n_rel, k)
            out[f"NDCG@{k}"] = ndcg_at_k(hits, n_rel, k)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Relative lift of mean(b) over mean(a), with a 95% CI from resampling users."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    lifts = b[idx].mean(1) / a[idx].mean(1) - 1
    return b.mean() / a.mean() - 1, np.percentile(lifts, [2.5, 97.5])
