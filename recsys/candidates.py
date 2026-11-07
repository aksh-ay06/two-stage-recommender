"""Multi-source candidate generation: union of two-tower and EASE top-k lists.

Each source scores the full catalog with seen items masked to -inf. Each keeps its own
top-``k``; the union (k to 2k items per user) goes to the ranker. Every candidate gets
both sources' scores, plus each source's rank capped at ``k`` ("not retrieved by this
source"). So the GBM can learn how much to trust each source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def top_k(S: np.ndarray, k: int) -> np.ndarray:
    """[n, k] item ids sorted by score, descending."""
    top = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, top, 1), axis=1, kind="stable")
    return np.take_along_axis(top, order, 1)


def rank_matrix(top: np.ndarray, n_items: int, cap: int) -> np.ndarray:
    """[n, n_items] rank of each item in a top list, ``cap`` if absent."""
    R = np.full((top.shape[0], n_items), cap, dtype=np.int16)
    np.put_along_axis(R, top, np.broadcast_to(np.arange(top.shape[1], dtype=np.int16), top.shape), axis=1)
    return R


@dataclass
class Candidates:
    rows: np.ndarray        # [m] row index into ``users`` (sorted: grouped by row)
    items: np.ndarray       # [m]
    tt_score: np.ndarray
    tt_rank: np.ndarray
    ease_score: np.ndarray
    ease_rank: np.ndarray
    tt_top: np.ndarray      # [n, k] two-tower top-k (the single-source candidate set)
    ease_top: np.ndarray    # [n, k]

    def subset(self, mask: np.ndarray) -> "Candidates":
        return Candidates(self.rows[mask], self.items[mask], self.tt_score[mask], self.tt_rank[mask],
                          self.ease_score[mask], self.ease_rank[mask], self.tt_top, self.ease_top)


def merge_candidates(S_tt: np.ndarray, S_ease: np.ndarray, k: int) -> Candidates:
    """S_* are [n_users_in_batch, n_items] scores with already-seen items set to -inf."""
    n, n_items = S_tt.shape
    tt_top, ease_top = top_k(S_tt, k), top_k(S_ease, k)
    R_tt, R_ease = rank_matrix(tt_top, n_items, k), rank_matrix(ease_top, n_items, k)
    rows, items = np.nonzero((R_tt < k) | (R_ease < k))  # row-major: grouped by user row
    return Candidates(rows, items, S_tt[rows, items].astype(np.float32), R_tt[rows, items].astype(np.float32),
                      S_ease[rows, items].astype(np.float32), R_ease[rows, items].astype(np.float32),
                      tt_top, ease_top)
