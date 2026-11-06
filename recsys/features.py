"""Stage 2 feature engineering for (user, candidate item) pairs.

``FeatureStore`` mirrors how this would run in production:

* ``FeatureStore(ds, hist_df, V)`` is the offline batch job. It precomputes every
  per-user and per-item statistic from the interaction log available at serving
  time (``hist_df``), so no label information can leak into features.
* ``store.transform(users, cand_items, cand_scores)`` is the online path. It only
  gathers precomputed arrays and computes a few cheap cross features, so it can
  run per request (see ``scripts/benchmark_serving.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GENRES, Dataset

# user_occ was removed after an ablation showed it overfit (see RESULTS.md).
CATEGORICAL = ["user_gender", "user_age"]
FEATURE_EXCLUDE = {"user", "item", "label"}
RECENCY_FEATURES = ("sim_recent_max", "sim_recent_mean", "sim_last")
TWO_TOWER_FEATURES = ("tt_score", "tt_rank")


class FeatureStore:
    def __init__(self, ds: Dataset, hist_df: pd.DataFrame, V: np.ndarray, n_recent: int = 5):
        self.V, self.G, self.n_recent = V, ds.genre_matrix, n_recent
        nu, ni = ds.n_users, ds.n_items
        u_idx, i_idx = hist_df.user.to_numpy(), hist_df.item.to_numpy()

        # ---- item statistics
        item_cnt = np.bincount(i_idx, minlength=ni).astype(np.float32)
        item_sum = np.bincount(i_idx, weights=hist_df.rating, minlength=ni)
        self.item_mean = np.where(item_cnt > 0, item_sum / np.maximum(item_cnt, 1), np.nan)
        self.log_pop = np.log1p(item_cnt)
        recent = hist_df.ts >= hist_df.ts.quantile(0.9)  # trend: last 10% of the log
        self.log_pop_recent = np.log1p(np.bincount(i_idx[recent.to_numpy()], minlength=ni))
        self.item_first_ts = hist_df.groupby("item").ts.min().reindex(range(ni)).to_numpy()
        self.year = ds.items.year.to_numpy()

        # ---- user statistics (indexed by user id; NaN for users absent from the log)
        g = hist_df.groupby("user")
        self.u_cnt = g.size().reindex(range(nu)).to_numpy(np.float32)
        self.u_mean = g.rating.mean().reindex(range(nu)).to_numpy()
        self.u_last_ts = g.ts.max().reindex(range(nu)).to_numpy()
        genre = np.zeros((nu, len(GENRES)), np.float32)
        np.add.at(genre, u_idx, self.G[i_idx])
        self.u_genre = genre / np.maximum(genre.sum(1, keepdims=True), 1)
        by_user = lambda a: pd.Series(a).groupby(u_idx).mean().reindex(range(nu)).to_numpy()  # noqa: E731
        self.u_year = by_user(self.year[i_idx])
        self.u_logpop = by_user(self.log_pop[i_idx])
        self.u_gender = ds.users.gender.to_numpy()
        self.u_age = ds.users.age.to_numpy()

        # ---- each user's n_recent most recent items, right-aligned (last col = most recent), -1 = none
        # assumes hist_df sorted by (user, ts)
        from_end = hist_df.groupby("user").cumcount(ascending=False).to_numpy()
        keep = from_end < n_recent
        self.recent = np.full((nu, n_recent), -1, dtype=np.int64)
        self.recent[u_idx[keep], n_recent - 1 - from_end[keep]] = i_idx[keep]

    def transform(self, users: np.ndarray, cand_items: np.ndarray, cand_scores: np.ndarray) -> pd.DataFrame:
        n, k = cand_items.shape
        it = cand_items.ravel()
        rep = lambda a: np.repeat(a[users], k)  # noqa: E731

        # similarity of each candidate to the user's most recent items (two-tower item space)
        R = self.recent[users]                              # [n, r]
        valid = R >= 0
        Vr = self.V[np.maximum(R, 0)] * valid[..., None]    # [n, r, d]
        sims = np.einsum("nkd,nrd->nkr", self.V[cand_items], Vr)
        v = valid[:, None, :]
        sim_max = np.where(v, sims, -np.inf).max(-1)
        sim_mean = sims.sum(-1) / np.maximum(v.sum(-1), 1)
        sim_last = np.where(valid[:, -1:], sims[:, :, -1], 0.0)

        cand_genres = self.G[it]
        affinity = (np.repeat(self.u_genre[users], k, axis=0) * cand_genres).sum(1) / np.maximum(cand_genres.sum(1), 1)

        feats = {
            "user": np.repeat(users, k),
            "item": it,
            "tt_score": cand_scores.ravel(),
            "tt_rank": np.tile(np.arange(k), n),
            "sim_recent_max": sim_max.ravel(),
            "sim_recent_mean": sim_mean.ravel(),
            "sim_last": sim_last.ravel(),
            "item_log_pop": self.log_pop[it],
            "item_log_pop_recent": self.log_pop_recent[it],
            "item_mean_rating": self.item_mean[it],
            "item_age_days": (rep(self.u_last_ts) - self.item_first_ts[it]) / 86400.0,
            "item_year": self.year[it],
            "user_cnt": rep(self.u_cnt),
            "user_mean_rating": rep(self.u_mean),
            "user_gender": rep(self.u_gender),
            "user_age": rep(self.u_age),
            "genre_affinity": affinity,
            "year_gap": self.year[it] - rep(self.u_year),
            "rating_gap": self.item_mean[it] - rep(self.u_mean),
            "pop_gap": self.log_pop[it] - rep(self.u_logpop),
        }
        for gi, gname in enumerate(GENRES):
            feats[f"g_{gname}"] = cand_genres[:, gi]
        return pd.DataFrame(feats)


def build_features(ds, hist_df, users, cand_items, cand_scores, V, n_recent: int = 5) -> pd.DataFrame:
    return FeatureStore(ds, hist_df, V, n_recent).transform(users, cand_items, cand_scores)
