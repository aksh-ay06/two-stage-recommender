"""Stage 2 feature engineering for (user, candidate item) pairs.

All statistics are computed from ``hist_df`` only (the log available at serving
time for that split), so no label information leaks into features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GENRES, Dataset


def build_features(
    ds: Dataset,
    hist_df: pd.DataFrame,
    users: np.ndarray,          # [n]
    cand_items: np.ndarray,     # [n, k]
    cand_scores: np.ndarray,    # [n, k] two-tower scores
    V: np.ndarray,              # item embeddings from the two-tower model
    n_recent: int = 5,
) -> pd.DataFrame:
    n, k = cand_items.shape
    G = ds.genre_matrix

    # ---- item stats from history
    item_cnt = np.bincount(hist_df.item, minlength=ds.n_items).astype(np.float32)
    item_sum = np.bincount(hist_df.item, weights=hist_df.rating, minlength=ds.n_items)
    item_mean = np.where(item_cnt > 0, item_sum / np.maximum(item_cnt, 1), np.nan)
    # popularity among the most recent 10% of interactions in the log (trend signal)
    recent_cut = hist_df.ts.quantile(0.9)
    item_cnt_recent = np.bincount(hist_df.item[hist_df.ts >= recent_cut], minlength=ds.n_items)
    item_first_ts = hist_df.groupby("item").ts.min().reindex(range(ds.n_items)).to_numpy()
    log_pop = np.log1p(item_cnt)

    # ---- user stats from history
    g = hist_df.groupby("user")
    u_cnt = g.size().reindex(users).to_numpy(np.float32)
    u_mean = g.rating.mean().reindex(users).to_numpy()
    u_last_ts = g.ts.max().reindex(users).to_numpy()
    u_genre = np.zeros((ds.n_users, len(GENRES)), np.float32)
    np.add.at(u_genre, hist_df.user.to_numpy(), G[hist_df.item.to_numpy()])
    u_genre = u_genre[users] / np.maximum(u_genre[users].sum(1, keepdims=True), 1)
    years = ds.items.year.to_numpy()
    h_year = years[hist_df.item.to_numpy()]
    u_year = pd.Series(h_year).groupby(hist_df.user.to_numpy()).mean().reindex(users).to_numpy()
    u_logpop = pd.Series(log_pop[hist_df.item.to_numpy()]).groupby(hist_df.user.to_numpy()).mean().reindex(users).to_numpy()

    # ---- similarity of candidate to the user's most recent items (two-tower item space)
    recent = hist_df.groupby("user").tail(n_recent)
    recent_items = recent.groupby("user").item.apply(list).reindex(users)
    R = np.full((n, n_recent), -1, dtype=np.int64)
    for r, its in enumerate(recent_items):
        R[r, :len(its)] = its
    Vr = V[np.maximum(R, 0)]                            # [n, r, d]
    Vr[R < 0] = 0.0
    sims = np.einsum("nkd,nrd->nkr", V[cand_items], Vr)  # [n, k, r]
    valid = (R >= 0)[:, None, :]
    sim_max = np.where(valid, sims, -np.inf).max(-1)
    sim_mean = (sims * valid).sum(-1) / np.maximum(valid.sum(-1), 1)
    sim_last = sims[:, :, -1]  # tail() keeps time order, so last col = most recent

    # ---- assemble (row-major: user r, candidate j)
    it = cand_items.ravel()
    rep = lambda a: np.repeat(a, k)  # noqa: E731
    cand_genres = G[it]
    n_genres = np.maximum(cand_genres.sum(1), 1)
    affinity = (np.repeat(u_genre, k, axis=0) * cand_genres).sum(1) / n_genres

    feats = {
        "user": rep(users),
        "item": it,
        "tt_score": cand_scores.ravel(),
        "tt_rank": np.tile(np.arange(k), n),
        "sim_recent_max": sim_max.ravel(),
        "sim_recent_mean": sim_mean.ravel(),
        "sim_last": sim_last.ravel(),
        "item_log_pop": log_pop[it],
        "item_log_pop_recent": np.log1p(item_cnt_recent[it]),
        "item_mean_rating": item_mean[it],
        "item_age_days": (rep(u_last_ts) - item_first_ts[it]) / 86400.0,
        "item_year": years[it],
        "user_cnt": rep(u_cnt),
        "user_mean_rating": rep(u_mean),
        "user_gender": rep(ds.users.gender.to_numpy()[users]),
        "user_age": rep(ds.users.age.to_numpy()[users]),
        "genre_affinity": affinity,
        "year_gap": years[it] - rep(u_year),
        "rating_gap": item_mean[it] - rep(u_mean),
        "pop_gap": log_pop[it] - rep(u_logpop),
    }
    for gi, gname in enumerate(GENRES):
        feats[f"g_{gname}"] = cand_genres[:, gi]
    return pd.DataFrame(feats)


FEATURE_EXCLUDE = {"user", "item", "label"}
# user_occ was removed after an ablation showed it overfit (see RESULTS.md).
CATEGORICAL = ["user_gender", "user_age"]
