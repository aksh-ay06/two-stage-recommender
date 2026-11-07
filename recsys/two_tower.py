"""Stage 1: two-tower retrieval model trained with in-batch sampled softmax.

User tower : user-id emb + demographics + mean-pooled embeddings of the user's
             last ``hist_len`` items *strictly before* the target interaction.
Item tower : item-id emb + genre multi-hot + release-year bucket.
Loss       : in-batch softmax over L2-normalised embeddings with a temperature,
             logQ correction for popularity bias of in-batch negatives, and
             masking of accidental in-batch positives (same item twice in a batch).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset


@dataclass
class TwoTowerConfig:
    dim: int = 64
    hist_len: int = 30
    epochs: int = 6
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-6
    temperature: float = 0.05
    dropout: float = 0.1
    seed: int = 42
    logq: bool = True  # subtract log item frequency from in-batch logits


def _mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, d_out))


class TwoTower(nn.Module):
    def __init__(self, ds: Dataset, cfg: TwoTowerConfig):
        super().__init__()
        d = cfg.dim
        self.pad = ds.n_items  # padding index for history
        # --- user tower
        self.user_id = nn.Embedding(ds.n_users, 32)
        self.gender = nn.Embedding(2, 4)
        self.age = nn.Embedding(7, 8)
        self.occ = nn.Embedding(21, 8)
        self.hist_item = nn.Embedding(ds.n_items + 1, d, padding_idx=self.pad)
        self.user_mlp = _mlp(32 + 4 + 8 + 8 + d, d, cfg.dropout)
        # --- item tower
        self.item_id = nn.Embedding(ds.n_items, d)
        self.genre = nn.Linear(ds.genre_matrix.shape[1], 16)
        self.year = nn.Embedding(20, 8)
        self.item_mlp = _mlp(d + 16 + 8, d, cfg.dropout)

        # static side features as buffers
        u = ds.users
        self.register_buffer("u_gender", torch.tensor(u.gender.to_numpy(), dtype=torch.long))
        self.register_buffer("u_age", torch.tensor(u.age.to_numpy(), dtype=torch.long))
        self.register_buffer("u_occ", torch.tensor(u.occupation.to_numpy(), dtype=torch.long))
        self.register_buffer("i_genre", torch.tensor(ds.genre_matrix))
        year_bucket = np.clip((ds.items.year.to_numpy() - 1905) // 5, 0, 19).astype(np.int64)
        self.register_buffer("i_year", torch.tensor(year_bucket))

    def user_vec(self, users: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist != self.pad).float().unsqueeze(-1)
        pooled = (self.hist_item(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([
            self.user_id(users), self.gender(self.u_gender[users]),
            self.age(self.u_age[users]), self.occ(self.u_occ[users]), pooled,
        ], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_id(items), self.genre(self.i_genre[items]), self.year(self.i_year[items])], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)


class HistoryIndex:
    """Gathers 'last L items before position p' for rows of a (user, ts)-sorted log."""

    def __init__(self, df: pd.DataFrame, pad: int, hist_len: int):
        self.items = torch.tensor(df.item.to_numpy(), dtype=torch.long)
        users = df.user.to_numpy()
        first = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
        counts = np.diff(np.r_[first, len(users)])
        self.user_start = np.repeat(first, counts)               # per row
        self.user_ids = users[first]                             # per user block
        self.user_first, self.user_end = first, first + counts   # per user block
        self.pad, self.L = pad, hist_len
        self._offsets = torch.arange(1, hist_len + 1)

    def gather(self, pos: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        idx = pos.unsqueeze(1) - self._offsets.unsqueeze(0)
        valid = idx >= start.unsqueeze(1)
        out = self.items[idx.clamp(min=0)]
        out[~valid] = self.pad
        return out


def train_two_tower(ds: Dataset, train_df: pd.DataFrame, cfg: TwoTowerConfig = TwoTowerConfig(),
                    min_rating: int = 1, model: nn.Module | None = None) -> tuple[TwoTower, HistoryIndex]:
    """History is built from every interaction; only rows rated >= min_rating are training targets.

    ``model`` lets another dataset supply its own towers (anything with ``pad``, ``user_vec``,
    ``item_vec``); logs without a ``rating`` column use every row as a target.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = model if model is not None else TwoTower(ds, cfg)
    hidx = HistoryIndex(train_df, model.pad, cfg.hist_len)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    users_all = torch.tensor(train_df.user.to_numpy(), dtype=torch.long)
    items_all = hidx.items
    start_all = torch.tensor(hidx.user_start, dtype=torch.long)
    eligible = (np.flatnonzero(train_df.rating.to_numpy() >= min_rating) if "rating" in train_df
                else np.arange(len(train_df)))
    n = len(eligible)
    # logQ: sampling probability of each item as an in-batch negative ~ its frequency.
    freq = np.bincount(train_df.item.to_numpy()[eligible], minlength=ds.n_items) + 1.0
    log_q = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32)

    steps = int(np.ceil(n / cfg.batch_size)) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)

    for epoch in range(cfg.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.tensor(rng.permutation(eligible))
        for b in range(0, n, cfg.batch_size):
            rows = perm[b:b + cfg.batch_size]
            u, it = users_all[rows], items_all[rows]
            hist = hidx.gather(rows, start_all[rows])
            uv, iv = model.user_vec(u, hist), model.item_vec(it)
            logits = uv @ iv.T / cfg.temperature
            if cfg.logq:
                logits = logits - log_q[it].unsqueeze(0)
            same_item = (it.unsqueeze(0) == it.unsqueeze(1)) & ~torch.eye(len(it), dtype=torch.bool)
            logits = logits.masked_fill(same_item, float("-inf"))
            loss = F.cross_entropy(logits, torch.arange(len(it)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(rows)
        print(f"  [two-tower] epoch {epoch + 1}/{cfg.epochs} loss={total / n:.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    return model, hidx


@torch.no_grad()
def embed_all(model: TwoTower, hidx: HistoryIndex, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """User embeddings use each user's most recent ``hist_len`` items in the given log."""
    users = torch.tensor(hidx.user_ids, dtype=torch.long)
    hist = hidx.gather(torch.tensor(hidx.user_end), torch.tensor(hidx.user_first))
    U = np.zeros((n_users, model.item_id.embedding_dim), dtype=np.float32)
    U[hidx.user_ids] = model.user_vec(users, hist).numpy()
    V = model.item_vec(torch.arange(n_items)).numpy()
    return U, V


def retrieve(U: np.ndarray, V: np.ndarray, seen: pd.DataFrame, k: int, users: np.ndarray):
    """Top-k unseen items per user by dot product. Returns (items[n,k], scores[n,k])."""
    scores = U[users] @ V.T
    row_of = {u: r for r, u in enumerate(users)}
    s = seen[seen.user.isin(row_of)]
    scores[s.user.map(row_of).to_numpy(), s.item.to_numpy()] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top, 1)
    order = np.argsort(-top_scores, axis=1)
    return np.take_along_axis(top, order, 1), np.take_along_axis(top_scores, order, 1)
