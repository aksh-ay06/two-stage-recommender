"""Two-tower towers for H&M; trained with the shared loop in ``two_tower.train_two_tower``.

User tower : age bucket, club status, newsletter frequency + mean-pooled embeddings of the
             customer's last ``hist_len`` purchases strictly before the target row.
             No customer-id embedding: 1.37M ids, most with a handful of purchases.
Item tower : article-id embedding + embeddings of 8 catalog attributes (product type and
             group, colour, department, index, section, garment group).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hm_data import HMData
from .two_tower import HistoryIndex, TwoTowerConfig, _mlp


class HMTwoTower(nn.Module):
    def __init__(self, d: HMData, cfg: TwoTowerConfig):
        super().__init__()
        dim = cfg.dim
        self.pad = d.n_items
        # --- user tower
        self.age = nn.Embedding(16, 8)
        self.club = nn.Embedding(int(d.users.club.max()) + 1, 4)
        self.news = nn.Embedding(int(d.users.news.max()) + 1, 4)
        self.hist_item = nn.Embedding(d.n_items + 1, dim, padding_idx=self.pad)
        self.user_mlp = _mlp(8 + 4 + 4 + dim, dim, cfg.dropout)
        # --- item tower
        self.item_id = nn.Embedding(d.n_items, dim)
        self.cat_names = [c for c in d.items.columns if c != "product_code"]
        self.cats = nn.ModuleList(nn.Embedding(d.n_cats[c], 8) for c in self.cat_names)
        self.item_mlp = _mlp(dim + 8 * len(self.cat_names), dim, cfg.dropout)

        age = d.users.age.to_numpy()
        age_bucket = np.where(age < 0, 0, np.clip((age - 15) // 5 + 1, 1, 15)).astype(np.int64)
        self.register_buffer("u_age", torch.tensor(age_bucket))
        self.register_buffer("u_club", torch.tensor(d.users.club.to_numpy(np.int64)))
        self.register_buffer("u_news", torch.tensor(d.users.news.to_numpy(np.int64)))
        self.register_buffer("i_cats", torch.tensor(d.items[self.cat_names].to_numpy(np.int64)))

    def user_vec(self, users: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist != self.pad).float().unsqueeze(-1)
        pooled = (self.hist_item(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.age(self.u_age[users]), self.club(self.u_club[users]),
                       self.news(self.u_news[users]), pooled], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        c = self.i_cats[items]
        x = torch.cat([self.item_id(items), *[emb(c[:, j]) for j, emb in enumerate(self.cats)]], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)


@torch.no_grad()
def item_vectors(model: HMTwoTower, n_items: int, batch: int = 32768) -> np.ndarray:
    return np.concatenate([model.item_vec(torch.arange(b, min(b + batch, n_items))).numpy()
                           for b in range(0, n_items, batch)])


@torch.no_grad()
def user_vectors(model: HMTwoTower, hidx: HistoryIndex, users: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Embeddings from each user's most recent purchases in ``hidx`` (users must appear in it)."""
    pos = np.searchsorted(hidx.user_ids, users)
    assert (hidx.user_ids[pos] == users).all(), "user without history"
    out = []
    for b in range(0, len(users), batch):
        p = pos[b:b + batch]
        hist = hidx.gather(torch.tensor(hidx.user_end[p]), torch.tensor(hidx.user_first[p]))
        out.append(model.user_vec(torch.tensor(users[b:b + batch], dtype=torch.long), hist).numpy())
    return np.concatenate(out) if out else np.zeros((0, model.item_id.embedding_dim), np.float32)
