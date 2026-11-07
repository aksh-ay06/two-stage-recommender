"""H&M stage 1 candidate sources and stage 2 features, computed per chunk of users.

``HMStore(d, stage, V)`` is the offline batch job: every statistic comes from the stage's
history window (strictly before the target week). ``candidates`` + ``transform`` are the
per-request path. Everything is chunked by user so memory stays bounded (~1M rows at a time).

Candidate sources (each keeps its own rank as a feature; a source that did not retrieve the
item gets rank = its cap):
    tt     two-tower top-k over articles sold in the last 4 weeks
    ease   EASE top-k over the best-selling ``n`` articles
    rebuy  the customer's own past purchases, most recent first (re-buying is common on H&M)
    pop    last week's best-sellers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .hm_data import HMData, Stage

SOURCES = ("tt", "ease", "rebuy", "pop")
CATEGORICAL = ["i_product_type_no", "i_product_group_name", "i_colour_group_code", "i_perceived_colour_master_id",
               "i_department_no", "i_index_group_no", "i_section_no", "i_garment_group_no", "u_club", "u_news"]
TWO_TOWER_FEATURES = ("tt_score", "tt_rank", "sim_last", "sim_recent_max", "sim_recent_mean")
OTHER_SOURCE_FEATURES = ("ease_score", "ease_rank", "rebuy_rank", "pop_rank")


class PairLookup:
    """Vectorised (a, b) -> value lookup via sorted int64 keys."""

    def __init__(self, a, b, n_b: int, **values):
        key = np.asarray(a, np.int64) * n_b + np.asarray(b, np.int64)
        order = np.argsort(key, kind="stable")
        self.key, self.n_b = key[order], n_b
        self.vals = {k: np.asarray(v, np.float32)[order] for k, v in values.items()}

    def get(self, a, b, name: str, default=np.nan) -> np.ndarray:
        q = np.asarray(a, np.int64) * self.n_b + np.asarray(b, np.int64)
        pos = np.minimum(np.searchsorted(self.key, q), len(self.key) - 1)
        return np.where(self.key[pos] == q, self.vals[name][pos], default).astype(np.float32)


class EASE:
    """EASE on the ``n`` articles with the most sales in the last 4 weeks of the window."""

    def __init__(self, d: HMData, st: Stage, n: int = 8000):
        h = st.hist
        recent = (st.start - h.day.to_numpy()) <= 28
        sales = np.bincount(h.item.to_numpy()[recent], minlength=d.n_items)
        self.items = np.argsort(-sales, kind="stable")[:n]
        self.col = np.full(d.n_items, -1, np.int64)
        self.col[self.items] = np.arange(len(self.items))
        c = self.col[h.item.to_numpy()]
        m = c >= 0
        X = sp.csr_matrix((np.ones(m.sum(), np.float32), (h.user.to_numpy()[m], c[m])), shape=(d.n_users, len(self.items)))
        X.data[:] = 1.0  # binary after summing duplicates
        self.X = X
        self.G = (X.T @ X).toarray().astype(np.float32)
        self.B = None

    def fit(self, lam: float) -> "EASE":
        self.B = None
        P = self.G.copy()
        P[np.diag_indices_from(P)] += lam
        P = np.linalg.inv(P)
        P /= -np.diag(P)[None, :]
        P[np.diag_indices_from(P)] = 0.0
        self.B, self.lam = P, lam
        return self

    def scores(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[users] @ self.B, dtype=np.float32)


class HMStore:
    def __init__(self, d: HMData, st: Stage, V: np.ndarray, n_recent: int = 5, rebuy_k: int = 30, pop_k: int = 30):
        h, start, ni, nu = st.hist, st.start, d.n_items, d.n_users
        self.start, self.n_items, self.V = start, ni, V
        u, i, day = h.user.to_numpy(), h.item.to_numpy(), h.day.to_numpy()
        price = h.price.to_numpy(np.float64)
        ago = (start - day).astype(np.int32)  # 1 = the last day before the target week

        def cnt(mask):
            return np.bincount(i[mask], minlength=ni).astype(np.float32)

        s1, s2, s4, sall = cnt(ago <= 7), cnt((ago > 7) & (ago <= 14)), cnt(ago <= 28), cnt(ago > 0)
        g = h.groupby("item").day.agg(["min", "max"])
        first, last = np.full(ni, np.nan, np.float32), np.full(ni, np.nan, np.float32)
        first[g.index], last[g.index] = g["min"], g["max"]
        item = {
            "i_log_sales_1w": np.log1p(s1), "i_log_sales_4w": np.log1p(s4), "i_log_sales_all": np.log1p(sall),
            "i_trend": np.log1p(s1) - np.log1p(s2),
            "i_mean_price": np.where(sall > 0, np.bincount(i, weights=price, minlength=ni) / np.maximum(sall, 1), np.nan),
            "i_days_since_first": start - first, "i_days_since_last": start - last,
        }
        for c in d.items.columns:
            if c != "product_code":
                item[f"i_{c}"] = d.items[c].to_numpy()
        self.item = {k: np.asarray(v, np.float32) for k, v in item.items()}

        ucnt = np.bincount(u, minlength=nu).astype(np.float32)
        ulast = np.full(nu, np.nan, np.float32)
        gu = h.groupby("user").day.max()
        ulast[gu.index] = gu.to_numpy()
        safe = np.maximum(ucnt, 1)
        age = d.users.age.to_numpy(np.float32)
        self.user = {
            "u_age": np.where(age < 0, np.nan, age), "u_club": d.users.club.to_numpy(np.float32),
            "u_news": d.users.news.to_numpy(np.float32), "u_fn": d.users.fn.to_numpy(np.float32),
            "u_active": d.users.active.to_numpy(np.float32),
            "u_n_tx": ucnt, "u_n_tx_4w": np.bincount(u[ago <= 28], minlength=nu).astype(np.float32),
            "u_days_since_last": start - ulast,
            "u_mean_price": (np.bincount(u, weights=price, minlength=nu) / safe).astype(np.float32),
            "u_online_share": (np.bincount(u, weights=(h.channel.to_numpy() == 2), minlength=nu) / safe).astype(np.float32),
        }
        self.user["u_mean_price"][ucnt == 0] = np.nan

        # (user, item) history, most recent first -> pair features, rebuy source, recent items
        ui = h.groupby(["user", "item"], sort=False).day.agg(["size", "max"]).reset_index()
        ui = ui.sort_values(["user", "max", "size"], ascending=[True, False, False], kind="stable")
        ui["rank"] = ui.groupby("user").cumcount()
        self.ui = PairLookup(ui.user, ui.item, ni, cnt=ui["size"], last=ui["max"])
        rb = ui[ui["rank"] < rebuy_k]
        self.rb_user, self.rb_item, self.rb_rank = rb.user.to_numpy(), rb.item.to_numpy(), rb["rank"].to_numpy()
        rec = ui[ui["rank"] < n_recent]
        self.recent = np.full((nu, n_recent), -1, np.int32)
        self.recent[rec.user.to_numpy(), rec["rank"].to_numpy()] = rec.item.to_numpy()
        self.user["u_n_items"] = np.bincount(ui.user.to_numpy(), minlength=nu).astype(np.float32)

        # user's purchase counts per catalog group (share = count / u_n_tx)
        self.groups = {}
        for c in ("product_code", "product_type_no", "index_group_no", "garment_group_no"):
            codes = d.items[c].to_numpy()
            t = pd.DataFrame({"u": u, "c": codes[i]}).groupby(["u", "c"]).size()
            self.groups[c] = (codes, PairLookup(t.index.get_level_values(0), t.index.get_level_values(1),
                                                int(codes.max()) + 1, n=t.to_numpy()))

        self.active = np.flatnonzero(s4 > 0)                         # catalog the two-tower retrieves from
        self.V_active = np.ascontiguousarray(V[self.active])
        self.pop = np.lexsort((-s4, -s1))[:pop_k]                   # last week's best-sellers
        self.caps = {"tt": None, "ease": None, "rebuy": rebuy_k, "pop": pop_k}

    # ------------------------------------------------------------------ sources
    def rebuy(self, users: np.ndarray):
        lo, hi = np.searchsorted(self.rb_user, users[0]), np.searchsorted(self.rb_user, users[-1], side="right")
        m = np.isin(self.rb_user[lo:hi], users)
        return self.rb_user[lo:hi][m], self.rb_item[lo:hi][m], self.rb_rank[lo:hi][m]


def _top(S: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(-S, k, axis=1)[:, :k]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)


def candidates(store: HMStore, ease: EASE, users: np.ndarray, U: np.ndarray, tt_k: int = 100, ease_k: int = 50):
    """Union of the four sources for a sorted chunk of users -> (frame, per-source top lists, EASE scores)."""
    n = len(users)
    tt_top = store.active[_top(U @ store.V_active.T, tt_k)]          # [n, tt_k] item codes
    Se = ease.scores(users)
    ease_top = ease.items[_top(Se, ease_k)]                             # [n, ease_k]
    rb_u, rb_i, rb_r = store.rebuy(users)
    pop_k = len(store.pop)

    caps = {"tt": tt_k, "ease": ease_k, "rebuy": store.caps["rebuy"], "pop": pop_k}
    blocks = [
        ("tt", np.repeat(users, tt_k), tt_top.ravel(), np.tile(np.arange(tt_k), n)),
        ("ease", np.repeat(users, ease_k), ease_top.ravel(), np.tile(np.arange(ease_k), n)),
        ("rebuy", rb_u, rb_i, rb_r),
        ("pop", np.repeat(users, pop_k), np.tile(store.pop, n), np.tile(np.arange(pop_k), n)),
    ]
    frames = []
    for src, bu, bi, br in blocks:
        f = pd.DataFrame({"user": bu.astype(np.int32), "item": bi.astype(np.int32)})
        for s in SOURCES:
            f[f"{s}_rank"] = np.float32(br) if s == src else np.float32(caps[s])
        frames.append(f)
    cand = pd.concat(frames, ignore_index=True).groupby(["user", "item"], sort=True).min().reset_index()
    rb_lists = pd.Series(rb_i).groupby(rb_u).apply(list).reindex(users).tolist()  # rank order kept
    lists = {"tt": tt_top, "ease": ease_top, "rebuy": [r if isinstance(r, list) else [] for r in rb_lists]}
    return cand, lists, Se


def transform(store: HMStore, ease: EASE, cand: pd.DataFrame, users: np.ndarray, U: np.ndarray, Se: np.ndarray,
              block: int = 100_000) -> pd.DataFrame:
    u, i = cand.user.to_numpy(), cand.item.to_numpy()
    row = np.searchsorted(users, u)
    out = cand
    n = len(cand)
    tt, s_last, s_max, s_mean = (np.empty(n, np.float32) for _ in range(4))
    for b in range(0, n, block):
        sl = slice(b, b + block)
        Vi = store.V[i[sl]]
        tt[sl] = np.einsum("ij,ij->i", U[row[sl]], Vi)
        rec = store.recent[u[sl]]
        valid = rec >= 0
        sims = np.einsum("ijk,ik->ij", store.V[np.maximum(rec, 0)], Vi)
        s_last[sl] = np.where(valid[:, 0], sims[:, 0], np.nan)
        s_max[sl] = np.where(valid.any(1), np.where(valid, sims, -np.inf).max(1), np.nan)
        s_mean[sl] = (sims * valid).sum(1) / np.maximum(valid.sum(1), 1)
    out["tt_score"], out["sim_last"], out["sim_recent_max"], out["sim_recent_mean"] = tt, s_last, s_max, s_mean
    col = ease.col[i]
    out["ease_score"] = np.where(col >= 0, Se[row, np.maximum(col, 0)], np.nan).astype(np.float32)

    for k, v in store.item.items():
        out[k] = v[i]
    for k, v in store.user.items():
        out[k] = v[u]
    out["ui_cnt"] = store.ui.get(u, i, "cnt", 0.0)
    out["ui_days_since"] = store.start - store.ui.get(u, i, "last")
    for c, (codes, lk) in store.groups.items():
        cnt = lk.get(u, codes[i], "n", 0.0)
        out[f"u_{c}_cnt" if c == "product_code" else f"u_{c}_share"] = (
            cnt if c == "product_code" else cnt / np.maximum(out.u_n_tx.to_numpy(), 1))
    out["price_ratio"] = out.i_mean_price / out.u_mean_price
    return out


def label(cand: pd.DataFrame, users: np.ndarray, targets: list[set], n_items: int) -> np.ndarray:
    tu = np.repeat(users, [len(t) for t in targets]).astype(np.int64)
    ti = np.fromiter((x for t in targets for x in t), np.int64, count=len(tu))
    keys = cand.user.to_numpy(np.int64) * n_items + cand.item.to_numpy(np.int64)
    return np.isin(keys, tu * n_items + ti).astype(np.int8)


def fill(lists, pop: np.ndarray, k: int) -> np.ndarray:
    """Top-k per user from each list, de-duplicated, then padded with best-sellers."""
    out = np.empty((len(lists), k), np.int64)
    for r, lst in enumerate(lists):
        seen, row = set(), []
        for x in list(lst) + list(pop):
            if x not in seen:
                seen.add(x)
                row.append(x)
                if len(row) == k:
                    break
        out[r] = row
    return out
