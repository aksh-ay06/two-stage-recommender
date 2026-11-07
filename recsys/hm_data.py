"""H&M Personalized Fashion Recommendations: compact loading and weekly time splits.

The raw CSVs (3.7 GB, 64-char hex customer ids) are converted once into compact numpy
arrays (~0.5 GB): customer and article ids become int32 codes, dates become day offsets.

Protocol (the competition's own): predict what each customer buys in a target week from
everything before it. ``make_stage`` builds one such stage: a history window of ``weeks``
weeks strictly before ``start``, and the purchases in [start, start + 7) as targets.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HM_DIR = Path(os.environ.get("HM_DATA", Path.home() / "data" / "hm"))
DAY0 = pd.Timestamp("2018-09-20")
ITEM_CATS = ["product_type_no", "product_group_name", "colour_group_code", "perceived_colour_master_id",
             "department_no", "index_group_no", "section_no", "garment_group_no"]
TX_COLS = ("day", "user", "item", "price", "channel")


def _codes(s: pd.Series) -> np.ndarray:
    """Dense integer codes; missing values get code 0."""
    return (pd.factorize(s)[0] + 1).astype(np.int32)


def prepare(hm_dir: Path = HM_DIR) -> Path:
    out = hm_dir / "hm_compact.npz"
    if out.exists():
        return out
    t0 = time.time()
    cust = pd.read_csv(hm_dir / "customers.csv")
    arts = pd.read_csv(hm_dir / "articles.csv", usecols=["article_id", "product_code", *ITEM_CATS])
    cidx, aidx = pd.Index(cust.customer_id), pd.Index(arts.article_id)
    parts = {k: [] for k in TX_COLS}
    reader = pd.read_csv(hm_dir / "transactions_train.csv", chunksize=2_000_000,
                         dtype={"article_id": np.int64, "price": np.float32, "sales_channel_id": np.int8})
    for ch in reader:
        u, i = cidx.get_indexer(ch.customer_id), aidx.get_indexer(ch.article_id)
        assert (u >= 0).all() and (i >= 0).all(), "transaction with unknown customer/article"
        parts["day"].append((pd.to_datetime(ch.t_dat) - DAY0).dt.days.to_numpy(np.int16))
        parts["user"].append(u.astype(np.int32))
        parts["item"].append(i.astype(np.int32))
        parts["price"].append(ch.price.to_numpy(np.float32))
        parts["channel"].append(ch.sales_channel_id.to_numpy(np.int8))
        print(f"  [prepare] {sum(len(p) for p in parts['day']):,} transactions ({time.time() - t0:.0f}s)")
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    arrays.update(
        u_age=cust.age.fillna(-1).to_numpy(np.float32),
        u_club=_codes(cust.club_member_status),
        u_news=_codes(cust.fashion_news_frequency.str.upper().replace("NONE", np.nan)),
        u_fn=cust.FN.fillna(0).to_numpy(np.int8),
        u_active=cust.Active.fillna(0).to_numpy(np.int8),
        i_product_code=_codes(arts.product_code),
        article_id=arts.article_id.to_numpy(np.int64),
        **{f"i_{c}": _codes(arts[c]) for c in ITEM_CATS},
    )
    np.savez(out, **arrays)
    print(f"  [prepare] saved {out} ({time.time() - t0:.0f}s)")
    return out


@dataclass
class HMData:
    tx: pd.DataFrame        # day, user, item, price, channel (only rows on/after ``from_day``)
    users: pd.DataFrame     # static customer attributes, indexed by user code
    items: pd.DataFrame     # static article attributes, indexed by item code
    article_id: np.ndarray  # item code -> original article_id
    n_users: int
    n_items: int
    last_day: int

    @property
    def n_cats(self) -> dict[str, int]:
        return {c: int(self.items[c].max()) + 1 for c in self.items.columns}


def load_hm(hm_dir: Path = HM_DIR, from_day: int | None = None) -> HMData:
    z = np.load(prepare(hm_dir))
    day = z["day"]
    keep = slice(None) if from_day is None else day >= from_day
    tx = pd.DataFrame({k: z[k][keep] for k in TX_COLS})
    users = pd.DataFrame({k[2:]: z[k] for k in ("u_age", "u_club", "u_news", "u_fn", "u_active")})
    items = pd.DataFrame({k[2:]: z[k] for k in ["i_product_code", *[f"i_{c}" for c in ITEM_CATS]]})
    return HMData(tx, users, items, z["article_id"], len(users), len(items), int(day.max()))


@dataclass
class Stage:
    name: str
    start: int               # first day of the target week
    hist: pd.DataFrame       # purchases in [start - 7*weeks, start), deduplicated, sorted by (user, day)
    users: np.ndarray        # evaluated users: bought in the target week AND have history (sorted)
    targets: list[set]       # articles each evaluated user bought in the target week
    n_buyers: int            # all buyers in the target week, including cold-start customers


def make_stage(d: HMData, start: int, weeks: int, name: str) -> Stage:
    tx = d.tx
    hist = tx[(tx.day >= start - 7 * weeks) & (tx.day < start)]
    # repeated rows = several units of one article in one day; keep one purchase event
    hist = hist.drop_duplicates(["user", "item", "day"]).sort_values(["user", "day"], kind="stable")
    hist = hist.reset_index(drop=True)
    week = tx[(tx.day >= start) & (tx.day < start + 7)]
    tgt = week.groupby("user").item.apply(lambda s: set(s.tolist()))
    warm = np.intersect1d(tgt.index.to_numpy(), hist.user.unique())
    return Stage(name, start, hist, warm, tgt.loc[warm].tolist(), len(tgt))
