"""MovieLens-1M loading, id encoding and a leakage-safe temporal split."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_ml1m(data_dir: Path) -> Path:
    target = data_dir / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)
    return target


@dataclass
class Dataset:
    interactions: pd.DataFrame  # user, item, rating, ts  (encoded 0-based ids)
    users: pd.DataFrame         # indexed by encoded user id: gender, age, occupation
    items: pd.DataFrame         # indexed by encoded item id: year, genre_* columns
    n_users: int
    n_items: int

    @property
    def genre_matrix(self) -> np.ndarray:
        return self.items[[f"g_{g}" for g in GENRES]].to_numpy(np.float32)


def load_ml1m(data_dir: Path) -> Dataset:
    root = download_ml1m(data_dir)
    read = dict(sep="::", engine="python", encoding="latin-1", header=None)
    ratings = pd.read_csv(root / "ratings.dat", names=["user", "item", "rating", "ts"], **read)
    users = pd.read_csv(root / "users.dat", names=["user", "gender", "age", "occupation", "zip"], **read)
    movies = pd.read_csv(root / "movies.dat", names=["item", "title", "genres"], **read)

    # Encode ids to contiguous 0..N-1 (only items that were rated).
    user_codes = {u: i for i, u in enumerate(sorted(ratings.user.unique()))}
    item_codes = {m: i for i, m in enumerate(sorted(ratings.item.unique()))}
    ratings["user"] = ratings.user.map(user_codes)
    ratings["item"] = ratings.item.map(item_codes)

    users = users[users.user.isin(user_codes)].copy()
    users["user"] = users.user.map(user_codes)
    users["gender"] = (users.gender == "M").astype(int)
    users["age"] = users.age.astype("category").cat.codes
    users = users.set_index("user").sort_index()[["gender", "age", "occupation"]]

    movies = movies[movies.item.isin(item_codes)].copy()
    movies["item"] = movies.item.map(item_codes)
    movies["year"] = movies.title.str.extract(r"\((\d{4})\)\s*$").astype(float).fillna(1990)
    for g in GENRES:
        movies[f"g_{g}"] = movies.genres.str.split("|").apply(lambda gs, g=g: int(g in gs))
    movies = movies.set_index("item").sort_index()

    ratings = ratings.sort_values(["user", "ts", "item"]).reset_index(drop=True)
    return Dataset(ratings, users, movies, len(user_codes), len(item_codes))


def temporal_leave_last_k(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per user, hold out the k most recent interactions. Assumes df sorted by (user, ts)."""
    rank_from_end = df.groupby("user").cumcount(ascending=False)
    return df[rank_from_end >= k].copy(), df[rank_from_end < k].copy()


def global_time_split(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before the (1-frac) timestamp quantile vs. everything at/after it."""
    cut = df.ts.quantile(1 - frac)
    return df[df.ts < cut].copy(), df[df.ts >= cut].copy()


@dataclass
class Splits:
    train: pd.DataFrame        # history for the VAL stage (stage-1 fit, features)
    val: pd.DataFrame          # VAL-stage targets: ranker labels + model selection
    train_val: pd.DataFrame    # history for the TEST stage (everything before test)
    test: pd.DataFrame         # TEST-stage targets: final reporting only
    protocol: str


def _targets(window: pd.DataFrame, hist: pd.DataFrame, min_rating: int, min_hist: int) -> pd.DataFrame:
    """Positives in the target window, for users with enough prior history to personalise."""
    cnt = hist.user.value_counts()
    warm = cnt.index[cnt >= min_hist]
    return window[(window.rating >= min_rating) & window.user.isin(warm)].copy()


def make_splits(ds: Dataset, protocol: str = "last_k", k: int = 5, time_frac: float = 0.1,
                min_rating: int = 1, min_hist: int = 5) -> Splits:
    """train | val | test, two protocols.

    * ``last_k``  : per user, the last ``k`` interactions are test and the ``k`` before are val.
                    Standard on MovieLens, but training can contain *other* users' later events.
    * ``global``  : one global clock. The last ``time_frac`` of interactions is test and the
                    ``time_frac`` before it is val. Nothing after a cutoff is ever used to predict
                    what happens before it, which is how a deployed model actually sees data.

    History always contains every interaction (any rating). Targets keep only ratings
    ``>= min_rating`` (positives) for users with at least ``min_hist`` history events.
    Stage 1 and features are fit on ``train`` for the VAL stage, then refit on ``train_val``
    for the TEST stage, so the ranker sees the same "history -> next items" semantics.
    """
    df = ds.interactions
    if protocol == "last_k":
        train_val, test = temporal_leave_last_k(df, k)
        train, val = temporal_leave_last_k(train_val, k)
    elif protocol == "global":
        train_val, test = global_time_split(df, time_frac)
        train, val = global_time_split(train_val, time_frac / (1 - time_frac))
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return Splits(
        train=train,
        val=_targets(val, train, min_rating, min_hist),
        train_val=train_val,
        test=_targets(test, train_val, min_rating, min_hist),
        protocol=protocol,
    )
