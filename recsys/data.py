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


def make_splits(ds: Dataset, k: int = 5):
    """train | val | test, each user's history in time order.

    * Stage-1 model + features fit on ``train``; ranker labels come from ``val``.
    * For the final test, everything is refit on ``train + val`` and scored on ``test``,
      so the ranker sees features with the same "history -> next k items" semantics.
    """
    train_val, test = temporal_leave_last_k(ds.interactions, k)
    train, val = temporal_leave_last_k(train_val, k)
    return train, val, train_val, test
