"""Unit tests for the parts that are cheap to test and expensive to get wrong."""
import numpy as np
import pandas as pd
import pytest
import torch

from recsys.baselines import binary_matrix, ease_scores, itemknn_scores, top_k_unseen
from recsys.data import GENRES, Dataset, make_splits
from recsys.features import FeatureStore
from recsys.metrics import evaluate, ndcg_at_k, paired_bootstrap, recall_at_k
from recsys.two_tower import HistoryIndex, retrieve


# ---------------------------------------------------------------- fixtures
def make_ds(n_users=6, n_items=10, per_user=14, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False) if per_user <= n_items else rng.permutation(n_items)
        ts = np.sort(rng.choice(np.arange(1000, 100000), size=len(items), replace=False))
        rows += [(u, int(i), int(rng.integers(1, 6)), int(t)) for i, t in zip(items, ts)]
    inter = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"]).sort_values(["user", "ts"]).reset_index(drop=True)
    users = pd.DataFrame({"gender": rng.integers(0, 2, n_users), "age": rng.integers(0, 7, n_users),
                          "occupation": rng.integers(0, 21, n_users)})
    items = pd.DataFrame({"year": rng.integers(1950, 2000, n_items).astype(float)})
    for g in GENRES:
        items[f"g_{g}"] = rng.integers(0, 2, n_items)
    return Dataset(inter, users, items, n_users, n_items)


@pytest.fixture
def ds():
    return make_ds(n_users=8, n_items=40, per_user=20)


# ---------------------------------------------------------------- metrics
def test_ndcg_recall_hand_computed():
    ranked = np.array([[5, 1, 7, 3]])
    targets = [{1, 3, 9}]
    m = evaluate(ranked, targets, ks=(2, 4))
    assert m["Recall@2"][0] == pytest.approx(1 / 3)
    assert m["Recall@4"][0] == pytest.approx(2 / 3)
    # hits at ranks 2 and 4; ideal = 3 relevant items, capped at k
    dcg4 = 1 / np.log2(3) + 1 / np.log2(5)
    idcg4 = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert m["NDCG@4"][0] == pytest.approx(dcg4 / idcg4)
    assert m["NDCG@2"][0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_perfect_and_empty_rankings():
    hits_perfect = np.array([[1, 1, 0, 0]], dtype=bool)
    assert ndcg_at_k(hits_perfect, np.array([2.0]), 4)[0] == pytest.approx(1.0)
    assert recall_at_k(hits_perfect, np.array([2.0]), 4)[0] == pytest.approx(1.0)
    hits_none = np.zeros((1, 4), dtype=bool)
    assert ndcg_at_k(hits_none, np.array([2.0]), 4)[0] == 0.0


def test_paired_bootstrap_identical_systems_has_zero_lift():
    a = np.random.default_rng(0).random(500)
    lift, ci = paired_bootstrap(a, a.copy())
    assert lift == pytest.approx(0.0)
    assert ci[0] == pytest.approx(0.0) and ci[1] == pytest.approx(0.0)


# ---------------------------------------------------------------- splits: no leakage
def test_last_k_split_is_per_user_temporal(ds):
    sp = make_splits(ds, "last_k", k=3, min_hist=1)
    for u, g in ds.interactions.groupby("user"):
        tv, te = sp.train_val[sp.train_val.user == u], sp.test[sp.test.user == u]
        tr, va = sp.train[sp.train.user == u], sp.val[sp.val.user == u]
        assert len(te) == 3 and len(va) == 3
        assert tv.ts.max() < te.ts.min()
        assert tr.ts.max() < va.ts.min()
        assert not set(te.item) & set(tv.item)
    assert len(sp.train_val) + len(sp.test) == len(ds.interactions)


def test_global_split_respects_one_clock(ds):
    sp = make_splits(ds, "global", time_frac=0.2, min_hist=1)
    assert sp.train.ts.max() < sp.val.ts.min()
    assert sp.train_val.ts.max() < sp.test.ts.min()
    assert sp.val.ts.max() <= sp.train_val.ts.max()


def test_min_rating_and_warm_user_filters(ds):
    sp = make_splits(ds, "last_k", k=3, min_rating=4, min_hist=1)
    assert (sp.test.rating >= 4).all() and (sp.val.rating >= 4).all()
    # history keeps every rating
    assert (sp.train_val.rating < 4).any()
    sp2 = make_splits(ds, "global", time_frac=0.2, min_hist=10_000)
    assert sp2.test.empty


# ---------------------------------------------------------------- two-tower helpers
def test_history_gather_only_sees_strictly_earlier_items_of_same_user():
    df = pd.DataFrame({"user": [0, 0, 0, 1, 1], "item": [10, 11, 12, 20, 21]})
    h = HistoryIndex(df, pad=99, hist_len=3)
    out = h.gather(torch.arange(5), torch.tensor(h.user_start))
    assert out[0].tolist() == [99, 99, 99]          # first event: no history
    assert out[2].tolist() == [11, 10, 99]          # most recent first, never itself
    assert out[3].tolist() == [99, 99, 99]          # user 1 does not see user 0's items
    assert out[4].tolist() == [20, 99, 99]
    # serving-time history: everything up to the end of each user's log
    end = h.gather(torch.tensor(h.user_end), torch.tensor(h.user_first))
    assert end[0].tolist() == [12, 11, 10] and end[1].tolist() == [21, 20, 99]


def test_retrieve_never_returns_seen_items():
    rng = np.random.default_rng(0)
    U, V = rng.normal(size=(3, 8)).astype(np.float32), rng.normal(size=(30, 8)).astype(np.float32)
    seen = pd.DataFrame({"user": [0, 0, 1], "item": [int(np.argmax(V @ U[0])), 5, int(np.argmax(V @ U[1]))]})
    items, scores = retrieve(U, V, seen, k=10, users=np.array([0, 1, 2]))
    assert not set(items[0]) & {seen.item[0], 5}
    assert seen.item[2] not in items[1]
    assert (np.diff(scores, axis=1) <= 0).all()      # sorted descending


# ---------------------------------------------------------------- baselines
def test_ease_has_zero_diagonal_and_ranks_cooccurring_item_first():
    # items 0 and 1 always co-occur; item 2 is independent
    X = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    S = ease_scores(X, lam=0.5)
    assert S[4, 1] > S[4, 2]
    K = itemknn_scores(X, k=1)
    assert K[4, 1] > K[4, 2]


def test_top_k_unseen_masks_history():
    X = np.array([[1, 0, 0, 1]], dtype=np.float32)
    scores = np.array([[9.0, 1.0, 2.0, 8.0]])
    assert top_k_unseen(scores, X, np.array([0]), 2).tolist() == [[2, 1]]


def test_binary_matrix():
    df = pd.DataFrame({"user": [0, 0, 1], "item": [1, 1, 2]})
    assert binary_matrix(df, 2, 3).tolist() == [[0, 1, 0], [0, 0, 1]]


# ---------------------------------------------------------------- features
def test_feature_store_recent_items_are_right_aligned_and_ordered(ds):
    V = np.random.default_rng(0).normal(size=(ds.n_items, 4)).astype(np.float32)
    store = FeatureStore(ds, ds.interactions, V, n_recent=3)
    for u, g in ds.interactions.groupby("user"):
        assert store.recent[u].tolist() == g.item.tolist()[-3:]
    cand = np.array([0, 1, 2])
    f = store.transform(np.zeros(3, dtype=np.int64), cand, {"tt_score": np.zeros(3)}, chunk=2)  # chunk < rows
    last = store.recent[0, -1]
    assert f.sim_last.to_numpy() == pytest.approx(V[cand] @ V[last], rel=1e-5)
    assert f.sim_recent_max.to_numpy() == pytest.approx((V[cand] @ V[store.recent[0]].T).max(1), rel=1e-5)
    assert "tt_score" in f and len(f) == 3
    assert not f.drop(columns=["item_mean_rating", "rating_gap"]).isna().any().any()


# ---------------------------------------------------------------- multi-source candidates
def test_merge_candidates_is_union_with_capped_ranks():
    from recsys.candidates import merge_candidates
    S_tt = np.array([[5, 4, 3, 2, 1, -np.inf], [1, 2, 3, 4, 5, 6]], dtype=np.float32)
    S_ease = np.array([[1, 2, 3, 4, 5, -np.inf], [6, 5, 4, 3, 2, 1]], dtype=np.float32)
    c = merge_candidates(S_tt, S_ease, k=2)
    got = {(int(r), int(i)) for r, i in zip(c.rows, c.items)}
    assert got == {(0, 0), (0, 1), (0, 4), (0, 3), (1, 5), (1, 4), (1, 0), (1, 1)}
    assert (np.diff(c.rows) >= 0).all()                       # grouped by user row
    row0 = {int(i): (tr, er) for r, i, tr, er in zip(c.rows, c.items, c.tt_rank, c.ease_rank) if r == 0}
    assert row0[0] == (0, 2) and row0[4] == (2, 0)            # absent from a source -> rank capped at k
    assert c.tt_top[0].tolist() == [0, 1] and c.ease_top[0].tolist() == [4, 3]
    assert 5 not in c.items[c.rows == 0]                      # seen (-inf) item never retrieved


def test_rerank_handles_variable_group_sizes():
    from recsys.ranker import rerank_by
    df = pd.DataFrame({"user": [0, 0, 0, 1, 1], "item": [7, 8, 9, 3, 4], "s": [0.1, 0.9, 0.5, 0.2, 0.3]})
    assert rerank_by(df, "s", 2).tolist() == [[8, 9], [4, 3]]


# ---------------------------------------------------------------- H&M pieces
def test_map_at_k_matches_kaggle_definition():
    from recsys.metrics import map_at_k
    hits = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    n_rel = np.array([2.0, 3.0, 5.0])
    # user 0: (1/1 + 2/3) / min(2, 4); user 1: 0; user 2: (1/2) / min(5, 4)
    np.testing.assert_allclose(map_at_k(hits, n_rel, 4), [(1 + 2 / 3) / 2, 0.0, 0.5 / 4])


def test_pair_lookup_hits_and_defaults():
    from recsys.hm_features import PairLookup
    lk = PairLookup([3, 1, 3], [7, 2, 0], 10, cnt=[5, 6, 7])
    np.testing.assert_array_equal(lk.get([1, 3, 3, 2], [2, 0, 7, 2], "cnt", default=-1), [6, 7, 5, -1])


def test_hm_label_and_fill():
    import pandas as pd
    from recsys.hm_features import fill, label
    cand = pd.DataFrame({"user": [0, 0, 2, 2], "item": [5, 6, 5, 9]})
    np.testing.assert_array_equal(label(cand, np.array([0, 2]), [{6}, {5, 1}], n_items=10), [0, 1, 1, 0])
    # de-duplicates the list, then pads with best-sellers it hasn't used
    np.testing.assert_array_equal(fill([[4, 4, 2], []], np.array([2, 7, 8]), 3), [[4, 2, 7], [2, 7, 8]])
