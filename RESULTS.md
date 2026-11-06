# Offline Results: MovieLens-1M

Every number on this page is a **test-set** number. All modelling decisions were made
on validation before the test set was scored:

- baseline hyperparameters (EASE λ, Item-kNN k);
- which ranker variant to use;
- early stopping.

## Setup

| | |
|---|---|
| Users / items / ratings | 6,040 / 3,706 / 1,000,209 |
| Candidates from stage 1 | 200 per user, from the full catalog, items already seen removed |
| Protocols | **leave-last-5** (per user), **global time split** (one clock), **leave-last-5 with ratings ≥ 4 as the only positives** |
| Seeds | 3 for leave-last-5 (42, 43, 44), 1 for each of the other protocols |
| Uncertainty | paired bootstrap over users (2,000 resamples) for every comparison; std across seeds |

**How a model decision is made.** Stage 1 and the features are fit on `train`, and
candidates are labeled with the `val` window. The validation users are then split 80/20:

- 80% train the LightGBM rankers;
- 20% are **ranker-holdout users**, used for early stopping and for picking the ranker
  variant with the best NDCG@10.

Baselines are tuned on the same validation window. For the test, everything is refit on
`train+val` and scored once.

## 1. Main result: leave-last-5, mean ± std over 3 seeds

| System | NDCG@10 | Recall@10 | Recall@50 | Recall@200 |
|---|---|---|---|---|
| Popularity | 0.0327 | 0.0390 | 0.1375 | 0.3378 |
| Item-kNN (k = 800, tuned) | 0.0584 | 0.0695 | 0.2145 | 0.4649 |
| EASE (λ = 1000, tuned) | 0.0712 | 0.0871 | 0.2902 | 0.6003 |
| Two-tower (stage 1 only) | 0.1130 ± 0.0016 | 0.1325 ± 0.0021 | 0.3629 ± 0.0009 | 0.6269 ± 0.0025 |
| **Two-stage: two-tower + GBM rerank** | **0.1385 ± 0.0010** | **0.1587 ± 0.0016** | **0.3987 ± 0.0008** | 0.6269 ± 0.0025 |

The baselines are deterministic, so they have no spread across seeds. Recall@200 is the
same for both two-tower rows because the reranker only reorders the 200 candidates.

**Paired lift in NDCG@10 for the two-stage system, by seed:**

| Compared with | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| Two-tower | +21.8% [+18.1, +25.4] | +23.8% [+20.3, +27.3] | +22.1% [+18.8, +25.9] |
| EASE | +94.0% [+85.1, +103.3] | +93.4% [+84.1, +103.7] | +95.9% [+87.1, +106.1] |
| Item-kNN | +136.5% [+124.8, +149.5] | +135.7% [+124.1, +149.4] | +138.8% [+127.5, +151.7] |

On every seed, the validation step chose the same ranker: the GBM with all features.

### Why does EASE lose here? Results by user activity

![NDCG@K](results/figures/ndcg_at_k.png)
![NDCG@10 by activity](results/figures/ndcg_by_activity.png)

| User activity quartile | Users | Two-tower | EASE | Two-stage | Two-stage vs two-tower | Two-stage vs EASE |
|---|---|---|---|---|---|---|
| Q1 (15–39 ratings) | 1,522 | 0.148 | 0.128 | 0.171 | +16% [+10, +21] | +34% [+25, +43] |
| Q2 (39–91) | 1,516 | 0.124 | 0.067 | 0.149 | +20% [+14, +27] | +121% [+100, +145] |
| Q3 (91–203) | 1,499 | 0.103 | 0.047 | 0.129 | +26% [+18, +34] | +172% [+141, +210] |
| Q4 (203–2,309) | 1,503 | 0.079 | 0.042 | 0.103 | +31% [+23, +41] | +148% [+117, +183] |

EASE scores a user's whole history as an unordered set. Leave-last-5 asks what the user
rates **next**. For a light user the whole history is a good proxy for current taste,
and EASE is close to the two-tower model. For a heavy user with hundreds of ratings, an
unordered history is dominated by old taste. The two-tower model only looks at the last
30 items, and the GBM's recency features look at the last 5, so they hold up. **The
reranker's lift grows with user activity**, which is where recency matters most.

## 2. Does the lift come from the model or from the recency features? (concern 2)

Same seeds and same candidates; only the reordering differs (mean over 3 seeds).

| Reorder the 200 two-tower candidates by… | NDCG@10 | vs two-tower order |
|---|---|---|
| Two-tower score (no reranking) | 0.1130 | – |
| `sim_recent_max` alone (heuristic) | 0.0872 | **−23%** |
| GBM without the recency features | 0.1145 | +1% |
| GBM without two-tower score and rank | 0.1267 | +12% |
| **GBM with all features** | **0.1385** | **+23%** |

- **The recency features are necessary.** Without them the GBM is no better than the
  retriever's own order.
- **They are not sufficient.** Sorting by recency similarity alone is 23% *worse* than
  the two-tower order.
- **The lift comes from learning how to combine them.** The GBM learns when to trust
  "similar to what you just watched" and when to trust the long-term two-tower score.

This supports the reviewer's hypothesis in part: a retriever that models the order of
recent items (SASRec-style) would probably absorb some of this lift. That is the next
experiment (see the README).

Top features by gain (seed 42): `tt_rank` 21.6%, `sim_recent_max` 11.3%,
`sim_last` 8.1%, `sim_recent_mean` 7.3%, `pop_gap` 6.6%, `user_cnt` 5.2%.

## 3. Global time split: the harder number (concern 4)

Here there is one clock. Training uses everything before 2000-12-02, validation covers
2000-12-02 to 2000-12-29, and test is everything after 2000-12-29. That's the last 10% of
ratings, spread over two years. The test covers 1,175 users with at least 5 earlier
ratings, averaging about 81 held-out items each.

| System | NDCG@10 | Recall@10 | Recall@50 | Recall@200 |
|---|---|---|---|---|
| Popularity | 0.2536 | 0.0393 | 0.1428 | 0.3418 |
| Item-kNN (k = 1600) | 0.2653 | 0.0427 | 0.1355 | 0.3273 |
| **EASE (λ = 32000)** | **0.2836** | 0.0508 | **0.1629** | **0.3857** |
| Two-tower (stage 1 only) | 0.1866 | 0.0370 | 0.1234 | 0.3166 |
| Two-stage (validation chose GBM without recency features) | 0.2534 | **0.0530** | 0.1548 | 0.3166 |

**On a strict time split, the two-stage system does not beat EASE.**

| Two-stage vs… | NDCG@10 | Recall@10 |
|---|---|---|
| EASE | **−10.7% [−15.0, −6.2]** | +4.3% [−6.9, +17.4], not significant |
| Two-tower | +35.8% [+28.5, +43.7] | +43.0% [+26.8, +61.3] |

What's going on:

- **The bottleneck is retrieval, not ranking.** The two-tower candidates contain 31.7%
  of the held-out items, against 38.6% for EASE's top 200. The reranker adds 36% over the
  retriever, but it can't rank items that were never retrieved.
- **The test is different in kind.** It mostly asks what the user will rate over the next
  months, not what they'll rate next. Popularity alone scores 0.254. The selected ranker's
  top feature is recent item popularity (28% of gain), and the recency features were
  dropped by validation. Both are consistent with "what's trending" mattering more than
  "what you just watched" here.
- **Strong regularisation wins.** EASE's best λ is 32× larger than under leave-last-5.
- **Validation data is thin.** Only 520 users are active in the validation window, and the
  ranker-holdout split has 104. The rankers early-stop at about 20 trees, so ranker
  selection on this protocol is noisy.

**Takeaway.** Under leave-last-5 the two-tower model is a clearly better retriever than
EASE. On a strict time split it isn't, so a production version should **use several
candidate sources**: merge the EASE and two-tower candidates, and give the GBM both
scores as features. This is the top next step.

## 4. Only ratings ≥ 4 count as positives (concern 5)

History still includes every rating, since the user has seen those films. Only ratings of
4 and 5 count as targets, both for the two-tower training loss and for evaluation. The test
covers 5,553 users.

| System | NDCG@10 | Recall@10 | Recall@50 |
|---|---|---|---|
| EASE (λ = 2000) | 0.0722 | 0.1027 | 0.3236 |
| Two-tower | 0.0864 | 0.1208 | 0.3373 |
| **Two-stage** | **0.1127** | **0.1516** | **0.3740** |

Two-stage vs two-tower: **+30.4%** NDCG@10 [+25.4, +35.4]. Two-stage vs EASE: **+56.2%**
[+47.1, +65.7]. The reranker's lift is *larger* when targets are things the user liked.
`item_mean_rating` and `rating_gap` now carry signal. EASE closes some of the gap
(−17% vs two-tower, from −37%), because liked items are more predictable from the full
history than any rated item is.

## 5. Serving latency (concern 6)

Measured with `scripts/benchmark_serving.py`: single thread, 1,000 requests after
warm-up, the real seed-42 model, and a LightGBM ranker with 151 trees.

| Step | p50 | p95 | p99 |
|---|---|---|---|
| User tower (history → embedding) | 1.2 ms | 1.6 ms | 1.8 ms |
| Retrieval, exact top-200 of 3.7k | 0.2 ms | 0.3 ms | 0.4 ms |
| Feature transform (200 candidates, precomputed store) | 1.5 ms | 2.1 ms | 2.4 ms |
| GBM scoring (200 × 36) | 7.2 ms | 9.2 ms | 10.5 ms |
| **End to end** | **10.1 ms** | **13.0 ms** | **16.1 ms** |

The GBM is 70% of the latency. Cheap ways to cut it: cap the number of trees, score fewer
candidates (e.g. 100), or compile the model (treelite or lleaves). The offline batch job,
which builds the history index, item embeddings and feature store, takes 1.3 s.

**Retrieval at scale.** The synthetic catalogs are real item vectors plus noise, with
500 queries each, top 200, on a single thread.

| Catalog | Exact p50 | HNSW ef=256 | HNSW ef=512 | HNSW ef=1024 | HNSW build |
|---|---|---|---|---|---|
| 3.7k | 0.15 ms | 0.75 ms, recall 1.000 | 1.5 ms, 1.000 | 3.9 ms, 1.000 | 1 s |
| 100k | 3.6 ms | 2.2 ms, 0.960 | 4.7 ms, 0.992 | 8.7 ms, 0.999 | 32 s |
| 1M | **38.5 ms** | 2.0 ms, 0.824 | 3.6 ms, 0.927 | **7.7 ms, 0.976** | 10 min |

![Retrieval scaling](results/figures/retrieval_scaling.png)

Brute force wins below about 50k items, so ML-1M correctly uses no ANN index. At 1M items,
HNSW is 5× faster at 97.6% recall, and the index needs to be rebuilt offline, which takes
10 minutes per rebuild on this machine.

## 6. The `user_occ` decision, restated on validation (concern 3)

The first version had a user-occupation feature. It was removed because it made **validation**
NDCG@10 worse on the users held out from ranker training: 0.1901 with it, 0.1974
without it. These are LightGBM's own NDCG@10 over each holdout user's candidates, from
the earlier pipeline's 10% holdout. The cause: early stopping triggered at 74 trees instead of 152, because splits
on a feature that is constant within each user overfit. The test-set comparison was
reported only after that decision. The pipeline now enforces this rule for every ranker
variant.

## Caveats

- **Seeds.** Only the leave-last-5 protocol has 3 seeds; global and ratings ≥ 4 have one
  each. The bootstrap CIs cover variation across users, not across runs.
- **Labels.** "Rated it" is still not "clicked it when shown". There's no impression or
  position data, which is why the A/B test is still needed (`AB_TEST_PLAN.md`).
- **Global split.** Its validation window is small (520 users), so ranker selection on
  that protocol is noisy.
- **Synthetic catalog.** The 100k/1M catalogs are synthetic, so HNSW recall on a real large
  catalog will differ.
