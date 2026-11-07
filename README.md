# Two-Stage Recommender: Two-Tower + EASE Retrieval, GBM Reranking

An end-to-end recommender on **MovieLens-1M** (1M ratings, 6,040 users, 3,706 movies),
repeated on **H&M** (31.8M purchases, 1.37M customers, 105k articles; see
[below](#hm-105k-articles-real-purchases)):

1. **Retrieval:** a PyTorch two-tower model and a closed-form EASE model each pick their top
   200 items per user from the full catalog. Their union, about 310 items, is the candidate set.
2. **Ranking:** a LightGBM LambdaRank model reorders the candidates using 38 user, item and
   user×item features, including both retrievers' scores and ranks.
3. **Offline evaluation:** NDCG@K and Recall@K against tuned **EASE** and **Item-kNN**
   baselines, on 3 protocols, with 3 seeds and paired-bootstrap confidence intervals. Every
   modelling decision is made on validation, never on test, including the choice of
   candidate pool.
4. **Serving:** per-request latency is measured, and FAISS HNSW is benchmarked against
   brute force up to 1M items.
5. **Online plan:** an A/B test design in [`AB_TEST_PLAN.md`](AB_TEST_PLAN.md).

## Headline results (test set; details in [`RESULTS.md`](RESULTS.md))

NDCG@10 by protocol:

| Protocol | Two-tower | EASE (tuned) | Two-tower cands + GBM | **Two-stage (selected)** | vs two-tower | vs EASE |
|---|---|---|---|---|---|---|
| Leave-last-5, 3 seeds | 0.113 | 0.071 | 0.139 | **0.158** | **+36% to +43%** | **+117% to +126%** |
| Ratings ≥ 4 only | 0.086 | 0.072 | 0.114 | **0.130** | **+50%** | **+80%** |
| Global time split | 0.187 | 0.284 | 0.254 | **0.303** | **+62%** | **+6.7% [+2.2, +11.4]** |

"Two-tower cands + GBM" is the earlier single-source version of the system. Adding EASE as
a second source beats it by 12–19% on every run.

![NDCG@K](results/figures/ndcg_at_k.png)

**What the experiments show:**

- **Two retrievers beat either one.** The two-tower model and EASE overlap less than their
  similar recall suggests. Their union holds 74.5% of the held-out items under
  leave-last-5, against 62.7% for the two-tower model alone. On the global time split it
  holds 48.7%, against 31.7%.
- **The multi-source system now beats EASE on the strict global time split.** The
  single-source version lost to it by 10.7%. There, EASE features on two-tower candidates
  aren't enough; the extra candidates are what make the difference.
- **Under leave-last-5, EASE helps mostly as a feature.** On the same two-tower
  candidates, adding EASE's score and rank as features gives +12%. Adding EASE's
  candidates gives another +3%.
- **Recency features are still the biggest single contributor.** Removing them costs 17%,
  yet sorting by recency alone is 23% worse than the two-tower order. The GBM's value is
  in weighing recency against long-term co-occurrence (EASE) and the two-tower score.
- **Serving fits a real latency budget:** 8 ms p50 and 17 ms p99 end to end on one CPU
  thread, with the GBM taking about 75% of that. Brute-force retrieval is fine up to about
  50k items. At 1M items, HNSW is 5× faster at 97.6% recall.

## H&M: 105k articles, real purchases

The same two-stage design on the H&M Kaggle data, with the competition's protocol: test is
the last week (2020-09-16 to 22), validation the week before, and each stage sees a 12-week
history window. Candidates are the union of four sources: two-tower top 100, EASE top 50,
the customer's past purchases (re-buy) and last week's best-sellers, about 165 articles
per user. Test results for the 49,219 test-week buyers with history, one seed:

| System | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|
| Popularity (last week) | 0.0091 | 0.0255 | 0.0161 |
| Re-buy, then popular | 0.0331 | 0.0634 | 0.0462 |
| EASE (tuned) | 0.0148 | 0.0394 | 0.0247 |
| Two-tower | 0.0069 | 0.0201 | 0.0125 |
| GBM on two-tower candidates | 0.0171 | 0.0346 | 0.0257 |
| **Two-stage (multi-source GBM, selected on validation)** | **0.0381** | **0.0780** | **0.0550** |

- **The two-stage system beats re-buy-then-popular, the strongest baseline, by 15.0% on
  MAP@12** [+12.6, +17.4], and EASE by 157%.
- **Retrieval is the bottleneck.** The four-source union holds only 18.5% of what users
  buy. Each source alone finds 5–9%, and they overlap little.
- **The two-tower model earns its place as a candidate source, not as a feature.** Alone it
  ranks below last-week popularity, but as a source its recall (7.7%) is close to EASE's
  (8.5%). Its scores add nothing to the ranker (+1.1% [−0.3, +2.6] without them), and
  validation chose the ranker without them.

Full setup, validation numbers and caveats are in [`RESULTS.md` §7](RESULTS.md#7-hm-105k-articles-real-purchases).

## Architecture
```
                         ┌─ user tower · item tower ─ dot product ─ top-200 ─┐
user history + profile ──┤                                                    ├─ union (~310) ─► FeatureStore.transform ─► LightGBM LambdaRank ─► top-K
                         └─ EASE: history · B (item-item) ─────────── top-200 ─┘                 (precomputed offline;
                                                                                                  both sources' score + rank)
```

**Two-tower model** (`recsys/two_tower.py`)
* User tower: user ID, gender, age, occupation, and the mean of the user's last 30 item
  embeddings. During training the history includes only items **before** the target, so
  the target can't leak into its own input.
* Item tower: item ID, genre multi-hot, and release-year bucket.
* Both towers output L2-normalized 64-dimensional vectors.
* Loss: in-batch softmax with temperature 0.05. A logQ correction offsets the popularity
  bias of in-batch negatives, and duplicate positives within a batch are masked.

**Candidate merge** (`recsys/candidates.py`): the union of both sources' top 200. Every
candidate gets `tt_score`, `tt_rank`, `ease_score` and `ease_rank`. A source that didn't
retrieve the item gets rank 200.

**Reranker** (`recsys/features.py`, `recsys/ranker.py`)
* `FeatureStore` precomputes every user and item statistic offline from the history log.
  `transform()` is the cheap online step, and it works in chunks to keep memory bounded.
* Features:
  * both retrievers' score and rank;
  * the candidate's similarity to the user's last 5 items;
  * item popularity (overall and recent), mean rating and age;
  * user activity and mean rating;
  * genre affinity;
  * popularity, year and rating gaps between the item and the user's usual taste.
* Validation users are split into ranker-train users and ranker-holdout users. The holdout
  users drive early stopping and the choice among six ranker variants: the two candidate
  pools, plus feature-group ablations.

**Baselines** (`recsys/baselines.py`): Popularity, Item-kNN (cosine, top-k neighbours) and
EASE (closed form). The tuning grids are wide enough that every chosen value is inside the
grid, not at its edge.

## Run it
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin on macOS/Linux; torch CPU wheel: --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python -m pytest tests                          # 17 unit tests, about 4 s
.venv/Scripts/python run_pipeline.py                          # leave-last-5, seed 42 (about 15-20 min on CPU)
.venv/Scripts/python run_pipeline.py --seed 43                # more seeds
.venv/Scripts/python run_pipeline.py --protocol global        # global time split
.venv/Scripts/python run_pipeline.py --min-rating 4           # ratings >= 4 as positives
.venv/Scripts/python run_pipeline.py --reuse-stage1           # reuse cached two-tower weights
.venv/Scripts/python scripts/benchmark_serving.py --run last_k_minr1_seed43   # latency + ANN benchmark
.venv/Scripts/python scripts/make_report.py                   # seed summary, segments, figures
```
Each run writes to `results/runs/<protocol>_minr<r>_seed<s>/`:
- `metrics_test.csv` and `metrics_val_holdout.csv`
- `results.json`, with the chosen hyperparameters, candidate recall per source, and paired comparisons
- `feature_importance.csv`
- `ranker.txt`

The dataset downloads automatically. A full run peaks at a few GB of RAM.

**H&M.** Download the Kaggle competition's `articles.csv`, `customers.csv` and
`transactions_train.csv` into `~/data/hm`, or set `HM_DATA` to another directory. The
first run compacts them into `hm_compact.npz`.
```bash
.venv/Scripts/python run_hm.py --smoke     # 4-week window, 1 epoch, 3,000 users: a few minutes
.venv/Scripts/python run_hm.py             # full run, about 30 min on CPU
.venv/Scripts/python run_hm.py --no-logq   # two-tower without the logQ correction
```
Results go to `results/hm/` (same files as above, with MAP@12).

## Layout
```
recsys/data.py          load ML-1M, leave-last-k and global time-split protocols
recsys/two_tower.py     stage 1 model, training, retrieval
recsys/candidates.py    multi-source candidate merge (two-tower ∪ EASE) with per-source score/rank
recsys/features.py      FeatureStore: offline precompute, online transform
recsys/ranker.py        LightGBM LambdaRank, validation-based variant selection
recsys/baselines.py     Popularity, Item-kNN, EASE
recsys/metrics.py       Recall@K, NDCG@K, paired bootstrap
recsys/hm_data.py       H&M loading, compaction, weekly stages
recsys/hm_model.py      H&M two-tower towers (catalog attributes, no customer-ID embedding)
recsys/hm_features.py   H&M candidate sources (two-tower, EASE, re-buy, popular) and ranker features
run_pipeline.py         end-to-end experiment (MovieLens)
run_hm.py               end-to-end experiment (H&M)
scripts/                serving benchmark, report and figures
tests/                  unit tests (metrics, split leakage, history gathering, baselines, features, candidate merge, reranking)
RESULTS.md              full results, ablations, caveats
AB_TEST_PLAN.md         online experiment design
```

## Next steps
1. **Sequential user tower** (SASRec-style): test whether it absorbs the recency lift the
   GBM gets today, and whether it closes the two-tower model's gap on the global split.
2. **Faster GBM:** cap the number of trees, or compile the model (treelite), to cut the
   6 ms scoring step. Also try scoring fewer candidates, such as the top 150 per source.
3. **Scalable EASE:** sparsify B, or precompute EASE candidates offline, before moving to
   a catalog of 100k+ items.
4. **H&M retrieval recall:** the four-source union finds only 18.5% of purchases. Extend
   the EASE λ grid below 50 (the chosen value is at the edge), add more sources (items
   bought together with recent purchases, trending items by age group), and run more seeds.
