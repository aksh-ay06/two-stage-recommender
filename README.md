# Two-Stage Recommender: Two-Tower Retrieval + GBM Reranking

An end-to-end recommender on **MovieLens-1M** (1M ratings, 6,040 users, 3,706 movies):

1. **Retrieval:** a PyTorch two-tower model picks 200 candidates per user from the full catalog.
2. **Ranking:** a LightGBM LambdaRank model reorders those candidates using 36 user, item and user×item features.
3. **Offline evaluation:** NDCG@K and Recall@K, with paired-bootstrap confidence intervals for the reranker's lift.
4. **Online plan:** an A/B test design in [`AB_TEST_PLAN.md`](AB_TEST_PLAN.md).

**Headline:** reranking improves NDCG@10 from 0.113 to **0.140 (+23%, 95% CI +20% to +27%)**.
Popularity scores 0.033. Full results are in [`RESULTS.md`](RESULTS.md).

## Architecture
```
user history ─┐
demographics ─┤ user tower ─┐
              │             ├─ dot product ─ top-200 ─► feature builder ─► LightGBM LambdaRank ─► top-K
item id/genre/year ─ item tower ┘   (unseen items)       (stats, recency sims,
                                                            affinities, tt score)
```

**Two-tower model** (`recsys/two_tower.py`)
* User tower: user ID, gender, age, occupation, and the mean of the user's last 30
  item embeddings. During training the history includes only items **before** the
  target interaction, so the target can't leak into its own input.
* Item tower: item ID, genre multi-hot, and release-year bucket.
* Both towers output L2-normalized 64-dimensional vectors.
* Loss: in-batch softmax with temperature 0.05. A logQ correction offsets the
  popularity bias of in-batch negatives, and duplicate positives within a batch are masked.

**Reranker** (`recsys/features.py`, `recsys/ranker.py`)
* Features: the two-tower score and rank; the candidate's similarity to the user's
  last 5 items; item popularity (overall and recent), mean rating and age; user
  activity and mean rating; genre affinity; popularity, year and rating gaps between
  the item and the user's usual taste.
* Ranker labels come from the validation window. The test stage refits stage 1 and
  all features on train+val, so feature semantics stay the same and test labels are
  never seen.

## Run it
```bash
python -m venv .venv
.venv/Scripts/pip install numpy pandas scikit-learn lightgbm        # Windows path; use .venv/bin on macOS/Linux
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python run_pipeline.py              # about 18 minutes on CPU; downloads ML-1M if it's missing
.venv/Scripts/python run_pipeline.py --epochs 1   # about 7-minute smoke run
```
Outputs are written to `results/`: `metrics.csv`, `results.json` and `feature_importance.csv`.

## Layout
```
recsys/data.py        load + encode ML-1M, per-user temporal split
recsys/two_tower.py   stage 1 model, training, retrieval
recsys/features.py    stage 2 features
recsys/ranker.py      LightGBM LambdaRank training + reranking
recsys/metrics.py     Recall@K, NDCG@K, paired bootstrap
run_pipeline.py       end-to-end experiment
AB_TEST_PLAN.md       online experiment design
RESULTS.md            offline results + caveats
```

## Next steps
* Use a global time split, and run several seeds to measure run-to-run variance.
* More ablations: measure how much each feature group contributes. Use `--reuse-stage1 --ablate <features>`, about 2 minutes per run.
* Add hard negatives or an ANN index (FAISS) for retrieval. Add a sequential user tower (SASRec-style).
* Try the H&M dataset, which has a much larger catalog and real purchase data.
