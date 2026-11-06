# Offline Results: MovieLens-1M

**Protocol:** each user's interactions are sorted by time, and the last 5 are held
out for **test**. The 5 before those are **validation**, used to label the ranker's
training data. The test stage refits the two-tower model and all features on
train+val, so no test labels are seen anywhere. All 6,040 users are evaluated
against the full catalog of 3,706 items, with items the user has already seen
removed. Retrieval returns 200 candidates per user.

## Test metrics (mean over 6,040 users)

| System | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Recall@200 |
|---|---|---|---|---|---|---|---|
| Popularity | 0.0390 | 0.0327 | 0.0705 | 0.0461 | 0.1376 | 0.0684 | 0.3377 |
| Two-tower (stage 1 only) | 0.1344 | 0.1134 | 0.2135 | 0.1471 | 0.3631 | 0.1974 | 0.6266 |
| **Two-tower + LightGBM rerank** | **0.1608** | **0.1400** | **0.2499** | **0.1780** | **0.3988** | **0.2279** | 0.6266 |

These are the numbers for the current ranker, which no longer uses `user_occ` (see
the ablation below). With `user_occ`, NDCG@10 was 0.1388.

Recall@200 is the same for both two-tower systems because the reranker only
reorders the 200 candidates. That value, 0.627, is the most the reranker can
recover.

## Reranker lift over two-tower ordering (paired bootstrap over users, 2,000 resamples)

| Metric | Lift | 95% CI |
|---|---|---|
| NDCG@10 | **+23.4%** | [+19.9%, +26.9%] |
| Recall@10 | +19.7% | [+16.5%, +22.9%] |
| NDCG@20 | +21.0% | [+18.4%, +23.6%] |
| Recall@20 | +17.0% | [+14.9%, +19.3%] |
| Recall@50 | +9.8% | [+8.6%, +11.1%] |

The lift is largest at the top of the list, which is the part the reranker is
trained for (LambdaRank truncated at 20).

## What the reranker uses (share of total gain)

| Feature | Gain | Meaning |
|---|---|---|
| `tt_rank` | 22.4% | stage-1 rank (the GBM refines the retriever's order, it doesn't replace it) |
| `sim_recent_max` / `sim_last` / `sim_recent_mean` | 27.1% combined | similarity of the candidate to the user's last 5 items (recency) |
| `pop_gap` | 6.0% | candidate popularity relative to the user's usual popularity taste |
| `item_mean_rating`, `item_log_pop_recent` | 9.2% combined | quality and trending signals |
| `user_cnt` | 4.7% | user activity level (interaction feature) |

Full list: `results/feature_importance.csv`.

**Why `user_occ` was removed:** in the first version, occupation was the
5th-largest feature by gain. A feature that is the same for all of a user's
candidates can't reorder them by itself. It only matters through interactions with
other features. With 21 categorical levels it was a likely overfitting risk, so we
tested dropping it (below).

## Ablation: drop `user_occ`

Both rankers reorder the same cached candidates, and each user is compared with
themselves under both rankers (paired bootstrap). The command was
`run_pipeline.py --ablate user_occ`, and the outputs are in `results/ablation_no_user_occ/`.

| Metric | Full ranker | Without `user_occ` | Change | 95% CI |
|---|---|---|---|---|
| NDCG@10 | 0.1388 | 0.1400 | +0.9% | [-0.6%, +2.4%] |
| Recall@10 | 0.1602 | 0.1608 | +0.4% | [-1.2%, +1.9%] |
| NDCG@20 | 0.1754 | 0.1780 | **+1.5%** | [+0.4%, +2.6%] |
| Recall@20 | 0.2461 | 0.2499 | **+1.6%** | [+0.5%, +2.6%] |
| Recall@50 | 0.3948 | 0.3988 | **+1.0%** | [+0.4%, +1.6%] |

Removing it never hurts, and it helps a little at @20 and @50. Without it, early
stopping runs to 152 trees instead of 74, and NDCG@10 on the users held out from
ranker training rises from 0.190 to 0.197. That fits the overfitting explanation:
splits on occupation stopped training early before the more general signals were
fully learned. **Decision: `user_occ` has been removed from the ranker, and all
headline numbers above use the version without it.** Stage-1 numbers
were the same to four decimal places in both runs, so the two-tower training is
deterministic with the fixed seed.

## Caveats
* **Leave-last-k per user is the standard MovieLens protocol, but it is not a strict global
  time split.** Training can include other users' interactions that happened after a
  given user's test items. A global time cutoff would give lower numbers that are more
  realistic for production.
* **Labels are "rated the movie", not "clicked when recommended".** No impression or
  position data exists, which is why the A/B test is still needed
  (see `AB_TEST_PLAN.md`).
* Single seed. Rerunning with several seeds would give the run-to-run variance in
  addition to the user-level bootstrap CIs.

Runtime: about 18 minutes on CPU. Raw outputs: `results/metrics.csv`, `results/results.json`, `results/run.log`.
