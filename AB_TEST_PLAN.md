# A/B Test Plan: Two-Stage Recommender (Two-Tower + GBM Rerank)

## 1. Hypothesis
Reranking two-tower candidates with a LightGBM LambdaRank model puts more relevant
titles at the top of the "Recommended for you" row, and that increases engagement
with it. Offline the reranker improves NDCG@10 over two-tower ordering alone
(see `RESULTS.md`). This test checks whether that gain carries over to real users.

* **H0:** the treatment does not change plays from recommendations per user.
* **H1:** the treatment changes it (two-sided test; we ship only if the effect is positive).

## 2. Variants
| Arm | Retrieval | Ranking | Traffic |
|---|---|---|---|
| **A: control** | current production system (or the two-tower model alone if there's no incumbent) | two-tower score | 50% |
| **B: treatment** | same two-tower model (same version and candidate count) | LightGBM reranker | 50% |

Both arms share retrieval, so the test isolates the reranker. If we also need to know
whether the two-tower model beats the incumbent retriever, run that as its own
experiment or as a third arm, and don't bundle the two changes together.

## 3. Unit, assignment, exposure
* **Randomization unit:** logged-in user (`hash(user_id, salt) mod 100`). This keeps the
  experience the same for each user across sessions and avoids carryover between arms.
* **Analysis population:** users who are *exposed*, meaning they actually loaded a surface
  with the recommendation row. We log exposure in both arms at the same point, before
  ranking, so it can't be affected by the treatment. This keeps the power of a
  triggered analysis without biasing the comparison.

## 4. Metrics
**Primary (decision) metric:** *plays started from the recommendation row per exposed user*
over the test window. It's the closest online counterpart to NDCG/Recall@K.

**Secondary:**
* CTR on the row (clicks / impressions); position-weighted CTR on the top 10 slots
* Completion rate of plays started from recs (checks for clickbait)
* Catalog coverage and the share of plays on long-tail items (checks for popularity collapse)
* Total watch time per user across the whole product (checks for cannibalization)

**Guardrails (must not get worse, non-inferiority margin of -1% relative):**
* p95 recommendation latency (the reranker adds feature fetch + GBM scoring of 200 candidates)
* Overall sessions per user and 7-day retention
* Error or empty-row rate

## 5. Sample size and duration
For a ratio metric like plays per user, estimate the variance from 2 to 4 weeks of
historical data. Use the delta method if the metric is computed per impression.

Example: baseline is 0.80 rec-plays per user per week, SD is 1.6, and the minimum
detectable effect is +2% relative (0.016). With α = 0.05 (two-sided) and 80% power:

n per arm ≈ 2 · (1.96 + 0.84)² · σ² / δ² = 2 · 7.84 · 2.56 / 0.016² ≈ **157k exposed users per arm**

* **CUPED:** use each user's pre-period rec-plays as the covariate. That typically
  removes 30–50% of the variance and cuts the required n by the same share.
* **Duration:** run at least **2 full weeks**, even if we reach n earlier, to cover
  weekly seasonality and let novelty effects fade. Fix the end date before launch,
  with no peeking. If early stopping is needed, use a group-sequential design
  (O'Brien–Fleming spending) that we decide on before launch.

## 6. Validity checks (before reading results)
* **Sample-ratio mismatch:** χ² test on exposed users per arm. Investigate if p < 0.001.
* **A/A test or pre-period check:** confirm the arms didn't differ on the primary metric
  before the experiment started.
* **Novelty and primacy:** plot the treatment effect by day since first exposure. If it
  decays, report the steady-state effect.
* **Logging parity:** both arms emit the same impression and click events and the same exposure trigger.

## 7. Analysis
* Welch t-test (or CUPED-adjusted regression) on per-user means, randomized at the
  user level, so ordinary standard errors are valid. For per-impression ratio
  metrics like CTR, use the delta method or a user-clustered bootstrap.
* Correct secondary metrics for multiple comparisons (Benjamini–Hochberg). Only the
  primary metric drives the decision.
* Pre-registered segments: new vs. tenured users (cold start is where the history
  features are weakest), heavy vs. light users, and device. These are for
  understanding the result, not for fishing.

## 8. Decision rule
* **Ship** if the primary metric goes up with a 95% CI excluding 0 **and** every
  guardrail passes non-inferiority.
* **Iterate** if the primary metric is flat but secondaries move (e.g. CTR up, completion
  down). That suggests the ranker is optimizing for the wrong label. Retrain on
  completions or watch time instead of "rated it".
* **Don't ship** if any guardrail fails. That includes latency, even if engagement
  improves. In that case, cut the candidate count or simplify the features first.

## 9. Risks and follow-ups
* **Offline/online gap:** offline labels are "rated it later", not "clicked when shown".
  After launch, log impressions with their positions and retrain the ranker on real
  impression→click data, debiasing for position bias (e.g. inverse propensity weighting).
* **Feedback loops:** a ranker trained on its own exposures narrows over time. Keep a
  small (1–2%) exploration bucket with randomized ordering to collect unbiased training data.
* **Interleaving** (team-draft) is a fast, high-sensitivity pre-test for ranker variants.
  Use it to pick the best candidate ranker before spending a full A/B slot on it.
