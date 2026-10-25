# Block 1: technology and research watch

Four resources from the plan, each checked at its primary source on 2026-10-25 from an unrestricted network
(the plan was written from a sandbox where `huggingface.co` and Zenodo downloads were blocked). Every URL in the
plan resolves and no source was invented. Quotes are verbatim. Anything beyond an abstract is marked with the
section or table it comes from.

## 1. Timestral (Mistral, 15 October 2026): a CPU-servable TSFM family

- Announcement: <https://mistral.ai/news/timestral/> · model card: <https://docs.mistral.ai/models/timestral-1-0-209m>
  · code: <https://github.com/mistralai/timestral> · weights: `mistralai/Timestral-{5M,28M,123M,209M}` on the Hub.

Verified facts:

| Item | Value | Source |
|---|---|---|
| Sizes | 5M / 28M / 123M / 209M, same architecture and recipe | announcement |
| fp32 weights | 21 / 111 / 490 / 837 MB (safetensors: 20.8 / 111.3 / 490.4 / 837.3 MB) | README, Hub API |
| Context | 32,768 steps (pre-trained at 8,192, post-trained with "a 4x longer context") | announcement |
| Output | "Probabilistic forecasts, 21 quantiles" | announcement |
| Latency, 8-thread server CPU, 1 day ahead from 3 weeks of 15 min history | 10 ms (5M), 140 ms (209M); 28M "five times faster" than 209M; 123M "within 0.2% of the 209M score at three quarters of its latency" | announcement |
| Rankings | first on GIFT-Eval, TIME, fev-bench among zero-shot models; 28M beats Chronos-2 (119.5M parameters) on GIFT-Eval and TIME | announcement, Hub metadata |
| Status | "Public Preview"; on-prem, VPC or air-gapped | model card |

The 28 ms and 105 ms latencies used in block 6 are derived (140 / 5 and 140 x 3/4), not published.

**The weights licence, read at the source.** The code is Apache-2.0. The weights are not: the four Hub repos ship
an identical `LICENSE` file titled "Modified MIT License" (`license: other` in the card metadata). It is MIT-like
with one decisive clause:

> 2. You are not authorized to exercise any rights under this license if the global consolidated monthly revenue
> of your company (or that of your employer) exceeds $20 million (or its equivalent in another currency) for the
> preceding month. This restriction in (b) applies to the Model and any derivatives, modifications, or combined
> works based on it [...]

So the weights are free for commercial use only below about $240M of annual revenue, and the test applies to the
company *or the user's employer*. For a consultant, that means the client's revenue. Above the cap, the only paths
are a discretionary commercial licence or Mistral's hosted services. The cap also covers fine-tunes.

## 2. arXiv 2604.06438: cost-sensitive retraining via posterior learning debt

Harrison Katz, v1 7 Apr 2026, v2 3 May 2026, stat.AP / cs.LG. <https://arxiv.org/abs/2604.06438>

Retraining as a costly decision: the monitored quantity is the KL divergence from a continuously updated shadow
posterior to the frozen deployed posterior. Retrain when the expected regret of waiting exceeds the retraining
cost; the usual two-state excess-loss rule is a special case. Abstract figures: "Under the primary
75th-percentile score-unit scaling, an age-adjusted debt-threshold policy improves on tuned calendar retraining in
all 72 non-stable scenario cells and on tuned CUSUM in 58 of 72 cells, with mean relative objectives 0.677 and
0.975". The abstract adds that "the CUSUM comparison remains policy-dependent" and calls the work an
"exact-state proof-of-concept in a synthetic conjugate simulation", "not a universal replacement for drift
detection".

Why it matters here: it is the formalism of the `kl_debt` policy of block 2, and 0.975 against CUSUM already says
the debt barely beats a well-tuned CUSUM. What it adds is a threshold expressed in units of cost.

## 3. arXiv 2506.05776 and 2505.00356: retraining frequency and forecast stability

Marco Zanotti. 2506.05776 (v3 10 Feb 2026, stat.AP / stat.OT) introduces the **Scaled Multi-Quantile Change
(SMQC)**, "model-agnostic, distribution-free, and scale-free", to measure the stability of probabilistic forecasts
on M4, M5 and VN1 with ten global models. 2505.00356 (v3 14 Jan 2026, DOI 10.1016/j.mlwa.2025.100769) studies the
same question on M5 and VN1, from continuous retraining to none.

The abstracts contain no result numbers; the plan cited them for their framework only. The numbers below come from
the paper bodies (v3 HTML):

- 2506.05776, section 4: for point forecasts, "even under extreme settings (such as the no-retraining scenario) any
  degradation in accuracy remains below 5%"; probabilistic accuracy "is more sensitive to longer retraining
  intervals", with differences "typically negligible (below 2%)" for low periodic retraining. Recommendation: about
  21 to 30 days for daily data, up to 8 weeks for weekly data.
- 2506.05776, SMQC (lower is more stable), baseline cadence then no retraining: VN1 (Table 15) NBEATSx 0.765 to
  0.369, CatBoost 2.265 to 1.243, but MLP 0.219 to 0.283 and LR 1.559 to 1.894 get *less* stable; M5 (Table 11) is
  essentially flat (LGBM 0.118 to 0.118).
- 2505.00356, section 4: RMSSE degrades by less than 5% even without retraining; compute time roughly halves at
  the first periodic step and drops about 75% with monthly retraining. Optimal frequency (Figure 6): 3 to 4 weeks
  (M5) and 8 to 10 weeks (VN1) for point forecasts, about 2 weeks (M5) and 4 weeks (VN1) for probabilistic ones.

Why it matters here: both papers find that **probabilistic objectives want more frequent retraining than point
objectives**, which is the direction of block 5. They also find the effect is moderate (2 weeks against 3 to 4),
which is closer to the corrected block 5 figure (factor about 2) than to the plan's factor 7.

## 4. arXiv 2609.39789: PILOT, pseudo-label-triggered retraining

Kwak, Jung, Park, 30 Sep 2026, cs.LG / cs.AI. <https://arxiv.org/abs/2609.39789>

PILOT "constructs a pseudo-label from future increases in forecast error", trains a light scorer on the observed
error state, and at deployment "uses only completed forecast errors", as a plug-in for any backbone. Evaluated on
eight multivariate benchmarks with DLinear, iTransformer and TimesNet, with the best average rank among retraining
policies. The abstract gives no other number. Kept for one idea: the trigger's *training label* is a future
quantity, while the deployed trigger only reads closed errors. That is the only correct way to build a learned
trigger, and it is the rule behind `forbid_trigger_on_future_error` in `configs/forecast_service.yaml`.

## Discarded: arXiv 2601.08928 (DriftGuard)

Alam, Rahman, Sadeqa, 13 Jan 2026. "97.8% detection recall within 4.2 days" and "up to 417 return on investment",
with no unit for the latter. A quantified claim without a unit is not a figure. Kept as a lead, not as a source.
Block 2 also shows why detection recall is the wrong KPI.

## Synthesis

1. **What moves.** TSFM serving moves from GPU to CPU (140 ms for 209M on 8 threads), which moves the binding
   constraint from compute cost to the compute window (block 6) and to the weights licence. The 2026 literature
   treats retraining cadence as an economic decision (2604.06438) and stability as a separate dimension
   (2506.05776).
2. **What changes for me.** A forecast service is a contract: freshness budget, stability clause, licence gate,
   permanent holdout. Blocks 3 and 5 write it down.
3. **What I refuse to believe without measuring it:** that a drift detector tells you when to retrain. Block 2
   tests a perfect detector.

## What I correct relative to the plan

| Plan claim | Finding at the source |
|---|---|
| Timestral weights restriction: "must not use them in a manner that infringes [...] any third party's rights" | That sentence is in the GitHub README only. The `LICENSE` file on the Hub carries the real restriction: **no rights at all above $20M of monthly revenue** (company or employer), derivatives included. |
| Model card licence "Modified MIT v1.0" | The licence is "Modified MIT"; "v1.0" is the model version shown next to it. |
| "The same structure holds for Chronos-2 [...] three families, three times the same trap" | **Wrong.** `amazon/chronos-2` and `chronos-bolt-{small,base}` declare `license: apache-2.0` on the Hub, same as the code. The W42 debt on Chronos-2 is closed: permissive. The general rule stays (read the weights licence on the Hub, not on the repo), but the trap is Timestral's, not a pattern. |
| DriftGuard "classified cs.CV" | Primary category is cs.LG; cs.CV is a cross-list. The other reasons to discard it stand. |
| 2604.06438 figures | Correct, but they hold "under the primary 75th-percentile score-unit scaling" and the CUSUM comparison "remains policy-dependent". |
| Zenodo downloads blocked (403) | Sandbox artefact. `zenodo.org/api/records/4656140/files/electricity_hourly_dataset.zip/content` returns 200 (11.8 MB). Block 5 now runs on real data too. |
| bentoml "10 releases in 11 months" (block 7) | True, but 1.4.39 (7 May 2026) to 1.4.40 (23 Oct 2026) is a 5.5-month gap. |
| Zanotti numbers "not verified" | Now extracted from the bodies (section 3 above). |
