# W44: retraining is not hygiene, it is a decision

Weekly study (ISO week 44, 2026) on forecast serving as a system of contracts: accuracy, freshness and stability,
and the retraining cadence each of them asks for. The original plan, in French, is [`plan_w44.md`](plan_w44.md).
Blocks 1 to 8 are implemented here; block 9 (positioning) is out of scope.

Every script of the plan was first run verbatim and reproduces its published figures exactly; the package
reproduces them too (`tests/integration/test_plan_reproduction.py`). Each block then checks the plan's reading
of its own numbers, and several readings do not survive.

## What changes relative to the plan

| Plan claim | What the checks say | Block |
|---|---|---|
| "A perfect drift detector loses to a weekly cron, because slow seasonal drift is not a break: drift detection answers the wrong question" | It loses (1.3685 vs 0.9829), but not for that reason: it still loses with the seasonal drift switched off (1.2620 vs 0.9425). It refits on a window closed 48 h *before* the break, and a single refit of a 240-step mean absorbs almost none of it. With a remedy that uses the detection (window restarted at the break) plus a weekly cron, it **beats continuous retraining on 5/5 seeds with 1% of its refits** (0.8789 vs 0.8915). | [2](02_retrain_policy_harness/README.md) |
| CUSUM targets breaks 3.4x better than a calendar (40.9% vs 12.2%) | Half of that is a leak: the CUSUM reads actuals that are not closed. With closed errors: 20.0%, 1.6x. | [2](02_retrain_policy_harness/README.md) |
| The freshness budget depends on the served quantile by a factor 7 (k_max 168 at the median, 24 at tau = 0.95) | Two artefacts: under-covered gaussian tails (0.95 covers 83.5% at k = 720), and a seasonal-naive baseline that reads 48 h of actuals the model may not. Corrected: factor 2 on the DGP, stable over 5 seeds. On real data (Monash Electricity, 314 series, 2014), under a protocol committed before the results: the pre-registered rule gives NEITHER (against the plan's lag-24 naive the model never wins at the median, at any cadence); against the closed naive, a deviation reported as such, the ratio is 1.0 with calibrated quantiles: **contradicted**. | [5](05_mini_project/README.md) |
| Stability clause "revision p95 <= 10%" forces continuous retraining | True for gaussian quantiles only; with conformal quantiles every cadence meets it, and the frequency of visible revisions becomes the binding clause. | [5](05_mini_project/README.md) |
| Churn per step reaches a constant floor: cadence stops being a lever | Only for k much smaller than w; at k = w churn is 0.82 of the floor and keeps falling as sqrt(q/k) beyond. | [4](04_fundamentals/README.md) |
| Timestral weights: "Modified MIT v1.0", restriction "must not infringe third-party rights"; the same trap for Chronos-2 | Read on the Hub: **no rights at all above 20M USD of monthly revenue** (company or employer), derivatives included. Chronos-2 weights are Apache-2.0: no trap there. | [1](01_tech_watch/README.md) |
| Refitting 100% of triggers makes the effect incalculable; the 5% holdout costs 0.0057 | A threshold trigger is a sharp regression discontinuity: the effect at the threshold is identified with no holdout (RDD recovers -0.060 to -0.064 for a true -0.0625, placebo passes). The holdout costs 0.0012 per park series (0.0057 is per triggered series). | [8](08_causal_inference/README.md) |
| mlserver "does not install" on Python 3.13 | pip silently installs mlserver 1.3.5 (July 2023); uv installs 1.7.1 despite the bound. | [7](07_tech_stack/README.md) |
| 209M is justified above 0.16 EUR of decision loss per series per month | Only if a 0.2% benchmark-score gap equals a 0.2% decision-loss gap. And the real cost is provisioned capacity: 7,862 EUR/month for the window vs 163 EUR consumed (x48). | [6](06_tsfm_serving/README.md) |

What holds as stated: the closing-lag illusion of the backtest (+6.3% MASE; +12.8% at tau = 0.95 on real data);
rare retraining concentrates churn (jump x223.6 while churn per step only falls x3.2); the closed forms of block 4
(within Monte Carlo error); the regression-to-the-mean dashboard bias (placebo at 197% of the true effect); the OIP
V2 has no time field.

## Blocks

| Block | Folder | Command | Deliverable |
|---|---|---|---|
| 1 Watch | [`01_tech_watch`](01_tech_watch/README.md) | | 4 resources checked at the source, weights licences read on the Hub, paper bodies' numbers |
| 2 Engineering | [`02_retrain_policy_harness`](02_retrain_policy_harness/README.md) | `w44 harness` | 10-policy bench + 4 checks (refit timing, no seasonal drift, break-aware remedies, closed-error trigger, 5 seeds) |
| 3 System design | [`03_system_design`](03_system_design/README.md) | `w44 gate` | `configs/forecast_service.yaml` + executable deployment gate G1 to G6 |
| 4 Fundamentals | [`04_fundamentals`](04_fundamentals/README.md) | `w44 churn` | closed forms of accuracy and stability, Monte Carlo error bars |
| 5 Mini-project | [`05_mini_project`](05_mini_project/README.md) | `w44 freshness`, `w44 real download\|run` | freshness budget per quantile, DGP and real data, protocol first |
| 6 TSFM serving | [`06_tsfm_serving`](06_tsfm_serving/README.md) | `w44 serving` | cost per size, compute window, licence gate with the revenue cap |
| 7 Tech stack | [`07_tech_stack`](07_tech_stack/README.md) | | mlserver / bentoml / kserve sheet, install behaviour, gateway cache key |
| 8 Causal inference | [`08_causal_inference`](08_causal_inference/README.md) | `w44 causal` | placebo, A/B, holdout sizing, RDD at the trigger threshold |

## Notebooks

Study material, written to understand the topic rather than to show results: each one builds the intuition, runs
the experiment from the package, and reads the result, with fixed callouts (**Intuition**, **Example**, **Watch
out**, **Takeaway**). They are committed executed, and export the figures of each block's `figures/` folder.

| Block | Notebook | What it explains |
|---|---|---|
| 2 | [`01_the_bench`](02_retrain_policy_harness/notebooks/01_the_bench.ipynb) | the two drifts, the closing lag on a timeline, every metric, the amplitude vs frequency trade-off |
| 2 | [`02_the_perfect_detector`](02_retrain_policy_harness/notebooks/02_the_perfect_detector.ipynb) | why a perfect detector loses, and the remedy that makes it win |
| 4 | [`01_closed_forms`](04_fundamentals/notebooks/01_closed_forms.ipynb) | accuracy and stability of a window mean, derived step by step and checked by Monte Carlo |
| 5 | [`01_quantiles_and_freshness`](05_mini_project/notebooks/01_quantiles_and_freshness.ipynb) | pinball loss, coverage, why the factor 7 is two artefacts |
| 5 | [`02_real_data`](05_mini_project/notebooks/02_real_data.ipynb) | Monash electricity under a pre-registered protocol |
| 6 | [`01_serving_cost`](06_tsfm_serving/notebooks/01_serving_cost.ipynb) | consumed vs provisioned CPU, break-even, the licence gate |
| 8 | [`01_regression_to_the_mean`](08_causal_inference/notebooks/01_regression_to_the_mean.ipynb) | the fake improvement, placebo, holdout, RDD at the threshold |

## Reproduce

```bash
uv sync                      # Python 3.13, versions pinned in uv.lock
uv run pytest                # 55 tests, a few seconds (integration tests marked `slow`)
uv run ruff check src tests
make harness gate churn freshness real serving causal   # every table of the READMEs
make notebooks               # re-execute the notebooks in place (run `make real` first: block 5 reads its cache)
```

Seeds are fixed in the code (44 for block 2, 544 for block 5, 844 for block 8, plus the seed ranges listed in each
README). The real dataset is checked against the sha256 written in `configs/freshness_real.yaml` before use.
