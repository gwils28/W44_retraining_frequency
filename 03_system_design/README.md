# Block 3: the forecast service as a system of contracts

```bash
uv run w44 gate             # checks configs/forecast_service.yaml; exit code 1 when blocked
```

Code: `src/w44_retraining/gate.py`, contract: [`configs/forecast_service.yaml`](../configs/forecast_service.yaml).
Tests: `tests/unit/test_gate.py` (each rule can block on its own).

## Decision

A park of forecasts served behind an API, with orchestrated retraining and a cache. Which service contract do we
write, given that the industry's standard inference protocol cannot express a forecast?

## The checked fact that drives the architecture

The **Open Inference Protocol V2** (KServe, Seldon, MLServer, Triton) has no field for time. Checked field by field
on the specification ([docs.seldon.ai/seldon-core-2/user-guide/v2](https://docs.seldon.ai/seldon-core-2/user-guide/v2),
2026-10-25):

- request: `id`, `parameters`, `inputs[]`, `outputs[]`; each input: `name`, `shape`, `datatype`, `parameters`, `data`;
- response: `model_name`, `model_version`, `id`, `parameters`, `outputs[]`;
- model metadata: `name`, `versions`, `platform`, `inputs`, `outputs`;
- the only extension point is `parameters`, of which the spec says "Currently no parameters are defined", so
  anything put there is server-specific.

Consequences: two identical requests sent on two dates are not the same request, so **a forecast cache cannot be
specified without an `as_of` in its key**; `model_version` exists but `training_asof` and `feature_version` do
not, so traceability stops at the model binary; and the served quantile level lives in a tensor whose meaning is
a server convention.

## Architecture

```
 +------------------ DEPLOYMENT GATE (deterministic, `w44 gate`) -----------------------+
 | G1 weights licence read at the source, valid for THIS client (revenue cap)         |
 | G2 the compute window holds the park with the provisioned workers                  |
 | G3 cadence <= freshness budget of every served quantile (vs the CLOSED naive)      |
 | G4 measured stability meets the ONE signed clause (amplitude OR frequency)         |
 | G5 retraining effect measurable: permanent holdout >= 2% OR RDD logging            |
 | G6 closing lag declared and applied to backtest, baseline AND trigger              |
 +--------------------------------------+-------------------------------------------------+
                                        v
 +------------------ DATA PLANE: bitemporality lives in serving ----------------------+
 |  event --(closed at t + d)--> warehouse --(window closed at t - d)--> TRAINING     |
 |  the backtest, the baseline and the retraining trigger read the SAME closed data   |
 +--------------------------------------+-------------------------------------------------+
                                        v
 +------------------ SERVING: the contract sits ABOVE the protocol -------------------+
 |  client --> forecast gateway --> OIP V2 runtime --> model                            |
 |              adds as_of, horizon, tau[], training_asof, model_version,             |
 |              feature_version; cache key = (series, as_of, horizon, tau, versions)  |
 |              refuses to serve if age(model) > budget(tau): falls back to the       |
 |              closed seasonal naive                                                 |
 |  cache TTL indexed on the horizon; blue/green by cohort of series + holdout        |
 +--------------------------------------+-------------------------------------------------+
                                        v
 +------------------ OBSERVABILITY --------------------------------------------------+
 | 1. served model age, p95 per series (the tail, not the mean)                       |
 | 2. jump p95 and share of origins with a visible revision (the signed clause)       |
 | 3. coverage of each served quantile (a 0.95 quantile covered 0.91 on real data)    |
 | 4. retraining effect against the holdout, or by RDD at the trigger threshold       |
 +-----------------------------------------------------------------------------------+
```

## Three explicit trade-offs

| # | Trade-off | Option A | Option B | Decision | Criterion |
|---|---|---|---|---|---|
| 1 | Where the time semantics live | in OIP `parameters`, by server convention | in a **forecast gateway** in front of the runtime | **B** | "Currently no parameters are defined": anything in `parameters` is non-portable, and without `as_of` in the cache key two requests sent on two dates are confused. |
| 2 | Retraining trigger and remedy | drift detector + refit on the usual window | **CUSUM on closed errors + calendar, full-window refit; window reset only on an externally confirmed break** | **B** | Block 2: a perfect detector with a fixed-window refit loses to a weekly cron (1.3685 vs 0.9829); with a break-aware window plus a weekly cron it beats continuous retraining on 5/5 seeds with 1% of its refits. A CUSUM-driven reset is worse than plain CUSUM (1.0445 vs 0.9179, churn x7), so the reset needs evidence a CUSUM does not give (e.g. a known event). |
| 3 | Measuring the effect of retraining | refit every trigger, read the dashboard | **5% permanent holdout, and log the trigger's running variable** | **B** | Block 8: the naive dashboard shows 197% of the true effect on a placebo. The holdout costs 0.0012 per park series and gives the average effect; the logged running variable gives an RDD estimate at the threshold for free. |

## The contract, checked

The committed contract describes a real candidate: Timestral-28M on 20,000 hourly series, daily cadence, client
under the licence cap, with the stability figures measured in block 5 on real data (gaussian quantiles, k = 24).
The gate output:

```
PASS  G1 weights licence     ALLOW under the cap (Modified MIT), re-check monthly; derivatives inherit the cap
PASS  G2 compute window      1 workers needed, 2 provisioned
PASS  G3 freshness budget    cadence 24 vs tightest budget 168 ({'0.50': 336, '0.80': 336, '0.95': 168})
FAIL  G4 stability clause    jump p95 0.540 vs clause 0.1
PASS  G5 measurable effect   holdout 5%, rdd logging True
PASS  G6 closing lag         d = 48, missing: none

BLOCKED (fail_closed)
```

It is blocked on purpose: a daily refit publishes 95th-percentile jumps of 54% of the scale at tau = 0.95, so the
amplitude clause "jump p95 <= 10%" cannot be signed at this cadence. Signing the frequency clause instead (4.2% of
origins with a visible revision, against a 15% cap) passes. Choosing which clause to sign is a business decision,
and the gate makes it explicit.

## What I correct relative to the plan

1. **`freshness.budget_by_tau`**: the plan's 168 / 120 / 24 (a factor 7) came from a baseline reading unclosed
   actuals and from under-covered tails (block 5). Measured against the closed naive on real data with gaussian
   quantiles: 720 / 720 / 168, or 336 / 336 / 168 with the refit calendar shifted by 240 h; the contract takes
   the lower. With calibrated (conformal) quantiles the budget does not depend on the quantile at all. The point
   that a scalar freshness SLA is a type error is much weakened; the stronger point is that the budget must be
   measured per park and per period (Q2 2014: no cadence beats the closed naive at tau = 0.95).
2. **G6 now covers the baseline and the trigger**, not only the backtest: the plan's own baseline and CUSUM read
   data that the closing lag forbids (blocks 2 and 5).
3. **Trade-off 2** was "CUSUM because 40.9% of its refits are near a break". With closed errors that share is 20.0%,
   and the stronger lever is the remedy, not the detector (block 2).
4. **G1 is a function of the client.** The Timestral weights licence has a 20M USD monthly revenue cap (block 1):
   the gate needs the client's revenue, and must re-check it.
5. **G5 accepts an RDD** as an alternative to the holdout (block 8), and the holdout costs 0.0012 per park series,
   not 0.0057.
6. **Quantile coverage is an observability signal.** The plan monitors "noise share" of revisions; on real data the
   more urgent signal is that a nominal 0.95 quantile covers 0.91.
