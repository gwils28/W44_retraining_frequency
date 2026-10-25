# Block 8: a retraining is an intervention, and its dashboard is biased by construction

```bash
uv run w44 causal           # about 2 s
```

Code: `src/w44_retraining/causal.py`. Tests: `tests/unit/test_causal.py`, `tests/integration/test_plan_reproduction.py`.

## The mechanism

Retraining fires when the recent error (mean over 14 steps) exceeds a threshold, and the dashboard compares the
error before and after. Selection is on a high error, part of which is noise, so the error falls afterwards even
when the refit does nothing: regression to the mean. Here the instrument is perfect; **the selection rule makes
the effect**.

Simulation: each series carries a staleness bias `b_i = |N(0, 0.5)|`; observed error `|b_i + N(0, 1)|`; a refit
removes 30% of the bias. True effect on the triggered series, by randomisation on 400,000 series: **-0.1053**.

## Results (the plan's figures reproduce)

| Design | Estimator | Estimate | Bias |
|---|---|---|---|
| refit everything above the threshold (20.4% of the park) | naive pre/post | -0.3119 | -0.2066 (196% of the effect) |
| same, placebo (identical refit, true effect 0) | naive pre/post | -0.2072 | 197% of the effect from zero change |
| A/B among triggered series (2,080 refit / 2,007 controls) | naive | -0.3033 | -0.1980 |
| | DiD | -0.0992 | +0.0061 |
| | A/B | -0.0993, CI [-0.1141, -0.0844] | +0.0060 |
| A/B, placebo | A/B | +0.0066, CI [-0.0102, +0.0234] | passes |

66% of the gain the dashboard displays (-0.2072 out of -0.3119) is regression to the mean. The stricter the
threshold, the larger the fake effect: 85% of the true effect when 63.7% of the park triggers, 284% at 1.3%.

**Holdout size** (share of triggered series left un-refit):

| p_treat | controls | A/B | CI half-width | extra error, per triggered series | per park series |
|---|---|---|---|---|---|
| 0.50 | 2,007 | -0.0993 | 0.0148 | 0.0496 | 0.0101 |
| 0.90 | 389 | -0.1045 | 0.0275 | 0.0104 | 0.0021 |
| 0.95 | 187 | -0.1144 | 0.0352 | 0.0057 | 0.0012 |
| 0.99 | 46 | -0.1079 | 0.0728 | 0.0011 | 0.0002 |

## What I add: the trigger rule is itself an identification strategy

The plan concludes that refitting 100% of the triggered series makes the effect "incalculable by construction",
and that a permanent holdout is the price of a meaningful number. That holds for the **average** effect on the
triggered series. But a deterministic threshold rule is a **sharp regression discontinuity**: series just below and
just above the threshold are comparable, and only the refit differs. A local linear fit of the post-error on the
pre-error on each side (triangular kernel) estimates the jump at the threshold **with no holdout at all**.

On the dashboard design (everything above 1.05 refit), 200,000 series, against a true effect of **-0.0625** for
triggered series with a pre-error in [1.05, 1.10):

| bandwidth | seed 844 | seed 845 | placebo (seed 844) | placebo (seed 845) |
|---|---|---|---|---|
| 0.05 | -0.0589 (s.e. 0.0059) | -0.0621 (0.0057) | +0.0006 (0.0064) | -0.0029 (0.0062) |
| 0.10 | -0.0638 (0.0042) | -0.0615 (0.0040) | -0.0044 (0.0045) | -0.0031 (0.0044) |
| 0.20 | -0.0631 (0.0029) | -0.0600 (0.0029) | -0.0053 (0.0032) | -0.0028 (0.0032) |

The RDD recovers the effect at the threshold and passes the placebo, at zero service cost. Its limit is what it
identifies: the effect **at the threshold** (-0.0625), not the average over the triggered series (-0.1053), because
the effect grows with the bias and series far above the threshold carry more of it. The two designs answer
different questions:

- "should we lower the threshold a little?" is the RDD's question, answered for free from existing logs;
- "what does our retraining policy buy on average?" needs the randomised holdout.

## What I correct relative to the plan

1. **"Cost of a 5% holdout: 0.0057 of extra error on the park"**: 0.0057 is per *triggered* series. Per park series
   it is 0.0012 (20.4% of the park triggers). The holdout is five times cheaper than the plan says.
2. **"Refitting 100% of the triggers makes the effect incalculable"** is true for the average effect, not for the
   effect at the threshold (section above). The contract can require *either* a holdout *or* a logged running
   variable (the pre-error that triggered, or not, each series) for an RDD.
3. **The single-seed A/B at p = 0.95** (-0.1144) sits 0.009 from the truth, inside its CI (half-width 0.0352); its
   distance from the other rows is noise, not a trend with p.
