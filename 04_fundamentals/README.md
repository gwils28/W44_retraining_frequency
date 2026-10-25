# Block 4: the stability of a forecast service, in closed form

```bash
uv run w44 churn            # under 1 s
```

Code: `src/w44_retraining/fundamentals.py`. Tests: `tests/unit/test_fundamentals.py` (closed forms against Monte
Carlo within three standard errors, vectorised simulation equal to the plan's loop).

## The model

`y_t = L_t + e_t`, with a random-walk level (increments `u ~ N(0, q)`) and noise `e ~ N(0, sigma^2)`. The service
publishes the mean of the last `w` observations, re-estimated every `k` steps; the deployed model has mean age
`(k - 1) / 2`. Here `sigma = 1`, `q = 0.004`.

**Accuracy.**

```
MSE(k, w) = sigma^2 (1 + 1/w) + q [ (a_bar + 1) + (w - 1)(2w - 1) / (6w) ],    a_bar = (k - 1) / 2
```

The last term is the **centroid lag**: a window mean estimates the level `(w - 1) / 2` steps in the past, not now.
Setting the derivative in `w` to zero gives `w* = sqrt(3 (sigma^2 + q/6) / q)` = 27.4, and the numerical argmin is
27 for k = 1, 24 and 168. The two knobs act on two additive terms: **window length fixes the window bias, cadence
fixes model age**, and tightening the cron does nothing for a too-long window.

**Stability.** The jump between two re-estimations `k` steps apart (`a = min(k, w)`, `b = max(k, w)`):

```
Var(R) = 2 a sigma^2 / w^2                                     estimation jitter (noise)
       + (q / w^2) [ a^2 (b - a + 1) + (a - 1) a (2a - 1) / 3 ]  real level drift (information)
```

The second term counts how many times each level increment appears in the difference of the two window means.
`noise share = jitter / Var(R)` is the share of the variance of published revisions that carries no information.

## Results (all reproduce the plan)

Closed forms against a 300,000-step Monte Carlo, eight (k, w) configurations: MSE within 0.6%, Var(R) within
1.8% on the plan's seed. Over five seeds, the relative standard error of a single-seed Var(R) is 0.4% to 2.9% and
every gap is below it. The 1.8% is Monte Carlo noise, not a defect of the formula.

| k | E\|R\| (jump) | churn / step | x jump vs k=1 | x churn vs k=1 | noise share |
|---|---|---|---|---|---|
| 1 | 0.00572 | 0.005720 | 1.0 | 1.000 | 67.6% |
| 4 | 0.01604 | 0.004010 | 2.8 | 0.701 | 34.4% |
| 24 | 0.08024 | 0.003343 | 14.0 | 0.585 | 8.2% |
| 168 | 0.48302 | 0.002875 | 84.4 | 0.503 | 1.6% |
| 240 | 0.64245 | 0.002677 | 112.3 | 0.468 | 1.3% |
| 720 | 1.27869 | 0.001776 | 223.6 | 0.310 | 0.3% |

From continuous to monthly, the published jump grows 223.6 times while churn per step only falls 3.2 times. At
continuous cadence on a 240-step window, two thirds of the revision variance is noise; at weekly cadence, 1.6%.

Trend counter-example (`y_t = 0.01 t + noise`, w = 240): the mean error is `b [(w - 1)/2 + (k - 1)/2 + 1]`,
confirmed by simulation (1.205 against 1.205 at k = 1, 4.800 against 4.797 at k = 720). Going from monthly to
continuous removes 75% of the bias; the remaining 1.205 is entirely window bias, which no cadence touches. The fix
is a trend term or a weighting, not a tighter cron.

## What I correct relative to the plan

1. **The "constant churn floor, no lever left" holds only for k much smaller than w.** The plan states that in the
   drift regime churn per step tends to `sqrt(2/pi) sqrt(q/w)` = 0.003257, a constant. Exactly, for `k <= w`,
   `Var(R) ~ q k^2 (w - k/3) / w^2`: the floor is reached only when `k << w`, and at `k = w` churn is
   `sqrt(2/3)` = 0.82 of it (drift part exactly 0.002660; 0.002677 with the noise part). For `k > w`,
   `Var(R) ~ q (k - w/3)` and churn keeps falling, towards `sqrt(2/pi) sqrt(q/k)` when `k >> w` (0.001776 at
   k = 720, the plan's own printout, below its "floor"). Cadence stays a lever for churn; it is a weak one (x0.47 from k = 1 to
   k = 240).
2. **The 67.6% noise share belongs to w = 240, which is nine times w\*.** On the same process with the window at
   its optimum, the noise share at k = 1 is 94.3% (the `w` free table in the CLI output). The shorter the window,
   the more of what is republished is noise. That strengthens the plan's point, but the figure to quote depends on
   the window, and the plan quotes one without saying so.
3. **"Window and cadence are independent" is a property of this model.** It follows from the additive MSE of a
   window mean on a random walk. It is the right first-order intuition, not a general law: with a breaking level
   (block 2) the window choice and the refit timing interact (the break-aware window of block 2 is exactly that
   interaction).
4. **Scope of the SMQC link.** The plan says this block "closes analytically" what SMQC measures. SMQC
   (arXiv 2506.05776) is a scaled multi-quantile change on real global models; this block is the mean of a local
   level model. Same concept (the revision of a published forecast), not the same quantity, and the closed form here is not a
   validation of the papers' tables.
