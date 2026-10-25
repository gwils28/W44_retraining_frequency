"""Block 8: a retraining is an intervention, and the dashboard that measures it is biased by construction.

Retraining fires when the recent error exceeds a threshold; the dashboard compares the error before and after.
Selection on a high error, part of which is noise, makes the error fall afterwards even if the refit does nothing
(regression to the mean). Estimators: naive pre/post, DiD against triggered-but-not-refit series, A/B among
triggered series, a placebo (identical refit), and a sharp regression discontinuity at the threshold, which needs
no holdout because the trigger rule itself is the identification strategy (but only identifies the effect AT the
threshold).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

L = 14          # length of the before / after windows (two weeks of daily monitoring)
SIGMA = 1.0     # error noise per step


def simulate(n=20_000, thr=1.05, delta_true=0.30, seed=844, bias_scale=0.50, placebo=False, randomize=False,
             p_treat=0.5) -> dict:
    """Series i carries a staleness bias b_i >= 0; observed error = |b_i + noise|, averaged over L steps.

    delta_true: share of the bias a refit really removes. placebo: the refit changes nothing. randomize: among
    TRIGGERED series, refit a random share p_treat (A/B); otherwise refit every triggered series.
    """
    rng = np.random.default_rng(seed)
    b = np.abs(rng.normal(0, bias_scale, n))
    pre = np.abs(b[:, None] + rng.normal(0, SIGMA, (n, L))).mean(axis=1)
    fired = pre > thr
    treat = fired & (rng.random(n) < p_treat) if randomize else fired.copy()
    eff = 0.0 if placebo else delta_true
    b_post = np.where(treat, b * (1 - eff), b)
    post = np.abs(b_post[:, None] + rng.normal(0, SIGMA, (n, L))).mean(axis=1)
    return dict(b=b, pre=pre, post=post, fired=fired, treat=treat)


def truth(delta_true, bias_scale=0.50, n=400_000, seed=11, thr=1.05, band: tuple[float, float] | None = None) -> float:
    """True causal effect on the triggered population (or on triggered series with pre in `band`), by
    randomisation on a very large sample."""
    d = simulate(n=n, thr=thr, delta_true=delta_true, seed=seed, bias_scale=bias_scale, randomize=True, p_treat=0.5)
    keep = d["fired"] if band is None else d["fired"] & (d["pre"] >= band[0]) & (d["pre"] < band[1])
    t, c = keep & d["treat"], keep & ~d["treat"]
    return float(d["post"][t].mean() - d["post"][c].mean())


def _groups(d):
    return d["treat"], d["fired"] & ~d["treat"]


def est_naive(d) -> float:
    t = d["treat"]
    return float(d["post"][t].mean() - d["pre"][t].mean())


def est_did(d) -> float:
    t, c = _groups(d)
    if c.sum() == 0:
        return float("nan")
    return float((d["post"][t] - d["pre"][t]).mean() - (d["post"][c] - d["pre"][c]).mean())


def est_ab(d) -> float:
    t, c = _groups(d)
    if c.sum() == 0:
        return float("nan")
    return float(d["post"][t].mean() - d["post"][c].mean())


def ci_ab(d, alpha=0.05) -> tuple[float, float]:
    t, c = _groups(d)
    a, bb = d["post"][t], d["post"][c]
    se = np.sqrt(a.var(ddof=1) / a.size + bb.var(ddof=1) / bb.size)
    z = norm.ppf(1 - alpha / 2)
    diff = a.mean() - bb.mean()
    return float(diff - z * se), float(diff + z * se)


def est_rdd(d, thr: float, bandwidth: float) -> tuple[float, float]:
    """Sharp RDD: local linear fits of post on pre on each side of the threshold (triangular kernel), jump at thr.
    Valid when every series above the threshold is refit (the dashboard design). Returns (estimate, s.e.)."""
    (right, var_r), (left, var_l) = rdd_lines(d, thr, bandwidth)
    return float(right[0] - left[0]), float(np.sqrt(var_r + var_l))


def rdd_lines(d, thr: float, bandwidth: float):
    """Local linear fits on each side of the threshold, in x = pre - thr: ((beta_right, var_intercept_right),
    (beta_left, var_intercept_left)), beta = (intercept, slope), HC0 sandwich variance of the intercept."""
    x, y = d["pre"] - thr, d["post"]
    out = []
    for side in (x >= 0, x < 0):
        m = side & (np.abs(x) < bandwidth)
        wts = 1 - np.abs(x[m]) / bandwidth
        X = np.column_stack([np.ones(m.sum()), x[m]])
        W = wts[:, None]
        beta = np.linalg.solve(X.T @ (W * X), X.T @ (wts * y[m]))
        resid = y[m] - X @ beta
        bread = np.linalg.inv(X.T @ (W * X))
        meat = X.T @ ((wts**2 * resid**2)[:, None] * X)
        out.append((beta, float((bread @ meat @ bread)[0, 0])))
    return out[0], out[1]


def effect_by_band(delta: float, edges, n: int = 400_000, seed: int = 11, thr: float = 1.05) -> list[float]:
    """True effect among triggered series, by band of pre-error (one randomised simulation, as in `truth`)."""
    d = simulate(n=n, thr=thr, delta_true=delta, seed=seed, randomize=True, p_treat=0.5)
    res = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        keep = d["fired"] & (d["pre"] >= lo) & (d["pre"] < hi)
        t, c = keep & d["treat"], keep & ~d["treat"]
        res.append(float(d["post"][t].mean() - d["post"][c].mean()))
    return res


def study(delta: float = 0.30) -> dict:
    out: dict = {"truth": truth(delta)}
    d = simulate(delta_true=delta)
    out["dashboard"] = dict(fired=float(d["fired"].mean()), naive=est_naive(d))
    dp = simulate(placebo=True)
    out["placebo_naive"] = est_naive(dp)
    out["threshold"] = []
    for thr in (0.80, 0.90, 1.00, 1.05, 1.15, 1.30, 1.50):
        dd = simulate(thr=thr, placebo=True)
        if dd["treat"].sum() >= 50:
            out["threshold"].append(dict(thr=thr, fired=float(dd["fired"].mean()), naive=est_naive(dd)))
    da = simulate(randomize=True, delta_true=delta)
    out["ab"] = dict(n_t=int(da["treat"].sum()), n_c=int((da["fired"] & ~da["treat"]).sum()), naive=est_naive(da),
                     did=est_did(da), ab=est_ab(da), ci=ci_ab(da))
    dap = simulate(randomize=True, placebo=True)
    out["ab_placebo"] = dict(naive=est_naive(dap), did=est_did(dap), ab=est_ab(dap), ci=ci_ab(dap))
    out["holdout"] = []
    for p in (0.5, 0.7, 0.9, 0.95, 0.99):
        dz = simulate(randomize=True, delta_true=delta, p_treat=p)
        t, c = _groups(dz)
        lo, hi = ci_ab(dz)
        loss_triggered = (dz["post"][c].mean() - dz["post"][t].mean()) * (1 - p)
        out["holdout"].append(dict(p=p, n_c=int(c.sum()), ab=est_ab(dz), half_width=(hi - lo) / 2,
                                   loss_per_triggered=float(loss_triggered),
                                   loss_per_park=float(loss_triggered * dz["fired"].mean())))
    # RDD on the dashboard design (everything above the threshold refit): no holdout needed
    out["rdd"] = []
    for bw in (0.05, 0.10, 0.20):
        for seed in (844, 845):
            dd = simulate(delta_true=delta, seed=seed, n=200_000)
            e, se = est_rdd(dd, 1.05, bw)
            pl, pse = est_rdd(simulate(placebo=True, seed=seed, n=200_000), 1.05, bw)
            out["rdd"].append(dict(bw=bw, seed=seed, est=e, se=se, placebo=pl, placebo_se=pse))
    out["truth_at_threshold"] = truth(delta, band=(1.05, 1.10))
    return out


def print_study(out: dict) -> None:
    tv = out["truth"]
    print(f"true causal effect on triggered series (randomised, 400,000 series) = {tv:+.4f}\n")
    db = out["dashboard"]
    print(f"1) dashboard: refit everything above the threshold ({db['fired']:.1%} of the park triggered)")
    print(f"   naive pre/post {db['naive']:+.4f} (bias {db['naive'] - tv:+.4f}, {abs(db['naive'] - tv) / abs(tv):.0%}"
          " of the true effect); DiD and A/B cannot be computed (no control)")
    print(f"\n2) placebo (identical refit): naive {out['placebo_naive']:+.4f} = "
          f"{abs(out['placebo_naive']) / abs(tv):.0%} of the true effect; share of the displayed gain that is "
          f"regression to the mean: {out['placebo_naive'] / db['naive']:.0%}")
    print("\n3) placebo naive effect by threshold")
    for r in out["threshold"]:
        print(f"   thr {r['thr']:.2f}: {r['fired']:>6.1%} triggered, naive {r['naive']:+.4f} "
              f"({abs(r['naive']) / abs(tv):.0%} of the true effect)")
    ab = out["ab"]
    print(f"\n4) A/B among triggered series: {ab['n_t']} refit / {ab['n_c']} controls")
    print(f"   naive {ab['naive']:+.4f}  DiD {ab['did']:+.4f}  A/B {ab['ab']:+.4f} "
          f"CI [{ab['ci'][0]:+.4f}, {ab['ci'][1]:+.4f}]")
    ap = out["ab_placebo"]
    print(f"5) A/B placebo: naive {ap['naive']:+.4f}  DiD {ap['did']:+.4f}  A/B {ap['ab']:+.4f} "
          f"CI [{ap['ci'][0]:+.4f}, {ap['ci'][1]:+.4f}]")
    print("\n6) holdout size: A/B estimate, CI half-width, extra error carried by the controls")
    for r in out["holdout"]:
        print(f"   p_treat {r['p']:.2f}: {r['n_c']:>5d} controls, A/B {r['ab']:+.4f}, half-width {r['half_width']:.4f},"
              f" loss {r['loss_per_triggered']:.4f} per triggered series = {r['loss_per_park']:.4f} per park series")
    print(f"\n7) sharp RDD at the threshold, dashboard design (no holdout), 200,000 series; true effect for "
          f"triggered series with pre in [1.05, 1.10): {out['truth_at_threshold']:+.4f}")
    for r in out["rdd"]:
        print(f"   bandwidth {r['bw']:.2f}, seed {r['seed']}: {r['est']:+.4f} (s.e. {r['se']:.4f}); "
              f"placebo {r['placebo']:+.4f} (s.e. {r['placebo_se']:.4f})")
