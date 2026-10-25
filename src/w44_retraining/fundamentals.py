"""Block 4: stability of a forecast service in closed form, and its split into noise and information.

Model. y_t = L_t + e_t, L_t a random walk with increments u ~ N(0, q), e ~ N(0, sigma^2). The service publishes
mu_hat = mean of the last w observations, re-estimated every k steps. Model age a has mean (k - 1) / 2.

  MSE(k, w) = sigma^2 (1 + 1/w) + q [ (a_bar + 1) + (w - 1)(2w - 1) / (6w) ]
      the last term is the CENTROID LAG of the window; w* = sqrt(3 (sigma^2 + q/6) / q) does not depend on k.

  Var(R) between two re-estimations k steps apart, a = min(k, w), b = max(k, w):
      2 a sigma^2 / w^2                                  estimation jitter (noise)
    + (q / w^2) [ a^2 (b - a + 1) + (a - 1) a (2a - 1) / 3 ]   real drift (information)

  churn per step = E|R| / k, with E|R| = sqrt(2/pi) sqrt(Var(R)) (R is Gaussian).
"""

from __future__ import annotations

import numpy as np

SQ2PI = np.sqrt(2.0 / np.pi)


def centroid_lag_var(w: int) -> float:
    return (w - 1) * (2 * w - 1) / (6.0 * w)


def mse_closed(k: int, w: int, sigma: float, q: float) -> float:
    a_bar = (k - 1) / 2.0
    return sigma**2 * (1 + 1 / w) + q * ((a_bar + 1) + centroid_lag_var(w))


def w_star(sigma: float, q: float) -> float:
    return float(np.sqrt(3.0 * (sigma**2 + q / 6.0) / q))


def var_revision_parts(k: int, w: int, sigma: float, q: float) -> tuple[float, float]:
    """(estimation jitter, real drift) parts of Var(R)."""
    a, b = min(k, w), max(k, w)
    noise = 2.0 * a * sigma**2 / w**2
    drift = q * (a**2 * (b - a + 1) + (a - 1) * a * (2 * a - 1) / 3.0) / w**2
    return noise, drift


def var_revision(k, w, sigma, q) -> float:
    return sum(var_revision_parts(k, w, sigma, q))


def mean_jump(k, w, sigma, q) -> float:
    return float(SQ2PI * np.sqrt(var_revision(k, w, sigma, q)))


def churn_per_step(k, w, sigma, q) -> float:
    return mean_jump(k, w, sigma, q) / k


def noise_share(k, w, sigma, q) -> float:
    """Share of the variance of published revisions that is pure estimation noise."""
    n, d = var_revision_parts(k, w, sigma, q)
    return n / (n + d)


def monte_carlo(k: int, w: int, sigma: float, q: float, T: int = 300_000, seed: int = 44) -> tuple[float, float, float]:
    """Simulated (MSE, Var(R), E|R|), same loop as the plan (vectorised window means)."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, np.sqrt(q), T)) + rng.normal(0, sigma, T)
    c = np.concatenate([[0.0], np.cumsum(y)])
    t_fit = np.arange(w, T - 1, k)                        # re-estimation times
    means = (c[t_fit + 1] - c[t_fit + 1 - w]) / w
    t = np.arange(w, T - 1)
    served = means[(t - w) // k]
    se = (y[t + 1] - served) ** 2
    revs = np.diff(means)
    return float(se.mean()), float(revs.var()), float(np.abs(revs).mean())


def trend_bias(k: int, w: int, b: float, T: int = 60_000, seed: int = 7) -> tuple[float, float]:
    """Under y_t = b t + noise, (theoretical, simulated) mean one-step error of the served window mean."""
    rng = np.random.default_rng(seed)
    y = b * np.arange(T) + rng.normal(0, 1.0, T)
    c = np.concatenate([[0.0], np.cumsum(y)])
    t_fit = np.arange(w, T - 1, k)
    means = (c[t_fit + 1] - c[t_fit + 1 - w]) / w
    t = np.arange(w, T - 1)
    err = y[t + 1] - means[(t - w) // k]
    return b * ((w - 1) / 2 + (k - 1) / 2 + 1), float(err.mean())


MC_CONFIGS = ((1, 20), (1, 100), (10, 100), (50, 100), (100, 100), (200, 100), (24, 240), (168, 240))


def study(sigma: float = 1.0, q: float = 0.004, w: int = 240, seeds=(44, 45, 46, 47, 48)) -> dict:
    out: dict = {"w_star": w_star(sigma, q)}
    out["mc"] = []
    for k, ww in MC_CONFIGS:
        sims = np.array([monte_carlo(k, ww, sigma, q, seed=s) for s in seeds])   # (seeds, 3)
        out["mc"].append(dict(
            k=k, w=ww, mse=mse_closed(k, ww, sigma, q), var_r=var_revision(k, ww, sigma, q),
            mse_mc=sims[0, 0], var_r_mc=sims[0, 1],
            gap_mse=mse_closed(k, ww, sigma, q) / sims[0, 0] - 1,
            gap_var=var_revision(k, ww, sigma, q) / sims[0, 1] - 1,
            # relative standard error of the single-seed Var(R) estimate, from the spread over seeds
            se_var=float(sims[:, 1].std(ddof=1) / sims[:, 1].mean()),
            gap_var_pooled=var_revision(k, ww, sigma, q) / sims[:, 1].mean() - 1,
        ))
    out["argmin_w"] = {k: min(range(3, 600), key=lambda ww, k=k: mse_closed(k, ww, sigma, q)) for k in (1, 24, 168)}
    base_j, base_c = mean_jump(1, w, sigma, q), churn_per_step(1, w, sigma, q)
    out["decomposition"] = [
        dict(k=k, jump=mean_jump(k, w, sigma, q), churn=churn_per_step(k, w, sigma, q),
             x_jump=mean_jump(k, w, sigma, q) / base_j, x_churn=churn_per_step(k, w, sigma, q) / base_c,
             noise=noise_share(k, w, sigma, q))
        for k in (1, 2, 4, 24, 48, 168, 240, 720)
    ]
    out["churn_floor"] = float(SQ2PI * np.sqrt(q / w))
    out["lambda_fixed_w"] = []
    for lam in (0.0, 1.0, 10.0, 100.0, 1000.0, 10_000.0):
        kb = min(range(1, 721),
                 key=lambda k, lam=lam: mse_closed(k, w, sigma, q) + lam * churn_per_step(k, w, sigma, q))
        out["lambda_fixed_w"].append(dict(lam=lam, k=kb, mse=mse_closed(kb, w, sigma, q),
                                          churn=churn_per_step(kb, w, sigma, q), noise=noise_share(kb, w, sigma, q)))
    out["lambda_free_w"] = []
    for lam in (0.0, 10.0, 100.0, 1000.0):
        kb, wb = min(((k, ww) for k in range(1, 400) for ww in range(10, 500, 5)),
                     key=lambda kw, lam=lam: mse_closed(*kw, sigma, q) + lam * churn_per_step(*kw, sigma, q))
        out["lambda_free_w"].append(dict(lam=lam, k=kb, w=wb, mse=mse_closed(kb, wb, sigma, q),
                                         noise=noise_share(kb, wb, sigma, q)))
    out["trend"] = {k: trend_bias(k, w, 0.01) for k in (1, 24, 168, 720)}
    return out


def print_study(out: dict, sigma: float = 1.0, q: float = 0.004, w: int = 240) -> None:
    print(f"sigma = {sigma}, q = {q}  ->  w* = {out['w_star']:.1f}\n")
    print("1) closed forms against Monte Carlo (300,000 steps; se = relative s.e. of Var(R) over 5 seeds)")
    print(f"{'k':>5s} {'w':>5s} {'MSE':>8s} {'MC':>8s} {'gap':>7s} {'Var(R)':>10s} {'MC':>10s} {'gap':>7s} "
          f"{'se':>6s} {'gap 5 seeds':>12s}")
    for r in out["mc"]:
        print(f"{r['k']:>5d} {r['w']:>5d} {r['mse']:>8.4f} {r['mse_mc']:>8.4f} {r['gap_mse']:>+7.1%} "
              f"{r['var_r']:>10.6f} {r['var_r_mc']:>10.6f} {r['gap_var']:>+7.1%} {r['se_var']:>6.1%} "
              f"{r['gap_var_pooled']:>+12.1%}")
    print("\n2) numerical argmin over w: " + ", ".join(f"k={k}: {v}" for k, v in out["argmin_w"].items()))
    print(f"\n3) decomposition of revisions (w = {w})")
    print(f"{'k':>6s} {'E|R|':>9s} {'churn/step':>11s} {'x jump':>8s} {'x churn':>8s} {'noise share':>12s}")
    for r in out["decomposition"]:
        print(f"{r['k']:>6d} {r['jump']:>9.5f} {r['churn']:>11.6f} {r['x_jump']:>7.1f}x {r['x_churn']:>7.3f}x "
              f"{r['noise']:>11.1%}")
    print(f"   drift-regime churn floor sqrt(2/pi) sqrt(q/w) = {out['churn_floor']:.6f} (valid for k <= w)")
    print(f"\n4) J = MSE + lam * churn/step, w = {w} fixed")
    for r in out["lambda_fixed_w"]:
        print(f"   lam {r['lam']:>7.0f}: k* = {r['k']:>3d}  MSE {r['mse']:.4f}  churn {r['churn']:.6f}  "
              f"noise {r['noise']:.1%}")
    print("   with w free:")
    for r in out["lambda_free_w"]:
        print(f"   lam {r['lam']:>7.0f}: k* = {r['k']:>3d}  w* = {r['w']:>3d}  MSE {r['mse']:.4f}  "
              f"noise {r['noise']:.1%}")
    print(f"\n5) trend counter-example (b = 0.01, w = {w}): mean error, theory vs simulation")
    floor = 0.01 * ((w - 1) / 2 + 1)
    ref = out["trend"][720][0]
    for k, (th, sim) in out["trend"].items():
        print(f"   k={k:>4d}: {th:.3f} vs {sim:.3f}  ({th / ref:.1%} of the k=720 bias, window share {floor / th:.1%})")
