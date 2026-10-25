"""Block 5 on the calibrated DGP: the plan's tables, then calibration and a closed baseline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .freshness import TAUS, closed_lag, euro_table, k_max, seasonal_naive, serve_quantiles, stability
from .metrics import seasonal_scale
from .park import ParkSpec, make_park

SPEC = ParkSpec(n=200, seed=544)
KS = (1, 12, 24, 36, 48, 72, 120, 168, 336, 720)
W, M, H, D = 240, 24, 24, 48
WARM = W + M + D
PRICES = (0.05, 0.50, 2.00, 10.00)
SEEDS = (544, 545, 546, 547, 548)


def _bench(y: np.ndarray, ks=KS, methods=("gaussian", "conformal")) -> dict:
    T = y.shape[1]
    out = {"model": {}, "refits": {}}
    for method in methods:
        for k in ks:
            r = serve_quantiles(y, k, start=WARM, stop=T, method=method)
            out["model"][method, k] = r
            out["refits"][k] = r["n_refit"]
    out["snaive"] = seasonal_naive(y, start=WARM, stop=T)
    out["snaive_closed"] = seasonal_naive(y, start=WARM, stop=T, lag=closed_lag(M, H, D))
    return out


def _mean_loss(b: dict, method: str, ks=KS) -> dict:
    return {k: {t: float(b["model"][method, k]["loss"][t].mean()) for t in TAUS} for k in ks}


def study(spec: ParkSpec = SPEC, seeds=SEEDS) -> dict:
    park = make_park(spec)
    y = park.y
    T = y.shape[1]
    scale = seasonal_scale(y, M)
    b = _bench(y)
    base = {name: {t: float(b[name]["loss"][t].mean()) for t in TAUS} for name in ("snaive", "snaive_closed")}
    out: dict = {"base": base, "base_cov": {t: float(b["snaive"]["cov"][t].mean()) for t in TAUS}}
    out["loss"] = {meth: _mean_loss(b, meth) for meth in ("gaussian", "conformal")}
    out["cov"] = {meth: {k: {t: float(b["model"][meth, k]["cov"][t].mean()) for t in TAUS} for k in KS}
                  for meth in ("gaussian", "conformal")}
    out["refits"] = b["refits"]
    out["k_max"] = {(meth, bn): {t: k_max(out["loss"][meth], base[bn], t, KS) for t in TAUS}
                    for meth in ("gaussian", "conformal") for bn in ("snaive", "snaive_closed")}
    steps = T - WARM - H
    out["euros"] = {}
    for price in PRICES:
        out["euros"][price] = {
            t: euro_table({k: out["loss"]["gaussian"][k][t] for k in KS}, b["refits"], n=spec.n, steps=steps,
                          price=price) for t in TAUS}
    out["stability"] = {meth: {k: stability(b["model"][meth, k]["rev"][0.95], scale) for k in KS}
                        for meth in ("gaussian", "conformal")}
    out["illusion"] = {}
    for meth in ("gaussian", "conformal"):
        for k in (24, 168):
            real = b["model"][meth, k]["loss"][0.95].mean()
            leak = serve_quantiles(y, k, start=WARM, stop=T, fit_lag=0, method=meth)["loss"][0.95].mean()
            out["illusion"][meth, k] = (float(real), float(leak), float((real - leak) / leak))
    # dispersion of k_max over seeds
    out["seeds"] = {}
    for s in seeds:
        yy = make_park(replace(spec, seed=s)).y
        bb = _bench(yy)
        bs = {name: {t: float(bb[name]["loss"][t].mean()) for t in TAUS} for name in ("snaive", "snaive_closed")}
        out["seeds"][s] = {(meth, bn): {t: k_max(_mean_loss(bb, meth), bs[bn], t, KS) for t in TAUS}
                           for meth in ("gaussian", "conformal") for bn in ("snaive", "snaive_closed")}
    return out


def print_study(out: dict) -> None:
    print(f"park: {SPEC.n} series x {SPEC.T} steps (seed {SPEC.seed}) | H = {H} | W = {W} | d = {D}\n")
    for bn, lab in (("snaive", "seasonal naive (plan, reads unclosed actuals)"),
                    ("snaive_closed", f"seasonal naive, closed (lag {closed_lag(M, H, D)})")):
        print(f"{lab:>50s}: " + "  ".join(f"tau={t:.2f} {out['base'][bn][t]:.4f}" for t in TAUS))
    print("   coverage of the plan's naive: " + "  ".join(f"{t:.2f}: {out['base_cov'][t]:.3f}" for t in TAUS))

    for meth in ("gaussian", "conformal"):
        print(f"\n1) [{meth}] pinball | coverage | gain vs naive (plan) | gain vs closed naive")
        for k in KS:
            L, C = out["loss"][meth][k], out["cov"][meth][k]
            cells = []
            for t in TAUS:
                g1 = 1 - L[t] / out["base"]["snaive"][t]
                g2 = 1 - L[t] / out["base"]["snaive_closed"][t]
                cells.append(f"{L[t]:.4f} {C[t]:.3f} {g1:+6.1%} {g2:+6.1%}")
            print(f"   k={k:>4d} refits {out['refits'][k]:>4d} | " + " | ".join(cells))
        deg = {t: out["loss"][meth][720][t] / out["loss"][meth][1][t] - 1 for t in TAUS}
        print("   degradation k=1 -> k=720: " + "  ".join(f"tau={t:.2f} {deg[t]:+.0%}" for t in TAUS))

    print("\n2) k_max (largest cadence that still beats the baseline), seed 544")
    for (meth, bn), v in out["k_max"].items():
        ratio = v[0.5] / v[0.95] if v[0.95] else float("inf")
        print(f"   {meth:>9s} vs {bn:<13s} " + "  ".join(f"tau={t:.2f}: {v[t]:>3d}" for t in TAUS)
              + f"   ratio k_max(0.50)/k_max(0.95) = {ratio:.1f}")
    print(f"   over seeds {list(out['seeds'])}:")
    for key in out["k_max"]:
        rows = [out["seeds"][s][key] for s in out["seeds"]]
        print(f"   {key[0]:>9s} vs {key[1]:<13s} " + "  ".join(
            f"tau={t:.2f}: {[r[t] for r in rows]}" for t in TAUS))

    print("\n3) the plan's money model (gaussian quantiles), optimum k per price and tau")
    for price, tab in out["euros"].items():
        print(f"   price {price:>5.2f} EUR/refit/series: " + "  ".join(
            f"tau={t:.2f} k*={min(tab[t], key=tab[t].get)}" for t in TAUS))

    print("\n4) stability clause at tau = 0.95: jump p95 (% of scale) and % origins > 1%")
    for k in KS:
        g, c = out["stability"]["gaussian"][k], out["stability"]["conformal"][k]
        print(f"   k={k:>4d}  gaussian {100 * g['jump_p95']:>6.1f} {g['pct_visible']:>6.1%}   "
              f"conformal {100 * c['jump_p95']:>6.1f} {c['pct_visible']:>6.1%}")

    print("\n5) closing-lag illusion at tau = 0.95 (pinball real d=48 vs leak d=0)")
    for (meth, k), (r, l_, ill) in out["illusion"].items():
        print(f"   {meth:>9s} k={k:>3d}: {r:.4f} vs {l_:.4f} -> {ill:+.1%}")
