"""Block 2 study: the plan's tables, then the checks that change their reading."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .harness import (
    Bench,
    BreakReset,
    Calendar,
    Cusum,
    CusumReset,
    KLDebt,
    KnownBreaks,
    Never,
    plan_policies,
    run,
    total_cost,
)
from .park import ParkSpec, make_park

LAMBDAS = ((0.0, 0.0), (0.05, 0.0), (0.05, 2.0), (0.05, 10.0), (0.20, 10.0), (0.05, 40.0))
SEEDS = (44, 45, 46, 47, 48)


def _remedies() -> list:
    return [
        Calendar(k=168, name="calendar k=168 (weekly)"),
        Calendar(k=1, name="calendar k=1 (continuous)"),
        Cusum(name="cusum on realised error"),
        KnownBreaks(delay=0, name="known breaks (plan)"),
        BreakReset(offsets=(24, 72, 240), name="known breaks + window reset"),
        BreakReset(offsets=(24, 72, 240), k=168, name="known breaks + window reset + weekly"),
        CusumReset(name="cusum + window reset"),
    ]


def study(spec: ParkSpec = ParkSpec(), bench: Bench = Bench(), seeds=SEEDS) -> dict:
    park = make_park(spec)
    out: dict = {"n_breaks": int(park.cps.sum()), "breaks_per_series": float(park.cps.sum(axis=1).mean())}

    out["plan"] = {p.name: run(park, p, bench) for p in plan_policies()}

    out["closing_lag"] = {}
    for p in (Calendar(k=24, name="calendar k=24 (daily)"), Calendar(k=168, name="calendar k=168 (weekly)"),
              Cusum(name="cusum on realised error"), KLDebt(thr=2.0, name="kl debt (thr 2)")):
        real = run(park, p, bench, fit_lag=bench.d)["mase"]
        leak = run(park, p, bench, fit_lag=0)["mase"]
        out["closing_lag"][p.name] = dict(real=real, leak=leak, illusion=(real - leak) / leak)

    out["composite"] = []
    for lam_c, lam_k in LAMBDAS:
        name, r = min(out["plan"].items(), key=lambda kv: total_cost(kv[1], lam_c, lam_k))
        out["composite"].append(dict(lam_c=lam_c, lam_k=lam_k, best=name, J=total_cost(r, lam_c, lam_k)))

    # -- checks -------------------------------------------------------------------------------------------------
    # (1) timing of the perfect detector: with d = 48 a refit at the break cannot see it
    out["oracle_delay"] = {
        delay: run(park, KnownBreaks(delay=delay), bench) for delay in (0, bench.d, 120, 168, bench.d + bench.w)
    }
    # (2) is the seasonal drift the mechanism? switch it off
    flat = make_park(replace(spec, seas_drift=0.0))
    out["no_seasonal_drift"] = {
        p.name: run(flat, p, bench)
        for p in (Never(), Calendar(k=168, name="calendar k=168 (weekly)"),
                  Calendar(k=24, name="calendar k=24 (daily)"), KnownBreaks(name="known breaks (plan)"),
                  BreakReset(name="known breaks + window reset"))
    }
    # (3) remedies that use the detection
    out["remedies"] = {p.name: run(park, p, bench) for p in _remedies()}
    # (4) the trigger reads actuals that are not closed yet: give it the same lag
    out["cusum_err_lag"] = {
        lag: run(park, Cusum(), bench, err_lag=lag) for lag in (0, bench.d)
    } | {f"reset, err_lag={bench.d}": run(park, CusumReset(), bench, err_lag=bench.d)}
    # (5) dispersion over seeds of the remedies
    disp: dict[str, list[float]] = {}
    for s in seeds:
        pk = make_park(replace(spec, seed=s))
        for p in _remedies():
            disp.setdefault(p.name, []).append(run(pk, p, bench)["mase"])
    out["seeds"] = {k: np.asarray(v) for k, v in disp.items()}
    return out


def print_study(out: dict, bench: Bench = Bench()) -> None:
    print(f"park: {ParkSpec().n} series x {ParkSpec().T} steps | H = {bench.h} | W = {bench.w} | d = {bench.d}")
    print(f"level breaks: {out['breaks_per_series']:.1f} per series ({out['n_breaks']} in total)\n")

    hdr = f"{'policy':>38s} {'MASE':>7s} {'churn':>7s} {'jump p95':>9s} {'% >1%':>7s} {'refits':>8s} {'useful':>7s}"

    def row(name, r):
        return (f"{name:>38s} {r['mase']:>7.4f} {r['churn']:>7.4f} {r['jump_p95']:>9.4f} "
                f"{r['pct_visible']:>6.1%} {r['n_refit']:>8.1f} {r['useful']:>6.1%}")

    print("A. the plan's bench (d = 48)")
    print(hdr)
    for k, r in out["plan"].items():
        print(row(k, r))

    print("\nB. closing-lag illusion: MASE with the real lag vs the leaky backtest (d = 0)")
    for k, r in out["closing_lag"].items():
        print(f"{k:>38s} real {r['real']:.4f}  leak {r['leak']:.4f}  illusion {r['illusion']:+.1%}")

    print("\nC. composite cost J = MASE + lam_c * refits/1000 + lam_k * churn")
    for r in out["composite"]:
        print(f"   lam_c {r['lam_c']:.2f}  lam_k {r['lam_k']:>4.1f}  ->  {r['best']:<30s} J = {r['J']:.4f}")

    print("\nD1. the perfect detector, refit `delay` steps after the break (window closed at t - d)")
    for delay, r in out["oracle_delay"].items():
        print(f"   delay {delay:>3d}: MASE {r['mase']:.4f}, refits {r['n_refit']:.1f}")

    print("\nD2. seasonal drift switched off (only level breaks)")
    print(hdr)
    for k, r in out["no_seasonal_drift"].items():
        print(row(k, r))

    print("\nD3. remedies that use the detection")
    print(hdr)
    for k, r in out["remedies"].items():
        print(row(k, r))

    print(f"\nD4. trigger on errors closed with the same lag (err_lag = {bench.d})")
    for k, r in out["cusum_err_lag"].items():
        print(row(f"cusum err_lag={k}" if isinstance(k, int) else f"cusum {k}", r))

    print(f"\nD5. MASE over {len(next(iter(out['seeds'].values())))} seeds (mean, min, max)")
    for k, v in out["seeds"].items():
        print(f"{k:>38s} {v.mean():.4f}  [{v.min():.4f}, {v.max():.4f}]")
    print("   paired difference per seed (negative = better), wins over the seeds:")
    for ref in ("calendar k=168 (weekly)", "calendar k=1 (continuous)"):
        for k, v in out["seeds"].items():
            if k == ref:
                continue
            diff = v - out["seeds"][ref]
            print(f"{k:>38s} vs {ref:<26s} {diff.mean():+.4f}  [{diff.min():+.4f}, {diff.max():+.4f}]"
                  f"  wins {(diff < 0).sum()}/{diff.size}")
