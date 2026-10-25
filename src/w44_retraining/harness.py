"""Block 2: retraining-policy bench on a park of series served continuously.

Each policy decides, at every origin and for every series, whether to refit. We measure three families of
quantities, never one: accuracy (MASE), cost (refits per series) and stability (churn, jump p95, share of origins
with a visible revision).

Data closing lag `d`: actuals of the last d steps are not closed yet, so a model "refit now" is fit on a window
closed at t - d. `fit_lag = d` is production; `fit_lag = 0` is the backtest leak. `err_lag` is the same lag for
the realised errors that trigger policies read (the plan used 0, which reads actuals that are not closed yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import jump_stats, seasonal_scale
from .model import fit_window, forecast
from .park import Park


@dataclass(frozen=True)
class Bench:
    h: int = 24        # served horizon
    w: int = 240       # training window
    d: int = 48        # data closing lag
    useful_window: int = 72   # a refit is "useful" if a true break happened in the last 72 steps

    def warm(self, m: int) -> int:
        return self.w + m + self.d


# --------------------------------------------------------------------------------------------------------------
# Policies: called once per origin, return a (n,) bool mask "refit this series now".
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Policy:
    name: str

    def reset(self, park: Park, bench: Bench, mu0: np.ndarray) -> None:
        self.n = park.y.shape[0]
        self.sigma = park.sigma

    def __call__(self, t: int, ctx: dict) -> np.ndarray:
        raise NotImplementedError

    def refitted(self, fire: np.ndarray, mu_new: np.ndarray) -> None:
        """Hook called after a refit (for policies that track the deployed state)."""


@dataclass
class Never(Policy):
    name: str = "never"

    def __call__(self, t, ctx):
        return np.zeros(self.n, bool)


@dataclass
class Calendar(Policy):
    k: int = 168
    name: str = ""

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        self.warm = bench.warm(park.spec.m)

    def __call__(self, t, ctx):
        return np.full(self.n, t > self.warm and (t - self.warm) % self.k == 0)


@dataclass
class Cusum(Policy):
    """Two-sided CUSUM on the standardised realised error of the h-step forecast."""

    thr: float = 5.0
    slack: float = 0.5
    name: str = "cusum"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        self.sp = np.zeros(self.n)
        self.sm = np.zeros(self.n)

    def __call__(self, t, ctx):
        err = ctx["err"]
        ok = np.isfinite(err)
        z = np.where(ok, err / self.sigma, 0.0)
        self.sp = np.maximum(0.0, self.sp + z - self.slack)
        self.sm = np.maximum(0.0, self.sm - z - self.slack)
        fire = ((self.sp > self.thr) | (self.sm > self.thr)) & ok
        self.sp = np.where(fire, 0.0, self.sp)
        self.sm = np.where(fire, 0.0, self.sm)
        return fire


@dataclass
class CusumReset(Cusum):
    """Same CUSUM, but the refit window starts at the change-point estimate: the last time the statistic that
    fired was at zero (the classic CUSUM change-time estimator). A realistic detector with a matching remedy."""

    name: str = "cusum + window reset"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        self.zero_p = np.zeros(self.n, int)
        self.zero_m = np.zeros(self.n, int)
        self.change = np.zeros(self.n, int)

    def __call__(self, t, ctx):
        tc = ctx["err_time"]
        ok = np.isfinite(ctx["err"])
        z = np.where(ok, ctx["err"] / self.sigma, 0.0)
        sp = np.maximum(0.0, self.sp + z - self.slack)
        sm = np.maximum(0.0, self.sm - z - self.slack)
        self.zero_p = np.where(sp == 0.0, tc, self.zero_p)
        self.zero_m = np.where(sm == 0.0, tc, self.zero_m)
        fire_p, fire_m = (sp > self.thr) & ok, (sm > self.thr) & ok
        self.change = np.where(fire_p, self.zero_p + 1, np.where(fire_m, self.zero_m + 1, self.change))
        fire = fire_p | fire_m
        self.sp, self.sm = np.where(fire, 0.0, sp), np.where(fire, 0.0, sm)
        return fire

    def window_start(self, t_end: int) -> np.ndarray:
        return np.minimum(self.change, t_end)


@dataclass
class KLDebt(Policy):
    """Learning debt = KL(shadow posterior || deployed posterior), Gaussian with known variance
    (arXiv 2604.06438 writes it with a normal-inverse-gamma). Refit when the debt exceeds `thr`."""

    thr: float = 2.0
    name: str = "kl_debt"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        self.m_d = mu0.copy()
        self.v = self.sigma**2 / bench.w

    def __call__(self, t, ctx):
        v_s = v_d = self.v
        kl = (ctx["shadow_mean"] - self.m_d) ** 2 / (2 * v_d) + 0.5 * np.log(v_d / v_s) + v_s / (2 * v_d) - 0.5
        return kl > self.thr

    def refitted(self, fire, mu_new):
        self.m_d = np.where(fire, mu_new, self.m_d)


@dataclass
class KnownBreaks(Policy):
    """Perfect drift DETECTOR: fires `delay` steps after each true level break.

    delay = 0 is the plan's version. With a closing lag d, a refit at the break is fit on a window closed d steps
    BEFORE the break, so it cannot see it; delay >= d is the earliest refit that contains post-break data.
    """

    delay: int = 0
    name: str = "known_breaks"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        cps = park.cps
        self.mask = np.zeros_like(cps)
        self.mask[:, self.delay :] = cps[:, : cps.shape[1] - self.delay]

    def __call__(self, t, ctx):
        return self.mask[:, t]


@dataclass
class BreakReset(Policy):
    """Perfect detector with a remediation that USES the detection: after a break at c, refit at c + d + o for
    each o in `offsets`, on a window that starts at the break (truncated to whole seasonal cycles). Optionally
    also refits every `k` steps (with the same break-aware window) for the gradual drift no detector sees."""

    offsets: tuple[int, ...] = (24, 72, 240)
    k: int | None = None
    name: str = "break reset"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        self.warm = bench.warm(park.spec.m)
        T = park.cps.shape[1]
        self.mask = np.zeros_like(park.cps)
        for i, c in zip(*np.nonzero(park.cps), strict=True):
            for o in self.offsets:
                if c + bench.d + o < T:
                    self.mask[i, c + bench.d + o] = True
        idx = np.where(park.cps, np.arange(T)[None, :], 0)
        self.last_break = np.maximum.accumulate(idx, axis=1)   # last break at or before each step (0 if none)

    def __call__(self, t, ctx):
        fire = self.mask[:, t].copy()
        if self.k is not None and t > self.warm and (t - self.warm) % self.k == 0:
            fire[:] = True
        return fire

    def window_start(self, t_end: int) -> np.ndarray:
        return self.last_break[:, t_end]


@dataclass
class Either(Policy):
    """Fires when any of its sub-policies fires."""

    parts: list[Policy] = field(default_factory=list)
    name: str = "either"

    def reset(self, park, bench, mu0):
        super().reset(park, bench, mu0)
        for p in self.parts:
            p.reset(park, bench, mu0)

    def __call__(self, t, ctx):
        out = np.zeros(self.n, bool)
        for p in self.parts:
            out |= p(t, ctx)
        return out

    def refitted(self, fire, mu_new):
        for p in self.parts:
            p.refitted(fire, mu_new)


# --------------------------------------------------------------------------------------------------------------
def run(park: Park, policy: Policy, bench: Bench = Bench(), fit_lag: int | None = None, err_lag: int = 0) -> dict:
    """Serve the park from `warm` to T - h with `policy`.

    fit_lag: age of the newest usable observation for a refit (default bench.d; 0 is the leak).
    err_lag: age of the newest realised error the policy may read (0 reproduces the plan; bench.d is consistent
    with the closing lag).
    """
    y = park.y
    n, T = y.shape
    m = park.spec.m
    h, w = bench.h, bench.w
    fit_lag = bench.d if fit_lag is None else fit_lag
    warm = bench.warm(m)

    mu, prof, _ = fit_window(y, warm - fit_lag, w, m)
    policy.reset(park, bench, mu)

    pending: dict[int, np.ndarray] = {}   # target time -> h-step forecast issued for it
    f_prev = None
    abs_err = np.zeros(n)
    revs = []
    n_refit = np.zeros(n)
    hit = np.zeros(n)
    count = 0
    for t in range(warm, T - h):
        tc = t - err_lag
        err = y[:, tc] - pending.pop(tc) if tc in pending else np.full(n, np.nan)
        shadow = y[:, t - fit_lag - w + 1 : t - fit_lag + 1].mean(axis=1)
        ctx = dict(err=err, err_time=tc, shadow_mean=shadow)

        f = forecast(mu, prof, t, h, m)
        abs_err += np.abs(f - y[:, t + 1 : t + h + 1]).mean(axis=1)
        count += 1
        pending[t + h] = f[:, -1].copy()
        if f_prev is not None:
            revs.append(np.abs(f[:, :-1] - f_prev[:, 1:]).mean(axis=1))
        f_prev = f

        fire = policy(t, ctx)
        if fire.any():
            mu_n, pr_n, _ = fit_window(y, t - fit_lag, w, m)
            mu = np.where(fire, mu_n, mu)
            prof = np.where(fire[:, None], pr_n, prof)
            if hasattr(policy, "window_start"):
                t_end = t - fit_lag
                start = policy.window_start(t_end)
                for i in np.nonzero(fire & (start > t_end - w + 1))[0]:
                    length = t_end - start[i] + 1
                    length = max(m, length - length % m)
                    a, b, _ = fit_window(y[i : i + 1], t_end, length, m)
                    mu[i], prof[i] = a[0], b[0]
                    mu_n[i] = a[0]
            policy.refitted(fire, mu_n)
            n_refit += fire
            hit += fire & park.cps[:, max(t - bench.useful_window, 0) : t + 1].any(axis=1)

    scale = seasonal_scale(y, m)
    rel = np.asarray(revs) / scale[None, :]
    return dict(
        mase=float((abs_err / count / scale).mean()),
        **jump_stats(rel),
        n_refit=float(n_refit.mean()),
        useful=float(hit.sum() / max(n_refit.sum(), 1)),
    )


def plan_policies() -> list[Policy]:
    """The ten policies of the plan, in its order."""
    return [
        Never(),
        Calendar(k=720, name="calendar k=720 (monthly)"),
        Calendar(k=168, name="calendar k=168 (weekly)"),
        Calendar(k=48, name="calendar k=48"),
        Calendar(k=24, name="calendar k=24 (daily)"),
        Calendar(k=1, name="calendar k=1 (continuous)"),
        Cusum(thr=5.0, slack=0.5, name="cusum on realised error"),
        KLDebt(thr=8.0, name="kl debt (thr 8)"),
        KLDebt(thr=2.0, name="kl debt (thr 2)"),
        KnownBreaks(delay=0, name="known breaks"),
    ]


def total_cost(r: dict, lam_c: float, lam_k: float) -> float:
    """J = MASE + lam_c * refits / 1000 + lam_k * churn (the plan's composite objective)."""
    return r["mase"] + lam_c * r["n_refit"] / 1000 + lam_k * r["churn"]
