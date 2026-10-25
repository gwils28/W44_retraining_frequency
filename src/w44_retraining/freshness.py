"""Block 5: what a day of staleness costs, per served quantile, and the stability clause it implies.

Chain: freshness -> quantile error -> pinball loss -> euros -> cadence. Shared by the DGP study (plan) and the
real-data study (`realdata.py`). Two ways to place the quantiles around the point forecast:

- `gaussian`: z_tau * in-window residual std (the plan). It ignores that out-of-sample error grows with model age,
  so the tails are under-covered, more so as the model gets older.
- `conformal`: empirical tau-quantile of the model's own CLOSED errors (per series and lead time, last `buf`
  closed origins, recomputed every `recal` steps). Calibrated by construction on what the model actually does.

Two seasonal-naive baselines: `snaive` (the plan) reads y[t0 - m + 1 .. t0], actuals that are not closed yet under
a closing lag d; `snaive_closed` only reads closed actuals (lag m * ceil((h + d) / m), 72 for h = 24, d = 48).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import norm

from .metrics import pinball, seasonal_scale
from .model import fit_window

TAUS = (0.50, 0.80, 0.95)


def serve_quantiles(
    y: np.ndarray, k: int, *, start: int, stop: int, w: int = 240, m: int = 24, h: int = 24, fit_lag: int = 48,
    taus=TAUS, method: str = "gaussian", buf: int = 240, recal: int = 24, min_closed: int = 48,
) -> dict:
    """Serve quantile forecasts at origins start .. stop - h - 1, refit every k steps.

    Returns per tau: `loss` (n,) mean pinball per series (`loss_by_origin` (origins, n) before averaging),
    `cov` (n,) coverage, `rev` (origins, n) mean absolute
    revision of the shared targets between consecutive origins; plus `n_refit`.
    """
    n = y.shape[0]
    z = {t: float(norm.ppf(t)) for t in taus}
    mu, prof, sd = fit_window(y, start - fit_lag, w, m)
    loss = {t: [] for t in taus}
    cov = {t: np.zeros(n) for t in taus}
    revs = {t: [] for t in taus}
    prev = {t: None for t in taus}
    errors: deque = deque()          # (closing time, error (n, h)) of past origins
    closed: deque = deque(maxlen=buf)
    offset = None
    n_refit = cnt = 0
    for t0 in range(start, stop - h):
        ph = np.arange(t0 + 1, t0 + h + 1) % m
        base = mu[:, None] + prof[:, ph]
        act = y[:, t0 + 1 : t0 + h + 1]
        if method == "conformal":
            while errors and errors[0][0] <= t0 - fit_lag:
                closed.append(errors.popleft()[1])
            if len(closed) >= min_closed and (offset is None or (t0 - start) % recal == 0):
                stack = np.stack(closed)
                offset = {t: np.quantile(stack, t, axis=0) for t in taus}
        for tau in taus:
            if method == "conformal" and offset is not None and len(closed) >= min_closed:
                q = base + offset[tau]
            else:
                q = base + z[tau] * sd[:, None]
            loss[tau].append(pinball(act, q, tau).mean(axis=1))
            cov[tau] += (act <= q).mean(axis=1)
            if prev[tau] is not None:
                revs[tau].append(np.abs(q[:, :-1] - prev[tau][:, 1:]).mean(axis=1))
            prev[tau] = q
        if method == "conformal":
            errors.append((t0 + h, act - base))
        cnt += 1
        if t0 > start and (t0 - start) % k == 0:
            mu, prof, sd = fit_window(y, t0 - fit_lag, w, m)
            n_refit += 1
    return dict(
        loss={t: np.mean(loss[t], axis=0) for t in taus},
        loss_by_origin={t: np.asarray(loss[t]) for t in taus},
        cov={t: cov[t] / cnt for t in taus},
        rev={t: np.asarray(revs[t]) for t in taus},
        n_refit=n_refit,
    )


def seasonal_naive(
    y: np.ndarray, *, start: int, stop: int, w: int = 240, m: int = 24, h: int = 24, lag: int | None = None,
    taus=TAUS,
) -> dict:
    """Seasonal naive with the plan's dispersion (std of lag differences over the last w steps).

    lag = m reproduces the plan. A larger multiple of m only reads older actuals: the newest value used for any
    target is t0 + h - lag, and the dispersion window ends there too.
    """
    lag = m if lag is None else lag
    n = y.shape[0]
    newest = h - lag             # newest observation read, relative to t0 (<= 0)
    loss = {t: [] for t in taus}
    cov = {t: np.zeros(n) for t in taus}
    cnt = 0
    for t0 in range(start, stop - h):
        q0 = y[:, np.arange(t0 + 1, t0 + h + 1) - lag]
        act = y[:, t0 + 1 : t0 + h + 1]
        e = t0 + newest
        ww = min(w, e - lag + 1)     # a long lag shortens the first dispersion windows (history is finite)
        s = (y[:, e - ww + 1 : e + 1] - y[:, e - ww + 1 - lag : e + 1 - lag]).std(axis=1)
        for tau in taus:
            q = q0 + norm.ppf(tau) * s[:, None]
            loss[tau].append(pinball(act, q, tau).mean(axis=1))
            cov[tau] += (act <= q).mean(axis=1)
        cnt += 1
    return dict(loss={t: np.mean(loss[t], axis=0) for t in taus},
                loss_by_origin={t: np.asarray(loss[t]) for t in taus}, cov={t: cov[t] / cnt for t in taus})


def closed_lag(m: int, h: int, d: int) -> int:
    """Smallest multiple of m such that every target's same-phase value is closed at t0 - d."""
    return m * int(np.ceil((h + d) / m))


def stability(rev: np.ndarray, scale: np.ndarray, visible: float = 0.01) -> dict:
    rel = rev / scale[None, :]
    nz = rel[rel > 1e-12]
    return dict(jump_p95=float(np.percentile(nz, 95)) if nz.size else 0.0, pct_visible=float((rel > visible).mean()))


def k_max(model: dict, base: dict, tau: float, ks) -> int:
    """Largest cadence whose loss is below the baseline (0 if none)."""
    ok = [k for k in ks if model[k][tau] < base[tau]]
    return max(ok) if ok else 0


def euro_table(pinball_by_k: dict, refits_by_k: dict, *, n: int, steps: int, price: float,
               volume: float = 1_000.0, margin: float = 12.0) -> dict:
    """The plan's toy money model: loss = pinball x volume x margin / 100 x n x steps / 1000; cost = refits x
    price x n. The constants are the plan's, not calibrated on anything."""
    return {k: pinball_by_k[k] * volume * margin / 100.0 * n * steps / 1000 + refits_by_k[k] * price * n
            for k in pinball_by_k}


def scaled(loss: np.ndarray, scale: np.ndarray) -> float:
    return float((loss / scale).mean())


__all__ = ["TAUS", "closed_lag", "euro_table", "k_max", "scaled", "seasonal_naive", "seasonal_scale",
           "serve_quantiles", "stability"]
