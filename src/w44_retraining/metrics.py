"""Scale, pinball loss and revision metrics."""

from __future__ import annotations

import numpy as np


def seasonal_scale(y: np.ndarray, m: int) -> np.ndarray:
    """MASE denominator per series: mean |y_t - y_{t-m}|, written with an explicit lag.

    Not `np.diff(y, n=m)`: that is the m-th ORDER difference, not the lag-m difference.
    """
    return np.abs(y[..., m:] - y[..., :-m]).mean(axis=-1)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.where(d >= 0, tau * d, (tau - 1) * d)


def revision(f_new: np.ndarray, f_prev: np.ndarray) -> np.ndarray:
    """Mean absolute revision (per series) of the targets shared by two consecutive origins.

    f_prev covers t0 .. t0 + h - 1 and f_new covers t0 + 1 .. t0 + h, so the overlap is f_new[:, :-1] vs
    f_prev[:, 1:].
    """
    return np.abs(f_new[:, :-1] - f_prev[:, 1:]).mean(axis=1)


def jump_stats(rel_rev: np.ndarray, visible: float = 0.01) -> dict[str, float]:
    """Stability clause figures from relative revisions (origins x series).

    A frozen model has exactly zero churn: revisions are concentrated on retraining origins, so the mean is
    meaningless. What a client sees is the size of a jump when there is one (p95 over non-zero revisions) and how
    often a jump is visible (share of origins above `visible`).
    """
    nz = rel_rev[rel_rev > 1e-12]
    return dict(
        churn=float(rel_rev.mean()),
        jump_p95=float(np.percentile(nz, 95)) if nz.size else 0.0,
        pct_visible=float((rel_rev > visible).mean()),
    )
