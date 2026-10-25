"""The served model: window mean + seasonal profile (+ residual std for quantiles). Shared by blocks 2 and 5."""

from __future__ import annotations

import numpy as np


def fit_window(y: np.ndarray, t_end: int, w: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on the window closed at `t_end` included (t_end - w + 1 .. t_end). Never reads y[:, t_end + 1:].

    Returns (mu, prof, sd): level (n,), seasonal profile (n, m) indexed by absolute phase t % m, and the std of
    the residual AFTER removing the profile (keeping the profile in it inflates sd and wrecks high quantiles).
    """
    if t_end - w + 1 < 0:
        raise ValueError(f"window [{t_end - w + 1}, {t_end}] starts before the series")
    yw = y[:, t_end - w + 1 : t_end + 1]
    mu = yw.mean(axis=1)
    resid = yw - mu[:, None]
    phase = np.arange(t_end - w + 1, t_end + 1) % m
    prof = np.empty((y.shape[0], m))
    for k in range(m):
        prof[:, k] = resid[:, phase == k].mean(axis=1)
    sd = (resid - prof[:, phase]).std(axis=1)
    return mu, prof, sd


def forecast(mu: np.ndarray, prof: np.ndarray, t0: int, h: int, m: int) -> np.ndarray:
    """Point forecast (n, h) issued at origin t0 for t0 + 1 .. t0 + h."""
    ph = np.arange(t0 + 1, t0 + h + 1) % m
    return mu[:, None] + prof[:, ph]
