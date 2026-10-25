"""Synthetic park of hourly series with two drifts of different nature (shared by blocks 2 and 5).

Level: piecewise constant with Poisson breaks (abrupt). Seasonal profile: its amplitude drifts slowly along a
sinusoid with a series-specific phase (gradual). A policy that only reacts to breaks is blind to the second one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParkSpec:
    n: int = 120
    T: int = 2400
    m: int = 24                # seasonal period (hourly data, daily cycle)
    seed: int = 44
    rate: float = 1 / 600.0    # break probability per step
    jump: float = 0.8          # std of a level break
    sigma: float = 0.5         # observation noise
    seas_drift: float = 0.35   # relative amplitude of the slow seasonal drift (0 switches it off)
    warm: int = 312            # no break before this step (W + M + d in the plan)


@dataclass(frozen=True)
class Park:
    y: np.ndarray       # (n, T)
    cps: np.ndarray     # (n, T) bool, True at a level break
    sigma: float
    spec: ParkSpec


def make_park(spec: ParkSpec = ParkSpec()) -> Park:
    """Same draws, in the same order, as the plan's `make_park` (bit-for-bit for seas_drift = 0.35)."""
    rng = np.random.default_rng(spec.seed)
    t = np.arange(spec.T)
    seas_base = rng.normal(0, 1.2, (spec.n, spec.m))
    seas_base -= seas_base.mean(axis=1, keepdims=True)
    cps = rng.random((spec.n, spec.T)) < spec.rate
    cps[:, : spec.warm] = False
    level = 10.0 + np.cumsum(rng.normal(0, spec.jump, (spec.n, spec.T)) * cps, axis=1)
    amp = 1.0 + spec.seas_drift * np.sin(2 * np.pi * t / spec.T + rng.uniform(0, 6.28, (spec.n, 1)))
    y = level + seas_base[:, t % spec.m] * amp + rng.normal(0, spec.sigma, (spec.n, spec.T))
    return Park(y=y, cps=cps, sigma=spec.sigma, spec=spec)
