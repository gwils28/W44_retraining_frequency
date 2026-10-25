import numpy as np
import pytest

from w44_retraining.freshness import closed_lag, k_max, seasonal_naive, serve_quantiles


def test_closed_lag_reads_only_closed_actuals():
    assert closed_lag(24, 24, 48) == 72
    assert closed_lag(24, 24, 0) == 24


def test_closed_seasonal_naive_does_not_read_unclosed_actuals():
    rng = np.random.default_rng(0)
    y = 10 + rng.normal(size=(3, 800))
    start, stop = 400, 425     # a single origin: t0 = 400
    ref = seasonal_naive(y, start=start, stop=stop, lag=72)["loss_by_origin"][0.5]
    y2 = y.copy()
    y2[:, 400 - 48 + 1 : 401] += 100.0       # actuals not closed at t0 with d = 48
    assert np.array_equal(ref, seasonal_naive(y2, start=start, stop=stop, lag=72)["loss_by_origin"][0.5])
    assert not np.array_equal(ref, seasonal_naive(y2, start=start, stop=stop, lag=24)["loss_by_origin"][0.5])


def test_conformal_quantiles_are_calibrated_on_stationary_noise():
    rng = np.random.default_rng(1)
    y = 5 + rng.standard_t(df=3, size=(40, 3000))      # heavy tails: the gaussian rule is miscalibrated
    r = serve_quantiles(y, 24, start=312, stop=3000, method="conformal")
    assert r["cov"][0.95].mean() == pytest.approx(0.95, abs=0.01)
    assert r["cov"][0.5].mean() == pytest.approx(0.5, abs=0.01)


def test_k_max_is_the_largest_cadence_below_the_baseline():
    model = {1: {0.5: 1.0}, 24: {0.5: 1.5}, 168: {0.5: 2.5}}
    assert k_max(model, {0.5: 2.0}, 0.5, (1, 24, 168)) == 24
    assert k_max(model, {0.5: 0.5}, 0.5, (1, 24, 168)) == 0


def test_refit_phase_changes_slow_cadences_much_more_than_fast_ones():
    from w44_retraining.realdata import gaussian_kmax_at_phase

    rng = np.random.default_rng(3)
    t = np.arange(1500)
    y = 10 + np.sin(2 * np.pi * t / 24) + np.cumsum(rng.normal(0, 0.05, (4, 1500)), axis=1)
    cfg = {"model": {"window_w": 120, "season_m": 24, "horizon_h": 24, "closing_lag_d": 48},
           "span": {"history_before_evaluation": 500}, "cadences_k": [1, 24, 72]}
    a, b = gaussian_kmax_at_phase(y, cfg, 240), gaussian_kmax_at_phase(y, cfg, 0)
    assert abs(a["loss"][1][0.5] / b["loss"][1][0.5] - 1) < 1e-3
    assert a["loss"][72] != b["loss"][72]
    assert set(a["k_max"]) == {0.5, 0.8, 0.95}
