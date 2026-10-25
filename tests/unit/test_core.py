import numpy as np
import pytest

from w44_retraining.metrics import jump_stats, pinball, seasonal_scale
from w44_retraining.model import fit_window, forecast
from w44_retraining.park import ParkSpec, make_park


def test_seasonal_scale_is_a_lag_not_a_difference_order():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(3, 200))
    expected = np.array([np.mean([abs(s[t] - s[t - 24]) for t in range(24, 200)]) for s in y])
    assert np.allclose(seasonal_scale(y, 24), expected)
    assert not np.allclose(seasonal_scale(y, 24), np.abs(np.diff(y, n=24)).mean(axis=1))


def test_fit_window_does_not_read_the_future():
    y = make_park(ParkSpec(n=5, T=600)).y
    ref = fit_window(y, 400, 240, 24)
    y2 = y.copy()
    y2[:, 401:] = 1e6
    for a, b in zip(ref, fit_window(y2, 400, 240, 24), strict=True):
        assert np.array_equal(a, b), "a fit on the window closed at 400 changed when data after 400 changed"


def test_fit_window_refuses_a_window_before_the_series():
    with pytest.raises(ValueError):
        fit_window(np.zeros((1, 100)), 50, 240, 24)


def test_profile_recovers_a_pure_seasonal_signal():
    t = np.arange(480)
    prof = np.sin(2 * np.pi * np.arange(24) / 24)
    y = (5.0 + prof[t % 24])[None, :]
    mu, p, sd = fit_window(y, 479, 240, 24)
    assert np.isclose(mu[0], 5.0) and np.allclose(p[0], prof) and sd[0] < 1e-12
    assert np.allclose(forecast(mu, p, 479, 24, 24)[0], 5.0 + prof[np.arange(480, 504) % 24])


def test_pinball_is_asymmetric():
    assert pinball(np.array(1.0), np.array(0.0), 0.9) == pytest.approx(0.9)
    assert pinball(np.array(0.0), np.array(1.0), 0.9) == pytest.approx(0.1)


def test_jump_stats_ignore_zero_revisions_for_the_jump_but_not_for_the_mean():
    rel = np.zeros((100, 2))
    rel[10, 0] = 0.5
    s = jump_stats(rel)
    assert s["jump_p95"] == pytest.approx(0.5)
    assert s["churn"] == pytest.approx(0.5 / 200)
    assert s["pct_visible"] == pytest.approx(1 / 200)


def test_seasonal_drift_switch():
    a, b = make_park(ParkSpec(n=4, seas_drift=0.0)), make_park(ParkSpec(n=4))
    assert np.array_equal(a.cps, b.cps), "switching the seasonal drift off must keep the same breaks"
    assert not a.cps[:, : ParkSpec().warm].any()
