import numpy as np
import pytest

from w44_retraining import fundamentals as F

SIG, Q = 1.0, 0.004


def test_w_star_matches_the_numerical_argmin_for_every_cadence():
    for k in (1, 24, 168):
        assert min(range(3, 600), key=lambda w, k=k: F.mse_closed(k, w, SIG, Q)) == round(F.w_star(SIG, Q))


def test_revision_parts_at_k1_are_two_noise_points_and_one_drift_step():
    noise, drift = F.var_revision_parts(1, 240, SIG, Q)
    assert noise == pytest.approx(2 * SIG**2 / 240**2)
    assert drift == pytest.approx(Q / 240)
    assert F.noise_share(1, 240, SIG, Q) == pytest.approx(0.676, abs=5e-4)


def test_churn_floor_is_only_an_asymptote_for_k_much_smaller_than_w():
    floor = F.SQ2PI * np.sqrt(Q / 240)
    assert F.churn_per_step(240, 240, SIG, Q) / floor == pytest.approx(np.sqrt(2 / 3), rel=0.01)
    assert F.churn_per_step(720, 240, SIG, Q) < F.churn_per_step(240, 240, SIG, Q)


def test_vectorised_monte_carlo_equals_the_plan_loop_on_a_short_series():
    k, w, T = 7, 30, 3000
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0, np.sqrt(Q), T)) + rng.normal(0, SIG, T)
    se, revs, mu = [], [], None
    for t in range(w, T - 1):
        if (t - w) % k == 0:
            new = y[t - w + 1 : t + 1].mean()
            if mu is not None:
                revs.append(new - mu)
            mu = new
        se.append((y[t + 1] - mu) ** 2)
    assert F.monte_carlo(k, w, SIG, Q, T=T, seed=1) == pytest.approx((np.mean(se), np.var(revs), np.mean(np.abs(revs))))


@pytest.mark.slow
@pytest.mark.parametrize(("k", "w"), F.MC_CONFIGS)
def test_closed_forms_within_three_monte_carlo_standard_errors(k, w):
    sims = np.array([F.monte_carlo(k, w, SIG, Q, seed=s) for s in (44, 45, 46, 47, 48)])
    mse, var_r = sims[:, 0], sims[:, 1]
    assert abs(F.mse_closed(k, w, SIG, Q) - mse.mean()) < 3 * mse.std(ddof=1) / np.sqrt(5) + 0.01 * mse.mean()
    assert abs(F.var_revision(k, w, SIG, Q) - var_r.mean()) < 3 * var_r.std(ddof=1) / np.sqrt(5) + 0.01 * var_r.mean()
