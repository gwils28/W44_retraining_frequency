from w44_retraining import causal as C


def test_placebo_fools_the_naive_estimator_but_not_the_ab():
    d = C.simulate(randomize=True, placebo=True)
    assert C.est_naive(d) < -0.1
    lo, hi = C.ci_ab(d)
    assert lo <= 0 <= hi


def test_stricter_threshold_makes_a_larger_fake_effect():
    fake = [C.est_naive(C.simulate(thr=t, placebo=True)) for t in (0.9, 1.05, 1.3)]
    assert fake[0] > fake[1] > fake[2]


def test_rdd_needs_no_holdout_and_passes_the_placebo():
    d = C.simulate(placebo=True, n=200_000)
    est, se = C.est_rdd(d, 1.05, 0.10)
    assert abs(est) < 3 * se
    e2, se2 = C.est_rdd(C.simulate(n=200_000), 1.05, 0.10)
    assert e2 < -3 * se2


def test_effect_by_band_matches_truth_on_the_same_band():
    assert C.effect_by_band(0.30, [1.05, 1.10])[0] == C.truth(0.30, band=(1.05, 1.10))


def test_rdd_lines_give_the_rdd_estimate():
    d = C.simulate(n=50_000)
    (r, _), (l_, _) = C.rdd_lines(d, 1.05, 0.2)
    assert r[0] - l_[0] == C.est_rdd(d, 1.05, 0.2)[0]
