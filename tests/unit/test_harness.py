import numpy as np

from w44_retraining.harness import Bench, BreakReset, Calendar, KnownBreaks, Never, run
from w44_retraining.park import ParkSpec, make_park

SMALL = ParkSpec(n=6, T=900)


def test_frozen_model_has_exactly_zero_churn():
    r = run(make_park(SMALL), Never())
    assert r["churn"] == 0.0 and r["n_refit"] == 0.0


def test_calendar_fires_every_k_after_warm_up():
    park, bench = make_park(SMALL), Bench()
    p = Calendar(k=100)
    p.reset(park, bench, np.zeros(SMALL.n))
    warm = bench.warm(SMALL.m)
    fired = [t for t in range(warm, SMALL.T) if p(t, {}).all()]
    assert fired == list(range(warm + 100, SMALL.T, 100))


def test_known_breaks_delay_shifts_the_mask():
    park = make_park(ParkSpec(n=50))
    p = KnownBreaks(delay=48)
    p.reset(park, Bench(), np.zeros(50))
    i, c = map(int, np.argwhere(park.cps)[0])
    assert p.mask[i, c + 48] and p.mask.sum() <= park.cps.sum()


def test_break_reset_window_starts_at_the_last_break():
    park = make_park(ParkSpec(n=50))
    p = BreakReset()
    p.reset(park, Bench(), np.zeros(50))
    i, c = map(int, np.argwhere(park.cps)[0])
    assert p.window_start(c)[i] == c and p.window_start(c - 1)[i] < c


def test_leaky_backtest_looks_better():
    park = make_park(SMALL)
    assert run(park, Calendar(k=24), fit_lag=0)["mase"] < run(park, Calendar(k=24))["mase"]


def test_trace_records_the_served_level_at_every_origin_without_changing_results():
    park, bench = make_park(SMALL), Bench()
    tr: list = []
    a = run(park, Calendar(k=24), trace=tr)
    assert a == run(park, Calendar(k=24))
    assert len(tr) == SMALL.T - bench.h - bench.warm(SMALL.m) and tr[0].shape == (SMALL.n,)
