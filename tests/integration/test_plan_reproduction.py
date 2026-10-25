"""The package reproduces the plan's published figures to the printed precision (same seeds)."""

import pytest

from w44_retraining.harness import Calendar, KnownBreaks, run
from w44_retraining.park import make_park

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("policy", "mase", "jump_p95", "n_refit"),
    [
        (Calendar(k=168), 0.9829, 1.2647, 12.0),
        (Calendar(k=1), 0.8915, 0.0132, 2063.0),
        (KnownBreaks(delay=0), 1.3685, 2.2602, 3.7),
    ],
)
def test_block2_table(policy, mase, jump_p95, n_refit):
    r = run(make_park(), policy)
    assert round(r["mase"], 4) == mase
    assert round(r["jump_p95"], 4) == jump_p95
    assert round(r["n_refit"], 1) == n_refit


def test_block5_dgp_figures():
    from w44_retraining.freshness import seasonal_naive, serve_quantiles
    from w44_retraining.park import ParkSpec

    y = make_park(ParkSpec(n=200, seed=544)).y
    r = serve_quantiles(y, 24, start=312, stop=y.shape[1])
    assert [round(float(r["loss"][t].mean()), 4) for t in (0.5, 0.8, 0.95)] == [0.2579, 0.1857, 0.0757]
    b = seasonal_naive(y, start=312, stop=y.shape[1])
    assert [round(float(b["loss"][t].mean()), 4) for t in (0.5, 0.8, 0.95)] == [0.2880, 0.2032, 0.0758]
