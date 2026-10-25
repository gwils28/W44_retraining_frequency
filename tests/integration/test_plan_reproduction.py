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
