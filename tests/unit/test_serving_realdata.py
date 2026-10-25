import pytest

from w44_retraining.realdata import DataCheckError, parse_tsf
from w44_retraining.serving import compute_cost, licence_gate, workers

TSF = """# comment
@relation x
@frequency hourly
@data
T1:2012-01-01 00-00-01:1,2,3
T2:2012-01-01 00-00-01:4,?,6
"""


def test_parse_tsf():
    names, starts, values = parse_tsf(TSF)
    assert names == ["T1", "T2"] and starts[0] == "2012-01-01 00-00-01" and values[1] == ["4", "?", "6"]


def test_data_check_error_is_a_value_error():
    assert issubclass(DataCheckError, ValueError)


def test_workers_and_cost_arithmetic():
    assert workers(500_000, 30, 140.0) == 39       # 277.8 inf/s needed, 7.14 inf/s per worker
    assert compute_cost(500_000, 1, 140.0) == pytest.approx(3920.0)


@pytest.mark.parametrize(
    ("family", "revenue", "prefix"),
    [("Timestral", None, "REFUSE"), ("Timestral", 5e6, "ALLOW"), ("Timestral", 50e6, "REFUSE"),
     ("Chronos-2", 50e6, "ALLOW"), ("unknown", 1.0, "REFUSE")],
)
def test_licence_gate(family, revenue, prefix):
    assert licence_gate(family, revenue).startswith(prefix)


def test_weekly_profile_aligns_sunday_and_measures_the_weekend():
    import numpy as np

    from w44_retraining.realdata import weekend_ratio, weekly_profile

    hours = np.arange(1000 + 552 + 2 * 168)
    file_offset = 30                                    # first kept hour is hour 30 of the file
    how = (hours + file_offset) % 168                  # hour of the week, 0 = Sunday 00:00
    y = np.where((how < 24) | (how >= 144), 1.0, 2.0)[None, :]
    rep = {"T_file": file_offset + hours.size, "T_kept": hours.size}
    week = weekly_profile(y, rep, {"span": {"history_before_evaluation": 552}}, weeks=2)
    assert week[0, :24].max() < week[0, 24:144].min()
    assert weekend_ratio(week)[0] == pytest.approx(0.5)
