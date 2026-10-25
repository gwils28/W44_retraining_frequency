from dataclasses import replace

import yaml

from w44_retraining.gate import CONTRACT
from w44_retraining.gateway import NON_PORTABLE_PARAMETERS, ForecastRequest, cache_key, narrow_cache_key

R = ForecastRequest("T1", "2026-10-01T00:00", 24, (0.5, 0.95), "m-7", "f-3", "2026-09-28T00:00")


def test_a_month_later_or_after_a_retrain_is_another_cache_entry():
    later = replace(R, as_of="2026-11-01T00:00")
    retrained = replace(R, model_version="m-8", training_asof="2026-09-30T00:00")
    assert cache_key(later) != cache_key(R) and cache_key(retrained) != cache_key(R)
    assert narrow_cache_key(later) == narrow_cache_key(R) == narrow_cache_key(retrained)


def test_oip_payload_carries_time_only_in_parameters_and_the_contract_declares_them():
    p = R.to_oip([1.0, 2.0])
    assert set(p) == {"id", "parameters", "inputs"}
    assert set(p["parameters"]) == set(NON_PORTABLE_PARAMETERS)
    contract = yaml.safe_load(CONTRACT.read_text())
    assert set(NON_PORTABLE_PARAMETERS) <= set(contract["service"]["gateway_adds"])
    assert set(contract["serving"]["cache_key"]) >= {"as_of", "series_id", "horizon", "training_asof"}
