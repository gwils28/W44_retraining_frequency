import copy

import yaml

from w44_retraining.gate import CONTRACT, check

BASE = yaml.safe_load(CONTRACT.read_text())


def results(c):
    return {name.split()[0]: ok for name, ok, _ in check(c)}


def test_committed_contract_is_blocked_only_by_the_amplitude_clause():
    r = results(BASE)
    assert r == {"G1": True, "G2": True, "G3": True, "G4": False, "G5": True, "G6": True}


def test_signing_the_frequency_clause_instead_unblocks_it():
    c = copy.deepcopy(BASE)
    c["stability"]["clause"] = "max_pct_origins_visible"
    assert all(results(c).values())


def test_each_rule_can_block():
    cases = {
        "G1": lambda c: c["client"].update(monthly_revenue_usd=50e6),
        "G2": lambda c: c["serving"].update(park_size=500_000),
        "G3": lambda c: c["retraining"].update(cadence_steps=336),
        "G5": lambda c: c["retraining"].update(effect_measurement={}),
        "G6": lambda c: c["data"].update(baseline_must_use_closed_data=False),
    }
    for rule, mutate in cases.items():
        c = copy.deepcopy(BASE)
        c["stability"]["clause"] = "max_pct_origins_visible"
        mutate(c)
        assert not results(c)[rule], rule
