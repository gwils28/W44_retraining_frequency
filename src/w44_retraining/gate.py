"""Block 3: the deployment gate of `configs/forecast_service.yaml`. Deterministic, fail-closed, no model call.

G1 weights licence read at the source and valid for this client
G2 the compute window holds the park with the provisioned workers
G3 the cadence fits the freshness budget of every served quantile
G4 the measured stability meets the ONE signed clause
G5 the effect of retraining is measurable (holdout >= 2%, or RDD logging)
G6 the closing lag is declared and applied to the backtest, the baseline and the trigger
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .serving import licence_gate, workers

CONTRACT = Path(__file__).resolve().parents[2] / "configs" / "forecast_service.yaml"


def check(c: dict) -> list[tuple[str, bool, str]]:
    res = []
    sv, rt, data = c["serving"], c["retraining"], c["data"]

    lic = licence_gate(sv["model_family"], c.get("client", {}).get("monthly_revenue_usd"))
    res.append(("G1 weights licence", lic.startswith("ALLOW"), lic))

    need = workers(sv["park_size"], sv["compute_window_minutes"], sv["latency_ms"])
    res.append(("G2 compute window", need <= sv["provisioned_workers"],
                f"{need} workers needed, {sv['provisioned_workers']} provisioned"))

    budget = {tau: v["max_model_age_steps"] for tau, v in c["freshness"]["budget_by_tau"].items()}
    k = rt["cadence_steps"]
    worst = min(budget.values())
    res.append(("G3 freshness budget", k <= worst, f"cadence {k} vs tightest budget {worst} ({budget})"))

    st, ms = c["stability"], c["measured"]
    if st["clause"] == "jump_p95_max":
        ok, msg = ms["jump_p95_at_tau_095"] <= st["jump_p95_max"], (
            f"jump p95 {ms['jump_p95_at_tau_095']:.3f} vs clause {st['jump_p95_max']}")
    elif st["clause"] == "max_pct_origins_visible":
        vis, cap = ms["pct_origins_visible_at_tau_095"], st["max_pct_origins_visible"]
        ok, msg = vis <= cap, f"visible revisions {vis:.1%} vs clause {cap:.0%}"
    else:
        ok, msg = False, f"unknown or double clause {st['clause']!r}: sign exactly one"
    res.append(("G4 stability clause", ok, msg))

    em = rt.get("effect_measurement", {})
    frac = em.get("holdout", {}).get("fraction", 0.0) if em.get("holdout", {}).get("permanent") else 0.0
    rdd = bool(em.get("rdd_logging"))
    res.append(("G5 measurable effect", frac >= 0.02 or rdd, f"holdout {frac:.0%}, rdd logging {rdd}"))

    flags = ["backtest_must_use_same_lag", "baseline_must_use_closed_data", "trigger_must_use_closed_errors"]
    missing = [f for f in flags if not data.get(f)]
    ok = data.get("closing_lag_steps") is not None and not missing
    res.append(("G6 closing lag", ok, f"d = {data.get('closing_lag_steps')}, missing: {missing or 'none'}"))
    return res


def run(path: Path = CONTRACT) -> bool:
    c = yaml.safe_load(path.read_text())
    res = check(c)
    for name, ok, msg in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22s} {msg}")
    passed = all(ok for _, ok, _ in res)
    print(f"\n{'DEPLOY' if passed else 'BLOCKED'} ({c['fail_mode']})")
    return passed
