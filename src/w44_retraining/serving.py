"""Block 6: choose the size of a TSFM by its serving cost, not by the leaderboard.

Latencies are Mistral's published Timestral figures (15 Oct 2026, server CPU, 8 threads, one day ahead from three
weeks of 15 min history, i.e. a 2016-step context and a 96-step horizon). 28M and 123M are derived from ratios
in the announcement. No accuracy number is published as text, only "within 0.2%" between 123M and 209M on the
benchmark score, so we compute the accuracy gain a size WOULD need, and compare it with that 0.2%.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .model import fit_window, forecast

# name: (parameters in M, CPU latency ms with 8 threads, fp32 weights MB, how the latency is known)
MODELS = {
    "Timestral-5M": (5, 10.0, 21, "published"),
    "Timestral-28M": (28, 28.0, 111, "derived: 140 / 5"),
    "Timestral-123M": (123, 105.0, 490, "derived: 140 x 3/4"),
    "Timestral-209M": (209, 140.0, 837, "published"),
}
PUBLISHED_SCORE_GAP_123_209 = 0.002   # "within 0.2% of the 209M score": a BENCHMARK score gap
THREADS = 8
PRICE_VCPU_H = 0.035                  # EUR per vCPU-hour (declared, not published)
HOURS_MONTH = 720


def compute_cost(S: int, k: float, lat_ms: float, price: float = PRICE_VCPU_H, threads: int = THREADS) -> float:
    """EUR of CPU per month for S series refreshed every k hours, paying only the CPU-seconds consumed (one
    inference holds all threads for its whole latency: no batching)."""
    n_inf = S * HOURS_MONTH / k
    return n_inf * lat_ms / 1000.0 * threads / 3600.0 * price


def breakeven_gain(S: int, k: float, lat_hi: float, lat_lo: float, loss_eur_series_month: float) -> float:
    """Relative reduction of the decision loss the slower model must bring to repay its extra CPU."""
    return (compute_cost(S, k, lat_hi) - compute_cost(S, k, lat_lo)) / (S * loss_eur_series_month)


def workers(S: int, window_min: float, lat_ms: float) -> int:
    """8-vCPU workers needed to refresh S series within a window, one inference at a time per worker."""
    return math.ceil(S / (window_min * 60) / (1000.0 / lat_ms))


def provisioned_cost(n_workers: int, price: float = PRICE_VCPU_H, threads: int = THREADS) -> float:
    """EUR per month if the workers sized for the window are kept all month (no autoscaling to zero)."""
    return n_workers * threads * HOURS_MONTH * price


def measure_classic_latency(n: int = 2000, T: int = 600, w: int = 240, m: int = 24, h: int = 24,
                            repeats: int = 5) -> float:
    """Measured ms per series for one refit + one forecast of the block 2 model (numpy, vectorised over series,
    single process). Replaces the plan's declared 0.08 ms."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, T))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        mu, prof, _ = fit_window(y, T - 1, w, m)
        forecast(mu, prof, T - 1, h, m)
        best = min(best, time.perf_counter() - t0)
    return best / n * 1000.0


# -------------------------------------------------------------------------------------------------------------
# Licence gate: the weights licence, read on the Hub, with its conditions (block 1)
# -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightsLicence:
    model: str
    licence: str
    source: str
    revenue_cap_usd_month: float | None = None   # no rights at all above it (company or employer)


LICENCES = {
    "Timestral": WeightsLicence(
        "mistralai/Timestral-*", "Modified MIT", "huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
        revenue_cap_usd_month=20e6),
    "Chronos-2": WeightsLicence("amazon/chronos-2", "apache-2.0", "huggingface.co/amazon/chronos-2 (card metadata)"),
}
ALLOWED = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0"}


def licence_gate(family: str, client_revenue_usd_month: float | None) -> str:
    lic = LICENCES.get(family)
    if lic is None:
        return "REFUSE: weights licence not read at the source"
    if lic.licence.lower() in ALLOWED:
        return f"ALLOW ({lic.licence})"
    if lic.revenue_cap_usd_month is not None:
        if client_revenue_usd_month is None:
            return "REFUSE: client revenue unknown, licence has a revenue cap"
        if client_revenue_usd_month > lic.revenue_cap_usd_month:
            return f"REFUSE: revenue above the {lic.revenue_cap_usd_month / 1e6:.0f}M USD/month cap"
        return f"ALLOW under the cap ({lic.licence}), re-check monthly; derivatives inherit the cap"
    return f"REVIEW: {lic.licence} is not in the allow-list"


def study() -> dict:
    out: dict = {"cost": {}, "breakeven": {}, "window": {}, "provisioned": {}}
    for S in (1_000, 20_000, 500_000):
        out["cost"][S] = {nm: {k: compute_cost(S, k, lat) for k in (1, 6, 24, 168)} for nm, (_, lat, _, _) in
                          MODELS.items()}
    S = 20_000
    for c in (0.10, 1.0, 10.0, 100.0, 1000.0):
        out["breakeven"][c] = {k: breakeven_gain(S, k, 140.0, 105.0, c) for k in (1, 6, 24)}
    grid = np.logspace(-2, 4, 2000)
    out["breakeven_threshold"] = float(min(c for c in grid if breakeven_gain(S, 6, 140.0, 105.0, c)
                                           <= PUBLISHED_SCORE_GAP_123_209))
    for S, win in ((20_000, 30), (500_000, 30), (500_000, 120)):
        out["window"][S, win] = {nm: workers(S, win, lat) for nm, (_, lat, _, _) in MODELS.items()}
        out["provisioned"][S, win] = {nm: provisioned_cost(wk) for nm, wk in out["window"][S, win].items()}
    out["consumed_daily_500k"] = {nm: compute_cost(500_000, 24, lat) for nm, (_, lat, _, _) in MODELS.items()}
    out["classic_ms"] = measure_classic_latency()
    out["classic_breakeven"] = {c: {nm: breakeven_gain(20_000, 6, MODELS[nm][1], out["classic_ms"], c)
                                    for nm in ("Timestral-28M", "Timestral-209M")} for c in (0.10, 1.0, 10.0)}
    out["gate"] = {(fam, rev): licence_gate(fam, rev) for fam in ("Timestral", "Chronos-2", "TimesFM-3")
                   for rev in (None, 5e6, 50e6)}
    return out


def print_study(out: dict) -> None:
    print(f"A. CPU cost per month (EUR), {PRICE_VCPU_H} EUR/vCPU-h, {THREADS} threads, consumed CPU-seconds only")
    for S, tab in out["cost"].items():
        print(f"   park of {S:,} series")
        for nm, row in tab.items():
            print(f"   {nm:>16s} " + " ".join(f"k={k}h {v:>8,.0f}" for k, v in row.items()))
    print("\nB. accuracy gain 209M must bring over 123M to repay its CPU (20,000 series)")
    for c, row in out["breakeven"].items():
        print(f"   decision loss {c:>8,.2f} EUR/series/month: " + "  ".join(f"k={k}h {v:.3%}" for k, v in row.items()))
    print(f"   threshold at k=6h: 209M repays its CPU above {out['breakeven_threshold']:.2f} EUR/series/month,"
          " IF the 0.2% benchmark gap were a 0.2% decision-loss gap")
    print("\nC. compute window: 8-vCPU workers, and their cost if kept all month")
    for (S, win), row in out["window"].items():
        print(f"   S={S:,}, window {win} min: " + "  ".join(
            f"{nm.split('-')[1]} {wk} ({out['provisioned'][S, win][nm]:,.0f} EUR)" for nm, wk in row.items()))
    print("   consumed CPU for a daily refresh of 500,000 series: " + "  ".join(
        f"{nm.split('-')[1]} {v:,.0f} EUR" for nm, v in out["consumed_daily_500k"].items()))
    print(f"\nD. classic model (block 2), MEASURED: {out['classic_ms']:.4f} ms per series (refit + forecast)")
    for c, row in out["classic_breakeven"].items():
        print(f"   at {c:.2f} EUR/series/month a TSFM must beat it by: " + "  ".join(
            f"{nm.split('-')[1]} {v:.2%}" for nm, v in row.items()))
    print("\nE. weights licence gate (client monthly revenue in USD)")
    for (fam, rev), v in out["gate"].items():
        print(f"   {fam:>10s}, revenue {('unknown' if rev is None else f'{rev / 1e6:.0f}M'):>8s}: {v}")
