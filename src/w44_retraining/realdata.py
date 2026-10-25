"""Block 5 on real data: Monash Electricity Hourly, driven by `configs/freshness_real.yaml` (protocol fixed first).

`download` fetches the zip into data/raw/ and refuses a file whose sha256 differs from the protocol. `load` parses
the .tsf and runs the protocol's data checks; each failure names the series and how many values are at fault.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

from .freshness import TAUS, closed_lag, k_max, seasonal_naive, serve_quantiles, stability

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "freshness_real.yaml"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "data" / "results" / "real_study.pkl"


class DataCheckError(ValueError):
    pass


def protocol() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: dict | None = None) -> Path:
    cfg = cfg or protocol()
    ds = cfg["dataset"]
    path = RAW / (Path(ds["file"]).stem + ".zip")
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ds["url"], path)
    got = sha256(path)
    if got != ds["sha256_zip"]:
        raise DataCheckError(f"{path}: sha256 {got} differs from the protocol's {ds['sha256_zip']}")
    return path


def parse_tsf(text: str) -> tuple[list[str], list[str], list[list[str]]]:
    names, starts, values = [], [], []
    in_data = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not in_data:
            in_data = line.strip().lower() == "@data"
            continue
        name, start, vals = line.split(":", 2)
        names.append(name)
        starts.append(start)
        values.append(vals.split(","))
    return names, starts, values


def load(cfg: dict | None = None) -> tuple[np.ndarray, list[str], dict]:
    """Return (y (n, T) of the evaluation span + history, kept series names, check report)."""
    cfg = cfg or protocol()
    path = download(cfg)
    with zipfile.ZipFile(path) as z:
        text = io.TextIOWrapper(z.open(cfg["dataset"]["file"]), encoding="utf-8").read()
    names, starts, values = parse_tsf(text)

    if len(names) != cfg["dataset"]["series_expected"]:
        raise DataCheckError(f"{len(names)} series, protocol expects {cfg['dataset']['series_expected']}")
    lengths = {len(v) for v in values}
    if len(lengths) != 1 or len(set(starts)) != 1:
        bad = [nm for nm, v in zip(names, values, strict=True) if len(v) != len(values[0])]
        raise DataCheckError(f"unequal lengths or starts: {len(bad)} series differ from {names[0]} (e.g. {bad[:5]})")
    missing = [(nm, v.count("?")) for nm, v in zip(names, values, strict=True) if "?" in v]
    if missing:
        raise DataCheckError(f"missing values in {len(missing)} series, e.g. {missing[:5]}")
    y = np.asarray(values, dtype=float)
    neg = (y < 0).sum(axis=1)
    if neg.any():
        bad = [(names[i], int(neg[i])) for i in np.nonzero(neg)[0]]
        raise DataCheckError(f"negative values in {len(bad)} series, e.g. {bad[:5]}")

    span = cfg["span"]
    T_keep = span["evaluation_hours"] + span["history_before_evaluation"] + cfg["model"]["horizon_h"]
    y = y[:, -T_keep:]
    ev = y[:, -span["evaluation_hours"] - cfg["model"]["horizon_h"] :]
    zero_share = (ev == 0).mean(axis=1)
    keep = zero_share <= 0.10
    report = dict(n_series=len(names), T_file=lengths.pop(), start=starts[0], T_kept=T_keep,
                  excluded=[(names[i], float(zero_share[i])) for i in np.nonzero(~keep)[0]])
    return y[keep], [nm for nm, k in zip(names, keep, strict=True) if k], report


def study(cfg: dict | None = None) -> dict:
    cfg = cfg or protocol()
    y, names, report = load(cfg)
    mod, span = cfg["model"], cfg["span"]
    w, m, h, d = mod["window_w"], mod["season_m"], mod["horizon_h"], mod["closing_lag_d"]
    T = y.shape[1]
    start = span["history_before_evaluation"]
    stop = T
    ev = y[:, start:]
    scale = np.abs(ev[:, m:] - ev[:, :-m]).mean(axis=1)
    ks = cfg["cadences_k"]
    n_q = 4

    def summarise(r):
        lo = {t: r["loss_by_origin"][t] / scale[None, :] for t in TAUS}
        q = np.array_split(np.arange(lo[TAUS[0]].shape[0]), n_q)
        return dict(full={t: float(lo[t].mean()) for t in TAUS},
                    quarters=[{t: float(lo[t][ix].mean()) for t in TAUS} for ix in q],
                    cov={t: float(r["cov"][t].mean()) for t in TAUS})

    common = dict(start=start, stop=stop, w=w, m=m, h=h)
    out: dict = {"report": report, "n": y.shape[0]}
    out["base"] = {
        "snaive": summarise(seasonal_naive(y, **common)),
        "snaive_closed": summarise(seasonal_naive(y, **common, lag=closed_lag(m, h, d))),
        "snaive_168": summarise(seasonal_naive(y, **common, lag=168)),
    }
    out["model"], out["stability"], out["refits"] = {}, {}, {}
    for method in ("gaussian", "conformal"):
        for k in ks:
            # conformal warm-up as in the protocol: 240 origins before the evaluation collect closed errors
            r = serve_quantiles(y, k, start=start - 240, stop=stop, w=w, m=m, h=h, fit_lag=d, method=method,
                                score_from=start)
            out["model"][method, k] = summarise(r)
            out["stability"][method, k] = stability(r["rev"][0.95], scale)
            out["refits"][k] = r["n_refit"]
    out["k_max"] = {}
    for method in ("gaussian", "conformal"):
        for bn in ("snaive", "snaive_closed"):
            full = {k: out["model"][method, k]["full"] for k in ks}
            out["k_max"][method, bn] = dict(
                full={t: k_max(full, out["base"][bn]["full"], t, ks) for t in TAUS},
                quarters=[{t: k_max({k: out["model"][method, k]["quarters"][i] for k in ks},
                                    out["base"][bn]["quarters"][i], t, ks) for t in TAUS} for i in range(n_q)])
    out["illusion"] = {}
    for method in ("gaussian", "conformal"):
        for k in (24, 168):
            leak = serve_quantiles(y, k, start=start - 240, stop=stop, w=w, m=m, h=h, fit_lag=0, method=method,
                                   score_from=start)
            real = out["model"][method, k]["full"][0.95]
            lk = float((leak["loss"][0.95] / scale).mean())
            out["illusion"][method, k] = (real, lk, (real - lk) / lk)
    return out


def ratio(v: dict) -> float:
    """k_max(0.50) / k_max(0.95); nan when the model never wins at the median (0/0 or 0/x is not a ratio)."""
    if v[0.5] == 0:
        return float("nan")
    return v[0.5] / v[0.95] if v[0.95] else float("inf")


def verdict(out: dict) -> str:
    """The protocol's decision rule (conformal quantiles). As written it uses seasonal_naive_24; the closed
    lag-72 naive was added AFTER the protocol (block 5 finding) and is reported as a deviation."""
    lines = []
    for bn, label in (("snaive", "as written (seasonal_naive_24)"),
                      ("snaive_closed", "DEVIATION: closed naive lag 72, added after the protocol")):
        km = out["k_max"]["conformal", bn]
        r_full = ratio(km["full"])
        r_q = [ratio(q) for q in km["quarters"]]
        if np.isnan(r_full):
            v = "NEITHER (undefined: the model never beats the baseline at the median)"
        elif r_full >= 4 and sum(r >= 4 for r in r_q if not np.isnan(r)) >= 3:
            v = "SUPPORTED"
        elif r_full <= 1:
            v = "CONTRADICTED"
        else:
            v = "NEITHER (between the two thresholds)"
        lines.append(f"   {label}: ratio full year {r_full:.1f}, quarters {[round(r, 1) for r in r_q]} -> {v}")
    return "\n".join(lines)


def print_study(out: dict) -> None:
    rep = out["report"]
    print(f"Monash Electricity Hourly: {rep['n_series']} series x {rep['T_file']} h from {rep['start']}; "
          f"kept the last {rep['T_kept']} h; excluded {len(rep['excluded'])} series (> 10 % zeros): "
          f"{[nm for nm, _ in rep['excluded']]}")
    print(f"evaluated: {out['n']} series, 8760 origins, pinball scaled per series by the lag-24 MAE\n")
    for bn, b in out["base"].items():
        print(f"   {bn:>14s}: " + "  ".join(f"tau={t:.2f} {b['full'][t]:.4f} (cov {b['cov'][t]:.3f})" for t in TAUS))
    for method in ("gaussian", "conformal"):
        print(f"\n[{method}] scaled pinball (coverage) and gain vs naive (plan) / closed naive")
        for k in sorted({k for (mm, k) in out["model"] if mm == method}):
            r = out["model"][method, k]
            cells = [f"{r['full'][t]:.4f} ({r['cov'][t]:.3f}) "
                     f"{1 - r['full'][t] / out['base']['snaive']['full'][t]:+6.1%} "
                     f"{1 - r['full'][t] / out['base']['snaive_closed']['full'][t]:+6.1%}" for t in TAUS]
            print(f"   k={k:>4d} | " + " | ".join(cells))
    print("\nk_max per tau: full year, then the four quarters")
    for (method, bn), v in out["k_max"].items():
        print(f"   {method:>9s} vs {bn:<13s} full {[v['full'][t] for t in TAUS]}  quarters "
              f"{[[q[t] for t in TAUS] for q in v['quarters']]}")
    print("\ndecision rule of the protocol (conformal):")
    print(verdict(out))
    print("\nstability at tau = 0.95: jump p95 (% of scale), % origins > 1%")
    for (method, k), s in out["stability"].items():
        print(f"   {method:>9s} k={k:>4d}: {100 * s['jump_p95']:6.1f}  {s['pct_visible']:6.1%}")
    print("\nclosing-lag illusion at tau = 0.95")
    for (method, k), (r, lk, ill) in out["illusion"].items():
        print(f"   {method:>9s} k={k:>3d}: {r:.4f} vs {lk:.4f} -> {ill:+.1%}")


def study_cached(refresh: bool = False) -> dict:
    """The study, cached in data/results/ (gitignored; regenerable with `w44 real run`). Notebooks read this."""
    import pickle

    if RESULTS.exists() and not refresh:
        return pickle.loads(RESULTS.read_bytes())
    out = study()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_bytes(pickle.dumps(out))
    return out
