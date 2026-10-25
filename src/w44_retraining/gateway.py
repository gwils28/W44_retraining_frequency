"""Block 7: the forecast gateway layer that the Open Inference Protocol cannot carry.

OIP V2 standardises tensors, not time: no `as_of`, no horizon, no quantile levels. The gateway adds them as
server-specific `parameters` (declared non-portable in `configs/forecast_service.yaml`) and owns the cache key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# keys we put in OIP `parameters`; the spec defines none, so these are ours and not portable across servers
NON_PORTABLE_PARAMETERS = ("as_of", "horizon", "tau", "training_asof", "feature_version")


@dataclass(frozen=True)
class ForecastRequest:
    series_id: str
    as_of: str               # ISO timestamp of the newest closed observation the forecast may use
    horizon: int
    taus: tuple[float, ...]
    model_version: str
    feature_version: str
    training_asof: str       # end of the training window of the served model

    def to_oip(self, data: list[float]) -> dict:
        """OIP V2 inference request; the time semantics go in `parameters`."""
        return {
            "id": f"{self.series_id}@{self.as_of}",
            "parameters": {"as_of": self.as_of, "horizon": self.horizon, "tau": list(self.taus),
                           "training_asof": self.training_asof, "feature_version": self.feature_version},
            "inputs": [{"name": "history", "shape": [1, len(data)], "datatype": "FP32", "data": data}],
        }


def cache_key(r: ForecastRequest) -> tuple:
    """Everything that changes the answer. Without as_of, two calls a month apart share an entry; without
    training_asof / model_version, a retrained model keeps serving the old forecast."""
    d = asdict(r)
    return (d["series_id"], d["as_of"], d["horizon"], d["taus"], d["model_version"], d["feature_version"],
            d["training_asof"])


def narrow_cache_key(r: ForecastRequest) -> tuple:
    """The usual key: (series, horizon). A bug that makes the system calmer: nobody sees a stale model."""
    return (r.series_id, r.horizon)
