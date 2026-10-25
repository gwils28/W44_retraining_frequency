# Block 7: open-source model serving, and the protocol that cannot say "when"

Code: `src/w44_retraining/gateway.py`. Tests: `tests/unit/test_gateway.py`.

## The stack sheet (PyPI JSON API, 2026-10-25)

| Criterion | `mlserver` (Seldon) | `bentoml` | `kserve` |
|---|---|---|---|
| Latest stable | 1.7.1 | 1.4.40 | 0.21.0 |
| Released | 6 Jun 2025 (16 months ago) | 23 Oct 2026 | 25 Sep 2026 |
| Since then | only 1.7.2rc1 (16 Dec 2025) | 1.4.39 was 7 May 2026: a 5.5-month gap before 1.4.40 | 0.20.0 on 6 Aug 2026 |
| Licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Requires-Python | `>=3.9,<3.13` | `>=3.9` | `>=3.10,<3.14` |
| Role | reference implementation of the Open Inference Protocol V2 | general serving runtime, own protocol | Kubernetes serving, implements OIP V2 |

`mlserver` fails the "still maintained" criterion, and it is the implementation the whole ecosystem cites for the
protocol. That is a reason not to build on that implementation, not to drop the protocol: `kserve` (released a
month ago, Python up to 3.13) and Triton implement it too.

## The three 5-minute checks, done

**1. The Python bound on your own stack.** On Python 3.13 (this repo's interpreter), measured:

- `pip install mlserver` does **not fail**: it skips every release that declares `<3.13` and **silently installs
  `mlserver 1.3.5`**, released on 6 Jul 2023, the last version without a `Requires-Python`. You get a release more
  than three years old, without an error.
- `uv pip install mlserver` **installs 1.7.1** and `import mlserver` works: uv does not enforce the upper bound of
  `Requires-Python`. Whether it runs correctly on 3.13 is untested by its authors, by their own declaration.

So the right check is not "does it install" but "which version did I get, and is my interpreter inside its declared
bounds": pin `mlserver==1.7.1` and Python 3.12, or choose another runtime.

**2. The `parameters` layer, written and declared non-portable.** `ForecastRequest.to_oip()` puts `as_of`,
`horizon`, `tau`, `training_asof` and `feature_version` in the OIP `parameters` object. The test checks that the
contract lists exactly these keys under `service.gateway_adds`, next to `protocol_cannot_express`.

**3. The cache key.** `cache_key()` includes `as_of`, `model_version`, `feature_version` and `training_asof`; the
usual `(series_id, horizon)` key returns the same entry a month later and after a retrain (tested). A cache that
is too wide makes the system look calmer: nobody sees that the model was never retrained. Writing the test exposed
that the contract's own `serving.cache_key` lacked `training_asof`; it is fixed.

## Position

"We serve through the Open Inference Protocol, so it is standardised." The protocol standardises tensor shapes
and nothing about time: no `as_of`, no horizon, no quantile levels, and one escape hatch, `parameters`, of which
the spec says "Currently no parameters are defined". A protocol that cannot state when a forecast was made is a
tensor standard used to carry forecasts. Its practical consequence: the cache key, hence the cache's behaviour, is
outside the contract, and has to be written in the gateway.

## What I correct relative to the plan

1. **"mlserver does not install on Python 3.13"** is not what happens. pip silently installs a 2023 release
   (1.3.5); uv installs 1.7.1 despite the bound. Both outcomes are worse than an install error, because neither
   says anything.
2. **bentoml's "10 releases in 11 months"** hides a 5.5-month gap (May to October 2026). It is maintained, with
   irregular cadence.
3. **`kserve` was missing from the sheet.** It is the maintained OIP V2 implementation (0.21.0, September 2026) and
   the natural replacement when the protocol must be kept.
4. **The MCP comparison** ("an agent tool protocol published in July 2026 makes `ttlMs` and `cacheScope`
   mandatory") comes from W43 and was not re-checked this week.
