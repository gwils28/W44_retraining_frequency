.PHONY: install test test-fast lint harness churn freshness real serving causal

install:        ## create .venv and install the project + dev/notebooks groups (uv.lock)
	uv sync

test:           ## all tests (unit + integration)
	uv run pytest

test-fast:      ## unit tests only
	uv run pytest -m "not slow"

lint:
	uv run ruff check src tests

harness:        ## block 2: retraining-policy bench
	uv run w44 harness

churn:          ## block 4: stability in closed form
	uv run w44 churn

freshness:      ## block 5: cadence per served quantile, calibrated DGP
	uv run w44 freshness

real:           ## block 5: same question on Monash Electricity Hourly (download once, then run)
	uv run w44 real download
	uv run w44 real run

serving:        ## block 6: TSFM size by serving cost, licence gate
	uv run w44 serving

causal:         ## block 8: retraining as an intervention
	uv run w44 causal
