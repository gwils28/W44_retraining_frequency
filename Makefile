.PHONY: install test test-fast lint harness gate churn freshness real serving causal notebooks lab

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

gate:           ## block 3: check the service contract (exit 1 when blocked)
	uv run w44 gate

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

notebooks:      ## re-execute every notebook in place (block 5 real data reads data/results/, run `make real` first)
	for nb in 0*/notebooks/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

lab:            ## open JupyterLab at the root
	uv run jupyter lab
