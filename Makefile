.PHONY: install test test-fast lint harness churn

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
