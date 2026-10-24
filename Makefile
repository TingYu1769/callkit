# Common tasks. `make check` must pass before opening a PR.
unexport VIRTUAL_ENV  # always use the project's own .venv, not an outer (e.g. conda) environment
.PHONY: setup fmt check lint type test

setup:            ## install dependencies from uv.lock
	uv sync

fmt:              ## format code and sort imports
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint type test  ## run before every PR: format, lint, types, offline tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

type:
	uv run basedpyright

test:             ## offline tests only (no paid API calls)
	uv run pytest -m "not network" -q
