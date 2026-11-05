.PHONY: help dev test test-python bench docs build check clean release

# Prefer a local venv when present (override with PYTHON=...).
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP ?= $(PYTHON) -m pip

help:
	@echo "Targets:"
	@echo "  dev          editable install with [dev,docs]"
	@echo "  test         pytest with 100% coverage (numba when installed)"
	@echo "  test-python  same gate with PSNW_PY_BACKEND=python"
	@echo "  bench        time python / numba backends"
	@echo "  docs         build Sphinx HTML (warnings are errors)"
	@echo "  build        sdist + wheel into dist/"
	@echo "  check        twine check + reject scratch/ in sdist"
	@echo "  clean        remove build / coverage / docs artifacts"
	@echo "  release      clean test docs build check (does not upload)"

dev:
	$(PIP) install -e '.[dev,docs]'

test:
	$(PYTHON) -m pytest --cov=psnw_py --cov-fail-under=100 --cov-report=term-missing

test-python:
	PSNW_PY_BACKEND=python $(PYTHON) -m pytest --cov=psnw_py --cov-fail-under=100 --cov-report=term-missing

bench:
	$(PYTHON) bench/bench.py

docs:
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html

build:
	$(PYTHON) -m build

check:
	$(PYTHON) -m twine check dist/*
	@sdist=$$(ls -1 dist/*.tar.gz | head -n1); \
	  if tar -tzf "$$sdist" | grep -E '(^|/)scratch(/|$$)'; then \
	    echo "ERROR: scratch/ present in $$sdist"; exit 1; \
	  fi

clean:
	rm -rf dist build *.egg-info .coverage htmlcov docs/_build .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

release: clean test docs build check
	@echo "Artifacts ready in dist/. Tag v* to publish via CI."
