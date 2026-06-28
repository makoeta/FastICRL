.PHONY: help install test test-v build format lint clean

PYTHON := .venv/bin/python
UV     := uv

help:
	@echo "install   install dependencies via uv"
	@echo "test      run the test suite"
	@echo "test-v    run the test suite (verbose)"
	@echo "build     build distribution packages"
	@echo "format    format code with black"
	@echo "lint      check formatting (black --check)"
	@echo "clean     remove build artifacts and caches"

install:
	$(UV) sync

test:
	$(PYTHON) -m pytest test/

test-v:
	$(PYTHON) -m pytest test/ -v

build:
	$(UV) build

format:
	$(PYTHON) -m black src/ test/

lint:
	$(PYTHON) -m black --check src/ test/

clean:
	rm -rf dist/ .pytest_cache/
	find . -path ./.venv -prune -o -name "__pycache__" -type d -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -name "*.pyc" -print -delete
