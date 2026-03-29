.PHONY: help install test lint format check run clean

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@echo '  install   Install dependencies via uv'
	@echo '  run       Start the proxy server'
	@echo '  test      Run the test suite'
	@echo '  lint      Run ruff and mypy'
	@echo '  format    Format code with ruff'
	@echo '  check     Run lint + test'
	@echo '  clean     Remove build/cache artifacts'

install:
	uv sync

run:
	uv run python -m codex_proxy

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run mypy src/ || true

format:
	uv run ruff format src/ tests/

check: lint test

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
