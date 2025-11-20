.PHONY: help install test lint format clean run docs

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install dependencies with Poetry
	poetry install

install-dev:  ## Install development dependencies
	poetry install --with dev
	poetry run pre-commit install

test:  ## Run tests
	poetry run pytest

test-cov:  ## Run tests with coverage report
	poetry run pytest --cov=. --cov-report=html --cov-report=term

lint:  ## Run linters (flake8, pylint, mypy)
	poetry run flake8 indicators.py strategies.py risk_manager.py metrics.py backtester.py
	poetry run pylint indicators.py strategies.py risk_manager.py metrics.py backtester.py
	poetry run mypy indicators.py strategies.py risk_manager.py metrics.py backtester.py

format:  ## Format code with black and isort
	poetry run black *.py
	poetry run isort *.py

format-check:  ## Check code formatting
	poetry run black --check *.py
	poetry run isort --check-only *.py

pre-commit:  ## Run pre-commit hooks on all files
	poetry run pre-commit run --all-files

clean:  ## Clean up generated files
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

run:  ## Run advanced backtesting
	poetry run python advanced_backtest.py

run-simple:  ## Run simple backtest
	poetry run python mono_trade.py

run-triple:  ## Run triple trade strategy
	poetry run python triple_trade.py

docs:  ## Open documentation in browser
	@echo "Opening documentation..."
	@xdg-open README.md 2>/dev/null || open README.md 2>/dev/null || echo "Please open README.md manually"

build:  ## Build package
	poetry build

publish:  ## Publish package to PyPI (requires authentication)
	poetry publish --build

version:  ## Show current version
	@poetry version

bump-patch:  ## Bump patch version (x.x.X)
	poetry version patch

bump-minor:  ## Bump minor version (x.X.0)
	poetry version minor

bump-major:  ## Bump major version (X.0.0)
	poetry version major

shell:  ## Open Poetry shell
	poetry shell

update:  ## Update dependencies
	poetry update

lock:  ## Update poetry.lock
	poetry lock --no-update

check:  ## Run all checks (format-check, lint, test)
	make format-check
	make lint
	make test
