# Makefile for skdr-eval development

.PHONY: help install install-dev clean lint format typecheck test test-cov build docs docs-serve check validate smoke coverage-sim notebooks use-cases known-failures all

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  clean        Clean build artifacts and cache files"
	@echo "  lint         Run linting checks with ruff"
	@echo "  format       Format code with ruff"
	@echo "  typecheck    Run type checking with mypy"
	@echo "  test         Run tests with pytest"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  build        Build package for distribution"
	@echo "  docs         Build the MkDocs site (--strict; needs [docs] extra)"
	@echo "  docs-serve   Serve the docs site locally with live reload"
	@echo "  check        Run all quality checks (lint + typecheck + test)"
	@echo "  validate     Run comprehensive contribution validation (AI agent friendly)"
	@echo "  coverage-sim Run moving-block bootstrap coverage simulation (issues #81 #62)"
	@echo "  smoke        Run examples/preflight.py and examples/quickstart.py"
	@echo "  notebooks    Execute examples/notebooks/ via nbmake (needs [dev] extra)"
	@echo "  use-cases    Run all examples/use_cases/*.py scripts"
	@echo "  known-failures Run examples/known_failures/*.py demos (#134)"
	@echo "  all          Run clean + check + build"

# Installation targets
install:
	pip install .

install-dev:
	pip install -e .[dev]

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Code quality targets
lint:
	ruff check src/ tests/ examples/

format:
	ruff format src/ tests/ examples/

typecheck:
	mypy src/skdr_eval/

# Testing targets
test:
	pytest -v

test-cov:
	pytest -v --cov=skdr_eval --cov-report=html --cov-report=term-missing --cov-report=xml

# Build targets
build: clean
	python -m build

# Documentation targets (MkDocs Material + mkdocstrings). Requires [docs] extra.
docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

# Validation target for AI agents and contributors
validate:
	python scripts/validate_contribution.py

# Coverage simulation for moving-block bootstrap calibration (#81 #62)
# Uses n_reps=50 for speed in CI; increase to 500 for thorough local checks.
coverage-sim:
	python -m skdr_eval._simulation --dgp iid --n_reps 50
	python -m skdr_eval._simulation --dgp ar1 --n_reps 50
	python -m skdr_eval._simulation --dgp seasonal --n_reps 50

# Examples smoke (mirrors CI examples-smoke job)
smoke:
	python examples/preflight.py
	python examples/quickstart.py

# CLI smoke (mirrors CI cli-smoke job). Requires the [cli] extra.
cli-smoke:
	skdr-eval version
	python -c "import skdr_eval, tempfile, pathlib; \
		td = pathlib.Path(tempfile.mkdtemp()); \
		logs, _, _ = skdr_eval.make_synth_logs(n=800, n_ops=3, seed=0); \
		p = td / 'logs.parquet'; logs.to_parquet(p); print(p)" \
		| tail -1 | xargs -I {} skdr-eval doctor {} --json

# Notebook smoke (mirrors CI notebooks-smoke job; requires [dev] for nbmake)
notebooks:
	python -m pytest --nbmake examples/notebooks/ \
		--nbmake-timeout=300 \
		--override-ini="addopts=" \
		-p no:cacheprovider

# Use-case smoke (mirrors CI use-cases-smoke job)
use-cases:
	python examples/use_cases/01_ecommerce_ranking.py
	python examples/use_cases/02_ad_targeting.py
	python examples/use_cases/03_healthcare_cate.py
	python examples/use_cases/04_call_routing.py
	python examples/use_cases/05_logs_to_experiment_card.py
	python examples/use_cases/06_agent_routing_policy.py

# Known-failure-mode tutorials (#134). These scripts intentionally
# produce ``support_health=high_risk`` so a newcomer can see what a bad
# offline evaluation looks like. They are NOT part of the use-case
# gallery (which is the happy-path showcase) and NOT run by the CI
# smoke job — run on-demand.
known-failures:
	python examples/known_failures/poor_overlap.py
	python examples/known_failures/misspecified_q.py
	python examples/known_failures/non_stationary.py

# Composite targets
check: lint typecheck test smoke

all: clean check build

# Development workflow helpers
dev-setup: install-dev
	@echo "Development environment set up successfully!"
	@echo "Run 'make check' to verify everything works."

release-check: clean check build
	twine check dist/*
	@echo "Release artifacts ready for upload!"

# Git workflow helpers
feature:
	@read -p "Enter feature name: " feature_name; \
	git checkout main && \
	git pull origin main && \
	git checkout -b feature/$$feature_name

hotfix:
	@read -p "Enter hotfix name: " hotfix_name; \
	git checkout main && \
	git pull origin main && \
	git checkout -b hotfix/$$hotfix_name

# CI simulation (run what CI runs)
ci: clean lint typecheck test-cov build
	@echo "All CI checks passed locally!"
