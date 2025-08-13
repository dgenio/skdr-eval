# Makefile for skdr-eval development

.PHONY: help install install-dev clean lint format typecheck test test-cov build docs check validate all

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
	@echo "  docs         Generate documentation (future)"
	@echo "  check        Run all quality checks (lint + typecheck + test)"
	@echo "  validate     Run comprehensive contribution validation (AI agent friendly)"
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

# Documentation targets (placeholder for future)
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Future: sphinx-build -b html docs/ docs/_build/"

# Validation target for AI agents and contributors
validate:
	python scripts/validate_contribution.py

# Composite targets
check: lint typecheck test

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
	git checkout develop && \
	git pull origin develop && \
	git checkout -b feature/$$feature_name

hotfix:
	@read -p "Enter hotfix name: " hotfix_name; \
	git checkout main && \
	git pull origin main && \
	git checkout -b hotfix/$$hotfix_name

# CI simulation (run what CI runs)
ci: clean lint typecheck test-cov build
	@echo "All CI checks passed locally!"
