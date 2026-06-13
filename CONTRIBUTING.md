# Contributing to skdr-eval

Thank you for your interest in contributing to skdr-eval! This document outlines our development workflow and contribution guidelines.

> **🤖 For AI Agents**: See [DEVELOPMENT.md](./DEVELOPMENT.md) for detailed AI-optimized guidelines with step-by-step instructions, common patterns, and troubleshooting tips.

## 📖 Documentation Overview

- **[CONTRIBUTING.md](./CONTRIBUTING.md)** (this file): High-level workflow and guidelines
- **[DEVELOPMENT.md](./DEVELOPMENT.md)**: Detailed AI agent-friendly development guide
- **[README.md](./README.md)**: Project overview and usage examples
- **[CHANGELOG.md](./CHANGELOG.md)**: Version history and changes

## 🌟 Development Workflow

We follow **GitHub Flow**: short-lived feature branches off `main`, merged back via PR.

### Branch Structure
- **`main`**: Production-ready code, always deployable, protected branch
- **`feature/*`**: Short-lived feature development branches
- **`hotfix/*`**: Critical fixes for production
- **`release/*`**: Optional release preparation branches (cut from `main`)

### Workflow Steps

1. **Create Feature Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Development**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all checks pass locally

3. **Pre-commit Checks**
   ```bash
   # Run linting
   ruff check src/ tests/ examples/
   ruff format src/ tests/ examples/

   # Run type checking
   mypy src/skdr_eval/

   # Run tests
   pytest -v --cov=skdr_eval
   ```

4. **Submit Pull Request**
   - Push feature branch to origin
   - Create PR against `main` branch
   - Fill out PR template completely
   - Wait for CI checks and code review

5. **Code Review**
   - Address reviewer feedback
   - Ensure CI passes
   - Squash commits if requested

6. **Merge**
   - PR merged into `main` by maintainer
   - Feature branch deleted
   - Tagged releases cut from `main`

## 🔒 Branch Protection & CI Requirements

### Mandatory PR Process
**ALL code movement between branches MUST go through Pull Requests with:**

- ✅ **CI Pipeline Success**: All automated checks must pass
  - Linting (ruff check)
  - Formatting (ruff format --check)
  - Type checking (mypy)
  - Tests (pytest with ≥80% coverage)
  - Multi-Python version compatibility (3.11-3.14)

- ✅ **Required Approvals**:
  - `main` branch: **1 maintainer approval** required
  - No self-approvals allowed

- ✅ **Branch Status**:
  - Branch must be up-to-date with target branch
  - No merge conflicts
  - All conversations resolved

### Protected Branches
- **`main`**: Production branch - requires 1 approval + CI pass
- **Direct pushes are DISABLED** for protected branches

### CI Failure Policy
- **Zero tolerance**: PRs with failing CI cannot be merged
- **Auto-block**: GitHub automatically prevents merge until CI passes
- **Coverage enforcement**: PRs that reduce coverage below 80% are rejected

## 📋 Code Quality Standards

### Linting & Formatting
- **Ruff**: Code linting and formatting
- **mypy**: Static type checking
- **Line length**: 88 characters
- **Import sorting**: isort-style

### Testing
- **pytest**: Testing framework
- **Coverage**: Minimum 80% code coverage
- **Test types**: Unit tests, integration tests, smoke tests
- **Test naming**: `test_*.py` files

### Documentation
- **Docstrings**: Google-style docstrings for all public APIs
- **Type hints**: All functions must have type annotations
- **README**: Keep examples up-to-date
- **CHANGELOG**: Document all changes
- **Public API**: Any name added to `skdr_eval.__all__` must also be recorded
  in [`docs/api-stability.md`](docs/api-stability.md) — a test enforces this.

## 🧭 Paths to contribute

- **Add an estimator** (the highest-value research contribution): start with
  the [architecture tour](docs/architecture.md), then follow the
  [write-your-own-estimator guide](docs/extending/add-an-estimator.md). A new
  estimator needs a simulation proof under `tests/sim_studies/`.
- **Improve docs / examples**: docs live in [`docs/`](docs/); runnable
  examples in [`examples/`](examples/).
- **Triage good-first-issues**: see the issue tracker labels.

## 🔍 Code Review Guidelines

### For Contributors
- **Small PRs**: Keep changes focused and reviewable
- **Clear descriptions**: Explain what and why
- **Tests included**: New features need tests
- **Documentation**: Update docs for API changes

### For Reviewers
- **Be constructive**: Suggest improvements, don't just criticize
- **Check functionality**: Does it work as intended?
- **Verify tests**: Are edge cases covered?
- **Consider maintainability**: Is the code readable and maintainable?

## 🚀 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps
1. Create `release/vX.Y.Z` branch from `main`
2. Update version numbers and CHANGELOG
3. Create PR: `release/vX.Y.Z` → `main`
4. After merge: Tag release and publish to PyPI

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- Git
- GitHub account

### Local Setup
```bash
# Clone repository
git clone https://github.com/dgenio/skdr-eval.git
cd skdr-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Development Commands
```bash
# Run all checks
make check          # or: ruff check src/ tests/ examples/

# Format code
make format         # or: ruff format src/ tests/ examples/

# Run tests
make test           # or: pytest -v --cov=skdr_eval

# Type checking
make typecheck      # or: mypy src/skdr_eval/

# Build package
make build          # or: python -m build
```

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples
```
feat(core): add MAGIC estimator implementation
fix(synth): handle edge case in synthetic data generation
docs(readme): update installation instructions
test(core): add integration tests for DR estimation
```

## 🐛 Issue Reporting

### Bug Reports
- Use the bug report template
- Include minimal reproducible example
- Specify Python version and dependencies
- Include error messages and stack traces

### Feature Requests
- Use the feature request template
- Explain the use case and motivation
- Provide examples of desired API
- Consider implementation complexity

## 📚 Documentation

### API Documentation
- All public functions/classes need docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions raised

### Examples
- Keep `examples/` directory up-to-date
- Add new examples for major features
- Ensure examples run without errors
- Include explanatory comments

## 🤝 Community Guidelines

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome contributors of all backgrounds
- **Be patient**: Help newcomers learn our workflow
- **Be collaborative**: Work together to improve the project

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Ask questions in PR comments
- **Documentation**: Check README and API docs first

---

Thank you for contributing to skdr-eval! Your contributions help make offline policy evaluation more accessible to the research community. 🚀
