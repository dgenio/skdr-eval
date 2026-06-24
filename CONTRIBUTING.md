# Contributing to skdr-eval

Thank you for your interest in contributing to skdr-eval! This document outlines our development workflow and contribution guidelines.

> **🤖 For AI Agents**: See [DEVELOPMENT.md](./DEVELOPMENT.md) for detailed AI-optimized guidelines with step-by-step instructions, common patterns, and troubleshooting tips.

## 📖 Documentation Overview

- **[CONTRIBUTING.md](./CONTRIBUTING.md)** (this file): High-level workflow and guidelines
- **[DEVELOPMENT.md](./DEVELOPMENT.md)**: Detailed AI agent-friendly development guide
- **[README.md](./README.md)**: Project overview and usage examples
- **[CHANGELOG.md](./CHANGELOG.md)**: Version history and changes

## 🌟 Development Workflow

Every change lands on `main` through a pull request. `main` is the only
long-lived branch; all work happens on short-lived topic branches that are
deleted after merge.

### Branch Structure
- **`main`**: the only long-lived branch — always releasable, protected, no
  direct pushes.
- **Topic branches**: short-lived, branched from `main` and merged back via PR.
  Most work here is done by coding agents, so branches are usually namespaced by
  tool — `claude/*`, `copilot/*`, `cursor/*` — but any short descriptive branch
  off `main` is fine for human contributors.

> Earlier revisions of this guide described a `develop` branch and
> `feature/*`/`hotfix/*`/`release/*` conventions. The repo no longer uses those:
> branch off `main` and open a PR.

### Workflow Steps

1. **Create a topic branch off `main`**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b your-topic-branch   # e.g. fix-elig-mask-coercion
   ```

2. **Develop**
   - Write code following our style guidelines and add tests for new behaviour.
   - When behaviour changes, update `examples/` first, then docs (see AGENTS.md).
   - Add a **changelog fragment** under `changelog.d/` (`<issue>.<type>.md`) —
     do not edit `CHANGELOG.md` directly (see
     [`changelog.d/README.md`](changelog.d/README.md)).

3. **Run the checks locally**
   ```bash
   make check       # fast inner loop: lint + typecheck + test + smoke
   make ci-local    # CI-faithful pre-PR pass: everything ci.yml runs on one
                    # interpreter, and it prints what stays CI-only
   ```
   `ruff` and `mypy` are version-pinned (`pyproject.toml` /
   `.pre-commit-config.yaml`) so local lint/type results match CI exactly.

4. **Submit the pull request**
   - Push the branch and open a PR against `main`; fill out the template.
   - CI must pass and review threads must be resolved before a maintainer merges.

### Review
PRs are reviewed before merge — frequently with GitHub Copilot's automated
review in addition to a maintainer. Address the threads, keep CI green, and a
maintainer merges. No self-merges to `main`.

## 🔒 Branch Protection & CI Requirements

### Mandatory PR Process
**ALL code movement between branches MUST go through Pull Requests with:**

- ✅ **CI Pipeline Success**: All automated checks must pass
  - Linting (ruff check)
  - Formatting (ruff format --check)
  - Type checking (mypy)
  - Tests (pytest with ≥80% coverage)
  - Multi-Python version compatibility (3.10-3.14)
  - Minimum-dependency floors (`floor-deps` job) and a strict docs build

- ✅ **Required Approvals**:
  - `main` branch: review before merge (maintainer and/or Copilot)
  - No self-merges

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
- **CHANGELOG**: Don't edit `CHANGELOG.md` directly — add a fragment under
  `changelog.d/` (`<issue>.<type>.md`); it is compiled at release time.
- **Public API**: Any name added to `skdr_eval.__all__` must also be recorded
  in [`docs/api-stability.md`](docs/api-stability.md) — a test enforces this.
  Keep `__all__` in its sorted order (constants first, then the rest); a test
  enforces that too.

## 🧭 Paths to contribute

- **Add an estimator** (the highest-value research contribution): start with
  the [architecture tour](docs/architecture.md), then follow the
  [write-your-own-estimator guide](docs/extending/add-an-estimator.md). A new
  estimator needs a simulation proof under `tests/sim_studies/`.
- **Improve docs / examples**: docs live in [`docs/`](docs/); runnable
  examples in [`examples/`](examples/).
- **Triage good-first-issues**: see the issue tracker labels.

## 🔬 Statistical changes (extra bar)

`skdr-eval`'s value is statistical correctness, so changes to evaluation,
estimator, or gating logic clear a higher bar than ordinary code. This encodes
the rules from [`docs/agent-context/review-checklist.md`](docs/agent-context/review-checklist.md)
and [`invariants.md`](docs/agent-context/invariants.md) at the point of
contribution. If your PR touches that logic:

- [ ] **Simulation proof** — add/extend a test under `tests/sim_studies/` that
      recovers a known ground-truth parameter (template below). A change to the
      math without a recovery proof will not be merged.
- [ ] **Default flips** — any changed default on a statistical entry point gets
      a `changed` changelog fragment that names the old and new behaviour.
- [ ] **Invariants** — re-check the `treatment` vs `policy` vocabulary and the
      documented invariants; do not "simplify" math without a proof.

### Simulation-proof template

Copy this into a `tests/sim_studies/test_<thing>_recovery.py` and adapt the DGP.
The shape is always: **define a data-generating process with a known target →
run the estimator → assert it recovers the target within Monte-Carlo error.**

```python
"""Simulation proof: <estimator/quantity> recovers its known ground truth."""

import numpy as np


def test_<thing>_recovers_known_value() -> None:
    rng = np.random.default_rng(0)  # fixed seed → deterministic CI

    # 1. Data-generating process with an ANALYTICALLY KNOWN target.
    #    Build logs where the true target-policy value is computable in closed
    #    form (or by a large-sample oracle), independent of the estimator.
    true_value = ...          # the ground truth you will recover
    logs = ...                # synthetic logs for this DGP

    # 2. Run the estimator under test on the logs.
    estimate = ...            # e.g. a DR/SNDR V_hat from the public API

    # 3. Recovery assertion: the estimate must match the truth within a tight,
    #    justified tolerance (Monte-Carlo SE), NOT a loose "close enough".
    assert abs(estimate - true_value) < tol, (estimate, true_value)
```

See `tests/sim_studies/test_policy_value_recovery.py` and the others in that
directory for worked examples, and `make coverage-sim` for the bootstrap
coverage simulations.

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

The version is derived from the Git tag by `setuptools_scm` — there is no
version string to bump by hand.

### Release Steps
1. From an up-to-date `main`, compile the changelog fragments:
   `make changelog VERSION=X.Y.Z` (renders `changelog.d/` into `CHANGELOG.md`
   and deletes the fragments). Commit the result via a PR.
2. Update the citation metadata version (`CITATION.cff`, `CITATION.bib`, README
   citation block); `make citation-check` must pass.
3. After merge, tag `vX.Y.Z` on `main`. The tag push triggers `release.yml`,
   which builds, publishes to PyPI, and creates the GitHub Release.

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+ (CI tests 3.10–3.14)
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
