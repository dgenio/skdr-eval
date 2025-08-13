# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2025-08-13

### Fixed
- **Release Workflow Versioning**: Fixed setuptools-scm version mismatch in CI builds
- **Checkout Configuration**: Added explicit ref checkout to ensure workflow builds from exact tag
- **Version Validation**: Resolved issue where built version didn't match expected tag version
- **CI Pipeline**: Enhanced release workflow to prevent version calculation errors

### Infrastructure
- **Release Process**: Improved workflow reliability for consistent version generation
- **Quality Assurance**: Strengthened version validation in build pipeline

## [0.3.1] - 2025-08-13

### Fixed
- **CI Build Failure**: Added missing `setuptools_scm` dependency to release workflow
- **Release Pipeline**: Fixed ModuleNotFoundError that prevented v0.3.0 from building successfully
- **Package Publication**: Ensured dynamic versioning works correctly in GitHub Actions

### Infrastructure
- **Release Workflow**: Enhanced build dependencies to include all required packages for successful builds

## [0.3.0] - 2025-08-13

### Added
- **State-of-the-Art (SOTA) Development Guidelines** optimized for AI agents and human developers
- Comprehensive `DEVELOPMENT.md` with 400+ lines of AI agent-friendly development workflows
- Automated validation script (`scripts/validate_contribution.py`) for contribution quality assurance
- **Error Prevention Strategy** with comprehensive documentation and prevention mechanisms
- Branch protection setup guide (`.github/BRANCH_PROTECTION_SETUP.md`) for maintainers
- Enhanced `Makefile` with `validate` command for comprehensive contribution checking
- **CI-strict validation** that mirrors GitHub Actions behavior exactly

### Enhanced
- **CONTRIBUTING.md** with branch protection requirements and mandatory PR process
- Validation script with centralized configuration, AST-based docstring detection, and current branch display
- Pre-commit hooks integration with automated quality checks
- **Zero tolerance for CI failures** policy with preventive measures

### Infrastructure
- **Comprehensive quality gates**: linting, formatting, type checking, testing (80% coverage minimum)
- **Git Flow branching strategy** with protected main and develop branches
- **Conventional commit message format** requirements
- **AI agent-specific guidelines** with step-by-step workflows and troubleshooting
- **Enterprise-grade development practices** ensuring code movement via PRs with approvals

### Fixed
- Import order issues (PLC0415) in test files
- Code formatting consistency across all source directories
- Validation script encoding issues and Path usage improvements

## [0.2.0] - 2025-08-12

### Added
- **Pairwise evaluation system** with comprehensive autoscaling strategies
- New `PairwiseDesign` class for pairwise comparison experiments
- Multiple autoscaling algorithms: `uniform`, `proportional`, `sqrt`, `log`, `inverse`
- Choice modeling functionality with propensity score estimation
- Comprehensive test suite for pairwise evaluation features
- Example notebook demonstrating pairwise evaluation usage

### Fixed
- Resolved all mypy type annotation errors across codebase
- Fixed type incompatibilities between pandas and numpy types
- Improved type safety with proper conversions and annotations

### Infrastructure
- Enhanced pre-commit hooks configuration and installation
- Updated development workflow documentation
- Improved GitHub templates and CI workflows

## [0.1.2] - 2025-08-12

### Added
- Professional development workflow with `develop` branch
- Comprehensive contributing guidelines (`CONTRIBUTING.md`)
- Pull request and issue templates (bug report, feature request)
- Development Makefile with common tasks (check, lint, test, build, etc.)
- Comprehensive DEVELOPMENT.md guide for contributors
- Updated pre-commit configuration with latest tool versions

### Changed
- CI workflow now runs on both `main` and `develop` branches
- CI workflow accepts PRs to both `main` and `develop` branches

### Fixed
- Updated deprecated GitHub Actions to latest versions
- Resolved mypy type annotation issues for Python 3.9 compatibility
- Applied comprehensive ruff formatting to all source files
- Permanently excluded auto-generated `_version.py` from ruff checks

## [0.1.1] - 2025-08-12

### Fixed
- Fixed ruff configuration error in `.ruff.toml` (moved `line-length` to top-level)
- Resolved 257+ linting issues across the codebase
- Updated GitHub Actions workflows to use latest action versions
- Fixed deprecated `actions/upload-artifact@v3` to `v4`
- Fixed deprecated `actions/setup-python@v4` to `v5`

### Changed
- Updated type annotations to modern syntax (`dict`/`tuple` instead of `Dict`/`Tuple`)
- Applied comprehensive code style improvements
- All ruff quality checks now pass

### Added
- Comprehensive v0.1.1 release notes
- Manual PyPI upload process documentation

## [0.1.0] - 2025-01-12

### Added
- Initial release of skdr-eval library
- Core implementation of Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators
- Time-aware cross-validation with proper timestamp sorting for offline policy evaluation
- Synthetic data generation for testing and evaluation (`make_synth_logs`)
- Design matrix construction with context and action features (`build_design`)
- Propensity score fitting with time-aware calibration (`fit_propensity_timecal`)
- Outcome model fitting with cross-validation (`fit_outcome_crossfit`)
- Policy induction from sklearn models (`induce_policy_from_sklearn`)
- Bootstrap confidence intervals with moving-block bootstrap for time-series data
- Comprehensive evaluation function for sklearn models (`evaluate_sklearn_models`)
- Complete test suite with 17 tests covering all major functionality
- CI/CD workflows for automated testing and building
- Comprehensive documentation with examples and API reference
- Quickstart example demonstrating full evaluation workflow

### Features
- 🎯 **Doubly Robust Estimation**: Implements both DR and Stabilized DR (SNDR) estimators
- ⏰ **Time-Aware Evaluation**: Uses time-series splits and calibrated propensity scores
- 🔧 **Sklearn Integration**: Easy integration with scikit-learn models
- 📊 **Comprehensive Diagnostics**: ESS, match rates, propensity score analysis
- 🚀 **Production Ready**: Type-hinted, tested, and documented
- 📈 **Bootstrap Confidence Intervals**: Moving-block bootstrap for time-series data

### Technical Details
- Supports Python 3.9+
- Dependencies: numpy, pandas, scikit-learn>=1.1
- Full type hints and comprehensive error handling
- 74% test coverage
- Follows modern Python packaging standards

[0.1.0]: https://github.com/dandrsantos/skdr-eval/releases/tag/v0.1.0
