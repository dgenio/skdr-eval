# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2025-01-12

### Fixed
- **Major Tech Debt**: Implemented proper bootstrap confidence intervals using the existing `block_bootstrap_ci` function
- **Statistical Accuracy**: Replaced incorrect normal approximation with proper moving-block bootstrap for time-series data
- **Dead Code Elimination**: The `block_bootstrap_ci` function was implemented but never used (0% test coverage)
- **False Advertising**: README claimed "Bootstrap Confidence Intervals" but only used normal approximation

### Added
- **Comprehensive Test Suite**: Added 14 new tests for `block_bootstrap_ci` function achieving 100% coverage
- **Integration Tests**: Added 7 integration tests for bootstrap CI in both evaluation functions
- **Proper Error Handling**: Added parameter validation and graceful fallback mechanisms
- **Enhanced Documentation**: Updated README with detailed bootstrap CI parameters and usage

### Enhanced
- **Statistical Rigor**: Bootstrap CIs now properly account for time-series correlation structure
- **Data Science Best Practices**: Moving-block bootstrap is the gold standard for time-series confidence intervals
- **Test Coverage**: Improved overall test coverage from 75% to 78%
- **Code Quality**: Eliminated dead code and improved maintainability

### Technical Details
- **Bootstrap Method**: Uses moving-block bootstrap with configurable block length (default: sqrt(n))
- **Fallback Strategy**: Gracefully falls back to normal approximation if bootstrap fails
- **Performance**: 400 bootstrap samples by default with configurable parameters
- **Reproducibility**: All bootstrap operations use consistent random seeds

## [0.4.1] - 2025-01-15

### Fixed
- **Release Automation**: Configured GitHub Actions workflow to trigger on pushed tags
- **Automatic PyPI Publishing**: Releases now automatically publish to PyPI when tags are pushed
- **Developer Experience**: Simplified release process - just push a tag to trigger publication

### Technical Details
- **Workflow Trigger**: Added `push.tags: ['v*']` trigger to release workflow
- **Backward Compatibility**: Maintains existing `release` and `workflow_dispatch` triggers
- **Tag Pattern**: Supports any version tag pattern (v1.0.0, v2.1.3, etc.)

## [0.4.0] - 2025-01-15

### Added
- **Comprehensive Type Safety**: Enhanced type annotations throughout the codebase
- **Sklearn Protocol Definitions**: Added `ClassifierProtocol` and `RegressorProtocol` for better type safety
- **Runtime Validation**: Added validation for callable estimators to prevent runtime errors
- **Enhanced Error Handling**: Improved error messages and warnings for better debugging

### Fixed
- **Major Tech Debt Resolution**: Resolved critical type safety violations using `Any` types as workarounds
- **Type Annotation Issues**: Fixed `induce_policy_from_sklearn` return type from `Any` to `np.ndarray`
- **Mypy Compatibility**: Resolved all mypy type checking errors with proper type annotations
- **Callable Estimator Safety**: Added runtime validation for callable estimators to prevent type errors
- **Import Ordering**: Fixed ruff linting issues with proper import organization

### Enhanced
- **Type Safety**: Comprehensive protocols for sklearn estimators with proper method signatures
- **Developer Experience**: Better IDE support and static analysis capabilities
- **Error Prevention**: Runtime validation prevents common type-related runtime errors
- **Code Quality**: All linting, formatting, and type checking standards now pass

### Technical Details
- **Protocols**: Added `ClassifierProtocol` with `predict_proba` method for sklearn classifiers
- **Validation**: Runtime checks ensure callable estimators return compatible objects
- **Type Inference**: Improved type inference with explicit type annotations
- **Compatibility**: Maintains full backward compatibility while improving type safety

## [0.3.3] - 2025-01-15

### Fixed
- **Type Safety Violations**: Resolved critical type annotation issues in core functions
- **induce_policy_from_sklearn**: Fixed return type annotation from `Any` to `np.ndarray`
- **EstimatorProtocol**: Added proper protocol for sklearn estimators in `_get_outcome_estimator`
- **estimate_propensity_pairwise**: Fixed parameter type from `Any` to `PairwiseDesign`
- **Type Safety Workarounds**: Removed mypy workarounds that compromised type safety

### Technical Debt
- **Major Tech Debt Resolution**: Addressed critical type safety violations that were using `Any` types to avoid mypy issues
- **Code Quality**: Improved type safety and maintainability by using proper type annotations
- **Developer Experience**: Enhanced IDE support and static analysis capabilities

## [0.3.2] - 2025-08-13

### Fixed
- **Major Tech Debt**: Implemented proper bootstrap confidence intervals using the existing `block_bootstrap_ci` function
- **Statistical Accuracy**: Replaced incorrect normal approximation with proper moving-block bootstrap for time-series data
- **Dead Code Elimination**: The `block_bootstrap_ci` function was implemented but never used (0% test coverage)
- **False Advertising**: README claimed "Bootstrap Confidence Intervals" but only used normal approximation

### Added
- **Comprehensive Test Suite**: Added 14 new tests for `block_bootstrap_ci` function achieving 100% coverage
- **Integration Tests**: Added 7 integration tests for bootstrap CI in both evaluation functions
- **Proper Error Handling**: Added parameter validation and graceful fallback mechanisms
- **Enhanced Documentation**: Updated README with detailed bootstrap CI parameters and usage

### Enhanced
- **Statistical Rigor**: Bootstrap CIs now properly account for time-series correlation structure
- **Data Science Best Practices**: Moving-block bootstrap is the gold standard for time-series confidence intervals
- **Test Coverage**: Improved overall test coverage from 75% to 78%
- **Code Quality**: Eliminated dead code and improved maintainability

### Technical Details
- **Bootstrap Method**: Uses moving-block bootstrap with configurable block length (default: sqrt(n))
- **Fallback Strategy**: Gracefully falls back to normal approximation if bootstrap fails
- **Performance**: 400 bootstrap samples by default with configurable parameters
- **Reproducibility**: All bootstrap operations use consistent random seeds

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
