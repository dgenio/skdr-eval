# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
