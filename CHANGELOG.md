# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Per-decision V̂ contributions on the artifact ([#92]).** `evaluate_sklearn_models` and `evaluate_pairwise_models` accept `keep_contributions=True` (and a `max_kept_contributions=100_000_000` memory guard); when set, each `DRResult` carries a `contributions` dict (`decision_id, q_pi, q_hat, weight, reward, contribution_to_V`) and `EvaluationArtifact.contributions(model, *, estimator="DR", top_k=None)` returns a tidy DataFrame. By construction `contribution_to_V.mean() == V_hat` to float64 precision for both DR and SNDR — DR uses `q_pi + w·(Y-q_hat)`; SNDR rescales the residual term by `n / Σw` so the per-decision values average to the ratio estimator. `ci_bootstrap=True` implicitly turns capture on (the bootstrap path already needed the same arrays). The stakeholder card now includes top-5 contributors / bottom-5 detractors when contributions are present.
- **Public preflight validators** — new `skdr_eval.validate_logs(logs, *, cli_pref, st_pref, strict)` and `skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, *, metric_col, ...)` raise typed `DataValidationError` / `InsufficientDataError` on schema problems before evaluation begins. Strict mode adds monotonic-timestamp and elig-mask sanity checks. ([#24])
- **`skdr_eval.get_capabilities()`** — side-effect-free detection of optional extras (`viz`, `speed`). Returns booleans plus a `missing_extras` list pointing at the install command needed to enable each disabled capability. ([#26])
- **Temporal split controls** — `gap`, `test_size`, and `max_train_size` keyword-only arguments are now plumbed through `fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`, `evaluate_sklearn_models`, and `evaluate_pairwise_models`. The new default is `gap=1` (conservative adjacent-row leakage guard). ([#29])
- **`examples/preflight.py`** — runnable preflight script: capability dump + log + pairwise schema validation.

### Changed
- **Default fold gap is now `gap=1`** on every entry point that takes time-series CV (`fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`, `evaluate_sklearn_models`, `evaluate_pairwise_models`). Prior versions used sklearn's `TimeSeriesSplit(n_splits=...)` default of `gap=0`. The new default is a conservative adjacent-row leakage guard; it shifts fold boundaries by one sample and therefore changes DR/SNDR estimates relative to prior baselines on the same data and seed. To restore the pre-PR fold layout, pass `gap=0` explicitly. The constant-outcome identity simulation (`tests/test_temporal_split_controls.py`) still recovers the ground-truth value under both defaults; the change is a numerical-behavior shift, not a statistical-correctness fix.
- **GitHub Flow.** Dropped the `develop` integration branch. All feature branches now branch off and merge back to `main`. CI no longer triggers on `develop`; branch-protection docs, `CONTRIBUTING.md`, `DEVELOPMENT.md`, and `scripts/validate_contribution.py` updated accordingly. ([#54])
- **Python matrix tightened.** `requires-python = ">=3.11"`. Python 3.10 dropped from CI; ruff `target-version` bumped to `py311`, mypy `python_version` to `3.11`, black `target-version` to `py311`.
- **CI smoke jobs.** New `examples-smoke` job runs `examples/preflight.py` and `examples/quickstart.py` under the default install; new `viz-extra-smoke` job installs `.[viz]` and asserts the `viz` capability is enabled. Mirrored locally as `make smoke`. ([#25])
- **`matplotlib` moved from mandatory dependencies to the `[viz]` optional extra.** It was already wrapped in `try/except ImportError` at every import site, so the extra now reflects reality. Install with `pip install 'skdr-eval[viz]'` for plotting helpers and inline sensitivity sparklines.
- **Breaking: `induce_policy_from_sklearn` dropped the unused `idx` parameter.** Callers should remove the final `eval_design.idx` argument. The function is now fully vectorized — one `model.predict` call per invocation instead of `O(n_samples × n_eligible_ops)` calls. ([#46])
- **`induce_policy_stream_topk` is now fully day-vectorized.** The surrogate runs once per client-chunk (was once per client) and the function accepts a new `chunk_pairs` parameter controlling the per-day client-axis batch size. Output policies are unchanged on fixed seeds. ([#61], [#63])
- **`induce_policy(strategy="stream_topk", chunk_pairs=...)` is no longer a no-op.** The value is now forwarded into the streaming top-K loop and caps the size of the per-chunk feature matrix to `chunk_pairs * 4 bytes * n_features`. ([#61])

### Fixed
- **`induce_policy_stream_topk` now raises `DataValidationError` on duplicate `operator_id` rows within a day.** The day-vectorized eligibility mask uses a `dict[operator_id -> position]`, which would otherwise silently drop earlier duplicate positions (last-write-wins) and diverge from the prior `isin(...)` semantics. Surface the ambiguity instead of papering over it, consistent with the fail-loud bias in `docs/agent-context/invariants.md`. ([#66] review)

### Tests
- New parity and call-count tests for both rewrites: `test_induce_policy_from_sklearn_vectorized_matches_scalar_reference`, `test_induce_policy_from_sklearn_issues_single_predict_call`, `test_stream_topk_surrogate_predict_call_count_per_day`, `test_stream_topk_chunk_pairs_controls_batching_and_preserves_policy`, `test_stream_topk_chunk_pairs_forwarded_through_induce_policy`.
- New `test_stream_topk_duplicate_operator_ids_per_day_raises` covering the fail-loud precondition added to `_build_day_elig_mask` (PR #66 review).

[#24]: https://github.com/dgenio/skdr-eval/issues/24
[#25]: https://github.com/dgenio/skdr-eval/issues/25
[#26]: https://github.com/dgenio/skdr-eval/issues/26
[#29]: https://github.com/dgenio/skdr-eval/issues/29
[#46]: https://github.com/dgenio/skdr-eval/issues/46
[#54]: https://github.com/dgenio/skdr-eval/issues/54
[#61]: https://github.com/dgenio/skdr-eval/issues/61
[#63]: https://github.com/dgenio/skdr-eval/issues/63
[#66]: https://github.com/dgenio/skdr-eval/pull/66
[#92]: https://github.com/dgenio/skdr-eval/issues/92

## [0.6.0] - 2026-05-12

### Removed
- **Breaking: 2-tuple return from `evaluate_sklearn_models` / `evaluate_pairwise_models`.** Both top-level entry points now return a single `EvaluationArtifact` instead of `(report, detailed_results)`. Migrate by unpacking `artifact.report` and `artifact.detailed`. ([#22], [#23], [#27], [#28], [#30])

### Added
- **`EvaluationArtifact` (new in `skdr_eval.reporting`)** — bundled return type carrying `report`, `detailed`, `warnings`, `sensitivity`, `diagnostics`, and `metadata`. ([#22], [#23], [#27])
- **Support-health warnings** ([#22]) — `support_health` and `diagnostic_warnings` columns on every report, plus a dedicated `artifact.warnings` DataFrame. Severity is one of `"ok"`, `"caution"`, `"high_risk"`, computed from `LOW_ESS`, `EXTREME_CLIP`, `LOW_MATCH_RATE`, and `POOR_OVERLAP` codes. Defaults are grounded in Owen (2013) and Kang & Schafer (2007); tune via `support_thresholds=SupportHealthThresholds(...)`.
- **First-class propensity diagnostics** ([#23]) — `artifact.diagnostics[model]` exposes the existing `PropensityDiagnostics` from every evaluation run with no extra call.
- **Clip-grid sensitivity** ([#27]) — `artifact.sensitivity` summarizes per-(model, estimator) `V_min`/`V_max`/`V_range`, the MSE-argmin clip, DR/SNDR agreement at the chosen clip, and a `stable` flag. This is a heuristic stability indicator; CI-overlap variants are tracked in #62.
- **JSON & HTML export** ([#28]) — `EvaluationArtifact.to_json`, `to_html`, and `export(formats=...)`. Versioned schema (`SCHEMA_VERSION = "1.0.0"`) backed by Pydantic v2. Top-level `skdr_eval.export_results` and `skdr_eval.load_artifact_json` helpers.
- **Stakeholder evaluation cards** ([#30]) — `EvaluationArtifact.card(model_name)` and the top-level `skdr_eval.render_evaluation_card` render a single-model HTML card with headline `V̂`, CI, trust banner, interpretation, inline clip-sensitivity sparkline, all-estimator table, and propensity diagnostics. Use `EvaluationArtifact.save_card(path, model_name)` to write to disk.
- **New public symbols** exported from `skdr_eval`: `EvaluationArtifact`, `SupportHealthThresholds`, `ArtifactSchema`, `SCHEMA_VERSION`, `attach_warnings`, `build_evaluation_artifact`, `summarize_sensitivity`, `render_evaluation_card`, `export_results`, `load_artifact_json`.

### Changed
- **`evaluate_sklearn_models` and `evaluate_pairwise_models` accept `support_thresholds=SupportHealthThresholds(...)`** to customize warning sensitivity.
- **`jinja2>=3.1`, `pydantic>=2.0`, `matplotlib>=3.5` are now required dependencies.** `matplotlib` was previously optional (`[viz]` extra) and is now required for card rendering; it remains available via `[viz]` for back-compat with users pinning the extra.

### Statistical scope (read me)
- This release **does not change** the DR/SNDR estimator math, the influence-function variance formulas, the bootstrap, or the `q_hat` shape. The warnings, sensitivity, and card outputs read the existing per-clip grid and per-decision diagnostics. Known limitations of the underlying metrics remain tracked in #58, #60, and #62.

### Tests
- New `tests/test_reporting_artifact.py` (31 tests) covering artifact construction, warning thresholds (healthy, low ESS caution / high-risk escalation, extreme clip, low match rate, poor overlap), sensitivity invariants, JSON round-trip via Pydantic, HTML rendering, card rendering, and pairwise integration.
- Updated existing tests (`test_dr_sndr_smoke.py`, `test_api.py`, `test_bootstrap_integration.py`, `test_pairwise_api.py`) to unpack via `artifact.report` and `artifact.detailed`.

[#22]: https://github.com/dgenio/skdr-eval/issues/22
[#23]: https://github.com/dgenio/skdr-eval/issues/23
[#27]: https://github.com/dgenio/skdr-eval/issues/27
[#28]: https://github.com/dgenio/skdr-eval/issues/28
[#30]: https://github.com/dgenio/skdr-eval/issues/30

## [0.5.0] - 2026-04-08

### Removed
- **`strategy` parameter removed from `estimate_propensity_pairwise()`**: Callers using `strategy=` must switch to `method=`. The `strategy` parameter was previously validated but had no effect on behavior.

### Changed
- **`estimate_propensity_pairwise()` default `method` is now `"auto"`**: Previously defaulted to `"condlogit"`, which emitted a warning on every call in environments without SciPy. `"auto"` selects `condlogit` when SciPy is available and falls back to `multinomial` silently.
- **`estimate_propensity_pairwise()` parameters after `design` are now keyword-only**: All parameters except `design` must be passed as keyword arguments. This enforces clarity after the signature change and prevents silent positional-argument misuse.
- **`evaluate_pairwise_models()` `propensity` parameter now accepts `"auto"`**: Type hint expanded from `Literal["condlogit", "multinomial"]` to `Literal["auto", "condlogit", "multinomial"]`.

### Fixed
- **Documentation**: Fixed broken pairwise quick start example in README — corrected function signature (`strategy` replaces non-existent `autoscale_strategies`), added required positional args (`metric_col`, `task_type`, `direction`), fixed return-value unpacking (`(report, detailed)` tuple), and added explicit model fitting before evaluation.
- **evaluate_sklearn_models**: Raise `ValueError` immediately when `models` is empty or contains `None` values, preventing silent `(0, 0)` DataFrames and obscure downstream errors ([#45](https://github.com/dgenio/skdr-eval/issues/45))

## [0.4.2] - 2025-09-13

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

## [0.4.1] - 2025-09-13

### Fixed
- **Release Automation**: Configured GitHub Actions workflow to trigger on pushed tags
- **Automatic PyPI Publishing**: Releases now automatically publish to PyPI when tags are pushed
- **Developer Experience**: Simplified release process - just push a tag to trigger publication

### Technical Details
- **Workflow Trigger**: Added `push.tags: ['v*']` trigger to release workflow
- **Backward Compatibility**: Maintains existing `release` and `workflow_dispatch` triggers
- **Tag Pattern**: Supports any version tag pattern (v1.0.0, v2.1.3, etc.)

## [0.4.0] - 2025-09-13

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

## [0.3.3] - 2025-09-13

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
- **Release Pipeline**: Minor release pipeline fixes and dependency updates

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
