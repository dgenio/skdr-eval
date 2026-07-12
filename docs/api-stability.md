# Public API, stability policy, and the road to 1.0

`skdr-eval` is pre-1.0 and the API still moves, but "pre-1.0" should not mean
"anything can break without warning." This page is the contract: which names
are public, how they change, how the serialized artifact/card schemas are
versioned, and what 1.0 will require.

If you are building a pipeline on `skdr-eval`, depend only on the names listed
here. Everything else — modules and attributes prefixed with `_`, anything not
in this inventory — is internal and may change at any time.

## What "public" means

The public API is exactly the set of names re-exported from the top-level
`skdr_eval` namespace (its `__all__`), plus the two documented subpackages
`skdr_eval.estimators` and `skdr_eval.slate` and the `skdr_eval.adapters` /
`skdr_eval.datasets` entry points referenced in this documentation.

- **Public** — importable as `skdr_eval.<name>` and listed in the inventory
  below. Covered by the deprecation policy.
- **Private** — any module or attribute whose name starts with `_`
  (`skdr_eval.core` internals, `skdr_eval._simulation`, `_frames`, etc.) and
  anything not in the inventory. No stability guarantee.

A test (`tests/test_api.py::test_public_api_inventory_documents_every_export`)
fails if a name is added to `skdr_eval.__all__` without being recorded in the
inventory below, so this contract cannot drift silently.

## Deprecation policy

`skdr-eval` follows [SemVer 2.0.0](https://semver.org/). While pre-1.0 the
project treats the **minor** version as the breaking-change signal (0.x.0),
and applies this discipline to every public name:

1. A public name slated for removal or signature change emits a
   `DeprecationWarning` for **at least one minor release** before the change
   lands, with the warning naming the replacement.
2. The deprecation is recorded in `CHANGELOG.md` and the relevant
   `RELEASE_NOTES_*.md`.
3. The old behavior keeps working (possibly alongside the new one) for that
   deprecation window.

The precedent is the `policy_train` parameter of `evaluate_sklearn_models`:
calling it without an explicit `policy_train` emits a `DeprecationWarning`
pointing at the `"pre_split"` / `"all"` choices rather than silently changing
the default. New deprecations follow that shape.

## Schema versioning

The serialized artifact and card carry their own version numbers, independent
of the package version, so downstream consumers of the JSON/YAML can detect
incompatibilities:

- `skdr_eval.SCHEMA_VERSION` — version of the `EvaluationArtifact` JSON schema
  (`ArtifactSchema`). Bumped when the serialized report/warnings/sensitivity/
  diagnostics payload changes shape.
- `skdr_eval.CARD_SCHEMA_VERSION` — version of the machine-readable
  `EvaluationCard` schema.

Both follow SemVer for the *payload*: a minor bump adds fields (old readers
keep working); a major bump may remove or rename fields. The current values
are exported as constants and embedded in every artifact's `metadata` and on
every card.

The generated JSON Schema for the **artifact** payload is committed under
[`docs/schemas/`](schemas/) (`artifact.schema.json`) so downstream tooling can
validate `skdr-eval` output without importing the library. Emit it yourself with
`skdr-eval schema --kind artifact`, or from Python via
`ArtifactSchema.json_schema()`. A test (`tests/test_schema_publishing.py`) fails
if the committed file drifts from the live schema, and
`scripts/generate_schemas.py` regenerates it.

The **card** schema is intentionally *not* published yet: the card carries the
`deploy`/recommendation verdict, whose contract is under the July 2026
experiment-eligibility audit. Publishing it now would freeze a provisional
verdict shape; it will be added once that contract stabilizes.

## Road to 1.0

1.0 means "the public surface below is frozen under SemVer and safe to build
production pipelines on." Concrete, trackable criteria:

- [ ] First-class logged-propensity support (#167) decided in or out of the
      1.0 surface.
- [ ] Public API inventory frozen — no planned renames of the names below.
- [ ] No open correctness-class bugs against the estimators or validators.
- [ ] Every public name has a docstring rendered in the
      [API reference](api.md) and a one-line entry here.
- [ ] Artifact/card schema versions stable for one full minor cycle with no
      breaking payload change.

This is a living checklist; the milestone tracking it is **1.0 readiness**.

## Public API inventory

The names below are the supported public surface. Visualization helpers
(`plot_*`, `create_dashboard`) are only present when the optional `[viz]`
extra is installed.

<!-- api-inventory:start -->

### Top-level functions

`attach_warnings` `block_bootstrap_ci` `bootstrap_confidence_interval`
`build_design` `build_evaluation_artifact` `build_strategy` `chi_square_test`
`create_dashboard` `create_model_ensemble` `doctor` `dr_value_with_clip`
`dr_value_with_strategy` `embedding_sufficiency_diagnostic`
`evaluate_external_policies` `evaluate_pairwise_models`
`evaluate_propensity_diagnostics` `evaluate_sklearn_models`
`evaluate_slate_models` `explain_artifact_schema` `export_results`
`fit_outcome_crossfit` `fit_propensity_timecal` `gate_diagnostics`
`get_capabilities` `get_capability_matrix` `get_default_config`
`get_model_recommendations` `induce_policy_from_sklearn`
`kolmogorov_smirnov_test` `load_artifact_json` `load_config_from_file`
`load_criteo_counterfactual` `load_movielens_ope` `load_obd`
`make_pairwise_synth` `make_slate_synth` `make_synth_logs`
`mann_whitney_u_test` `median_bandwidth` `merge_configs` `mips_value`
`multiple_comparison_correction` `per_action_propensity_diagnostics`
`permutation_test` `plot_calibration_curve` `plot_diagnostics_summary`
`plot_dr_results` `plot_propensity_distribution` `plot_roc_curve`
`power_analysis` `pseudo_inverse_ips` `render_evaluation_card`
`reward_interaction_ips` `sample_size_calculation` `save_config_to_file`
`simulate_autoscaling_scenario` `simulate_coverage` `slate_cascade_dr`
`slate_standard_ips` `summarize_sensitivity` `t_test` `validate_config`
`validate_logs` `validate_pairwise_inputs`

### Classes and dataclasses

`ArtifactSchema` `BaselineBlock` `Capability` `Check` `ClipTransform`
`ConfigManager` `CoverageResult` `CoverageSimBlock` `DRResult`
`DRosShrinkTransform` `DataProfile` `DatasetBundle` `Design` `DiagnosticGate`
`DiagnosticsBlock` `DoctorReport` `EmbeddingSufficiencyReport` `EstimandBlock`
`EstimatorStrategy` `EvaluationArtifact` `EvaluationCard` `EvaluationConfig`
`Explanation` `FileTracker`
`GateResult` `HeadlineBlock` `IdentityTransform` `MIPSTransform`
`MRDRWeightedLoss` `MSEOutcomeLoss` `ModelConfig` `ModelEvaluator`
`ModelFactory` `ModelSelector` `NullTracker` `OutcomeLoss` `PairwiseDesign`
`PerActionDiagnostics` `PropensityDiagnostics` `ProvenanceBlock` `Reason`
`Recommendation` `RecommendationPolicy` `ReportRow` `SensitivityBlock` `SlateGroundTruth`
`SlateResult` `StatisticalTest` `SupportHealthThresholds` `SwitchTauTransform`
`Tracker` `TransformContext` `TrustBlock` `VisualizationConfig`
`WeightTransform`

### Exceptions

`BootstrapError` `ConfigurationError` `ConvergenceError` `DataValidationError`
`DatasetError` `EstimationError` `InsufficientDataError` `MemoryError`
`ModelValidationError` `OptionalDependencyError` `OutcomeModelError`
`PairwiseEvaluationError` `PolicyInductionError` `PropensityScoreError`
`SkdrEvalError` `VersionError`

### Constants

`CARD_SCHEMA_VERSION` `DEFAULT_ASSUMPTION_TAGS` `DEFAULT_ESTIMAND_SUMMARY`
`DEFAULT_ESTIMAND_TEX` `HAS_VISUALIZATION` `SCHEMA_VERSION`

### Subpackages

`adapters` `datasets` `estimators` `slate`

<!-- api-inventory:end -->

## See also

- [API reference](api.md) — rendered docstrings for every public name.
- [Choosing an estimator](choosing-an-estimator.md) — which estimator to run.
- [Architecture tour](architecture.md) — how the pieces fit together.
- `CONTRIBUTING.md` — public-name additions must update this inventory.
