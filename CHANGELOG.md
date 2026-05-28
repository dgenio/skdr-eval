# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Newcomer documentation & README onboarding sweep** ([#98], [#116], [#117],
  [#118], [#119], [#120], [#122], [#126]). One-PR pass over the
  adoption-facing docs:

  - **Metrics glossary** ([#116]): new `docs/metrics-glossary.md` with a
    plain-language, "how to read this" entry for every field in the report,
    warnings, sensitivity, and diagnostics — and all eight warning codes.
  - **Report interpretation guide** ([#117]): new
    `docs/report-interpretation.md` taking a reader from HTML output to a
    decision, with a reading order, a decision table, and a stakeholder
    explanation template.
  - **Comparison / positioning page** ([#119]): new `docs/comparisons.md`
    positioning skdr-eval honestly against OBP, SCOPE-RL, d3rlpy, and
    banditml, with a "when *not* to use" section; cross-linked with
    `docs/methods.md`.
  - **README hero rewrite** ([#120], [#126]): the opening now leads with a
    practitioner job-to-be-done and a workflow diagram (logs → DR/SNDR →
    trust diagnostics → decision artifact) before any estimator names, with
    "use this when / do not use this when" boxes.
  - **First-10-minutes onboarding path** ([#118]): a README section that
    points newcomers at the quickstart notebook, the interpretation guide,
    and the glossary in the order they are actually needed.
  - **Use-case clarity + repository metadata** ([#98]): README explains the
    practical use case and limits without OPE jargon; `pyproject.toml` URLs
    already point at `dgenio/skdr-eval`.

- **Statistical trust contract** ([#127], [#128], [#129], [#130], [#131],
  [#132], [#133], [#134]). One-PR sweep over the eight `area: trust`
  issues that anchor the maintainer-level statistical evidence map:

  - **Estimand block** ([#128]): `EvaluationArtifact` and `EvaluationCard`
    now carry `estimand_tex`, `estimand_summary`, and an `assumptions`
    list whose default value is the seven canonical tags
    (`unconfoundedness`, `overlap`, `sutva`, `double_robustness`,
    `stochastic_logging`, `bounded_weight_variance`,
    `time_structure_respected`). New `EstimandBlock` Pydantic block;
    full prose in `docs/concepts/estimands-and-assumptions.md`.
  - **Statistical validation matrix** ([#127]):
    `docs/statistical-validation-matrix.md` indexes every estimator,
    diagnostic, and failure-mode tutorial to its assumption tags,
    simulation proof, and test references.
  - **Per-action propensity diagnostics** ([#131]):
    `skdr_eval.per_action_propensity_diagnostics(...)` returning a list
    of `PerActionDiagnostics` (n, logged_frac, mean propensity taken vs
    global, per-action ECE/Brier/log-loss, `insufficient`, `rare`).
    `PropensityDiagnostics` gains `per_action`, `n_rare_actions`,
    `n_insufficient_actions`, `max_per_action_ece`. Two new warning
    codes — `PER_ACTION_MISCAL` (caution/high_risk on the maximum
    per-action ECE) and `RARE_ACTION_NO_SUPPORT` (high_risk when a
    target-support action has fewer than `_MIN_ACTION_COUNT_DISC=5`
    logged samples).
  - **Baseline / delta-vs-baseline first-class outputs** ([#132]):
    new `baseline=` kwarg on `evaluate_sklearn_models` and
    `evaluate_pairwise_models` (accepts `float`, `"logged"`, or `None`).
    The report gains `delta_V_hat` (and `delta_ci_lower` / `delta_ci_upper`
    when `ci_bootstrap=True`). The card carries a new `BaselineBlock`.
  - **Three-band stability grade** ([#133]):
    `summarize_sensitivity(...)` now emits `v_range_frac` and
    `stability_grade` ∈ {`stable`, `sensitive`, `unstable`} on every
    row. `SensitivityBlock` exposes both fields.
  - **Simulation studies** ([#129], [#130]): new `tests/sim_studies/`
    package — `test_dr_misspecification.py` verifies the double-robustness
    property (four regimes), `test_overlap_failure.py` sweeps logging-
    policy sharpness and shows Pareto-k crossing the PSIS 0.7 threshold,
    `test_bootstrap_validity.py` exercises moving-block bootstrap
    coverage under iid / AR(1) / small-n / seasonal DGPs (assumption-
    boundary case). All gated by `SIM_REPS` (default 30; bootstrap floors
    at 50 for stable empirical coverage).
  - **Known-failure-mode tutorials** ([#134]): new `examples/known_failures/`
    directory ships `poor_overlap.py`, `misspecified_q.py`, and
    `non_stationary.py` — three deliberately-failing offline evaluations
    so newcomers see what `high_risk` looks like before they ship one.
    A new `make known-failures` target runs all three; they are
    intentionally kept outside `examples/use_cases/` so the CI smoke job
    on that directory remains green.

  Card schema version bumped 1.0.0 → 1.1.0. `ConfigDict(extra="allow")`
  keeps older payloads forward-compatible; new optional fields default
  to `None`.

- **Composable estimator strategies** ([#86]). New `skdr_eval.estimators`
  subpackage introduces `WeightTransform` and `OutcomeLoss` protocols that
  decouple the DR pseudo-outcome from the clip-grid + MSE pair, plus three
  new doubly-robust variants:
  - `MRDR` — More-Robust DR (Farajtabar, Chow & Ghavamzadeh 2018). Uses
    `MRDRWeightedLoss` to refit the outcome model with `(π/e)^2` sample
    weights for variance-minimising q̂.
  - `SWITCH-DR` — Wang, Agarwal & Dudík 2017. Falls back to the direct
    method when the raw IPS weight exceeds the user-supplied threshold
    `tau`.
  - `DRos` — DR with optimistic shrinkage (Su et al. 2020). Replaces the
    importance weight with `w · λ / (w² + λ)` so `λ → 0` collapses to the
    direct method and `λ → ∞` recovers raw IPS.
  Each strategy plugs into the high-level `evaluate_sklearn_models` /
  `evaluate_pairwise_models` via the new `estimators=` kwarg and the
  per-estimator dataclasses `EstimatorStrategy`, `ClipTransform`,
  `IdentityTransform`, `SwitchTauTransform`, `DRosShrinkTransform`,
  `MSEOutcomeLoss`, `MRDRWeightedLoss`.

- **MIPS (Marginalized IPS) estimator** ([#85]). New `MIPSTransform`
  replaces the per-action propensity with an embedding-marginal
  propensity, restoring common support in large-action settings (operator
  pools, candidate-set rerankers). Exposed as `estimators=(..., "MIPS")`
  with the `action_embedding=`, `mips_bandwidth=` kwargs on
  `evaluate_sklearn_models` / `evaluate_pairwise_models`, plus the
  free-function `skdr_eval.mips_value(...)` convenience wrapper. Pairs with
  the new `embedding_sufficiency_diagnostic(...)` probe that flags when
  the embedding has lost too much action-specific reward signal for MIPS
  to be approximately unbiased.

- **Slate / top-K off-policy estimators** ([#75]). New `skdr_eval.slate`
  subpackage ships four ranking-OPE estimators —
  `slate_standard_ips`, `reward_interaction_ips`, `pseudo_inverse_ips`,
  `slate_cascade_dr` (Kiyohara et al. 2022) — plus the synthetic
  cascade-click generator `make_slate_synth(...)` with closed-form ground
  truth so unit tests can verify recovery.

- **`fit_outcome_crossfit(..., sample_weight=...)`**. Optional per-row
  non-negative weights forwarded to the underlying estimator's
  `fit(..., sample_weight=...)`. Required by MRDR; falls back to
  unweighted MSE when omitted (current default).

- **Statistical-integrity simulation proofs**
  (`tests/test_estimator_recovery_simulation.py`). Monte-Carlo recovery
  proofs for DR / SNDR / MRDR / SWITCH-DR / DRos / MIPS against a closed-form
  ground-truth `V*`, and slate-OPE recovery under the cascade click model,
  satisfying the `.claude/CLAUDE.md` §2 / `review-checklist.md`
  simulation-proof requirement.

- **Examples**: `examples/quickstart_estimators.py`,
  `examples/quickstart_mips.py`, `examples/quickstart_slate.py` walk
  through the new estimator family, MIPS workflow, and slate-OPE pipeline.

### Changed
- **Claims audit: conservative, precise wording** ([#122]). Dropped the
  "Production Ready" feature claim and the absolute "general-purpose"
  framing from the README and the `pyproject.toml` description; calibrated
  the positioning to "practitioner-focused" and made the standing OPE
  assumptions and the "offline evaluation does not replace online
  validation" caveat explicit near the top of the README.
- **`evaluate_sklearn_models` / `evaluate_pairwise_models` signature**.
  New kwargs `estimators=("DR", "SNDR")` (tuple of estimator names),
  `action_embedding=None`, `switch_tau=5.0`, `dros_lam=1.0`,
  `mips_bandwidth=1.0`. Default `estimators=("DR","SNDR")` preserves the
  historical report shape exactly. The new `estimators=` row order in the
  report follows the order in the kwarg.
- **Configurable reward column** ([#105]). `evaluate_sklearn_models`,
  `build_design`, and `validate_logs` accept `y_col` (default
  `"service_time"`) so general-purpose OPE logs can name the reward column
  anything (`reward`, `click`, `revenue`, ...). Default preserves prior
  behavior exactly; revert by removing the `y_col` kwarg.
- **`EvaluationArtifact.to_json()` / `.to_html()`** ([#108]) now follow the
  pandas convention: called with no argument they return the serialized
  string; called with a `path` they write the file and return its `Path`.
- **`models` contract validation** ([#109]). Both evaluators now raise a
  clear `DataValidationError` (was an opaque `AttributeError` / `ValueError`)
  when `models` is not a non-empty `{name: estimator}` dict, including a
  "did you mean `models={...}`?" hint for a bare estimator.

### Fixed
- **README quickstart no longer warns** ([#104]). The canonical snippets pass
  `policy_train="pre_split"` explicitly, so a copy-pasted quickstart runs
  without a `DeprecationWarning`.
- **Undefined `[choice]` extra** ([#107]). Removed the `pip install
  skdr-eval[choice]` instruction; conditional-logit works out of the box
  (SciPy is a core dependency). A test now guards README↔pyproject extra
  drift.
- **Stale repository owner / citation version** ([#110], [#111]). README
  Trusted-Publishing block points at `dgenio/skdr-eval`; the BibTeX `version`
  matches the released version.
- **Python version drift in docs** ([#112]). `CONTRIBUTING.md` /
  `DEVELOPMENT.md` now say Python 3.11+ to match `requires-python`.
- **`make_pairwise_synth` docstring shapes** ([#113]) corrected to the actual
  `(300, 14)` / `(30, 6)` output, with a regression test.
- **Pre-split small-input error** ([#114]). With `policy_train="pre_split"`,
  the `InsufficientDataError` now reports the original input row count and
  `policy_train_frac` instead of the opaque post-split count.
- **`validate_logs` int/str action mismatch** ([#115]). When actions are
  ints but `*_elig` column names parse as str, the error now names the dtype
  gap and suggests an `astype(str)` cast.

## [0.9.0] - 2026-05-22

### Added
- **`EvaluationCard` Pydantic v2 schema** ([#88]). New
  ``skdr_eval.EvaluationCard`` (the machine-readable sibling of the HTML
  evaluation card) bundles the headline result, trust signals, diagnostics,
  sensitivity, and provenance for a single ``(model, estimator)`` row into
  a typed payload that is JSON/YAML round-trippable and has a
  ``json_schema()`` export for downstream tooling. Build a card via
  ``EvaluationArtifact.card_schema(model_name, *, estimator='SNDR',
  baseline=None, include_gate=True, include_recommendation=True)``. Round-
  trip with ``card.to_yaml(path)`` / ``card.to_json(path)`` /
  ``EvaluationCard.from_yaml(...)`` / ``EvaluationCard.from_json(...)``.
  New constant ``CARD_SCHEMA_VERSION = "1.0.0"``. Block models
  ``HeadlineBlock``, ``TrustBlock``, ``DiagnosticsBlock``,
  ``SensitivityBlock``, ``ProvenanceBlock``, and ``CoverageSimBlock`` are
  all exported. ``ConfigDict(extra="allow")`` keeps schemas forward-
  compatible: unknown fields are preserved on round-trip.

- **`skdr_eval.doctor()`** ([#91]). New preflight diagnostic surface
  composing the existing public validators (``validate_logs`` /
  ``validate_pairwise_inputs``) and ``get_capabilities()`` into a single
  non-raising ``DoctorReport`` with concrete fix hints. Each row is a
  ``Check(name, status, message, fix_hint, category)``; the report
  exposes ``ok``, ``summary``, ``to_dict()``, ``to_markdown()``,
  ``to_text(color=…)``, and ``print(color=…)``. The doctor never mutates
  its input DataFrame and never raises — even on a non-DataFrame input
  the report surfaces the failure as a ``Check`` with ``status="fail"``.

- **`Tracker` protocol + `NullTracker` + `FileTracker`** ([#93]). The new
  ``skdr_eval.trackers`` package defines the minimum tracker surface
  (``log_metric`` / ``log_artifact`` / ``log_card`` / ``set_tag`` + context
  management) and ships a no-op ``NullTracker`` (default) plus a disk-based
  ``FileTracker`` that writes ``metrics.jsonl``, ``tags.json``,
  ``artifacts/``, and ``cards/<model>_<estimator>.card.yaml``. Both
  ``evaluate_sklearn_models`` and ``evaluate_pairwise_models`` accept a new
  ``tracker=`` kwarg (default ``None``); the evaluator auto-logs
  ``V_hat/ci_lower/ci_upper/ESS/match_rate/pareto_k`` per ``(model,
  estimator)`` row, run-level tags, and one ``EvaluationCard`` per row.
  Stubs ``skdr_eval.trackers.mlflow.MLflowTracker``,
  ``skdr_eval.trackers.wandb.WandbTracker``,
  ``skdr_eval.trackers.aim.AimTracker`` raise ``NotImplementedError`` on
  construction — full adapters tracked by umbrella issue #73.

- **`skdr-eval` CLI** ([#89]). New Typer-based CLI gated behind the
  ``[cli]`` extra. Subcommands:

  - ``skdr-eval evaluate LOGS --model NAME=PATH --out DIR`` — run the
    sklearn evaluator on local logs + pickled models, write artifact
    JSON + HTML report + one ``EvaluationCard`` YAML/JSON per row.
  - ``skdr-eval pairwise LOGS OP_DAILY --model NAME=PATH ...`` — the
    pairwise counterpart.
  - ``skdr-eval card ARTIFACT_JSON --model NAME --estimator DR|SNDR --out
    PATH --format yaml|json`` — re-render an ``EvaluationCard`` directly
    from a saved ``artifact.json``.
  - ``skdr-eval validate-schema LOGS [--kind standard|pairwise]
    [--op-daily PATH] [--metric-col COL] [--strict]`` — call the public
    validators and exit non-zero on failure.
  - ``skdr-eval doctor LOGS [--kind ...] [--json]`` — print the preflight
    report (text or JSON) and exit non-zero when the doctor reports
    ``fail``.
  - ``skdr-eval version`` — print the package version.

  Input format auto-detection covers ``.parquet`` / ``.csv`` / ``.tsv`` /
  ``.feather``. The CLI normalizes parquet-round-trip quirks (e.g. object
  cells coming back as ``ndarray``) before handing them to the public
  validators. Exit codes are stable for CI gates: ``0`` ok, ``1`` data /
  schema error, ``2`` env error, ``3`` ``do_not_deploy`` recommendation
  on at least one row.

- **New optional dependency extras**: ``[cli]`` (``typer>=0.12``,
  ``joblib>=1.3``, ``pyarrow>=14``), and three placeholder extras for
  the umbrella tracker adapters: ``[mlflow]``, ``[wandb]``, ``[aim]``.
  The ``skdr-eval`` script is registered via ``[project.scripts]``.

- **Repositioning, onboarding, and citability sweep** ([#67], [#69], [#77], [#78], [#90], [#98]).
  - **README**: rewrite the headline + opening paragraph for general-purpose
    offline policy evaluation (was service-time-only); add a "When should I
    use this?" plain-language section; add an Open-in-Colab badge table for
    five quickstart notebooks; add a use-case gallery section pointing at the
    new `examples/use_cases/` scripts; fix CI / Coverage / dev-clone badges
    and URLs from `dandrsantos` to `dgenio`. ([#67], [#98])
  - **`pyproject.toml`**: fix `[project.urls]` Homepage / Repository / Issues
    to `dgenio/skdr-eval`; add Changelog / Documentation URLs; broaden
    `description` from "service-time minimization" to general-purpose OPE;
    broaden `keywords`. ([#67], [#98])
  - **`examples/notebooks/`** (new) — five nbmake-tested Colab notebooks:
    `01_quickstart`, `02_pairwise_quickstart`, `03_ecommerce_ranking`,
    `04_ad_targeting`, `05_healthcare_cate`. Each is < 12 cells, runs in
    under 60 s, opens directly in Colab via the README badge table. ([#69], [#90])
  - **`examples/use_cases/`** (new) — four runnable domain walk-throughs:
    `01_ecommerce_ranking.py`, `02_ad_targeting.py`, `03_healthcare_cate.py`,
    `04_call_routing.py`. Each reuses the existing synth generators with
    domain-flavored framing and stakeholder summaries. ([#78])
  - **CI** — new `notebooks-smoke` job runs
    `pytest --nbmake examples/notebooks/` on every PR; new `use-cases-smoke`
    job runs the four domain scripts. Both gate on the `test` job so
    notebook / example rot is caught at PR time. ([#69], [#78], [#90])
  - **`Makefile`** — new `notebooks` and `use-cases` targets mirror the new
    CI jobs locally; help text updated.
  - **`CITATION.cff`** — bump `version` to `0.7.0`, fix URLs to
    `dgenio/skdr-eval`, broaden keywords / abstract, add `identifiers` block
    with a DOI placeholder, add a `preferred-citation` block. ([#77])
  - **`.zenodo.json`** (new) — Zenodo metadata for the next tagged release
    so the GitHub → Zenodo binding mints a concept DOI automatically. ([#77])
  - **`docs/zenodo.md`** (new) — one-time maintainer checklist for the
    GitHub → Zenodo binding. ([#77])
  - **`docs/methods.md`** (new) — short methods-note outline (positioning vs.
    Open Bandit Pipeline / SCOPE-RL / banditml; references). ([#77])
- **`[notebooks]` optional extra** in `pyproject.toml` — installs
  `jupyter`, `matplotlib`, `nbformat`, `nbclient` for users who want to run
  the new notebooks locally (`pip install 'skdr-eval[notebooks]'`).
- **`nbmake>=1.5` and `nbformat>=5.0`** added to the `[dev]` extra so CI
  can execute the new notebook smoke job.

### Changed
- ``[project.urls]`` Homepage/Repository/Issues now point at
  ``dgenio/skdr-eval`` (was the legacy ``dandrsantos`` URL); a new
  ``Changelog`` URL is published alongside.
- **`pyproject.toml`** `description` now reads "General-purpose offline policy
  evaluation for sklearn-compatible models with time-aware Doubly Robust (DR)
  and Stabilized DR (SNDR) estimators, calibrated propensities, and
  stakeholder evaluation cards" — reflects the actual scope (no statistical
  change).

### Tests
- New ``tests/test_card_schema.py`` (22 tests) covering ``card_schema()``,
  YAML/JSON round-trip (string and file paths), ``json_schema()`` export,
  forward-compatibility under ``ConfigDict(extra="allow")``, and the
  schema-version stability guard.
- New ``tests/test_doctor.py`` (18 tests) covering the per-check pass /
  warn / fail paths, the never-raise contract, idempotence, and the
  ``to_dict``/``to_markdown``/``to_text`` renderers.
- New ``tests/test_trackers.py`` (19 tests) covering ``NullTracker``
  no-op semantics, ``FileTracker`` metrics/tags/artifacts/cards writes,
  evaluator wiring (``tracker=None`` artifact-identical to
  ``tracker=NullTracker()``), the pairwise evaluator's ``tracker=`` kwarg,
  and the ``MLflowTracker`` / ``WandbTracker`` / ``AimTracker`` stub
  contracts.
- New ``tests/test_cli.py`` (15 tests) covering ``--help`` /
  ``version`` / ``doctor`` / ``validate-schema`` / ``evaluate`` (via the
  ``card`` round-trip) / ``card`` (YAML and JSON output formats) using
  Typer's ``CliRunner``; exit-code stability guard for
  ``EXIT_DO_NOT_DEPLOY == 3``.
- Additional coverage for ``_build_card_from_row`` fallback paths and
  ``OSError`` branch in ``reporting.py``; direct unit tests for CLI path
  guards, ``coerce_float``, ``doctor`` ``kind`` parameter, and the
  path-resolver helper ([#101], [#102]).

### Fixed
- **CLI path-traversal and filename-sanitization** ([#101]). Output paths
  in the ``evaluate``, ``pairwise``, and ``card`` subcommands are now
  validated to stay inside the declared ``--out`` directory; model
  filenames are sanitized before use in filesystem operations. ``doctor``
  column handling is corrected. Notebook ``pip install`` calls are guarded
  behind a Colab-detection check to avoid unintended side-effects in
  non-Colab environments.

[#67]: https://github.com/dgenio/skdr-eval/issues/67
[#69]: https://github.com/dgenio/skdr-eval/issues/69
[#77]: https://github.com/dgenio/skdr-eval/issues/77
[#78]: https://github.com/dgenio/skdr-eval/issues/78
[#88]: https://github.com/dgenio/skdr-eval/issues/88
[#89]: https://github.com/dgenio/skdr-eval/issues/89
[#90]: https://github.com/dgenio/skdr-eval/issues/90
[#91]: https://github.com/dgenio/skdr-eval/issues/91
[#93]: https://github.com/dgenio/skdr-eval/issues/93
[#98]: https://github.com/dgenio/skdr-eval/issues/98
[#101]: https://github.com/dgenio/skdr-eval/pull/101
[#102]: https://github.com/dgenio/skdr-eval/pull/102
[#104]: https://github.com/dgenio/skdr-eval/issues/104
[#105]: https://github.com/dgenio/skdr-eval/issues/105
[#107]: https://github.com/dgenio/skdr-eval/issues/107
[#108]: https://github.com/dgenio/skdr-eval/issues/108
[#109]: https://github.com/dgenio/skdr-eval/issues/109
[#110]: https://github.com/dgenio/skdr-eval/issues/110
[#111]: https://github.com/dgenio/skdr-eval/issues/111
[#112]: https://github.com/dgenio/skdr-eval/issues/112
[#113]: https://github.com/dgenio/skdr-eval/issues/113
[#114]: https://github.com/dgenio/skdr-eval/issues/114
[#115]: https://github.com/dgenio/skdr-eval/issues/115

## [0.8.0] - 2026-05-20

### Added
- **SNDR bootstrap CI fix** ([#58]). The moving-block bootstrap CI for the
  SNDR estimator now uses the correct normalised pseudo-outcome
  `q_pi + (n/Σw)·w·(Y−q̂)` instead of the DR pseudo-outcome
  `q_pi + w·(Y−q̂)`. The old code produced a CI anchored to the DR point
  estimate, not to V̂_SNDR. This was a statistical correctness bug silently
  noted in test comments. Both evaluators (`evaluate_sklearn_models`,
  `evaluate_pairwise_models`) are fixed.

- **`policy_train` defaults to `"pre_split"` with DeprecationWarning** ([#60],
  [#82]). The sentinel approach: passing `policy_train=None` (or omitting the
  argument) now emits `DeprecationWarning` and uses `"pre_split"`. Passing
  `policy_train="pre_split"` or `policy_train="all"` suppresses the warning.
  The old default of `"all"` is retained for backward compatibility but
  deprecated. This change affects both `evaluate_sklearn_models` and
  `evaluate_pairwise_models`.

- **`Recommendation` API** ([#83]). New `EvaluationArtifact.recommendation(
  model_name, *, estimator="SNDR", baseline=0.0, policy=None)` method that
  aggregates support-health warnings and CI position into a structured
  `Recommendation` dataclass with `verdict` (``"deploy"`` / ``"ab_test"`` /
  ``"do_not_deploy"`` / ``"insufficient_evidence"``), `confidence`
  (``"high"`` / ``"medium"`` / ``"low"``), `primary_blocker`, `reasons`
  (list of `Reason` objects), and `recommended_estimator`. Supports
  `to_dict()` / `from_dict()`. Also exported from the top-level package.

- **`DiagnosticGate` API** ([#99]). New `gate_diagnostics(artifact, model_name,
  estimator="DR", *, thresholds=None)` function that runs structured
  pass/warn/fail gates for overlap (`min_pscore` vs 1/n), effective sample
  size, and calibration (Pareto-k). Returns a `DiagnosticGate` dataclass with
  `overlap`, `ess`, `calibration` (`GateResult` objects) and `overall`.
  Supports `to_dict()` and `to_text()`. All symbols exported from the top-level
  package.

- **Coverage-probability simulation harness** ([#81]). New private module
  `src/skdr_eval/_simulation.py` with `simulate_coverage(dgp, n, n_reps,
  alpha, block_length_strategy, block_len, seed, tolerance) → CoverageResult`.
  Supports DGPs: ``"iid"``, ``"ar1"`` (ρ=0.5), ``"seasonal"`` (weekly, period=52).
  Returns `CoverageResult` with empirical coverage, Wilson 95% CI for the
  proportion, and a `passes_nominal` verdict. `CoverageResult` and
  `simulate_coverage` are exported from the top-level package.

- **`make coverage-sim` target** ([#81]). Runs the simulation harness for all
  three DGPs at `n_reps=50` (fast CI check). Increase `--n_reps` locally for
  thorough calibration checking.

### Changed
- `evaluate_sklearn_models` and `evaluate_pairwise_models` parameter
  `policy_train` changed from `str = "all"` to `str | None = None`.
  Callers relying on the old default should add `policy_train="all"` to
  suppress the `DeprecationWarning`, or migrate to `policy_train="pre_split"`
  for the statistically-sound choice.

### Tests
- New `tests/test_coverage_simulation.py` — structural, validation, and
  statistical coverage tests for `simulate_coverage`.
- Extended `tests/test_bootstrap_integration.py` with
  `TestBootstrapCICoverageDGP`: B=50 DGP seeds check that the DR CI brackets
  the oracle V* at ≥70% rate, and a new test checks that the post-fix SNDR CI
  contains V̂_SNDR across B=30 seeds.
- Extended `tests/test_reporting_artifact.py` with `TestRecommendation` (all
  verdict branches, `to_dict`/`from_dict`, error paths) and
  `TestDiagnosticGate` (all gate states, custom thresholds, serialization,
  error paths).
- Extended `tests/test_api.py` with `policy_train` deprecation warning tests
  and import checks for new public symbols.

[#58]: https://github.com/dgenio/skdr-eval/issues/58
[#60]: https://github.com/dgenio/skdr-eval/issues/60
[#81]: https://github.com/dgenio/skdr-eval/issues/81
[#82]: https://github.com/dgenio/skdr-eval/issues/82
[#83]: https://github.com/dgenio/skdr-eval/issues/83
[#99]: https://github.com/dgenio/skdr-eval/issues/99

## [0.7.0] - 2026-05-17


### Added
- **PSIS Pareto-k support-health diagnostic** ([#80]). Every `DRResult` and report row now carries `pareto_k`, the Generalized-Pareto shape parameter of the unclipped importance-weight tail (Vehtari, Simpson, Gelman, Yao & Gabry 2024, *Pareto Smoothed Importance Sampling*, JMLR 25:1–58). New warning code `HIGH_PARETO_K` fires as `caution` when `k ≥ 0.5` and escalates to `high_risk` when `k ≥ 0.7`. Thresholds tunable via `SupportHealthThresholds(high_pareto_k_caution=..., high_pareto_k=...)`. New public helper `skdr_eval.diagnostics.psis_pareto_k(weights)`.
- **Propensity calibration diagnostics in card** ([#84]). `PropensityDiagnostics` now exposes 15-bin `ece` (Expected Calibration Error, Naeini, Cooper & Hauskrecht 2015), `brier_score` (multiclass), `reliability_curve`, and `ece_n_bins`. New warning code `MISCAL_PROP` fires as `caution` when `ECE > 0.10` and escalates to `high_risk` when `ECE > 0.20`. Thresholds tunable via `SupportHealthThresholds(miscal_ece=...)`. New public helpers in `skdr_eval.diagnostics`: `compute_propensity_ece`, `compute_propensity_brier`, `compute_propensity_reliability_curve`.
- New columns in the HTML report/card (`pareto_k`, plus `ECE`/`Brier` on the propensity-diagnostics table) and a `Calibration` block in the card ([#84]).
- **Per-decision V̂ contributions on the artifact ([#92]).** `evaluate_sklearn_models` and `evaluate_pairwise_models` accept `keep_contributions=True` (and a `max_kept_contributions=100_000_000` memory guard); when set, each `DRResult` carries a `contributions` dict (`decision_id, q_pi, q_hat, weight, reward, contribution_to_V`) and `EvaluationArtifact.contributions(model, *, estimator="DR", top_k=None)` returns a tidy DataFrame. By construction `contribution_to_V.mean() == V_hat` to float64 precision for both DR and SNDR — DR uses `q_pi + w·(Y-q_hat)`; SNDR rescales the residual term by `n / Σw` so the per-decision values average to the ratio estimator. Capture is opt-in and independent of `ci_bootstrap` — bootstrap CIs continue to use local arrays inline and do not retain a payload on `DRResult`. The stakeholder card includes top-5 contributors / bottom-5 detractors when contributions are present.
- **Public preflight validators** — new `skdr_eval.validate_logs(logs, *, cli_pref, st_pref, strict)` and `skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, *, metric_col, ...)` raise typed `DataValidationError` / `InsufficientDataError` on schema problems before evaluation begins. Strict mode adds monotonic-timestamp and elig-mask sanity checks. ([#24])
- **`skdr_eval.get_capabilities()`** — side-effect-free detection of optional extras (`viz`, `speed`). Returns booleans plus a `missing_extras` list pointing at the install command needed to enable each disabled capability. ([#26])
- **Temporal split controls** — `gap`, `test_size`, and `max_train_size` keyword-only arguments are now plumbed through `fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`, `evaluate_sklearn_models`, and `evaluate_pairwise_models`. The new default is `gap=1` (conservative adjacent-row leakage guard). ([#29])
- **`examples/preflight.py`** — runnable preflight script: capability dump + log + pairwise schema validation.

### Changed
- **Schema bump:** `SCHEMA_VERSION` `1.0.0 → 1.1.0`. Additive only — `_ReportRowSchema` and `_DiagnosticsPayloadSchema` use `None` defaults for the new fields, so 1.0.0 payloads load unchanged via `load_artifact_json`.
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
- New `tests/test_diagnostics_trust.py` with simulation proofs that PSIS Pareto-k recovers known GPD shapes (`k ∈ {0.3, 0.7, 1.2}`), separates light- from heavy-tail weights, and that ECE → 0 under perfectly-calibrated DGPs and clears the 0.10 gate under temperature-distorted propensities. Required by `docs/agent-context/review-checklist.md` for any new statistical primitive.
- Extended `tests/test_propensity_diagnostics.py` with unit tests for `psis_pareto_k`, `compute_propensity_ece`, `compute_propensity_brier`, and `compute_propensity_reliability_curve` (degenerate inputs, sample floors, hand-computed Brier fixture, backward-compat dataclass construction).
- Extended `tests/test_reporting_artifact.py` with `HIGH_PARETO_K` / `MISCAL_PROP` threshold matrices, end-to-end column presence, and JSON round-trip coverage of the new fields.
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
[#80]: https://github.com/dgenio/skdr-eval/issues/80
[#84]: https://github.com/dgenio/skdr-eval/issues/84
[#75]: https://github.com/dgenio/skdr-eval/issues/75
[#85]: https://github.com/dgenio/skdr-eval/issues/85
[#86]: https://github.com/dgenio/skdr-eval/issues/86
[#92]: https://github.com/dgenio/skdr-eval/issues/92
[#127]: https://github.com/dgenio/skdr-eval/issues/127
[#128]: https://github.com/dgenio/skdr-eval/issues/128
[#129]: https://github.com/dgenio/skdr-eval/issues/129
[#130]: https://github.com/dgenio/skdr-eval/issues/130
[#131]: https://github.com/dgenio/skdr-eval/issues/131
[#132]: https://github.com/dgenio/skdr-eval/issues/132
[#133]: https://github.com/dgenio/skdr-eval/issues/133
[#134]: https://github.com/dgenio/skdr-eval/issues/134
[#116]: https://github.com/dgenio/skdr-eval/issues/116
[#117]: https://github.com/dgenio/skdr-eval/issues/117
[#118]: https://github.com/dgenio/skdr-eval/issues/118
[#119]: https://github.com/dgenio/skdr-eval/issues/119
[#120]: https://github.com/dgenio/skdr-eval/issues/120
[#122]: https://github.com/dgenio/skdr-eval/issues/122
[#126]: https://github.com/dgenio/skdr-eval/issues/126

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

[0.9.0]: https://github.com/dgenio/skdr-eval/compare/v0.8.0...v0.9.0
[0.1.0]: https://github.com/dandrsantos/skdr-eval/releases/tag/v0.1.0
