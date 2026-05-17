# skdr-eval v0.7.0 Release Notes

**Feature & Diagnostics Release — Trust, Contributions, and Temporal Controls**

This release ships several major additions: PSIS Pareto-k support-health diagnostics, propensity calibration diagnostics (ECE/Brier), per-decision V̂ contribution tracking, public preflight validators, a capabilities API, and temporal split controls — plus the `EvaluationArtifact` bundled return type and its export/card infrastructure introduced as part of this cycle.

This is also a **breaking-change release**: the 2-tuple return from both top-level entry points has been removed, `induce_policy_from_sklearn` drops the unused `idx` parameter, and the default fold gap has changed from 0 to 1 on all time-series CV paths.

---

## ⚠️ Breaking Changes

### `evaluate_sklearn_models` / `evaluate_pairwise_models` no longer return a 2-tuple

Both entry points now return a single `EvaluationArtifact` instead of `(report, detailed_results)`.

```python
# Before (raises TypeError in v0.7.0)
report, detailed = evaluate_sklearn_models(...)

# After
artifact = evaluate_sklearn_models(...)
report   = artifact.report
detailed = artifact.detailed
```

### `induce_policy_from_sklearn`: `idx` parameter removed

The unused final positional argument `idx` (previously `eval_design.idx`) has been dropped. Remove it from all call sites.

```python
# Before
induce_policy_from_sklearn(models, design, eval_design.idx)

# After
induce_policy_from_sklearn(models, design)
```

### Default fold gap is now `gap=1`

All time-series CV paths (`fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`, `evaluate_sklearn_models`, `evaluate_pairwise_models`) now default to `gap=1`. This shifts fold boundaries by one sample relative to prior baselines. Pass `gap=0` explicitly to restore previous fold layout.

---

## Added

### `EvaluationArtifact` — bundled evaluation result

A new return type (`skdr_eval.reporting.EvaluationArtifact`) bundles `report`, `detailed`, `warnings`, `sensitivity`, `diagnostics`, and `metadata`. Includes JSON/HTML export, stakeholder evaluation cards, and a versioned schema (`SCHEMA_VERSION = "1.1.0"`).

```python
artifact = evaluate_sklearn_models(logs, models)
artifact.to_json("results.json")
artifact.save_card("card.html", model_name="my_model")
```

### PSIS Pareto-k support-health diagnostic (`#80`)

Every `DRResult` and report row now carries `pareto_k` — the Generalized-Pareto shape parameter of the unclipped importance-weight tail. New warning code `HIGH_PARETO_K` fires as `caution` when `k ≥ 0.5` and escalates to `high_risk` when `k ≥ 0.7`. Thresholds are tunable via `SupportHealthThresholds`. New public helper: `skdr_eval.diagnostics.psis_pareto_k(weights)`.

### Propensity calibration diagnostics (`#84`)

`PropensityDiagnostics` now exposes `ece` (Expected Calibration Error, 15-bin), `brier_score`, `reliability_curve`, and `ece_n_bins`. New warning code `MISCAL_PROP` fires as `caution` when `ECE > 0.10` and escalates to `high_risk` when `ECE > 0.20`. New public helpers: `compute_propensity_ece`, `compute_propensity_brier`, `compute_propensity_reliability_curve`.

### Per-decision V̂ contributions (`#92`)

`evaluate_sklearn_models` and `evaluate_pairwise_models` accept `keep_contributions=True` (opt-in, with a `max_kept_contributions` memory guard). When set, each `DRResult` carries a `contributions` dict and `artifact.contributions(model, estimator="DR", top_k=None)` returns a tidy DataFrame. By construction `contribution_to_V.mean() == V_hat` for both DR and SNDR. The stakeholder card includes top-5/bottom-5 contributors when present.

### Public preflight validators (`#24`)

New top-level helpers: `skdr_eval.validate_logs()` and `skdr_eval.validate_pairwise_inputs()` raise typed `DataValidationError` / `InsufficientDataError` on schema problems before evaluation begins. Strict mode adds monotonic-timestamp and eligibility-mask sanity checks.

### `skdr_eval.get_capabilities()` (`#26`)

Side-effect-free detection of optional extras (`viz`, `speed`). Returns booleans and a `missing_extras` list with install instructions.

### Temporal split controls (`#29`)

`gap`, `test_size`, and `max_train_size` keyword-only arguments on all time-series CV entry points.

### Support-health warnings & diagnostics infrastructure

`support_health` and `diagnostic_warnings` columns on every report row; `artifact.warnings` DataFrame; clip-grid sensitivity via `artifact.sensitivity`; propensity diagnostics via `artifact.diagnostics[model]`.

### JSON & HTML export + stakeholder cards

`artifact.to_json()`, `artifact.to_html()`, `artifact.export(formats=...)`, `artifact.card(model_name)`, `artifact.save_card(path, model_name)`. Top-level helpers: `export_results`, `load_artifact_json`, `render_evaluation_card`.

---

## Changed

- **`matplotlib` moved to `[viz]` optional extra** — was required in 0.6.0, now optional again.
- **`induce_policy_stream_topk` is fully day-vectorized** — surrogate runs once per client-chunk; new `chunk_pairs` parameter controls batch size. Output policies unchanged on fixed seeds.
- **Schema version bumped** `1.0.0 → 1.1.0` (additive; old payloads load unchanged via `load_artifact_json`).

---

## Fixed

- `induce_policy_stream_topk` now raises `DataValidationError` on duplicate `operator_id` rows within a day (was silent last-write-wins).

---

## New Public Symbols

```python
from skdr_eval import (
    EvaluationArtifact, SupportHealthThresholds, ArtifactSchema, SCHEMA_VERSION,
    attach_warnings, build_evaluation_artifact, summarize_sensitivity,
    render_evaluation_card, export_results, load_artifact_json,
    validate_logs, validate_pairwise_inputs, get_capabilities,
)
from skdr_eval.diagnostics import (
    psis_pareto_k, compute_propensity_ece,
    compute_propensity_brier, compute_propensity_reliability_curve,
)
```

---

## Upgrade Guide

1. Replace any `report, detailed = evaluate_*_models(...)` with `artifact = evaluate_*_models(...)` and use `artifact.report` / `artifact.detailed`.
2. Remove `eval_design.idx` from `induce_policy_from_sklearn` calls.
3. If your baselines depend on fold layout, add `gap=0` explicitly to restore previous behavior; otherwise the new `gap=1` default is the conservative choice.
4. `matplotlib` is no longer required; add `pip install 'skdr-eval[viz]'` if you use plotting helpers.
