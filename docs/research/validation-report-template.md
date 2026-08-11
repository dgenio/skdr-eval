# Validation report template

Use this template for versioned validation evidence produced under #62 and #223. It is intentionally designed to make unsupported regimes and negative results visible.

A completed report should be reproducible from a committed/versioned configuration and identify the exact package and reference-backend versions used.

## 1. Identity

```yaml
report_id: skdr-validation-YYYY-MM-DD-001
skdr_eval_version: "..."
skdr_eval_commit: "..."
python_version: "..."
platform: "..."
validation_config_version: "..."
seed_policy: "..."
reference_backends:
  - estimator: DR
    backend: OBP
    version: "..."
    semantic_equivalence: "..."
review_status:
  external_methodological_review: pending
```

## 2. Claimed validation envelope

State the exact scope tested by this report:

- decision setting;
- action-space contract;
- target-policy contract;
- behavior-propensity contract;
- outcome/estimand;
- effective evaluation-population rule;
- nuisance-fitting/cross-fitting regime;
- dependence regime;
- CI/inference procedure;
- estimator(s);
- explicitly unsupported configurations.

Do not write “contextual OPE generally” when the matrix validates a narrower regime.

## 3. Promotion target

For each estimator/configuration:

```yaml
estimator: DR
current_maturity: validation_pending
requested_maturity: reference_validated
reference_agreement_required: true
independent_human_review_required_for_next_level: true
```

Implementation maturity and evidence status for individual runs are separate concepts.

## 4. Predeclared validation thresholds

Declare thresholds **before** inspecting the final validation results.

```yaml
criteria:
  max_absolute_bias: "..."
  max_relative_rmse: "..."
  ci_nominal_level: 0.95
  ci_coverage_tolerance: "..."
  max_failure_rate: "..."
  max_false_reassurance_rate: "..."
  max_false_abstention_rate: "..."
  reference_agreement_tolerance: "..."
```

Justify each threshold. Do not loosen a threshold after observing failure without recording the change as a new validation-config version.

## 5. Scenario matrix

Every scenario family must identify what is varied and what truth is known.

Suggested dimensions for validated-v1:

- sample size;
- action count;
- deterministic vs stochastic target policy;
- strong / moderate / weak overlap;
- outcome-model specification quality;
- logging-policy shape with exact logged propensities;
- heterogeneous outcomes;
- policy distance from behavior;
- specifically supported temporal/dependence regimes;
- clipping configuration if clipping is inside the claimed validated path.

Also include deliberately unsupported cases that should fail closed.

## 6. Known-ground-truth results

For every scenario family publish at least:

| Scenario | N runs | True value | Mean estimate | Bias | RMSE | Failure/abstention |
|---|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... |

Include full machine-readable results alongside summarized tables.

## 7. Inferential validation

For every supported CI regime publish:

| Scenario | Nominal coverage | Empirical coverage | Mean width | Median width | Failure rate |
|---|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... |

Explicitly distinguish:

- conditional intervals;
- end-to-end intervals;
- any uncertainty component held fixed or omitted.

If a scenario does not support the claimed interval semantics, mark it unsupported rather than silently showing a reassuring interval.

## 8. Evidence-health validation

The project claim depends on knowing when an estimate should *not* be believed. Diagnostics therefore need their own empirical validation.

For each run record:

- absolute/relative estimation error;
- overlap metrics;
- ESS / ESS fraction;
- weight-tail / Pareto-k diagnostics where applicable;
- propensity diagnostics where applicable;
- sensitivity/clip instability;
- normalized diagnostic states;
- final evidence state.

### False reassurance

Predefine what constitutes a materially wrong estimate for each DGP family.

Report:

> Among runs whose estimation error exceeded the material-error threshold, what fraction were nevertheless classified `estimate_supported` (or equivalent positive evidence state)?

| Scenario | Material-error runs | Falsely supported | False-reassurance rate |
|---|---:|---:|---:|
| ... | ... | ... | ... |

This metric must not be omitted because the result is unfavorable.

### False abstention / over-conservatism

Where a reliable ground truth permits it, also report cases where the evidence layer abstains/fails despite an accurate estimate and otherwise supported conditions.

### Diagnostic discrimination

Report whether individual and combined diagnostics actually enrich for large estimation error. Suitable summaries may include:

- error distributions by diagnostic state;
- risk ratios / enrichment;
- threshold curves;
- calibration plots;
- classification metrics only when their interpretation is appropriate.

Do not turn this into a single vanity score.

## 9. Reference agreement

Follow #281 estimator by estimator.

For each claimed comparison record:

- estimator definition on both sides;
- estimand;
- target-policy probabilities;
- logged propensities;
- effective row mask;
- action ordering;
- clipping/normalization;
- reference version/commit;
- declared tolerance;
- result.

If semantic equivalence is unavailable, state `reference_agreement: unavailable` rather than comparing a superficially similar estimator.

| Estimator | Reference | Equivalent scope | Scenarios | Pass/fail | Notes |
|---|---|---|---:|---|---|
| ... | ... | ... | ... | ... | ... |

## 10. Public-data evidence

Public logged-bandit data can demonstrate realism but cannot replace known-ground-truth validation.

Record:

- dataset/version/license;
- exact slice/filtering;
- logged-propensity semantics;
- action/eligibility mapping;
- what can and cannot be concluded because ground truth is unknown;
- any online-policy comparison available from the dataset design.

## 11. Failure cases

Publish important failures prominently.

For every failed validation criterion:

```yaml
scenario: "..."
criterion: "..."
observed: "..."
threshold: "..."
impact:
  - estimator remains validation_pending
  - scope narrowed
root_cause_status: known | suspected | unknown
follow_up: "#..."
```

A validation report with no visible failures should trigger scrutiny, not celebration.

## 12. Historical behavior comparison

When this report follows a correctness change such as #58 or #280, include a controlled comparison between the historical and corrected behavior and link the user-facing advisory (#300).

## 13. Conclusion by estimator/configuration

For each candidate promotion, use one of:

- `promotion_passed`
- `promotion_blocked`
- `scope_narrowed_and_passed`
- `validation_incomplete`

Explain the exact validation envelope earned by the evidence.

Do not conclude that a validated implementation makes every future evaluation trustworthy.

## 14. Reproduction

Provide exact commands/configuration needed to regenerate:

- simulations;
- reference agreement;
- summary tables;
- published machine-readable result bundle.

Record expected runtime/resource requirements and any external dataset download requirements.

## 15. External review

After independent methodological review (#298), append or link:

- reviewed report ID;
- reviewer findings;
- changes made;
- remaining limitations;
- final promotion decision.
