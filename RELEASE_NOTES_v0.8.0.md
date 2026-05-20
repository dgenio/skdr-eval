# skdr-eval v0.8.0 Release Notes

**Correctness & Decision-Intelligence Release — SNDR CI Fix, Recommendation/DiagnosticGate APIs, Coverage Simulation**

This release fixes a statistical correctness bug in the SNDR moving-block bootstrap confidence interval, adds three new decision-intelligence APIs (`Recommendation`, `DiagnosticGate`, `simulate_coverage`), and deprecates the `policy_train="all"` default in favour of the statistically-sounder `"pre_split"` split strategy.

There are **no breaking changes** in this release. All new API additions are additive, and the `policy_train` default change is protected by a `DeprecationWarning`.

---

## Bug Fixes

### SNDR bootstrap CI now anchored to V̂_SNDR (#58)

The moving-block bootstrap CI for the SNDR estimator was previously using the DR pseudo-outcome `q_pi + w·(Y−q̂)` instead of the SNDR-normalised pseudo-outcome `q_pi + (n/Σw)·w·(Y−q̂)`. This caused the CI to be centred on the DR point estimate rather than V̂_SNDR — a silent statistical correctness bug. Both `evaluate_sklearn_models` and `evaluate_pairwise_models` are fixed.

The fix is verified by a simulation proof (new `TestBootstrapCICoverageDGP` in `tests/test_bootstrap_integration.py`) that checks the SNDR CI brackets V̂_SNDR across B=30 independent DGP seeds.

---

## Added

### `Recommendation` API (#83)

New `EvaluationArtifact.recommendation(model_name, *, estimator="SNDR", baseline=0.0, policy=None)` aggregates support-health warnings and CI position into a structured decision:

```python
rec = artifact.recommendation("my_model")
print(rec.verdict)      # "deploy" | "ab_test" | "do_not_deploy" | "insufficient_evidence"
print(rec.confidence)   # "high" | "medium" | "low"
print(rec.reasons)      # list[Reason]
print(rec.to_dict())
```

`Recommendation` and `Reason` are exported from the top-level `skdr_eval` package.

### `DiagnosticGate` API (#99)

New `gate_diagnostics(artifact, model_name, estimator="DR", *, thresholds=None)` runs structured pass/warn/fail gates for overlap (`min_pscore` vs 1/n), effective sample size, and calibration (Pareto-k):

```python
from skdr_eval import gate_diagnostics

gate = gate_diagnostics(artifact, "my_model")
print(gate.overall)       # "pass" | "warn" | "fail"
print(gate.overlap)       # GateResult
print(gate.ess)           # GateResult
print(gate.calibration)   # GateResult
print(gate.to_text())
```

`DiagnosticGate`, `GateResult`, and `gate_diagnostics` are exported from the top-level `skdr_eval` package.

### Coverage-probability simulation harness (#81)

New private module `skdr_eval._simulation` with `simulate_coverage(dgp, n, n_reps, ...)` that empirically checks whether the moving-block bootstrap CI achieves nominal 95% coverage under three DGPs: `"iid"`, `"ar1"` (ρ=0.5), `"seasonal"` (weekly, period=52).

```python
from skdr_eval import simulate_coverage, CoverageResult

result: CoverageResult = simulate_coverage(dgp="ar1", n=500, n_reps=200)
print(result.empirical_coverage)   # e.g. 0.935
print(result.passes_nominal)       # True
```

`CoverageResult` and `simulate_coverage` are exported from the top-level `skdr_eval` package. The new `make coverage-sim` Makefile target runs the harness for all three DGPs at `n_reps=50` (fast CI check).

---

## Changed

### `policy_train` defaults to `"pre_split"` with `DeprecationWarning` (#60, #82)

Passing `policy_train=None` (or omitting the argument) now emits a `DeprecationWarning` and uses `"pre_split"`. The old `"all"` default is retained for backward compatibility but deprecated.

```python
# To suppress the warning, pass explicitly:
artifact = evaluate_sklearn_models(logs, models, policy_train="pre_split")  # recommended
# or:
artifact = evaluate_sklearn_models(logs, models, policy_train="all")        # old behavior, no warning
```

---

## Tests

- New `tests/test_coverage_simulation.py` — structural, validation, and statistical coverage tests for `simulate_coverage`.
- Extended `tests/test_bootstrap_integration.py` with `TestBootstrapCICoverageDGP` — DGP-seed sweep verifying DR CI bracketing rate ≥70% and SNDR CI correctness post-fix.
- Extended `tests/test_reporting_artifact.py` with `TestRecommendation` (all verdict branches, `to_dict`/`from_dict`, error paths) and `TestDiagnosticGate` (all gate states, custom thresholds, serialization, error paths).
- Extended `tests/test_api.py` with `policy_train` deprecation warning tests and import checks for new public symbols.
