# skdr-eval v0.11.0 Release Notes

**M3 Completeness Release — Real Data, External Policies, Agent Stack Integration, Corrected Estimators**

This release completes the surface sketched in v0.10.0: real public benchmark datasets, Polars/PyArrow input paths, non-sklearn model adapters, external-policy evaluation for pairwise OPE, what-if scenario simulation, and a full documentation site. It also fixes critical correctness issues in the DR/SNDR importance weight and propensity calibration that were present in earlier versions.

There are **no breaking changes** in this release. All additions are additive or correctness fixes with the same default parameters. The `CARD_SCHEMA_VERSION` is bumped to `1.1.0`; older payloads remain forward-compatible via `ConfigDict(extra="allow")`.

---

## Added

### Public dataset loaders (#70)

New `skdr_eval.datasets` package ships `load_obd` for the Open Bandit Dataset, returning the canonical `(logs, ops_all, ground_truth)` tuple so real benchmark data flows through `evaluate_sklearn_models` unchanged.

```python
from skdr_eval.datasets import load_obd

logs, ops_all, ground_truth = load_obd(
    behavior_policy="random",
    campaign="men",
    max_rows=100_000,
)
```

Downloads are cached under `~/.skdr_eval/datasets` (override via `SKDR_EVAL_CACHE_DIR`) with a sha256 `manifest.json`. A local `base_url` enables offline use. `load_criteo_counterfactual` and `load_movielens_ope` are documented stubs.

### Non-sklearn model adapters (#71)

New `skdr_eval.adapters` package makes GBDTs and bare callables first-class:

- `XGBRegressorAdapter`, `LGBMRegressorAdapter`, `CatBoostRegressorAdapter` — behind the `[boosting]` extra, forwarding native fit kwargs (early-stopping, categoricals, GPU flags).
- `CallableModelAdapter(predict_fn, predict_proba_fn=None, fit_fn=None)` — backend-free.

Missing backends raise `OptionalDependencyError`.

### Polars / PyArrow inputs (#72)

The public evaluators now accept Polars `DataFrame` and PyArrow `Table` inputs (converted once at the boundary; results identical to the pandas path). `EvaluationArtifact` gains `to_polars()` / `to_arrow()` accessors for the headline report. Both require the `[speed]` extra.

### External-policy evaluation for pairwise OPE (#56)

`evaluate_external_policies(logs_df, op_daily_df, policies, ...)` scores policies produced by an *external* decision process — e.g. a discrete-event call-centre simulator that accounts for queues and shifts — instead of inducing them greedily from candidate models. A simulation proof (`tests/sim_studies/test_external_policy_recovery.py`) shows the external-policy DR recovers the analytic value of a known target.

### What-if autoscaling scenario simulator (#34)

`simulate_autoscaling_scenario(logs_df, op_daily_df, models, scenario, ...)` re-evaluates a candidate policy under documented operational knobs — `capacity_multiplier` and `eligibility_mode` (`"as_logged"` / `"restricted"`). The applied scenario + its assumptions are recorded on `artifact.metadata["scenario"]`.

### Large-data execution path for pairwise evaluation (#33)

`evaluate_pairwise_models` gains `execution_mode` (`"auto"` / `"standard"` / `"large_data"`). `"large_data"` uses vectorized NumPy operations instead of a per-row `DataFrame.iterrows()` loop; it is **numerically identical** to `"standard"` (parity-tested to <1e-10).

### Generic trace → OPE-log adapter (#149)

`skdr_eval.adapters` gains `from_records` and `from_jsonl_trace`: map generic `(context, action, reward[, timestamp, propensity])` decision records into the canonical logs schema consumed by `evaluate_sklearn_models`. No hand-shaping required.

### Documentation site (#68)

MkDocs Material + `mkdocstrings` site (`mkdocs.yml`, `docs/index.md`, getting-started, concepts, recipes, API reference), a `--strict` CI build job, `.readthedocs.yaml`, and `make docs` / `make docs-serve` targets.

### Flagship LLM-reranker OPE recipe (#145)

`skdr_eval.recipes.llm_reranker` — `make_llm_reranker_synth`, `LLMRerankerLogSchema`, `induce_reranker_policy`, and `evaluate_reranker_mips`. Includes the `10_llm_reranker_ope.ipynb` walkthrough and a simulation proof. No runtime LLM dependency; CPU-only.

### Python 3.10 support (#123)

`requires-python` is now `>=3.10,<3.15`. The CI test matrix runs 3.10–3.14.

---

## Fixed

### DR/SNDR importance weight corrected to include target policy (#106)

**Critical**: the estimator computed the importance weight as `1/e(A|x)` instead of the textbook doubly-robust ratio `π(A|x)/e(A|x)`. Different candidate models produced byte-identical `V_hat`/`SE`/`ESS`. The fix applies the correct ratio at every weight site, with a simulation proof (`test_policy_value_recovery.py`) showing recovery of the analytic value.

### Propensity calibration switched from isotonic to sigmoid (Platt) (#106)

Isotonic `CalibratedClassifierCV(cv=2)` overfit small time-aware folds and drove estimated propensities to ≈0, tripping `POOR_OVERLAP` even on well-overlapped data. Sigmoid calibration (`cv=3`) now yields non-degenerate propensities.

### MIPS weight now marginalises target numerator over embedding kernel (#142)

For non-identity kernels the MIPS weight used the exact observed-action probability as numerator while marginalising only the denominator — returning `n_actions` instead of `1` under a uniform kernel. The numerator is now the symmetric marginal. A simulation proof (`test_mips_marginal_recovery.py`) recovers the analytic value.

### Pairwise `elig_mask` value type normalized at ingestion (#155, #158)

`set`, `frozenset`, `tuple`, and `np.ndarray` cells previously fell through to an incorrect "all eligible" fallback. Every `elig_mask` cell is now canonicalized to a Python `list` once at ingestion.

---

## Changed

- **Library-grade dependency constraints** (#152): lower bounds only (`numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`, `scikit-learn>=1.2`) with no upper-bound caps.
- **MIPS no-embedding graceful SNDR fallback** (#136): requesting `"MIPS"` without `action_embedding=` now falls back to SNDR with a `UserWarning` instead of raising `ValueError`.
- **Slate estimators vectorized** (#137): pure-Python loops replaced with NumPy batching.
