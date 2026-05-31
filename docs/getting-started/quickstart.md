# Quickstart

This mirrors [`examples/quickstart.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/quickstart.py),
the runnable source of truth (executed in CI).

## 1. Preflight

```python
import skdr_eval

logs, ops_all, _ = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)

# Confirm the logs match the schema evaluate_sklearn_models expects.
skdr_eval.validate_logs(logs)
```

A valid `logs` frame carries: an `arrival_ts` timestamp, context feature columns
(`cli_*`, `st_*`), per-operator eligibility columns (`op_*_elig`), the logged
`action`, and the observed outcome (`service_time` by default).

## 2. Define candidate models

Any scikit-learn-compatible regressor (`fit` / `predict`) works:

```python
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}
```

## 3. Evaluate

```python
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    n_splits=3,
    outcome_estimator="hgb",
    policy_train="pre_split",   # fit policy on the first 80%, evaluate on the tail
    random_state=42,
)
```

`policy_train="pre_split"` fits each candidate on the first
`policy_train_frac` of the data and evaluates on the held-out tail, which avoids
training-on-test bias when `fit_models=True`.

## 4. Read the result — in the right order

`evaluate_sklearn_models` returns a single
[`EvaluationArtifact`](../api.md) bundling the report, per-model results,
warnings, clip-grid sensitivity, and propensity diagnostics.

```python
artifact.report      # V_hat, SE_if, ESS, match_rate, support_health, ...
artifact.warnings    # the trust contract: ok / caution / high_risk + codes
artifact.sensitivity # how V_hat moves across the clip grid
artifact.diagnostics # per-model propensity calibration (ECE / Brier)
```

**Always read `support_health` before `V_hat`.** A `high_risk` row means the
logged data does not support the candidate policy and the point estimate is not
deployment evidence — see [reading the report](../report-interpretation.md) and
[good vs bad support](../recipes/good-vs-bad-support.md).

## 5. Export & hand off

```python
artifact.export("artifacts/quickstart", formats=["json", "html"])
artifact.save_card("artifacts/quickstart_card.html", "RandomForest")
```

## Next

- The end-to-end practitioner workflow:
  [logs → experiment-review card](../recipes/logs-to-experiment-card.md).
- The pairwise (operator-routing) API and the slate (ranking) API:
  [slate vs pairwise vs standard](../slate-vs-pairwise-vs-standard.md).
