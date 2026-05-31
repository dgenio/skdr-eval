# Recipe: logs → experiment-review card

This is the end-to-end practitioner workflow — the default way to use
`skdr-eval`. It is more narrative than the [quickstart](../getting-started/quickstart.md)
and less theoretical than the [methods note](../methods.md).

> **Runnable source of truth:**
> [`examples/use_cases/05_logs_to_experiment_card.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/use_cases/05_logs_to_experiment_card.py).
> The snippets below mirror that script, which runs in CI.

## The problem

> *We trained a new routing / recommender / targeting policy. Before we risk an
> A/B test, we want an offline estimate, honest trust diagnostics, and an
> artifact we can discuss with stakeholders.*

## 1. The data you already log

`skdr-eval` evaluates a candidate policy against **logged decisions**. You need,
per decision:

- **context features** (e.g. `cli_*` columns);
- the **action taken** by the baseline/logging policy (`action`);
- the **observed outcome** (`service_time` by default);
- **eligibility** of each action (`<action>_elig`) if it varies;
- a **timestamp** (`arrival_ts`) so splits respect time order.

The most important property of the logs is **exploration**: if the baseline
policy was near-deterministic, no method can tell you what an alternative policy
would have done. The recipe builds an epsilon-greedy baseline so every action
keeps a healthy logged probability.

## 2. Preflight

```python
import skdr_eval

skdr_eval.validate_logs(logs)   # fails fast if the schema is wrong
```

## 3. Fit candidate policies

Any scikit-learn-compatible regressor works:

```python
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

models = {
    "RandomForest": RandomForestRegressor(n_estimators=120, random_state=7),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=7),
}
```

## 4. Evaluate

```python
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    n_splits=3,
    policy_train="pre_split",
    baseline="logged",
)
```

## 5. Read the artifact — in this order

The single most important habit: **read the trust signals before the value
estimate.**

1. **`support_health`** — `ok` / `caution` / `high_risk`. If this is
   `high_risk`, stop: the logged data does not support the candidate policy.
2. **`artifact.warnings`** — which specific signals fired (`POOR_OVERLAP`,
   `HIGH_PARETO_K`, `MISCAL_PROP`, …).
3. **`artifact.sensitivity`** — does `V_hat` stay stable across the clip grid,
   or does it swing with the weight truncation?
4. **`artifact.diagnostics`** — propensity calibration (ECE / Brier). Honest
   weights require a well-calibrated propensity model.
5. **`artifact.report` `V_hat`** — only now, the value estimate (plus
   `delta_V_hat` vs the logged baseline).

```python
artifact.report[["model", "estimator", "support_health", "ESS", "match_rate"]]
artifact.warnings
artifact.sensitivity
artifact.diagnostics
artifact.report[["model", "estimator", "V_hat", "SE_if", "delta_V_hat"]]
```

On the well-explored logs in the example, the headline rows report
`support_health == "ok"` and the two candidate models produce **distinct**
`V_hat` values — the fix for [#106](https://github.com/dgenio/skdr-eval/issues/106)
means different policies are no longer collapsed to an identical estimate.

## 6. Hand off

```python
artifact.save_card("artifacts/recipe_card.html", "RandomForest")
```

The card is a self-contained HTML page a non-statistician can read: the value
estimate, the trust verdict, and the warnings, in plain language.

## The decision rule

| `support_health` | What it means | What to do |
|------------------|---------------|------------|
| `ok` | The logs support the candidate policy. | Proceed to A/B-test planning. |
| `caution` | Some signals are marginal. | Interpret carefully; consider more logging. |
| `high_risk` | The estimate is **not** deployment evidence. | Improve logging / exploration / the propensity model first. |

See [good vs bad support](good-vs-bad-support.md) for a side-by-side of a
healthy and an unhealthy evaluation on the same problem.
