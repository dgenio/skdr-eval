# Coming from Open Bandit Pipeline (OBP)

[Open Bandit Pipeline](https://github.com/st-tech/zr-obp) (OBP) is the
reference research library for off-policy evaluation. If you already have a
dataset prepared the OBP way — logged bandit feedback with `context`,
`action`, `reward`, and a logged propensity `pscore` — this guide maps it onto
the `skdr-eval` log schema, runs the same DR estimator, and shows what
`skdr-eval` adds on top of the point estimate.

For where the two libraries sit relative to each other, see
[comparisons](../comparisons.md). This page makes that positioning *runnable*.

The companion script is
[`examples/obp_interop.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/obp_interop.py).
It runs offline on synthetic OBP-shaped feedback (no OBP dependency) and can
also load a real Open Bandit Dataset slice with `--obd`.

!!! note "OBP stays a notebook-only dependency"
    Nothing in `skdr-eval` imports `obp`. The same-data DR cross-check against
    `obp.ope` belongs in your own notebook behind an optional import; the
    library and the example script never need OBP on the runtime path.

## Field mapping

OBP's `bandit_feedback` dict maps onto the `skdr-eval` log schema like this:

| OBP term | skdr-eval log schema | Notes |
|---|---|---|
| `context` (`n × dim` array) | `cli_*` feature columns | one column per context dimension |
| `action` (`n` ints) | `action` | integer action index `0..n_actions-1` |
| `reward` (`n` floats) | `reward` (the `y_col`) | the outcome being evaluated |
| `pscore` (`n` floats) | `propensity` | logged action probability — see below |
| `position` | *(no equivalent)* | `skdr-eval` is single-slot; use the [slate API](../slate-vs-pairwise-vs-standard.md) for ranked lists |
| *(none)* | `arrival_ts` | `skdr-eval` is time-aware; OBP is not |

!!! warning "Logged propensities are reported, not consumed (yet)"
    `skdr-eval` estimates its own calibrated propensities internally, so the
    logged `pscore` is recorded by the adapter (`had_logged_propensities`) but
    is **not** used by the DR/SNDR estimators today. First-class logged
    propensities are tracked in issue #167. OBP, by contrast, uses `pscore`
    directly.

## Mapping the data

The generic [trace adapter](../datasets.md) does the work — map each row of
the OBP feedback dict into a record, then `from_records` produces a
schema-valid logs frame:

```python
import skdr_eval

# `feedback` is an obp-style dict: context, action, reward, pscore arrays.
records = [
    {
        "context": feedback["context"][i].tolist(),
        "action": int(feedback["action"][i]),
        "reward": float(feedback["reward"][i]),
        "propensity": float(feedback["pscore"][i]),
    }
    for i in range(len(feedback["action"]))
]
adapted = skdr_eval.adapters.from_records(records)
print(adapted.had_logged_propensities)  # True — recorded, not consumed
```

## Running DR/SNDR with trust diagnostics

```python
from sklearn.ensemble import HistGradientBoostingRegressor

artifact = skdr_eval.evaluate_sklearn_models(
    logs=adapted.logs,
    models={"hgb": HistGradientBoostingRegressor(random_state=0)},
    fit_models=True,
    n_splits=3,
    random_state=0,
    policy_train="pre_split",
    y_col="reward",
)
print(artifact.report[["model", "estimator", "V_hat", "SE_if", "ESS"]])
print(artifact.warnings[["model", "estimator", "support_health"]])
```

You get the DR and SNDR point estimates OBP would also give you — plus the
`support_health` verdict, PSIS Pareto-k, ESS, calibration, and clip-grid
sensitivity that turn the estimate into a *decision*.

## Same-data DR cross-check (optional, in your notebook)

To convince yourself the estimators agree on the shared estimand, run OBP's DR
on the same frame in a notebook where `obp` is installed:

```python
# Notebook-only; requires `pip install obp`.
from obp.ope import OffPolicyEvaluation, DoublyRobust
# ... build OBP's bandit_feedback and action_dist, then compare its DR
# estimate to artifact.report's V_hat for the DR row.
```

The point of the comparison is not a winner: it is to show the **point
estimate is the commodity** and the trust diagnostics are what `skdr-eval`
adds.

## What's out of scope

- A general-purpose OBP adapter class in the library (revisit if demand shows
  up).
- Benchmark MSE tables across estimators — tracked in #94.

## See also

- [Comparisons: OBP / SCOPE-RL / d3rlpy / banditml](../comparisons.md)
- [Datasets, inputs & adapters](../datasets.md) — `load_obd` and the trace adapter
- [Choosing an estimator](../choosing-an-estimator.md)
