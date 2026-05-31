# skdr-eval

**Practitioner-focused offline policy evaluation for scikit-learn-compatible
contextual-bandit policies.**

`skdr-eval` answers one question: *I have logged decisions and a candidate
sklearn-like policy — before I risk an A/B test, what does an offline estimate,
with honest trust diagnostics, say?* It ships time-aware **Doubly Robust (DR)**
and **Stabilized DR (SNDR)** estimators, calibrated propensities, support-health
diagnostics, and a stakeholder-ready evaluation card.

```python
import skdr_eval
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

logs, ops_all, _ = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models={
        "RandomForest": RandomForestRegressor(random_state=42),
        "HistGB": HistGradientBoostingRegressor(random_state=42),
    },
    policy_train="pre_split",
)
print(artifact.report)        # V_hat, SE, ESS, match_rate, support_health, ...
print(artifact.warnings)      # the trust contract
```

## Why it exists

A single offline value estimate is dangerous without a trust contract. `skdr-eval`
makes the trust signals first-class:

- **DR / SNDR** with time-aware cross-fitting — no training-on-test leakage.
- **Calibrated propensities** so the importance weights are honest.
- **Support-health diagnostics** — `ESS`, `match_rate`, PSIS Pareto-k,
  propensity calibration (ECE) — surfaced as `ok` / `caution` / `high_risk`.
- **A stakeholder card** you can hand to a non-statistician.

## Where to go next

- New here? Start with **[Install](getting-started/install.md)** then the
  **[Quickstart](getting-started/quickstart.md)**.
- Want the full workflow? Read the
  **[logs → experiment-review card recipe](recipes/logs-to-experiment-card.md)**.
- Not sure whether to trust an estimate? See
  **[good vs bad support](recipes/good-vs-bad-support.md)** and
  **[reading the report](report-interpretation.md)**.
- Curious about the math? See
  **[estimands & assumptions](concepts/estimands-and-assumptions.md)** and
  **[methods](methods.md)**.
- Coming from another library? See
  **[comparisons vs OBP / SCOPE-RL / d3rlpy](comparisons.md)**.

!!! warning "When *not* to use offline evaluation as deployment evidence"
    If the logged data does not support the candidate policy — poor overlap,
    near-deterministic logging, heavy importance-weight tails — the estimate is
    not deployment evidence. `skdr-eval` is designed to *tell you that loudly*
    via `support_health`. Read the
    [good vs bad support tutorial](recipes/good-vs-bad-support.md) first.
