# The Daily Driver guide: logs → evaluation → decision

This is the practical, end-to-end workflow for an analyst or ML team using
`skdr-eval` as a routine pre-deployment check. It assumes you already have
logged decisions and a candidate policy; if you are new to the concepts, read
the [quickstart](getting-started/quickstart.md) first.

The loop is always the same five stages. Each one has an off-ramp: a point
where the honest answer might be "stop, the logs don't support this."

```
logs → 1. preflight → 2. wrap candidate → 3. evaluate → 4. read the card → 5. decide
```

## 1. Preflight your logs with `doctor`

Before any estimation, check the data is even shaped for OPE. The
[`doctor`](api.md) preflight runs a non-raising battery of schema,
time-ordering, support, and missingness checks.

```python
import skdr_eval

report = skdr_eval.doctor(logs, kind="standard", metric_col="reward")
print(report.to_text())          # human-readable
report_json = report.to_dict()    # machine-readable for CI
```

Or from the command line (text or `--json`):

```bash
skdr-eval doctor your_logs.parquet --json
```

**Off-ramp:** if `doctor` reports schema failures or no eligibility structure,
fix the data before going further — a clean estimate on malformed logs is
worse than no estimate.

## 2. Wrap your candidate policy

`skdr-eval` evaluates any policy you can express as a scikit-learn-compatible
model (`fit` / `predict`). A candidate is just a dict of named models:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

models = {"candidate_v2": HistGradientBoostingRegressor(random_state=0)}
```

If your decisions come from agent traces or another framework rather than a
tidy table, map them first with the
[trace adapter](datasets.md) (`skdr_eval.adapters.from_records` /
`from_jsonl_trace`).

## 3. Run the evaluation

```python
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    y_col="reward",
    fit_models=True,
    n_splits=3,
    policy_train="pre_split",   # explicit: avoids the deprecation warning
    random_state=0,
)
```

The returned `EvaluationArtifact` bundles everything: `artifact.report` (the
headline V̂ / SE / ESS table), `artifact.warnings` (per-row `support_health`),
`artifact.sensitivity` (the clip-grid sweep), and `artifact.diagnostics`
(calibration / overlap). See the
[metrics glossary](metrics-glossary.md) for every column.

## 4. Read the card before the number

The single most important habit: **check support health before you look at
V̂.** A better-looking estimate on unsupported logs is a trap.

```python
# Per-(model, estimator) trust verdict.
rec = artifact.recommendation("candidate_v2", estimator="SNDR")
print(rec.verdict)   # "deploy" | "ab_test" | "do_not_deploy" | "insufficient_evidence"

# Stakeholder-ready card (HTML report + machine-readable YAML/JSON).
artifact.to_html("report.html")
artifact.save_card("candidate_v2", estimator="SNDR", path="card.yaml")
```

Work top-down, exactly as in the
[report interpretation guide](report-interpretation.md):

1. **`support_health`** — `high_risk` means stop; the logs do not support the
   question.
2. **Warning codes** — `LOW_ESS`, `HIGH_PARETO_K`, `EXTREME_CLIP`,
   `MISCAL_PROP`, … each points at a specific failure mode.
3. **Sensitivity** — does V̂ move when the clip threshold changes? A stable
   estimate is a trustworthy one.
4. **Calibration & overlap** — is the propensity model believable?

## 5. Make the decision

Map the verdict to an action:

| Verdict | What it means | Next step |
|---|---|---|
| `deploy` | strong, well-supported improvement | proceed to a confirmatory A/B test |
| `ab_test` | promising and supported, uncertain magnitude | design the A/B test (see below) |
| `insufficient_evidence` | supported but no separation from baseline | gather more logs or accept no change |
| `do_not_deploy` | poor support or a regression | do **not** ship; fix logging/overlap first |

For CI gates, the CLI exits with code `3` on any `do_not_deploy`, so a
policy-change PR can be blocked automatically:

```bash
skdr-eval evaluate logs.parquet --model candidate_v2.pkl || echo "gate failed"
```

## When not to trust the estimate

- `support_health == "high_risk"` — overlap or ESS is too low to trust V̂.
- V̂ swings across the clip grid — the estimate is dominated by a few extreme
  weights, not the data.
- The propensity model is miscalibrated (`MISCAL_PROP`) — the importance
  weights are built on a shaky base.
- The candidate proposes actions the logging policy never took — there is no
  counterfactual to learn from.

In all of these, the right move is to improve the logs (more exploration,
better instrumentation) — not to reach for a different estimator.

## From offline evidence to an online test

A good offline verdict is the *start* of validation, not the end. When the
verdict is `deploy` or `ab_test`, move to a canary or A/B test; the
[production rollout checklist](rollout-checklist.md) walks through sizing,
monitoring, and rollback. Offline evaluation tells you what is *worth*
testing online — it does not replace the test.

## See also

- [Report interpretation guide](report-interpretation.md)
- [Choosing an estimator](choosing-an-estimator.md)
- [Production rollout checklist](rollout-checklist.md)
- [What skdr-eval claims](https://github.com/dgenio/skdr-eval/blob/main/CLAIMS.md)
