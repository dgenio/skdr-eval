# skdr-eval

[![PyPI version](https://badge.fury.io/py/skdr-eval.svg)](https://badge.fury.io/py/skdr-eval)
[![Python versions](https://img.shields.io/pypi/pyversions/skdr-eval.svg)](https://pypi.org/project/skdr-eval/)
[![CI](https://github.com/dgenio/skdr-eval/workflows/CI/badge.svg)](https://github.com/dgenio/skdr-eval/actions)
[![Coverage](https://codecov.io/gh/dgenio/skdr-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/dgenio/skdr-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Read the Weaver Stack overview on Towards AI](https://img.shields.io/badge/Read_the_overview-Towards_AI-black?logo=medium&logoColor=white)](https://pub.towardsai.net/the-weaver-stack-one-contract-layer-for-safe-llm-agents-7f733cad5eac)

**You trained a better recommender, routing model, treatment rule, or targeting
policy. Offline metrics look good — but deploying it directly is risky.**
`skdr-eval` estimates how that candidate policy *would have* performed on your
logged decisions, and tells you whether the logs have enough support to trust
the estimate before you spend an A/B test on it.

```text
logged decisions (context, action, outcome, time)  +  candidate sklearn-like policy
                              │
                              ▼
            offline policy evaluation  (DR / SNDR)
                              │
                              ▼
   trust diagnostics  (support-health · overlap · ESS · Pareto-k · calibration · sensitivity)
                              │
                              ▼
        decision artifact  (HTML report · machine-readable card)
```

```bash
pip install skdr-eval
```

> **Use this when:** you have logged `(context, action, reward)` decisions, a
> candidate policy you can wrap behind `fit`/`predict`, and you want a
> trustworthy pre-A/B-test read on it.
>
> **Do not use this when:** your problem is sequential / reinforcement learning
> (reach for [SCOPE-RL or d3rlpy](docs/comparisons.md)), or your logs have no
> overlap with what the candidate would do — no OPE method can rescue that, and
> `skdr-eval` will say so via `support_health = high_risk` rather than return a
> confident number.

Doubly Robust (DR) and Stabilized DR (SNDR) are the estimators under the hood;
you do not need to know the math to read the result. Full positioning vs. other
OPE/RL libraries: [`docs/comparisons.md`](docs/comparisons.md).

Try it in your browser — no install needed:

| Notebook | Open in Colab |
|---|---|
| Quickstart (contextual-bandit OPE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/01_quickstart.ipynb) |
| Pairwise / autoscaling quickstart | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/02_pairwise_quickstart.ipynb) |
| E-commerce ranking use case | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/03_ecommerce_ranking.ipynb) |
| Ad targeting use case | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/04_ad_targeting.ipynb) |
| Healthcare CATE use case | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/05_healthcare_cate.ipynb) |

## What is this?

`skdr-eval` is a Python library for **offline policy evaluation** — estimating how well a candidate decision policy would have performed *from logged data alone*, without deploying it. It implements Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators on top of `scikit-learn`-protocol models, with first-class support for time-correlated logs, calibrated propensities, moving-block bootstrap confidence intervals, and a single bundled `EvaluationArtifact` that exposes per-decision diagnostics, clip-grid sensitivity, PSIS Pareto-k support-health, propensity calibration (ECE / Brier), and a renderable HTML stakeholder card.

It started life as an internal tool for call-routing / service-time minimization (and still ships a pairwise / autoscaling layer for that use case), but the underlying machinery applies to contextual-bandit OPE generally — wherever you have logged one-shot decisions and a sklearn-compatible candidate policy.

Under standard OPE assumptions (unconfoundedness, overlap, a stable
data-generating process, and useful nuisance models), `skdr-eval` produces an
*estimate plus trust diagnostics* — not a guarantee. The diagnostics are signals
that help you decide whether an estimate is worth acting on; they do not prove
it is correct, and offline evaluation does not replace online validation.

## When should I use this?

Reach for `skdr-eval` when **all** of the following are true:

- You have **logged data** of the form `(context x, action a, reward y)` from a policy you no longer want to keep running unchanged.
- You want to evaluate a **candidate policy** (a recommender, a ranker, a clinical decision rule, a routing model, an ad targeter) **before** A/B testing it, because A/B testing has a real cost (lost revenue, patient risk, SLA violations, operator overtime).
- Your candidate policy is, or can be wrapped behind, a **scikit-learn-protocol** estimator — `fit` / `predict` (or `predict_proba`) is enough.
- The logged decisions cover the actions the candidate policy *would* take with non-trivial probability (i.e., there is reasonable **overlap** / positivity). `skdr-eval` will warn you when overlap is thin via PSIS Pareto-k, ESS, and match-rate diagnostics.

Typical use cases:

- **Recommender / ranking systems** — evaluate a new model against logged session data.
- **Ad targeting** — score a candidate bidding policy on Criteo-style counterfactual logs.
- **Healthcare CATE** — compare a treatment-assignment rule to standard-of-care on retrospective records.
- **Call routing / autoscaling** — choose between client-operator assignment policies on historical traffic (the original motivating use case, still first-class via `evaluate_pairwise_models`).
- **Any contextual-bandit decision** where re-running history would be too expensive or risky to do live.

If you need *slate* / top-K ranking estimators (Cascade-DR, Reward-Interaction IPS) or *MIPS* for very large action spaces, those are tracked on the roadmap (#75, #85) but not yet shipped.

**When *not* to use it:**

- Your problem is **sequential / reinforcement learning** (state transitions,
  long horizons) — use [SCOPE-RL or d3rlpy](docs/comparisons.md) instead.
- You need a wide bank of research estimators or to reproduce published bandit
  benchmarks — [Open Bandit Pipeline](docs/comparisons.md) is the reference.
- Your logs have **no overlap** with what the candidate policy would do; no OPE
  method can fix that, and `skdr-eval` will flag it as `high_risk` rather than
  hand you a confident number.

See [`docs/comparisons.md`](docs/comparisons.md) for an honest, side-by-side
comparison against OBP, SCOPE-RL, d3rlpy, and banditml.

## First 10 minutes: understand what skdr-eval does

If the purpose is not obvious yet, follow this path — it mirrors how the
library actually gets used, and it does not require reading any theory first:

1. **Run the quickstart notebook**
   ([`01_quickstart.ipynb`](examples/notebooks/01_quickstart.ipynb), or click
   the Colab badge above). Watch historical logs + candidate models turn into
   an `EvaluationArtifact`. Notice `artifact.report`, `support_health`, the
   warnings, and the exported card.
2. **Open the generated report / card** and ask *"is this estimate trustworthy
   enough to discuss?"*. The
   [report interpretation guide](docs/report-interpretation.md) walks you from
   the HTML output to an actual decision.
3. **Reach for the [metrics glossary](docs/metrics-glossary.md) only when a
   field is unclear** — `V_hat`, `ESS`, `pareto_k`, `support_health`, the
   warning codes. Don't force yourself through theory before the
   job-to-be-done makes sense.
4. **Then choose your path:** standard contextual-bandit evaluation, the
   pairwise / call-routing / autoscaling API, or a domain use case under
   [`examples/use_cases/`](examples/use_cases/). To *see* the difference
   between healthy and unhealthy support, run the
   [known-failure demos](examples/known_failures/README.md).

## Where to start

- **Full documentation:** **[skdr-eval.readthedocs.io](https://skdr-eval.readthedocs.io/)** (built from [`docs/`](docs/) with MkDocs Material).
- **Just want to see it work?** Click any "Open in Colab" badge above.
- **First time here?** Follow [First 10 minutes](#first-10-minutes-understand-what-skdr-eval-does) above.
- **Have logs already?** Skim [Quick Start](#quick-start) below; the standard / pairwise variants are both two screens long. The full workflow is the [Daily Driver guide](docs/daily-driver.md) (and the [logs → experiment-review card recipe](docs/recipes/logs-to-experiment-card.md)).
- **Not sure whether to trust an estimate?** Read the [report interpretation guide](docs/report-interpretation.md), the [metrics glossary](docs/metrics-glossary.md), and the [good-vs-bad support tutorial](docs/recipes/good-vs-bad-support.md).
- **Which estimator should I run?** See the [estimator selection guide](docs/choosing-an-estimator.md).
- **What does it actually claim?** Read [`CLAIMS.md`](CLAIMS.md) — receipts for every claim, plus the explicit non-claims.
- **Going to production?** Work through the [production rollout checklist](docs/rollout-checklist.md) before any online test.
- **Comparing against another OPE library?** See [`docs/comparisons.md`](docs/comparisons.md) for OBP / SCOPE-RL / d3rlpy / banditml, the [methodological positioning](docs/methods.md), and the runnable [coming-from-OBP interop guide](docs/recipes/obp-interop.md).
- **Looking for end-to-end examples by domain?** Browse [`examples/use_cases/`](examples/use_cases/) for runnable scripts (e-commerce ranking, ad targeting, healthcare CATE, call routing, logs→card, agent routing).
- **Evaluating an LLM reranker or an agent routing/tool-selection policy?** See the [agent routing & tool-selection guide](docs/agent-routing.md) and [Evaluate LLM / agent policies offline](#evaluate-llm--agent-policies-offline) below.
- **Building a pipeline on the API?** Check the [public API, stability policy & road to 1.0](docs/api-stability.md).
- **Want to add an estimator?** Start with the [architecture tour](docs/architecture.md) and the [write-your-own-estimator guide](docs/extending/add-an-estimator.md).

> The `skdr-eval` CLI (`pip install 'skdr-eval[cli]'`) makes the same
> evaluators reachable from a terminal — see [Command-line interface](#command-line-interface).
> Run `skdr-eval doctor logs.parquet` before evaluation to catch schema and
> environment problems early.

## Table of Contents

- [First 10 minutes](#first-10-minutes-understand-what-skdr-eval-does)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Standard Evaluation](#standard-evaluation)
  - [Pairwise Evaluation](#pairwise-evaluation)
- [Command-line interface](#command-line-interface)
- [Reference & theory](#reference--theory)
- [Examples](#examples)
- [Development & releasing](#development--releasing)
- [Citation](#citation)

## Features

- 🎯 **Doubly Robust Estimation**: Implements both DR and Stabilized DR (SNDR) estimators
- ⏰ **Time-Aware Evaluation**: Uses time-series splits and calibrated propensity scores
- 🔧 **Sklearn Integration**: Easy integration with scikit-learn models
- 📊 **Comprehensive Diagnostics**: ESS, match rates, propensity score analysis
- 🧰 **Engineered for reuse**: Fully type-hinted, tested, and documented (offline evaluation does not replace online validation)
- 📈 **Bootstrap Confidence Intervals**: Moving-block bootstrap for time-series data
- 🤝 **Pairwise Evaluation**: Client-operator pairwise evaluation with autoscaling strategies
- 🎛️ **Autoscaling**: Direct, stream, and stream_topk strategies with policy induction
- 🧮 **Choice Models**: Conditional logit models for propensity estimation

## Installation

```bash
pip install skdr-eval
```

Conditional-logit choice models work out of the box — SciPy is a core
dependency, so no extra install is required.

### Optional Dependencies

For speed optimizations (PyArrow, Polars):
```bash
pip install skdr-eval[speed]
```

For development:
```bash
git clone https://github.com/dgenio/skdr-eval.git
cd skdr-eval
pip install -e .[dev]
```

To run the Colab quickstart notebooks locally:
```bash
pip install 'skdr-eval[notebooks]'
jupyter notebook examples/notebooks/
```

## Quick Start

### Preflight

Before a real evaluation, confirm your environment + schema in one shot:

```python
import skdr_eval

# Which optional extras are installed?
print(skdr_eval.get_capabilities())
# {'viz': True, 'speed': False, 'missing_extras': ['speed']}

# Validate your logs match the schema evaluate_sklearn_models expects.
skdr_eval.validate_logs(logs, strict=True)

# For the pairwise API:
skdr_eval.validate_pairwise_inputs(
    logs_df, op_daily_df, metric_col="service_time", strict=True,
)
```

See `examples/preflight.py` for a runnable script — wire it into CI to catch
schema or extras drift before the long-running evaluation kicks off.

### Standard Evaluation

```python
import skdr_eval
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# 1. Generate synthetic service logs
logs, ops_all, true_q = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)

# 2. Define candidate models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}

# 3. Evaluate models using DR and SNDR
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    fit_models=True,
    n_splits=3,
    random_state=42,
    policy_train="pre_split",  # reserve a holdout slice for policy training
)
# `policy_train="pre_split"` fits the policy on the first 85% of the data and
# evaluates on the held-out tail, avoiding training-on-test bias when
# `fit_models=True`. Omitting it falls back to `"pre_split"` with a
# DeprecationWarning; pass it explicitly to keep the output clean.

# 4. View results
print(artifact.report[['model', 'estimator', 'V_hat', 'ESS', 'match_rate']])

# 5. Trust signals (issue #22 / #23)
print(artifact.warnings)        # per-(model, estimator) support_health + codes
print(artifact.sensitivity)     # clip-grid value range and stability flag
print(artifact.diagnostics)     # propensity overlap / calibration / discrimination

# 6. Export (issue #28) and stakeholder card (issue #30)
artifact.export("artifacts/run", formats=["json", "html"])
artifact.save_card("artifacts/run_card.html", "RandomForest")

# 7. Per-decision contributions (issue #92) — opt in with keep_contributions=True
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    fit_models=True,
    n_splits=3,
    random_state=42,
    policy_train="pre_split",
    keep_contributions=True,  # attach per-decision DR/SNDR pseudo-outcomes
)
contribs = artifact.contributions("RandomForest", estimator="DR", top_k=5)
print(contribs)  # decision_id, q_pi, q_hat, weight, reward, contribution_to_V
#  contribution_to_V.mean() == V_hat by construction (float64 precision)
```

If your logs name the reward column anything other than `service_time` (e.g.,
`reward`, `click`, `revenue`), pass it via the `y_col` keyword:

```python
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs.rename(columns={"service_time": "reward"}),
    models=models,
    fit_models=True,
    n_splits=3,
    policy_train="pre_split",
    y_col="reward",  # name of the reward column in your logs
)
```

> **Breaking change in 0.6.0:** `evaluate_sklearn_models` and
> `evaluate_pairwise_models` now return a single `EvaluationArtifact`
> instead of the legacy `(report, detailed)` tuple. Unpack
> `artifact.report` / `artifact.detailed` to migrate.

### Pairwise Evaluation

```python
import skdr_eval
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Generate synthetic pairwise data (client-operator pairs)
logs_df, op_daily_df = skdr_eval.make_pairwise_synth(n_days=3, n_clients_day=500, n_ops=10, seed=42)

# 2. Train model on observed data
feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
model = HistGradientBoostingRegressor(random_state=42)
model.fit(logs_df[feature_cols].values, logs_df["service_time"].values)

# 3. Run pairwise evaluation
artifact = skdr_eval.evaluate_pairwise_models(
    logs_df=logs_df,
    op_daily_df=op_daily_df,
    models={"HGB": model},
    metric_col="service_time",
    task_type="regression",
    direction="min",
    strategy="auto",
    n_splits=3,
    random_state=42,
    policy_train="pre_split",
)

# 4. View results
print(artifact.report[["model", "estimator", "V_hat", "ESS", "match_rate"]])
print(artifact.warnings)
```

#### External (simulator) policies, what-if scenarios, and large data

Three pairwise extensions share the same `EvaluationArtifact` and trust
diagnostics (see [`examples/use_cases/07_simulator_and_scenarios.py`](examples/use_cases/07_simulator_and_scenarios.py)):

```python
# Score policies from an external decision process (e.g. a call-centre
# simulator), instead of inducing them from candidate models:
artifact = skdr_eval.evaluate_external_policies(
    logs_df=logs_df,
    op_daily_df=op_daily_df,
    policies={"simulator_a": assignments_df},  # DataFrame[client_id, operator_id]
    metric_col="service_time", task_type="regression", direction="min",
)

# What-if: re-evaluate a policy under reduced operator capacity:
artifact = skdr_eval.simulate_autoscaling_scenario(
    logs_df, op_daily_df, models={"HGB": model},
    scenario={"capacity_multiplier": 0.6, "eligibility_mode": "as_logged"},
    metric_col="service_time", task_type="regression", direction="min",
)

# Large inputs: a vectorized path, numerically identical to the default:
artifact = skdr_eval.evaluate_pairwise_models(
    ..., execution_mode="large_data",
)
```

## Reference & theory

The full Python API — function signatures and parameters, DR/SNDR theory, the
autoscaling strategies, and bootstrap confidence intervals — lives in the docs
so it stays in sync with the code:

- **[Python API & theory](https://skdr-eval.readthedocs.io/en/latest/python-api/)** —
  narrative reference for the core functions plus the estimator math.
- **[API reference](https://skdr-eval.readthedocs.io/en/latest/api/)** —
  auto-generated symbol reference.
- **[Methods (DR / SNDR)](https://skdr-eval.readthedocs.io/en/latest/methods/)** —
  estimands, assumptions, and citations.
- **[Choosing an estimator](https://skdr-eval.readthedocs.io/en/latest/choosing-an-estimator/)**
  and the **[metrics glossary](https://skdr-eval.readthedocs.io/en/latest/metrics-glossary/)**.

The runnable [`examples/`](examples/) are the most trustworthy usage guide.

## Command-line interface

The `skdr-eval` CLI ships behind the `[cli]` extra and exposes the same
evaluation surface to teams that don't want to write Python.

```bash
pip install 'skdr-eval[cli]'

# Zero-to-card in one command (synth logs → doctor → evaluate → explain).
# Great first run; always exits 0 (it's a demo, not a CI gate).
skdr-eval quickstart --out ./quickstart

# See which optional extras are installed and what each one unlocks.
skdr-eval capabilities
skdr-eval capabilities --json | jq .

# Quick environment + schema probe before evaluation.
skdr-eval doctor logs.parquet
skdr-eval doctor logs.parquet --json | jq .
# Emit a copy-paste, data-free reproduction snippet for a bug report.
skdr-eval doctor logs.parquet --repro

# Validate logs against the schema (exit code 1 on failure — useful in CI).
skdr-eval validate-schema logs.parquet --strict
skdr-eval validate-schema pw_logs.parquet --kind pairwise \
    --op-daily pw_op.parquet --metric-col service_time

# Run a full evaluation from disk.
skdr-eval evaluate logs.parquet \
    --model HGB=model.joblib \
    --policy-train pre_split \
    --n-splits 3 \
    --out ./run \
    --tracker-dir ./tracker_runs/2026-05-20

# Re-render a card directly from a saved artifact.json.
skdr-eval card ./run/artifact.json --model HGB --estimator DR \
    --out ./run/card.yaml --format yaml

# Narrate *why* a saved artifact got its verdict (no re-evaluation).
skdr-eval explain ./run/artifact.json --model HGB --estimator SNDR
skdr-eval explain ./run/artifact.json --model HGB --json | jq .

# Stable exit codes (good for CI gates):
#   0 — success: no 'do_not_deploy' or 'insufficient_evidence' verdict was
#       produced (an uncomputable recommendation is logged and does not flip this)
#   1 — data / schema error
#   2 — environment / import error
#   3 — at least one evaluated estimator's verdict is 'do_not_deploy'
#       (checked across DR/SNDR/MRDR/SWITCH-DR/DRos/MIPS; takes precedence over 4)
#   4 — at least one estimator's verdict is 'insufficient_evidence' and none is
#       'do_not_deploy' — the logs can't yet support a deploy decision, so an
#       honest "we can't tell" does not pass the gate as green
```

> The verdict gate inspects **every** estimator present in the artifact, not
> just `DR`/`SNDR`. `insufficient_evidence` (exit 4) usually means no bootstrap
> CI was available — re-run with `--ci-bootstrap`. See
> [`docs/report-interpretation.md`](docs/report-interpretation.md#deployment-verdicts).

## Preflight diagnostics: `skdr_eval.doctor`

`skdr_eval.doctor(logs, *, kind='standard'|'pairwise', op_daily_df=None,
metric_col='service_time', n_splits=3, strict=False)` returns a non-raising
`DoctorReport` that surfaces environment + schema + statistical sanity
failures with actionable fix hints.

```python
import skdr_eval

logs, _, _ = skdr_eval.make_synth_logs(n=5000, n_ops=3, seed=0)
report = skdr_eval.doctor(logs)
report.print()            # text table with status glyphs + capability matrix
report.to_markdown()      # copy-pasteable Markdown
report.to_dict()          # JSON-serializable (checks + capabilities + profile)
report.to_repro()         # data-free minimal reproduction snippet for bug reports
assert report.ok          # True iff no Check has status='fail'
```

Checks cover environment, schema, duplicates, **time ordering** (time-aware CV
assumes chronological logs), **column missingness**, finite outcomes,
positivity, and sample size. The report also carries a full **optional-extra
capability matrix** — the same data behind `skdr_eval.get_capability_matrix()`
and `skdr-eval capabilities` — and a privacy-safe `DataProfile` (column
names/dtypes/shape only) used by `to_repro()`.

You can also narrate a verdict programmatically:

```python
artifact = skdr_eval.evaluate_sklearn_models(logs=logs, models=models)
print(artifact.explain("HGB", estimator="SNDR").to_text())
# Or, from a saved artifact.json, without re-running the evaluation:
schema = skdr_eval.load_artifact_json("run/artifact.json")
skdr_eval.explain_artifact_schema(schema, "HGB", estimator="SNDR").to_dict()
```

## Machine-readable cards: `EvaluationCard`

`EvaluationArtifact.card_schema(model_name, estimator='SNDR')` builds an
`EvaluationCard` — the typed sibling of the HTML stakeholder card. The card
is YAML/JSON round-trippable, exposes a stable `json_schema()` for downstream
tooling, and is ideal for CI gates and Git-pinned snapshots of an evaluation.

```python
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs, models=models, fit_models=True, policy_train="pre_split"
)
card = artifact.card_schema("RandomForest", estimator="DR")

card.to_yaml("artifacts/rf.card.yaml")
card.to_json("artifacts/rf.card.json")

# Round-trip
loaded = skdr_eval.EvaluationCard.from_yaml("artifacts/rf.card.yaml")
assert loaded == card

# CI gate
if card.trust.recommendation and card.trust.recommendation["verdict"] == "do_not_deploy":
    raise SystemExit(1)
```

## Experiment tracker

`evaluate_sklearn_models` and `evaluate_pairwise_models` both accept a
`tracker=` kwarg. The default `NullTracker` is a no-op (so the evaluator is
unchanged when omitted). `FileTracker` writes a deterministic run directory
to disk; external adapters (`MLflowTracker`, `WandbTracker`, `AimTracker`)
ship as stubs behind the `[mlflow]` / `[wandb]` / `[aim]` extras and are
filled in under umbrella issue #73.

```python
from skdr_eval import FileTracker

with FileTracker(root="runs/2026-05-20") as tracker:
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs, models=models, fit_models=True,
        policy_train="pre_split", tracker=tracker,
    )
# Writes:
#   runs/2026-05-20/metrics.jsonl          (one row per logged metric)
#   runs/2026-05-20/tags.json
#   runs/2026-05-20/artifacts/...
#   runs/2026-05-20/cards/<model>_<estimator>.card.yaml
```

## Evaluate LLM / agent policies offline

You don't need a classic tabular logging pipeline to use `skdr-eval`. If you
have **logged LLM-reranker or agent routing / tool-selection decisions**, you
can estimate whether a *candidate* policy would do better — offline, CPU-only,
before an A/B test.

- **LLM reranker:** *"How do I know my new reranker is actually better, without
  shipping it?"* The [LLM-reranker OPE recipe](docs/recipes/llm-reranker-ope.md)
  ([notebook](examples/notebooks/10_llm_reranker_ope.ipynb) ·
  [Colab](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/10_llm_reranker_ope.ipynb))
  evaluates a candidate reranker with embedding-**MIPS** and recovers a known
  ground-truth value within ±2 SE.
- **Agent routing / tool selection:** map generic `(context, action, reward)`
  traces with the trace adapter and evaluate a candidate routing policy:

  ```python
  import skdr_eval

  adapted = skdr_eval.adapters.from_jsonl_trace("agent_traces.jsonl", reward_col="cost")
  artifact = skdr_eval.evaluate_sklearn_models(
      logs=adapted.logs, models={"candidate": my_cost_model}, y_col=adapted.reward_col,
  )
  print(artifact.report[["model", "estimator", "V_hat", "support_health"]])
  ```

  See the [agent routing & tool-selection guide](docs/agent-routing.md) for
  the full conceptual walkthrough (actions, rewards, support, and when OPE is
  the wrong tool), [`examples/use_cases/06_agent_routing_policy.py`](examples/use_cases/06_agent_routing_policy.py)
  (healthy vs. `high_risk` support), and the
  [offline-evaluation companion guide](docs/weaver-stack.md).

The trace adapter (`skdr_eval.adapters.from_records` / `from_jsonl_trace`) maps
`(context, action, reward[, timestamp, propensity])` records — including JSONL
agent traces — into a schema-valid logs frame, so you don't hand-shape a
DataFrame. skdr-eval estimates calibrated propensities internally; a logged
propensity in the trace is reported but not consumed.

## Examples

`examples/` ships three kinds of runnable artifacts — pick the one that
matches how you want to consume them:

| Path | Format | Use when |
|---|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | `.py` | Headless / CI / no Jupyter installed. |
| [`examples/quickstart_pairwise.py`](examples/quickstart_pairwise.py) | `.py` | Same, for the pairwise / autoscaling API. |
| [`examples/preflight.py`](examples/preflight.py) | `.py` | One-shot capability + schema check before a long evaluation. |
| [`examples/notebooks/`](examples/notebooks/) | `.ipynb` × 5 | Colab-runnable; click the badges at the top of this README. |
| [`examples/use_cases/`](examples/use_cases/) | `.py` × 4 | Self-contained domain walk-throughs (e-commerce ranking, ad targeting, healthcare CATE, call routing). |

CI exercises `examples/preflight.py`, `examples/quickstart.py`, every
notebook in `examples/notebooks/`, and every script in
`examples/use_cases/` on every PR — they cannot silently rot.

To run a domain example locally:

```bash
python examples/use_cases/01_ecommerce_ranking.py
python examples/use_cases/02_ad_targeting.py
python examples/use_cases/03_healthcare_cate.py
python examples/use_cases/04_call_routing.py
```

To open the notebooks locally:

```bash
pip install 'skdr-eval[notebooks]'
jupyter notebook examples/notebooks/
```

### Estimator family (DR, SNDR, MRDR, SWITCH-DR, DRos, MIPS)

The strategy seam introduced in `skdr_eval.estimators` (issues #85, #86) lets
you opt into additional DR variants without leaving the high-level API:

```python
import skdr_eval
from sklearn.ensemble import HistGradientBoostingRegressor

logs, _, _ = skdr_eval.make_synth_logs(n=2000, n_ops=5, seed=0)
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models={"hgb": HistGradientBoostingRegressor(random_state=0)},
    fit_models=True,
    policy_train="pre_split",
    n_splits=3,
    random_state=0,
    estimators=("DR", "SNDR", "MRDR", "SWITCH-DR", "DRos"),
    switch_tau=10.0,
    dros_lam=2.0,
)
print(artifact.report[["estimator", "V_hat", "SE_if", "ESS"]])
```

For `MIPS` (Marginalized IPS), supply an `action_embedding` — either an
`(n_actions, embed_dim)` matrix **or a logs column name** holding a per-row
embedding vector (#136). The kernel and bandwidth are configurable:
`mips_kernel="rbf"` (default) | `"linear"` | a callable, and
`mips_bandwidth=0.5` or `"median"` for the median heuristic. The
`skdr_eval.embedding_sufficiency_diagnostic(...)` helper flags whether the
embedding captures enough of the action-driven reward signal for MIPS to be
approximately unbiased. If `"MIPS"` is requested without an
`action_embedding`, MIPS gracefully falls back to SNDR with a warning rather
than failing. The logs-column-name form is only supported when the action
index is a stable global action id; the pairwise evaluator (`A` is a
day-relative operator index) requires the explicit `(n_actions, embed_dim)`
array instead.

See `examples/quickstart_estimators.py` and `examples/quickstart_mips.py`
for runnable walkthroughs.

### Slate / top-K off-policy evaluation

The `skdr_eval.slate` subpackage (issue #75) ships four ranking-OPE
estimators — `slate_standard_ips`, `reward_interaction_ips`,
`pseudo_inverse_ips`, and `slate_cascade_dr` — plus a synthetic
cascade-click generator `make_slate_synth(...)` with closed-form ground
truth. The top-level `evaluate_slate_models(...)` entry point (#135) bundles
them into an `EvaluationArtifact` — report, support-health warnings,
sensitivity, and a renderable card — the same surface as
`evaluate_sklearn_models`:

```python
import skdr_eval

logs, attractiveness, truth = skdr_eval.make_slate_synth(
    n_impressions=500, n_items=10, slate_size=3, seed=0
)

def my_reranker(rank: int, item: int) -> float:
    return 1.0 / attractiveness.shape[1]  # your per-rank target policy

artifact = skdr_eval.evaluate_slate_models(
    logs,
    models={"my_reranker": my_reranker},
    estimators=("RIPS", "PI-IPS", "SlateCascadeDR"),
    baseline="logging",
)
print(artifact.report[["estimator", "V_hat", "SE_if", "ESS", "support_health"]])
```

See `examples/quickstart_slate.py` and
[`docs/slate-vs-pairwise-vs-standard.md`](docs/slate-vs-pairwise-vs-standard.md)
for when to reach for slate vs pairwise vs standard evaluation.

### Recipes / LLM-reranker OPE

`skdr_eval.recipes` ships opinionated, end-to-end workflows built on the core
estimators. The first recipe (#95) evaluates an **LLM reranker** offline with
**embedding-MIPS** — no runtime LLM dependency, CPU-only:

```python
from skdr_eval.recipes import (
    make_llm_reranker_synth, LLMRerankerLogSchema,
    induce_reranker_policy, evaluate_reranker_mips,
)

# Deterministic logs with a known ground-truth target value.
logs, candidate_embeddings, truth = make_llm_reranker_synth(
    n_queries=2000, candidates_per_query=20, embed_dim=32, seed=42
)
LLMRerankerLogSchema.validate_frame(logs)          # validate your own logs

# Candidate (new) reranker = sort candidates by query·candidate dot product.
policy = induce_reranker_policy(logs, candidate_embeddings)
result = evaluate_reranker_mips(logs, candidate_embeddings, policy)
print(result.V_hat, "vs true", truth.V_sort_by_dot)  # recovers within ±2 SE
```

A full walkthrough is in
[`examples/notebooks/10_llm_reranker_ope.ipynb`](examples/notebooks/10_llm_reranker_ope.ipynb).

## Development & releasing

```bash
git clone https://github.com/dgenio/skdr-eval.git
cd skdr-eval
pip install -e .[dev]
make check        # fast inner loop (lint + typecheck + test + smoke)
make ci-local     # CI-faithful pre-PR pass
```

Full contributor workflow, the statistical-change bar, and the
trusted-publishing release process are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Ecosystem

`skdr-eval` is **standalone-first and MIT-licensed**, with no runtime dependency
on any other framework. It also serves as an optional **offline-evaluation
companion** to agent stacks (e.g. the Weaver stack): those systems produce
logged agent decisions, and `skdr-eval` answers *"would this candidate
routing/tool-selection policy actually perform better, from these logs, before
we ship it?"* — including the honest *"your logs don't support that question
yet"* answer.

See [`docs/weaver-stack.md`](docs/weaver-stack.md) for the companion guide,
positioning boundary, and cross-links.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{santos2026skdreval,
  title   = {{skdr-eval}: Offline Policy Evaluation for {sklearn}-Compatible Models with Time-Aware Doubly Robust Estimators},
  author  = {Santos, Diogo},
  year    = {2026},
  url     = {https://github.com/dgenio/skdr-eval},
  version = {0.12.0},
  license = {MIT}
}
```

For one-click copy, the same entry — plus the foundational method references
(DR, PSIS, calibration, double machine learning) — is available as
[`CITATION.bib`](CITATION.bib) and as machine-readable
[`CITATION.cff`](CITATION.cff).

A Zenodo concept DOI is being minted on the next tagged release (see
[`docs/zenodo.md`](docs/zenodo.md)); after that, `CITATION.cff` will carry
the canonical DOI and replace the URL field above.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Uses [pandas](https://pandas.pydata.org/) for data manipulation
- Follows [PEP 621](https://peps.python.org/pep-0621/) for project metadata
