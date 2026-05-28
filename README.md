# skdr-eval

[![PyPI version](https://badge.fury.io/py/skdr-eval.svg)](https://badge.fury.io/py/skdr-eval)
[![Python versions](https://img.shields.io/pypi/pyversions/skdr-eval.svg)](https://pypi.org/project/skdr-eval/)
[![CI](https://github.com/dgenio/skdr-eval/workflows/CI/badge.svg)](https://github.com/dgenio/skdr-eval/actions)
[![Coverage](https://codecov.io/gh/dgenio/skdr-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/dgenio/skdr-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- **Just want to see it work?** Click any "Open in Colab" badge above.
- **First time here?** Follow [First 10 minutes](#first-10-minutes-understand-what-skdr-eval-does) above.
- **Have logs already?** Skim [Quick Start](#quick-start) below; the standard / pairwise variants are both two screens long.
- **Got a report and not sure what it means?** Read the [report interpretation guide](docs/report-interpretation.md) and the [metrics glossary](docs/metrics-glossary.md).
- **Comparing against another OPE library?** See [`docs/comparisons.md`](docs/comparisons.md) for OBP / SCOPE-RL / d3rlpy / banditml, and [`docs/methods.md`](docs/methods.md) for the methodological positioning.
- **Looking for end-to-end examples by domain?** Browse [`examples/use_cases/`](examples/use_cases/) for runnable scripts (e-commerce ranking, ad targeting, healthcare CATE, call routing).

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
- [API Reference](#api-reference)
- [Theory](#theory)
- [Implementation Details](#implementation-details)
- [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
- [Examples](#examples)
- [Development](#development)
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

## API Reference

### Core Functions

#### `make_synth_logs(n=5000, n_ops=5, seed=0)`
Generate synthetic service logs for evaluation.

**Returns:**
- `logs`: DataFrame with service logs
- `ops_all`: Index of all operator names
- `true_q`: Ground truth service times

#### `build_design(logs, cli_pref='cli_', st_pref='st_')`
Build design matrices from logs.

**Returns:**
- `Design`: Dataclass with feature matrices and metadata

#### `evaluate_sklearn_models(logs, models, **kwargs)`
Evaluate sklearn models using DR and SNDR estimators.

**Parameters:**
- `logs`: Service log DataFrame
- `models`: Dict of model name to sklearn estimator
- `fit_models`: Whether to fit models (default: True)
- `n_splits`: Number of time-series splits (default: 3)
- `random_state`: Random seed for reproducibility
- `y_col`: Name of the reward/outcome column (keyword-only, default:
  `"service_time"`). Set it for general-purpose OPE logs whose reward is named
  e.g. `"reward"`, `"click"`, or `"revenue"`:
  `evaluate_sklearn_models(logs=logs, models=models, y_col="reward")`.

**Temporal split controls (keyword-only):**
- `gap`: Samples skipped between train and test in each CV fold
  (default: `1`, conservative adjacent-row leakage guard; `0` for sklearn's
  unbuffered behavior).
- `test_size`: Per-fold test-window size in samples (default: `None`,
  defers to sklearn's automatic sizing).
- `max_train_size`: Cap on training-fold size in samples (default: `None`,
  expanding window). Set this to switch to a sliding-window CV — useful when
  early data is no longer representative.

The same trio is accepted by `evaluate_pairwise_models`,
`fit_propensity_timecal`, `fit_outcome_crossfit`, and
`estimate_propensity_pairwise`.

#### `evaluate_pairwise_models(logs_df, op_daily_df, models, metric_col, task_type, direction, **kwargs)`
Evaluate models using pairwise (client-operator) evaluation with autoscaling.

**Parameters:**

*Required:*
- `logs_df`: Pairwise decision log DataFrame
- `op_daily_df`: Daily operator availability DataFrame
- `models`: Dict of model name to fitted sklearn estimator
- `metric_col`: Target metric column name
- `task_type`: Type of prediction task (`"regression"` or `"binary"`)
- `direction`: Whether to minimize or maximize the metric (`"min"` or `"max"`)

*Optional:*
- `n_splits`: Number of time-series cross-validation splits (default: 3)
- `strategy`: Policy induction strategy (`"auto"`, `"direct"`, `"stream"`, or `"stream_topk"`; default: `"auto"`)
- `propensity`: Propensity estimation method (`"auto"`, `"condlogit"`, or `"multinomial"`; default: `"auto"`). `"auto"` lets `skdr-eval` choose an appropriate method based on the evaluation setup.
- `topk`: Top-K operators for `stream_topk` strategy (default: 20)
- `neg_per_pos`: Negative samples per positive for conditional logit (default: 5)
- `chunk_pairs`: Chunk size for streaming pair generation (default: 2,000,000)
- `min_ess_frac`: Minimum ESS fraction for clipping threshold selection (default: 0.02)
- `clip_grid`: Tuple of clipping thresholds (default: `(2, 5, 10, 20, 50, float("inf"))`)
- `ci_bootstrap`: Whether to compute bootstrap confidence intervals (default: False)
- `alpha`: Significance level for confidence intervals (default: 0.05)
- `outcome_estimator`: Outcome model (depends on `task_type`): for `"regression"`: `"hgb"`, `"ridge"`, `"rf"`; for `"binary"`: `"hgb"`, `"logistic"`; or a callable (default: `"hgb"`)
- `day_col`: Day column name (default: `"arrival_day"`)
- `client_id_col`: Client ID column name (default: `"client_id"`)
- `operator_id_col`: Operator ID column name (default: `"operator_id"`)
- `elig_col`: Eligibility mask column name (default: `"elig_mask"`)
- `random_state`: Random seed for reproducibility (default: 0)

**Returns:**
- `EvaluationArtifact`: bundled result. Use `.report` for the summary DataFrame, `.detailed` for per-model `DRResult`s, `.warnings` for support-health warnings, `.sensitivity` for clip-grid stability, `.diagnostics` for propensity diagnostics, and `.to_json` / `.to_html` / `.card` / `.export` for stakeholder artifacts. `.to_json()` / `.to_html()` return a string when called with no argument, and write the file (returning its `Path`) when given a `path`.

#### `make_pairwise_synth(n_days=14, n_clients_day=2000, n_ops=200, **kwargs)`
Generate synthetic pairwise (client-operator) data for evaluation.

**Parameters:**
- `n_days`: Number of days to simulate
- `n_clients_day`: Number of clients per day
- `n_ops`: Number of operators
- `seed`: Random seed for reproducibility
- `binary`: Whether to generate binary outcomes (default: False)

**Returns:**
- `logs_df`: DataFrame with pairwise decisions
- `op_daily_df`: DataFrame with daily operator data

### Advanced Functions

#### `fit_propensity_timecal(X_phi, A, n_splits=3, random_state=0)`
Fit propensity model with time-aware cross-validation and isotonic calibration.

#### `fit_outcome_crossfit(X_obs, Y, n_splits=3, estimator='hgb', random_state=0)`
Fit outcome model with cross-fitting. Supports `'hgb'`, `'ridge'`, `'rf'`, or custom estimators.

#### `dr_value_with_clip(propensities, policy_probs, Y, q_hat, A, elig, clip_grid=...)`
Compute DR and SNDR values with automatic clipping threshold selection.

#### `block_bootstrap_ci(values_num, values_den, base_mean, n_boot=400, **kwargs)`
Compute confidence intervals using moving-block bootstrap for time-series data.

## Theory

### Why DR and SNDR?

**Doubly Robust (DR)** estimation provides unbiased policy evaluation when either the propensity model OR the outcome model is correctly specified. The estimator is:

```
V̂_DR = (1/n) Σ [q̂_π(x_i) + w_i * (y_i - q̂(x_i, a_i))]
```

**Stabilized DR (SNDR)** reduces variance by normalizing importance weights:

```
V̂_SNDR = (1/n) Σ q̂_π(x_i) + [Σ w_i * (y_i - q̂(x_i, a_i))] / [Σ w_i]
```

Where:
- `q̂_π(x)` = expected outcome under evaluation policy π
- `q̂(x,a)` = outcome model prediction
- `w_i = π(a_i|x_i) / e(a_i|x_i)` = importance weight (clipped)
- `e(a_i|x_i)` = propensity score (calibrated)

## Implementation Details

### Autoscaling Strategies

- **Direct**: Uses the logging policy directly without modification
- **Stream**: Induces a policy from sklearn models and applies it to streaming decisions  
- **Stream TopK**: Similar to stream but restricts choices to top-K operators based on predicted service times

### Key Features

- **Time-Series Aware**: Uses `TimeSeriesSplit` for all cross-validation with temporal ordering
- **Calibrated Propensities**: Per-fold isotonic calibration via `CalibratedClassifierCV`
- **Automatic Clipping**: Smart threshold selection to minimize variance while maintaining ESS
- **Comprehensive Diagnostics**: ESS, match rates, propensity quantiles, and tail mass analysis

## Bootstrap Confidence Intervals

For time-series data, use moving-block bootstrap with proper statistical methodology:

```python
# Enable bootstrap CIs
artifact = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    ci_bootstrap=True,
    alpha=0.05,  # 95% confidence
    policy_train="pre_split",
)

print(artifact.report[['model', 'estimator', 'V_hat', 'ci_lower', 'ci_upper']])
```

**Key Features:**
- **Moving-block bootstrap**: Preserves time-series correlation structure
- **Proper statistical inference**: Uses bootstrap distribution of DR contributions
- **Automatic fallback**: Falls back to normal approximation if bootstrap fails
- **Configurable parameters**: Control bootstrap samples, block length, and significance level

## Command-line interface

The `skdr-eval` CLI ships behind the `[cli]` extra and exposes the same
evaluation surface to teams that don't want to write Python.

```bash
pip install 'skdr-eval[cli]'

# Quick environment + schema probe before evaluation.
skdr-eval doctor logs.parquet
skdr-eval doctor logs.parquet --json | jq .

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

# Stable exit codes (good for CI gates):
#   0 — success
#   1 — data / schema error
#   2 — environment / import error
#   3 — at least one model row's recommendation verdict is 'do_not_deploy'
```

## Preflight diagnostics: `skdr_eval.doctor`

`skdr_eval.doctor(logs, *, kind='standard'|'pairwise', op_daily_df=None,
metric_col='service_time', n_splits=3, strict=False)` returns a non-raising
`DoctorReport` that surfaces environment + schema + statistical sanity
failures with actionable fix hints.

```python
import skdr_eval

logs, _, _ = skdr_eval.make_synth_logs(n=5000, n_ops=3, seed=0)
report = skdr_eval.doctor(logs)
report.print()            # text table with status glyphs
report.to_markdown()      # copy-pasteable Markdown
report.to_dict()          # JSON-serializable
assert report.ok          # True iff no Check has status='fail'
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

For `MIPS` (Marginalized IPS), supply an `action_embedding` matrix of shape
`(n_actions, embed_dim)`; the
`skdr_eval.embedding_sufficiency_diagnostic(...)` helper flags whether the
embedding captures enough of the action-driven reward signal for MIPS to
be approximately unbiased.

See `examples/quickstart_estimators.py` and `examples/quickstart_mips.py`
for runnable walkthroughs.

### Slate / top-K off-policy evaluation

The `skdr_eval.slate` subpackage (issue #75) ships four ranking-OPE
estimators — `slate_standard_ips`, `reward_interaction_ips`,
`pseudo_inverse_ips`, and `slate_cascade_dr` — plus a synthetic
cascade-click generator `make_slate_synth(...)` with closed-form ground
truth. See `examples/quickstart_slate.py`.

## Development

### Setup
```bash
git clone https://github.com/dgenio/skdr-eval.git
cd skdr-eval
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### Testing
```bash
pytest -v
```

### Linting and Formatting
```bash
ruff check src/ tests/ examples/
ruff format src/ tests/ examples/
mypy src/skdr_eval/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Building
```bash
python -m build
```

## Publishing to PyPI

This package uses **Trusted Publishing** (PEP 740) for secure PyPI releases.

### Automatic (Recommended)
1. Create a GitHub release with a version tag (e.g., `v0.1.0`)
2. The `release.yml` workflow will automatically build and publish

### Manual Fallback
If Trusted Publishing is not configured:

1. Set up PyPI API token: https://pypi.org/manage/account/token/
2. Build the package: `python -m build`
3. Upload: `twine upload dist/*`

### Trusted Publishing Setup
1. Go to https://pypi.org/manage/project/skdr-eval/settings/publishing/
2. Add GitHub repository as trusted publisher:
   - **Repository**: `dgenio/skdr-eval`
   - **Workflow**: `release.yml`
   - **Environment**: `release`

## Citation

If you use this software in your research, please cite:

```bibtex
@software{santos2026skdreval,
  title   = {{skdr-eval}: Offline Policy Evaluation for {sklearn}-Compatible Models with Time-Aware Doubly Robust Estimators},
  author  = {Santos, Diogo},
  year    = {2026},
  url     = {https://github.com/dgenio/skdr-eval},
  version = {0.9.0},
  license = {MIT}
}
```

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
