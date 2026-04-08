# skdr-eval

[![PyPI version](https://badge.fury.io/py/skdr-eval.svg)](https://badge.fury.io/py/skdr-eval)
[![Python versions](https://img.shields.io/pypi/pyversions/skdr-eval.svg)](https://pypi.org/project/skdr-eval/)
[![CI](https://github.com/dandrsantos/skdr-eval/workflows/CI/badge.svg)](https://github.com/dandrsantos/skdr-eval/actions)
[![Coverage](https://codecov.io/gh/dandrsantos/skdr-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/dandrsantos/skdr-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Offline policy evaluation for service-time minimization using Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators with time-aware splits and calibration. Now with pairwise evaluation and autoscaling support.**

## What is this?

`skdr-eval` is a Python package for offline policy evaluation in service-time optimization scenarios. It implements state-of-the-art Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators with time-aware cross-validation and calibration. The package is designed for evaluating machine learning models that make decisions about service allocation, with special support for pairwise (client-operator) evaluation and autoscaling strategies.

## Table of Contents

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
- 🚀 **Production Ready**: Type-hinted, tested, and documented
- 📈 **Bootstrap Confidence Intervals**: Moving-block bootstrap for time-series data
- 🤝 **Pairwise Evaluation**: Client-operator pairwise evaluation with autoscaling strategies
- 🎛️ **Autoscaling**: Direct, stream, and stream_topk strategies with policy induction
- 🧮 **Choice Models**: Conditional logit models for propensity estimation

## Installation

```bash
pip install skdr-eval
```

### Optional Dependencies

For choice models (conditional logit):
```bash
pip install skdr-eval[choice]
```

For speed optimizations (PyArrow, Polars):
```bash
pip install skdr-eval[speed]
```

For development:
```bash
git clone https://github.com/dandrsantos/skdr-eval.git
cd skdr-eval
pip install -e .[dev]
```

## Quick Start

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
report, detailed_results = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    fit_models=True,
    n_splits=3,
    random_state=42,
)

# 4. View results
print(report[['model', 'estimator', 'V_hat', 'ESS', 'match_rate']])
```

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
report, detailed = skdr_eval.evaluate_pairwise_models(
    logs_df=logs_df,
    op_daily_df=op_daily_df,
    models={"HGB": model},
    metric_col="service_time",
    task_type="regression",
    direction="min",
    strategy="auto",
    n_splits=3,
    random_state=42,
)

# 4. View results
print(report[["model", "estimator", "V_hat", "ESS", "match_rate"]])
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
- `tuple[pd.DataFrame, dict[str, dict[str, DRResult]]]`: A summary report DataFrame and detailed results per model and estimator

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
report, _ = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    ci_bootstrap=True,
    alpha=0.05,  # 95% confidence
)

print(report[['model', 'estimator', 'V_hat', 'ci_lower', 'ci_upper']])
```

**Key Features:**
- **Moving-block bootstrap**: Preserves time-series correlation structure
- **Proper statistical inference**: Uses bootstrap distribution of DR contributions
- **Automatic fallback**: Falls back to normal approximation if bootstrap fails
- **Configurable parameters**: Control bootstrap samples, block length, and significance level

## Examples

See `examples/quickstart.py` for a complete example, or run:

```bash
python examples/quickstart.py
```

## Development

### Setup
```bash
git clone https://github.com/dandrsantos/skdr-eval.git
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
   - **Repository**: `dandrsantos/skdr-eval`
   - **Workflow**: `release.yml`
   - **Environment**: `release`

## Citation

If you use this software in your research, please cite:

```bibtex
@software{santos2024skdr,
  title = {skdr-eval: Offline Policy Evaluation for Service-Time Minimization},
  author = {Santos, Diogo},
  year = {2024},
  url = {https://github.com/dandrsantos/skdr-eval},
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Uses [pandas](https://pandas.pydata.org/) for data manipulation
- Follows [PEP 621](https://peps.python.org/pep-0621/) for project metadata
