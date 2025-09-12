# skdr-eval

[![PyPI version](https://badge.fury.io/py/skdr-eval.svg)](https://badge.fury.io/py/skdr-eval)
[![Python versions](https://img.shields.io/pypi/pyversions/skdr-eval.svg)](https://pypi.org/project/skdr-eval/)
[![CI](https://github.com/dandrsantos/skdr-eval/workflows/CI/badge.svg)](https://github.com/dandrsantos/skdr-eval/actions)
[![Coverage](https://codecov.io/gh/dandrsantos/skdr-eval/branch/main/graph/badge.svg)](https://codecov.io/gh/dandrsantos/skdr-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Offline policy evaluation for service-time minimization using Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators with time-aware splits and calibration. Now with pairwise evaluation and autoscaling support.**

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
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# 1. Generate synthetic pairwise data (client-operator pairs)
logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
    n_days=5,
    n_clients_day=1000,
    n_ops=20,
    seed=42
)

# 2. Define models for different tasks
models = {
    "ServiceTime": HistGradientBoostingRegressor(random_state=42),
    "Binary": HistGradientBoostingClassifier(random_state=42),
}

# 3. Run pairwise evaluation with autoscaling
results = skdr_eval.evaluate_pairwise_models(
    logs_df=logs_df,
    op_daily_df=op_daily_df,
    models=models,
    autoscale_strategies=["direct", "stream", "stream_topk"],
    n_splits=3,
    random_state=42
)

# 4. View autoscaling results
for strategy, result in results.items():
    print(f"{strategy}: V_hat = {result['V_hat']:.4f}, ESS = {result['ESS']:.1f}")
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

#### `evaluate_pairwise_models(logs_df, op_daily_df, models, **kwargs)`
Evaluate models using pairwise (client-operator) evaluation with autoscaling.

**Parameters:**
- `logs_df`: Pairwise decision log DataFrame
- `op_daily_df`: Daily operator availability DataFrame
- `models`: Dict of model name to sklearn estimator
- `autoscale_strategies`: List of strategies ("direct", "stream", "stream_topk")
- `n_splits`: Number of time-series splits (default: 3)
- `random_state`: Random seed for reproducibility

**Returns:**
- Dict mapping strategy names to evaluation results

#### `make_pairwise_synth(n_days=5, n_clients_day=1000, n_ops=20, **kwargs)`
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

**Parameters:**
- `values_num`: Numerator values for bootstrap
- `values_den`: Denominator values for ratio estimation (optional)
- `base_mean`: Base mean for centering
- `n_boot`: Number of bootstrap samples (default: 400)
- `block_len`: Block length for time-series correlation (default: sqrt(n))
- `alpha`: Significance level (default: 0.05)
- `random_state`: Random seed for reproducibility

**Returns:**
- `ci_lower`: Lower confidence bound
- `ci_upper`: Upper confidence bound

## Why DR and SNDR?

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

## Key Implementation Details

### Autoscaling Strategies

#### Direct Strategy
Uses the logging policy directly without modification.

#### Stream Strategy
Induces a policy from sklearn models and applies it to streaming decisions.

#### Stream TopK Strategy
Similar to stream but restricts choices to top-K operators based on predicted service times.

### Time-Series Considerations

- Uses `TimeSeriesSplit` for all cross-validation
- Propensity scores include standardized timestamps
- Respects temporal ordering in data splits
- Pairwise evaluation maintains temporal consistency across client-operator pairs

### Propensity Score Calibration
- Per-fold isotonic calibration via `CalibratedClassifierCV`
- Fallback to uncalibrated scores if calibration fails
- Handles class imbalance gracefully

### Clipping Threshold Selection
- **DR**: Minimize MSE proxy with ESS floor (default: 2% of samples)
- **SNDR**: Minimize |SNDR - DR| + MSE proxy
- Automatic selection from grid: `(2, 5, 10, 20, 50, ∞)`

### Diagnostics and Quality Checks
- **Effective Sample Size (ESS)**: `(Σw)² / Σw²`
- **Match Rate**: Fraction with positive propensity scores
- **Propensity Quantiles**: P01, P05, P10 for positivity assessment
- **Tail Mass**: Fraction of samples affected by clipping

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
