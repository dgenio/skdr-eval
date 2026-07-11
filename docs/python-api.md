# Python API reference & theory

This is the narrative reference for the most-used public functions, plus the
estimator theory and implementation notes. For the full auto-generated symbol
reference see [API reference](api.md); for the stability contract see
[API stability & road to 1.0](api-stability.md).

> The runnable [`examples/`](https://github.com/dgenio/skdr-eval/tree/main/examples)
> are the most trustworthy usage guide — when in doubt, trust the example
> scripts and the type signatures in `src/`.

## Core functions

### `make_synth_logs(n=5000, n_ops=5, seed=0)`
Generate synthetic service logs for evaluation.

**Returns:**
- `logs`: DataFrame with service logs
- `ops_all`: Index of all operator names
- `true_q`: Ground truth service times

### `build_design(logs, cli_pref='cli_', st_pref='st_')`
Build design matrices from logs.

**Returns:**
- `Design`: Dataclass with feature matrices and metadata

### `evaluate_sklearn_models(logs, models, **kwargs)`
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

**Scaling controls (keyword-only):**
- `n_jobs`: Parallel workers for the candidate-model loop, cross-fitting folds,
  and bootstrap replicates (default: `1` = serial; `-1` = all cores). Results
  are **deterministic and independent of `n_jobs`** — the model loop and folds
  are bit-identical to serial, and the bootstrap reseeds per replicate so the
  CIs reproduce exactly.
- `execution_mode`: `"auto"` (default), `"standard"`, or `"large_data"`.
  `"large_data"` predicts the policy-induction feature matrix in `chunk_size`
  row-blocks to bound peak memory on large logs; it is numerically identical to
  `"standard"`. `"auto"` selects `"large_data"` once the log is large.
- `chunk_size`: Max eligible (sample, operator) pairs materialised at once in
  `"large_data"` mode (default: `100_000`).
- `requires_overlap`: When `True`, run a cheap overlap/positivity precheck
  before the expensive fit and raise `InsufficientOverlapError` if the logs
  cannot support OPE at all (default: `False`). Tune with `overlap_floor` and
  `min_match_rate`.

### `evaluate_pairwise_models(logs_df, op_daily_df, models, metric_col, task_type, direction, **kwargs)`
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

### `make_pairwise_synth(n_days=14, n_clients_day=2000, n_ops=200, **kwargs)`
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

## Advanced functions

### `fit_propensity_timecal(X_phi, A, n_splits=3, random_state=0)`
Fit propensity model with time-aware cross-validation and isotonic calibration.

### `fit_outcome_crossfit(X_obs, Y, n_splits=3, estimator='hgb', random_state=0)`
Fit outcome model with cross-fitting. Supports `'hgb'`, `'ridge'`, `'rf'`, or custom estimators.

### `dr_value_with_clip(propensities, policy_probs, Y, q_hat, A, elig, clip_grid=...)`
Compute DR and SNDR values with automatic clipping threshold selection.

### `block_bootstrap_ci(values_num, values_den, base_mean, n_boot=400, **kwargs)`
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

See [Methods (DR / SNDR)](methods.md) for the full estimand, assumptions, and
references.

## Implementation details

### Autoscaling strategies

- **Direct**: Uses the logging policy directly without modification
- **Stream**: Induces a policy from sklearn models and applies it to streaming decisions
- **Stream TopK**: Similar to stream but restricts choices to top-K operators based on predicted service times

### Key features

- **Time-Series Aware**: Uses `TimeSeriesSplit` for all cross-validation with temporal ordering
- **Calibrated Propensities**: Per-fold isotonic calibration via `CalibratedClassifierCV`
- **Automatic Clipping**: Smart threshold selection to minimize variance while maintaining ESS
- **Comprehensive Diagnostics**: ESS, match rates, propensity quantiles, and tail mass analysis

## Bootstrap confidence intervals

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

**Key features:**
- **Moving-block bootstrap**: Preserves time-series correlation structure
- **Proper statistical inference**: Uses bootstrap distribution of DR contributions
- **Automatic fallback**: Falls back to normal approximation if bootstrap fails
- **Configurable parameters**: Control bootstrap samples, block length, and significance level
