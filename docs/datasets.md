# Datasets, inputs & model adapters

skdr-eval is pandas- and scikit-learn-first, but the boundaries are open: you
can feed Polars / PyArrow frames, plug in non-sklearn models, and load public
benchmark datasets — all returning to the same canonical schema so the rest of
the library is unchanged.

## Public dataset loaders

`skdr_eval.datasets` ships opinionated loaders that return the same
`(logs, ops_all, ground_truth)` tuple as `make_synth_logs`, so real benchmark
data flows straight through `evaluate_sklearn_models`.

```python
import skdr_eval

# Lazily downloads + caches the public Open Bandit Dataset sample.
logs, ops_all, ground_truth = skdr_eval.datasets.load_obd(
    behavior_policy="random",   # or "bts"
    campaign="all",             # or "men" / "women"
    max_rows=5000,              # truncate for a quick run
)

art = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models={"hgb": ...},
    y_col="click",              # OBD reward column
    policy_train="pre_split",
)
```

### Caching

Downloads are cached under `~/.skdr_eval/datasets` (override with the
`SKDR_EVAL_CACHE_DIR` environment variable). Each dataset directory carries a
`manifest.json` recording, per file, the `sha256`, `size_bytes`, and source —
so a cached dataset is reproducible and verifiable across runs. Pass a local
directory as `base_url` to load an already-downloaded copy offline:

```python
logs, ops_all, _ = skdr_eval.datasets.load_obd(
    "random", "all", base_url="/path/to/obd"
)
```

### Available datasets

| Loader | Dataset | Reward | License / citation | Size |
| --- | --- | --- | --- | --- |
| `load_obd` | Open Bandit Dataset (ZOZOTOWN) | `click` (binary) | CC BY 4.0, ZOZO Inc. — Saito et al., *Open Bandit Dataset and Pipeline*, arXiv:2008.07146; <https://research.zozo.com/data.html> | Sample slice loads in seconds; full set ≈ 26M rows |
| `load_criteo_counterfactual` | Criteo counterfactual logs | — | Requires license acceptance — <https://www.cs.cornell.edu/~adith/Criteo/> | *not yet implemented (tracked by #70)* |
| `load_movielens_ope` | MovieLens OPE recipe | — | GroupLens terms | *not yet implemented (tracked by #70)* |

`load_criteo_counterfactual` and `load_movielens_ope` currently raise
`DatasetError` directing you to `load_obd`; their full implementations are
tracked as follow-ups under #70.

### Failure modes

Loaders fail loud with actionable `DatasetError` messages: network
unavailable (with a hint to pass a local `base_url`), insufficient disk space,
or a missing local source.

## Polars / PyArrow inputs

With the `[speed]` extra installed, the public evaluators accept Polars
`DataFrame` and PyArrow `Table` inputs directly — they are converted to pandas
once at the boundary, so results are identical to the pandas path.

```python
import polars as pl

art = skdr_eval.evaluate_sklearn_models(logs=pl.from_pandas(logs), models=...)
```

The returned artifact exposes matching accessors for the headline report:

```python
art.to_polars()   # -> polars.DataFrame
art.to_arrow()    # -> pyarrow.Table
```

Both raise `OptionalDependencyError` when the `[speed]` extra is not installed.

## Non-sklearn models

The evaluators accept any object with the sklearn `fit` / `predict` surface.
For gradient-boosted trees and bespoke callables, `skdr_eval.adapters` provides
first-class wrappers (install the backends with the `[boosting]` extra):

```python
from skdr_eval.adapters import (
    XGBRegressorAdapter,
    LGBMRegressorAdapter,
    CatBoostRegressorAdapter,
    CallableModelAdapter,
)

models = {
    "lgbm": LGBMRegressorAdapter(n_estimators=200, num_leaves=31),
    # Forward native fit kwargs (early stopping, categoricals, GPU flags):
    "cat": CatBoostRegressorAdapter(
        iterations=300, fit_kwargs={"cat_features": [0, 3]}
    ),
    # Wrap a plain predict function:
    "my_model": CallableModelAdapter(predict_fn=my_fn),
}
```

See `examples/real_data_obd.py` and `examples/boosting_adapter.py` for runnable
end-to-end scripts.
