# skdr-eval v0.5.0 Release Notes

**Breaking Changes Release — Pairwise API Cleanup**

This minor release cleans up the pairwise evaluation API, removes a dead parameter, and fixes documentation and error-handling gaps. Any code using the `strategy=` keyword argument on `estimate_propensity_pairwise()` must be updated before upgrading.

---

## ⚠️ Breaking Changes

### `strategy` parameter removed from `estimate_propensity_pairwise()`

The `strategy` keyword argument has been **removed**. It was previously accepted and validated, but had no effect on behaviour. Update call sites to use `method=` instead:

```python
# Before (raises TypeError in v0.5.0)
estimate_propensity_pairwise(design, strategy="condlogit")

# After
estimate_propensity_pairwise(design, method="condlogit")
```

### Parameters after `design` are now keyword-only in `estimate_propensity_pairwise()`

All arguments except the first positional `design` must now be passed as keyword arguments. Positional calls beyond the first argument will raise `TypeError`.

```python
# Before (silently worked, prone to argument-order bugs)
estimate_propensity_pairwise(design, "condlogit", 100)

# After (required form)
estimate_propensity_pairwise(design, method="condlogit", n_iter=100)
```

---

## Changed

### `estimate_propensity_pairwise()` default `method` is now `"auto"`

The default method changed from `"condlogit"` to `"auto"`. With `"auto"`, the function selects `condlogit` when SciPy is available and falls back to `multinomial` silently — eliminating the warning that appeared on every call in environments without SciPy.

### `evaluate_pairwise_models()` accepts `"auto"` for `propensity`

The type hint for the `propensity` parameter has been expanded from `Literal["condlogit", "multinomial"]` to `Literal["auto", "condlogit", "multinomial"]` to align with the new default.

---

## Fixed

### Pairwise quick-start example in README

The pairwise section of the README quick-start example was broken. Fixes applied:
- Corrected function signature (`strategy` → non-existent `autoscale_strategies` removed)
- Added required positional arguments: `metric_col`, `task_type`, `direction`
- Fixed return-value unpacking to the correct `(report, detailed)` tuple
- Added explicit model fitting step before calling `evaluate_pairwise_models()`

### `evaluate_sklearn_models` empty/None models input

`evaluate_sklearn_models()` now raises `ValueError` immediately when `models` is empty or contains `None` values. Previously these inputs silently produced `(0, 0)` DataFrames and caused confusing downstream errors. Resolves [#45](https://github.com/dgenio/skdr-eval/issues/45).

---

## Migration Guide

| Old call | New call |
|---|---|
| `estimate_propensity_pairwise(design, strategy="condlogit")` | `estimate_propensity_pairwise(design, method="condlogit")` |
| `estimate_propensity_pairwise(design, "condlogit")` | `estimate_propensity_pairwise(design, method="condlogit")` |
| `evaluate_pairwise_models(..., propensity="condlogit")` | No change required; `"auto"` is now also valid |

---

## Installation

```bash
pip install skdr-eval==0.5.0
```

Or upgrade:

```bash
pip install --upgrade skdr-eval
```
