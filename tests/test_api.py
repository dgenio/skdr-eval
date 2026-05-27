"""Test API imports and basic functionality."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import skdr_eval
from skdr_eval.exceptions import (
    DataValidationError,
    InsufficientDataError,
    PolicyInductionError,
)


def test_imports():
    """Test that all required functions can be imported."""
    # Test main functions
    assert hasattr(skdr_eval, "make_synth_logs")
    assert hasattr(skdr_eval, "build_design")
    assert hasattr(skdr_eval, "fit_propensity_timecal")
    assert hasattr(skdr_eval, "fit_outcome_crossfit")
    assert hasattr(skdr_eval, "induce_policy_from_sklearn")
    assert hasattr(skdr_eval, "dr_value_with_clip")
    assert hasattr(skdr_eval, "block_bootstrap_ci")
    assert hasattr(skdr_eval, "evaluate_sklearn_models")

    # Test classes
    assert hasattr(skdr_eval, "Design")
    assert hasattr(skdr_eval, "DRResult")

    # Test version
    assert hasattr(skdr_eval, "__version__")


def test_make_synth_logs_signature():
    """Test make_synth_logs function signature and return types."""
    logs, ops_all, true_q = skdr_eval.make_synth_logs(n=100, n_ops=3, seed=42)

    # Check return types
    assert isinstance(logs, pd.DataFrame)
    assert isinstance(ops_all, pd.Index)
    assert isinstance(true_q, np.ndarray)

    # Check shapes
    assert len(logs) == 100
    assert len(ops_all) == 3
    assert true_q.shape == (100, 3)

    # Check required columns
    required_cols = ["arrival_ts", "action", "service_time"]
    for col in required_cols:
        assert col in logs.columns

    # Check eligibility columns
    elig_cols = [col for col in logs.columns if col.endswith("_elig")]
    assert len(elig_cols) == 3


def test_make_pairwise_synth_docstring_shapes():
    """#113: make_pairwise_synth output matches the shapes quoted in its docstring."""
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=100, n_ops=10
    )
    assert logs_df.shape == (300, 14)
    assert op_daily_df.shape == (30, 6)


def test_build_design_signature():
    """Test build_design function signature and Design dataclass."""
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=1)
    design = skdr_eval.build_design(logs)

    # Check return type
    assert isinstance(design, skdr_eval.Design)

    # Check Design fields
    assert hasattr(design, "X_base")
    assert hasattr(design, "X_obs")
    assert hasattr(design, "X_phi")
    assert hasattr(design, "A")
    assert hasattr(design, "Y")
    assert hasattr(design, "ts")
    assert hasattr(design, "ops_all")
    assert hasattr(design, "elig")
    assert hasattr(design, "idx")

    # Check shapes
    n_samples = len(logs)
    n_ops = len(ops_all)

    assert design.X_base.shape[0] == n_samples
    assert design.X_obs.shape[0] == n_samples
    assert design.X_phi.shape[0] == n_samples
    assert len(design.A) == n_samples
    assert len(design.Y) == n_samples
    assert len(design.ts) == n_samples
    assert design.elig.shape == (n_samples, n_ops)

    # Check that X_obs includes action one-hot
    assert design.X_obs.shape[1] == design.X_base.shape[1] + n_ops


def test_fit_propensity_timecal_signature():
    """Test fit_propensity_timecal function signature."""
    logs, _, _ = skdr_eval.make_synth_logs(n=100, n_ops=3, seed=2)
    design = skdr_eval.build_design(logs)

    propensities, fold_indices = skdr_eval.fit_propensity_timecal(
        design.X_phi, design.A, design.ts, n_splits=3, random_state=0
    )

    # Check return types and shapes
    assert isinstance(propensities, np.ndarray)
    assert isinstance(fold_indices, np.ndarray)
    assert propensities.shape == (len(design.A), 3)  # n_samples x n_actions
    assert len(fold_indices) == len(design.A)

    # Check propensities are valid probabilities
    assert np.all(propensities >= 0)
    assert np.all(propensities <= 1)
    # Row sums should be close to 1 (allowing for numerical precision)
    row_sums = propensities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)


def test_drresult_dataclass():
    """Test DRResult dataclass fields."""
    # Create a dummy DRResult
    grid_data = {
        "clip": [2.0, 5.0],
        "V_DR": [10.0, 11.0],
        "V_SNDR": [10.1, 11.1],
        "ESS": [50.0, 40.0],
    }
    grid = pd.DataFrame(grid_data)

    result = skdr_eval.DRResult(
        clip=2.0,
        V_hat=10.0,
        SE_if=1.0,
        ESS=50.0,
        tail_mass=0.1,
        MSE_est=1.0,
        match_rate=0.9,
        min_pscore=0.01,
        pscore_q10=0.02,
        pscore_q05=0.015,
        pscore_q01=0.01,
        grid=grid,
    )

    # Check all fields exist
    assert result.clip == 2.0
    assert result.V_hat == 10.0
    assert result.SE_if == 1.0
    assert result.ESS == 50.0
    assert result.tail_mass == 0.1
    assert result.MSE_est == 1.0
    assert result.match_rate == 0.9
    assert result.min_pscore == 0.01
    assert result.pscore_q10 == 0.02
    assert result.pscore_q05 == 0.015
    assert result.pscore_q01 == 0.01
    assert isinstance(result.grid, pd.DataFrame)


def test_evaluate_sklearn_models_signature():
    """Test evaluate_sklearn_models function signature."""

    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=3)
    models = {"rf": RandomForestRegressor(n_estimators=10, random_state=0)}

    _artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="all",  # explicit: suppress DeprecationWarning
    )

    report, detailed_results = _artifact.report, _artifact.detailed

    # Check return types
    assert isinstance(report, pd.DataFrame)
    assert isinstance(detailed_results, dict)

    # Check report structure
    expected_cols = [
        "model",
        "estimator",
        "V_hat",
        "SE_if",
        "clip",
        "ESS",
        "tail_mass",
        "MSE_est",
        "match_rate",
        "min_pscore",
        "pscore_q10",
        "pscore_q05",
        "pscore_q01",
    ]
    for col in expected_cols:
        assert col in report.columns

    # Check detailed results structure
    assert "rf" in detailed_results
    assert "DR" in detailed_results["rf"]
    assert "SNDR" in detailed_results["rf"]
    assert isinstance(detailed_results["rf"]["DR"], skdr_eval.DRResult)
    assert isinstance(detailed_results["rf"]["SNDR"], skdr_eval.DRResult)


def test_evaluate_sklearn_models_empty_models_raises():
    """#109: an empty models dict raises a clear DataValidationError."""
    logs, _, _ = skdr_eval.make_synth_logs(n=10, n_ops=3, seed=4)

    with pytest.raises(DataValidationError, match="models dict is empty"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={},
            fit_models=True,
            n_splits=2,
            random_state=42,
        )

    with pytest.raises(DataValidationError, match="must be estimators"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": None},
            fit_models=True,
            n_splits=2,
            random_state=42,
        )


def test_evaluate_sklearn_models_single_estimator_hint():
    """#109: passing a bare estimator (not a dict) raises with a did-you-mean hint."""
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=42)

    with pytest.raises(DataValidationError, match="did you mean"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=RandomForestRegressor(random_state=42),  # bug: should be a dict
            fit_models=True,
            n_splits=3,
            policy_train="pre_split",
        )


def test_evaluate_sklearn_models_non_estimator_value_raises():
    """#109: a dict value lacking a fit method raises a clear type error."""
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=42)

    with pytest.raises(DataValidationError, match="must be estimators with a 'fit'"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"x": 42},
            fit_models=True,
            n_splits=3,
            policy_train="pre_split",
        )


def test_evaluate_sklearn_models_y_col_matches_default(recwarn):
    """#105: a custom y_col reproduces the default service_time path numerically."""
    logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=42)
    logs_renamed = logs.rename(columns={"service_time": "reward"})

    art_default = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"rf": RandomForestRegressor(random_state=0)},
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
    )
    art_custom = skdr_eval.evaluate_sklearn_models(
        logs=logs_renamed,
        models={"rf": RandomForestRegressor(random_state=0)},
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
        y_col="reward",
    )

    np.testing.assert_allclose(
        art_custom.report["V_hat"].to_numpy(),
        art_default.report["V_hat"].to_numpy(),
    )


def test_evaluate_sklearn_models_y_col_missing_column_errors():
    """#105: requesting a y_col that is absent surfaces the missing-column error."""
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=42)

    with pytest.raises(DataValidationError, match="missing required columns"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": RandomForestRegressor(random_state=0)},
            fit_models=True,
            n_splits=3,
            policy_train="pre_split",
            y_col="reward",  # not present (column is 'service_time')
        )


def test_pre_split_small_input_error_mentions_context():
    """#114: a too-small pre_split input names the input count and policy_train_frac."""
    logs, _, _ = skdr_eval.make_synth_logs(n=20, n_ops=3, seed=42)

    with pytest.raises(InsufficientDataError) as excinfo:
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": RandomForestRegressor(random_state=0)},
            n_splits=3,
            policy_train="pre_split",
            policy_train_frac=0.85,
            fit_models=True,
        )
    message = str(excinfo.value)
    assert "pre_split" in message
    assert "20 input rows" in message
    assert "85%" in message


def test_all_policy_train_small_input_error_unchanged():
    """#114: with policy_train='all' the original split message is left intact."""
    # n=10 clears build_design's min_rows=10 floor; n_splits=8 makes the
    # time-series split (not build_design) the binding constraint.
    logs, _, _ = skdr_eval.make_synth_logs(n=10, n_ops=3, seed=42)

    with pytest.raises(InsufficientDataError) as excinfo:
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": RandomForestRegressor(random_state=0)},
            n_splits=8,
            policy_train="all",
            fit_models=True,
        )
    message = str(excinfo.value)
    assert "Need at least" in message
    assert "pre_split" not in message


def test_induce_policy_from_sklearn_vectorized_matches_scalar_reference():
    """#46: vectorized induce_policy_from_sklearn matches a scalar reference.

    The vectorized implementation must produce *identical* policy_probs as a
    per-sample, per-eligible-op reference loop (the original implementation
    style). This is the parity guarantee for the perf rewrite.
    """
    rng = np.random.RandomState(0)
    n_samples = 8
    n_features = 3
    n_ops = 5
    ops_all = [f"op{i}" for i in range(n_ops)]
    X_base = rng.randn(n_samples, n_features).astype(np.float64)
    # Mix of fully eligible / partially eligible / no-eligible rows
    elig = rng.randint(0, 2, size=(n_samples, n_ops))
    elig[0] = 1  # fully eligible
    elig[1] = 0  # no eligible ops at all (uniform fallback)
    elig[2] = [1, 0, 0, 0, 0]  # single eligible op

    class _DeterministicModel:
        """Deterministic, finite, strictly positive predictions."""

        def fit(self, X, y):  # required by validate_sklearn_estimator
            return self

        def predict(self, X):
            # Linear function of features + op-index encoded in one-hot block.
            base = X[:, :n_features].sum(axis=1)
            op_part = X[:, n_features:] @ (np.arange(n_ops) + 1.0)
            return np.abs(base) + op_part + 0.1  # strictly > 0, finite

    model = _DeterministicModel()

    def scalar_reference(model, X_base, ops_all, elig):
        n_samples, _ = X_base.shape
        n_ops = len(ops_all)
        probs = np.zeros((n_samples, n_ops))
        for i in range(n_samples):
            eligible_ops = np.where(elig[i])[0]
            if eligible_ops.size == 0:
                probs[i] = 1.0 / n_ops
                continue
            preds = []
            for op_idx in eligible_ops:
                onehot = np.zeros(n_ops, dtype=X_base.dtype)
                onehot[op_idx] = 1
                x = np.concatenate([X_base[i], onehot])
                preds.append(float(model.predict(x.reshape(1, -1))[0]))
            arr = np.asarray(preds)
            probs[i, eligible_ops] = 1.0 / (arr + 1e-8)
            probs[i] /= probs[i].sum()
        return probs

    vectorized = skdr_eval.induce_policy_from_sklearn(model, X_base, ops_all, elig)
    reference = scalar_reference(model, X_base, ops_all, elig)

    # Identical up to floating-point: same arithmetic, same order.
    np.testing.assert_allclose(vectorized, reference, rtol=0, atol=1e-12)


def test_induce_policy_from_sklearn_issues_single_predict_call():
    """#46: vectorized path must issue exactly one model.predict call."""
    rng = np.random.RandomState(1)
    n_samples = 6
    n_features = 2
    n_ops = 4
    ops_all = [f"op{i}" for i in range(n_ops)]
    X_base = rng.randn(n_samples, n_features)
    elig = rng.randint(0, 2, size=(n_samples, n_ops))
    elig[0] = 1
    elig[-1] = 1

    class _CountingModel:
        def __init__(self):
            self.calls: list[int] = []

        def fit(self, X, y):  # required by validate_sklearn_estimator
            return self

        def predict(self, X):
            self.calls.append(len(X))
            return np.full(len(X), 0.5, dtype=np.float64)

    model = _CountingModel()
    skdr_eval.induce_policy_from_sklearn(model, X_base, ops_all, elig)

    assert len(model.calls) == 1, (
        f"Expected exactly one model.predict call (vectorized); "
        f"got {len(model.calls)} with sizes {model.calls}"
    )
    # The single call must cover every eligible (sample, op) pair.
    assert model.calls[0] == int(elig.sum()), (
        f"Single predict call should cover {int(elig.sum())} eligible pairs; "
        f"got {model.calls[0]}"
    )


def test_induce_policy_from_sklearn_raises_on_non_finite_predictions():
    """Non-finite predictions must surface as ``PolicyInductionError`` (#46 audit).

    The vectorized rewrite of ``induce_policy_from_sklearn`` issues a single
    ``model.predict`` call; if any cell of the result is non-finite the helper
    must fail loud per ``docs/agent-context/invariants.md``, naming the
    offending (sample, operator) so the caller can localize the bad row.
    """
    rng = np.random.RandomState(11)
    n_samples = 4
    n_features = 2
    n_ops = 3
    ops_all = [f"op{i}" for i in range(n_ops)]
    X_base = rng.randn(n_samples, n_features)
    elig = np.ones((n_samples, n_ops), dtype=int)

    class _NaNModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            preds = np.zeros(len(X), dtype=np.float64)
            # NaN somewhere in the middle so np.argmax(~isfinite) points at
            # a real sample/op index pair, not row 0.
            preds[len(X) // 2] = np.nan
            return preds

    with pytest.raises(PolicyInductionError, match="Non-finite prediction"):
        skdr_eval.induce_policy_from_sklearn(_NaNModel(), X_base, ops_all, elig)


def test_induce_policy_from_sklearn_handles_negative_predictions(caplog):
    """Negative predictions trigger the abs-fallback warning, not a raise.

    A regression model can produce a small negative service-time prediction
    on the eligible-pair grid; the policy is still computed as
    ``1 / (|pred| + eps)`` and a warning is emitted so the operator can see
    the model is misbehaving. Covers the negative-pred branch of #46.
    """
    rng = np.random.RandomState(12)
    n_samples = 4
    n_features = 2
    n_ops = 3
    ops_all = [f"op{i}" for i in range(n_ops)]
    X_base = rng.randn(n_samples, n_features)
    elig = np.ones((n_samples, n_ops), dtype=int)

    class _NegativeModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), -1.0, dtype=np.float64)

    with caplog.at_level(logging.WARNING, logger="skdr_eval"):
        policy = skdr_eval.induce_policy_from_sklearn(
            _NegativeModel(), X_base, ops_all, elig
        )

    assert "Negative predictions" in caplog.text, (
        f"Expected the negative-prediction warning; got: {caplog.text!r}"
    )
    # All preds collapse to -1 -> abs -> 1 -> uniform 1/n_ops per row.
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        policy, np.full((n_samples, n_ops), 1.0 / n_ops), atol=1e-9
    )


def test_induce_policy_from_sklearn_uniform_fallback_for_no_eligible_samples():
    """Samples with zero eligible ops get the uniform 1/n_ops distribution.

    Matches the prior scalar behavior. Covers the ``if no_elig.any():`` branch
    of the vectorized rewrite (#46).
    """
    rng = np.random.RandomState(13)
    n_samples = 4
    n_features = 2
    n_ops = 3
    ops_all = [f"op{i}" for i in range(n_ops)]
    X_base = rng.randn(n_samples, n_features)
    elig = np.ones((n_samples, n_ops), dtype=int)
    # Row 1 has no eligible operators — should fall back to uniform.
    elig[1] = 0

    class _ConstModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), 0.5, dtype=np.float64)

    policy = skdr_eval.induce_policy_from_sklearn(_ConstModel(), X_base, ops_all, elig)

    # Eligible rows: uniform over their eligible ops (all 1s -> uniform across all).
    expected_eligible_row = np.full(n_ops, 1.0 / n_ops)
    np.testing.assert_allclose(policy[0], expected_eligible_row, atol=1e-9)
    np.testing.assert_allclose(policy[2], expected_eligible_row, atol=1e-9)
    np.testing.assert_allclose(policy[3], expected_eligible_row, atol=1e-9)
    # Ineligible row: uniform 1/n_ops fallback.
    np.testing.assert_allclose(policy[1], np.full(n_ops, 1.0 / n_ops), atol=1e-9)


# --------------------------------------------------------------------------- #
# policy_train deprecation warning (#82 + #60)                               #
# --------------------------------------------------------------------------- #


def test_evaluate_sklearn_models_no_policy_train_warns() -> None:
    """Calling evaluate_sklearn_models without policy_train emits DeprecationWarning."""
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
    models = {"rf": RandomForestRegressor(n_estimators=5, random_state=0)}
    with pytest.warns(DeprecationWarning, match="policy_train"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=2,
            # policy_train intentionally omitted
        )


def test_evaluate_sklearn_models_explicit_policy_train_no_warning() -> None:
    """Passing policy_train explicitly suppresses the DeprecationWarning."""
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
    models = {"rf": RandomForestRegressor(n_estimators=5, random_state=0)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Should NOT raise — no warning expected when policy_train is explicit.
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=2,
            policy_train="pre_split",
        )


def test_evaluate_sklearn_models_none_policy_train_uses_pre_split() -> None:
    """evaluate_sklearn_models(policy_train=None) defaults to 'pre_split' behaviour."""
    logs, _, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=1)
    models = {"rf": RandomForestRegressor(n_estimators=5, random_state=0)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        art_none = skdr_eval.evaluate_sklearn_models(
            logs=logs, models=models, fit_models=True, n_splits=2, policy_train=None
        )
    art_pre = skdr_eval.evaluate_sklearn_models(
        logs=logs, models=models, fit_models=True, n_splits=2, policy_train="pre_split"
    )
    # Both should produce the same artifact structure.
    assert list(art_none.report.columns) == list(art_pre.report.columns)
    assert list(art_none.report["estimator"]) == list(art_pre.report["estimator"])


def test_evaluate_sklearn_models_new_public_symbols_importable() -> None:
    """New public symbols from issues #83 and #99 are accessible from the package."""
    assert hasattr(skdr_eval, "Recommendation")
    assert hasattr(skdr_eval, "RecommendationPolicy")
    assert hasattr(skdr_eval, "Reason")
    assert hasattr(skdr_eval, "DiagnosticGate")
    assert hasattr(skdr_eval, "GateResult")
    assert hasattr(skdr_eval, "gate_diagnostics")
    assert hasattr(skdr_eval, "simulate_coverage")
    assert hasattr(skdr_eval, "CoverageResult")
