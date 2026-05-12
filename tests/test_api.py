"""Test API imports and basic functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import skdr_eval


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
    """Test that evaluate_sklearn_models raises ValueError for an empty models dict."""
    logs, _, _ = skdr_eval.make_synth_logs(n=10, n_ops=3, seed=4)

    with pytest.raises(ValueError, match="models dict must not be empty"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={},
            fit_models=True,
            n_splits=2,
            random_state=42,
        )

    with pytest.raises(ValueError, match="models dict values must not be None"):
        skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": None},
            fit_models=True,
            n_splits=2,
            random_state=42,
        )


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
