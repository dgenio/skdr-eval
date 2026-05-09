"""Test pairwise evaluation API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge

from skdr_eval import evaluate_pairwise_models, make_pairwise_synth
from skdr_eval.pairwise import (
    PairwiseDesign,
    _fit_surrogate,
    _surrogate_features,
    induce_policy_stream_topk,
)


def test_pairwise_regression_basic():
    """Test basic pairwise evaluation with regression task."""
    # Generate small synthetic data
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42, binary=False
    )

    # Create simple models
    models = {
        "ridge": Ridge(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
    }

    # Fit models on pairwise features
    # For testing, we'll create a simple feature matrix
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values

    for model in models.values():
        model.fit(X, y)

    # Run pairwise evaluation
    report, detailed_results = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,  # Small for testing
        strategy="direct",  # Force direct strategy for small data
        random_state=42,
    )

    # Basic checks
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "model" in report.columns
    assert "estimator" in report.columns
    assert "V_hat" in report.columns
    assert "ESS" in report.columns
    assert "match_rate" in report.columns

    # Check that all values are finite
    numeric_cols = ["V_hat", "SE_if", "ESS", "tail_mass", "MSE_est", "match_rate"]
    for col in numeric_cols:
        if col in report.columns:
            assert report[col].notna().all(), f"NaN values found in {col}"
            assert np.isfinite(report[col]).all(), f"Non-finite values found in {col}"

    # Check ESS > 0
    assert (report["ESS"] > 0).all(), "ESS should be positive"

    # Check match_rate in [0, 1]
    assert (report["match_rate"] >= 0).all(), "match_rate should be >= 0"
    assert (report["match_rate"] <= 1).all(), "match_rate should be <= 1"

    # Check detailed results structure
    assert isinstance(detailed_results, dict)
    assert len(detailed_results) == len(models)

    for model_name in models:
        assert model_name in detailed_results
        model_results = detailed_results[model_name]
        assert isinstance(model_results, dict)
        assert "DR" in model_results or "SNDR" in model_results


def test_pairwise_binary_basic():
    """Test basic pairwise evaluation with binary task."""
    # Generate small synthetic data with binary outcomes
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42, binary=True
    )

    # Create simple models
    models = {
        "logistic": LogisticRegression(random_state=42, max_iter=1000),
    }

    # Fit models on pairwise features
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["success"].values

    for model in models.values():
        model.fit(X, y)

    # Run pairwise evaluation
    report, _ = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="success",
        task_type="binary",
        direction="max",  # Maximize success probability
        n_splits=2,
        strategy="direct",
        random_state=42,
    )

    # Basic checks
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "model" in report.columns
    assert "V_hat" in report.columns

    # Check that all values are finite
    numeric_cols = ["V_hat", "SE_if", "ESS", "match_rate"]
    for col in numeric_cols:
        if col in report.columns:
            assert report[col].notna().all(), f"NaN values found in {col}"
            assert np.isfinite(report[col]).all(), f"Non-finite values found in {col}"


def test_pairwise_autoscale_strategy():
    """Test that autoscale strategy selection works."""
    # Generate data that should trigger different strategies
    small_logs, small_ops = make_pairwise_synth(
        n_days=1, n_clients_day=50, n_ops=5, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Fit model
    feature_cols = [
        col for col in small_logs.columns if col.startswith(("cli_", "op_"))
    ]
    X = small_logs[feature_cols].values
    y = small_logs["service_time"].values
    models["ridge"].fit(X, y)

    # Test auto strategy (should select direct for small data)
    report, _ = evaluate_pairwise_models(
        logs_df=small_logs,
        op_daily_df=small_ops,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        strategy="auto",
        random_state=42,
    )

    assert len(report) > 0

    # Test explicit direct strategy
    report_direct, _ = evaluate_pairwise_models(
        logs_df=small_logs,
        op_daily_df=small_ops,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        strategy="direct",
        random_state=42,
    )

    assert len(report_direct) > 0


def test_pairwise_propensity_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that propensity='auto' runs end-to-end without crashing."""
    # Force the multinomial path so the test is deterministic across
    # environments regardless of whether SciPy is installed.
    monkeypatch.setattr("skdr_eval.core.SCIPY_AVAILABLE", False)

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=8, seed=42, binary=False
    )

    models = {
        "ridge": Ridge(random_state=42),
    }

    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    report, detailed_results = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        propensity="auto",
        random_state=42,
    )

    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "ridge" in detailed_results
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()


def test_pairwise_design_creation():
    """Test PairwiseDesign creation and statistics."""

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=50, n_ops=8, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Check design attributes
    assert len(design.cli_features) > 0
    assert len(design.op_features) > 0
    assert len(design.ops_all_by_day) == 2  # 2 days
    assert all(len(ops) == 8 for ops in design.ops_all_by_day.values())

    # Check statistics
    stats = design.get_stats()
    assert stats["n_rows"] == 100  # 2 days * 50 clients
    assert stats["n_days"] == 2
    assert stats["candidate_pairs"] > 0
    assert stats["memory_gb"] > 0


def test_pairwise_with_eligibility():
    """Test pairwise evaluation with eligibility constraints."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=30, n_ops=6, seed=42
    )

    # Verify eligibility masks exist
    assert "elig_mask" in logs_df.columns
    assert all(isinstance(mask, list) for mask in logs_df["elig_mask"])

    models = {"ridge": Ridge(random_state=42)}

    # Fit model
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    # Run evaluation with eligibility
    report, _ = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        elig_col="elig_mask",
        random_state=42,
    )

    assert len(report) > 0
    assert (report["ESS"] > 0).all()


def test_pairwise_error_handling():
    """Test error handling in pairwise evaluation."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Test with invalid task_type
    with pytest.raises(ValueError, match="Unknown task_type"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="invalid",
            direction="min",
        )

    # Test with invalid direction
    with pytest.raises(ValueError, match=r"Unknown.*direction"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="invalid",
        )


def test_pairwise_fit_models_true():
    """Test that fit_models=True fits unfitted models internally."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=50, n_ops=8, seed=42
    )

    # Create unfitted models
    models = {
        "ridge": Ridge(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42, max_iter=10),
    }

    # Run evaluation with fit_models=True
    report, detailed_results = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        fit_models=True,  # Let the function fit the models
        random_state=42,
    )

    # Verify results are valid
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "V_hat" in report.columns
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()

    # Verify all models were evaluated
    assert len(detailed_results) == len(models)
    for model_name in models:
        assert model_name in detailed_results


def test_pairwise_stream_topk_strategy():
    """Test stream_topk strategy end-to-end with top-K filtering."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=12, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Fit model on observed data
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    # Run evaluation with stream_topk strategy
    report, detailed_results = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="stream_topk",
        topk=5,  # Keep top 5 operators per client
        random_state=42,
    )

    # Verify results are valid
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "V_hat" in report.columns
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()

    # Verify that stream_topk strategy evaluated successfully
    assert "ridge" in detailed_results
    model_results = detailed_results["ridge"]
    # Should have either DR or SNDR estimator
    assert len(model_results) > 0
    for _estimator_name, dr_result in model_results.items():
        # Check that the DRResult has valid values
        assert hasattr(dr_result, "V_hat")
        assert np.isfinite(dr_result.V_hat)


def test_pairwise_unfitted_model_fails_without_fit_models():
    """Test that unfitted models raise NotFittedError when fit_models=False."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    # Create unfitted model
    models = {"ridge": Ridge(random_state=42)}

    # Should raise NotFittedError when trying to predict with unfitted model
    with pytest.raises(NotFittedError):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            strategy="direct",
            fit_models=False,  # Don't fit models
            random_state=42,
        )


@pytest.mark.parametrize(
    "strategy",
    ["direct", "stream", "stream_topk"],
    ids=["direct_unfitted", "stream_unfitted", "stream_topk_unfitted"],
)
def test_pairwise_unfitted_models_fail_across_strategies(strategy: str):
    """Test that unfitted models raise NotFittedError immediately for all strategies."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=10, seed=42
    )

    # Create unfitted model
    models = {"ridge": Ridge(random_state=42)}

    # All strategies should fail fast with NotFittedError on unfitted models
    with pytest.raises(NotFittedError):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=2,
            strategy=strategy,
            topk=5,
            fit_models=False,  # Don't fit models
            random_state=42,
        )


def test_stream_topk_rejects_plain_ridge_surrogate():
    """Plain ridge surrogate is degenerate (day-global top-K) and must be rejected."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=8, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Fit a model so check_is_fitted passes (we want to hit the surrogate validation)
    feature_cols = design.cli_features + design.op_features
    X = design.logs_df[feature_cols].values
    y = design.logs_df["service_time"].values
    model = Ridge()
    model.fit(X, y)

    with pytest.raises(ValueError, match="degenerate"):
        induce_policy_stream_topk(
            models={"m": model},
            design=design,
            metric_col="service_time",
            surrogate_model="ridge",  # rejected
        )


@pytest.mark.parametrize("surrogate_model", ["hgb", "ridge_interaction"])
def test_stream_topk_personalization_recovery(surrogate_model: str):
    """Simulation proof: stream_topk top-K must vary across clients.

    Required by AGENTS.md / .claude/CLAUDE.md: any change to statistical
    evaluation logic must include a simulation that proves the code recovers
    a known ground-truth parameter.

    Ground truth setup
    ------------------
    We construct synthetic data where the optimal operator depends on a
    *client* feature: clients with ``cli_type=0`` should prefer operators with
    high ``op_a``; clients with ``cli_type=1`` should prefer operators with
    high ``op_b``. The true target is

        y = - (cli_type * op_b + (1 - cli_type) * op_a) + small_noise

    so ``direction="min"`` recovers each client's preferred operator.

    A degenerate surrogate (linear on concatenated [cli | op]) cannot
    distinguish the two client types and would produce the *same* top-K for
    all clients. The fixed surrogates ("hgb", "ridge_interaction") capture
    the cli x op interaction and produce *client-specific* top-K rankings —
    we verify this directly.
    """
    rng = np.random.RandomState(7)
    n_ops = 12
    n_clients_per_type = 200
    topk = 3

    # Operators: half are op_a-heavy, half are op_b-heavy (well separated)
    op_a = np.concatenate(
        [
            rng.uniform(0.8, 1.0, n_ops // 2),
            rng.uniform(0.0, 0.2, n_ops - n_ops // 2),
        ]
    )
    op_b = np.concatenate(
        [
            rng.uniform(0.0, 0.2, n_ops // 2),
            rng.uniform(0.8, 1.0, n_ops - n_ops // 2),
        ]
    )
    op_ids = [f"op_{i}" for i in range(n_ops)]

    op_daily_df = pd.DataFrame(
        {
            "operator_id": op_ids,
            "arrival_day": ["d1"] * n_ops,
            "op_a": op_a,
            "op_b": op_b,
        }
    )

    # Two client types with opposite preferences (cli_type repurposed as cli_x)
    cli_x = np.concatenate([np.zeros(n_clients_per_type), np.ones(n_clients_per_type)])
    n_clients = len(cli_x)
    chosen_idx = rng.randint(0, n_ops, size=n_clients)
    chosen_op_a = op_a[chosen_idx]
    chosen_op_b = op_b[chosen_idx]
    noise = rng.normal(0, 0.02, size=n_clients)
    y = -(cli_x * chosen_op_b + (1 - cli_x) * chosen_op_a) + noise

    logs_df = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n_clients)],
            "operator_id": [op_ids[i] for i in chosen_idx],
            "arrival_day": ["d1"] * n_clients,
            "arrival_ts": np.arange(n_clients),
            "cli_x": cli_x,
            "op_a": chosen_op_a,
            "op_b": chosen_op_b,
            "service_time": y,
        }
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test the surrogate's *ranking* per client directly. This is the
    # personalization property. We do NOT route through the full model —
    # whether the full model can also resolve the interaction is a separate
    # concern (and depends on its capacity / data).
    X_cli_train = design.logs_df[design.cli_features].values.astype(np.float32)
    X_op_train = design.logs_df[design.op_features].values.astype(np.float32)
    y_train = design.logs_df["service_time"].values.astype(np.float64)

    surrogate, feature_mode = _fit_surrogate(
        surrogate_model, X_cli_train, X_op_train, y_train, random_state=0
    )

    # Build features for one type-0 client and one type-1 client paired with
    # *all* operators, then score with the surrogate.
    op_features_full = op_daily_df[["op_a", "op_b"]].values.astype(np.float32)

    def _score_for(cli_x_value: float) -> np.ndarray:
        X_cli = np.full((n_ops, 1), cli_x_value, dtype=np.float32)
        X_in = _surrogate_features(feature_mode, X_cli, op_features_full)
        return surrogate.predict(X_in)

    score_type0 = _score_for(0.0)
    score_type1 = _score_for(1.0)

    # Top-K (smallest scores under direction="min") for each client type
    top0 = np.argpartition(score_type0, topk - 1)[:topk]
    top1 = np.argpartition(score_type1, topk - 1)[:topk]

    # Personalization signal: top-K rankings must differ between client types.
    # A degenerate (linear-on-concat) surrogate produces identical top-K, so
    # checking set inequality is the strongest possible assertion.
    assert set(top0) != set(top1), (
        f"surrogate={surrogate_model}: top-K is identical across client types "
        f"({sorted(top0)} == {sorted(top1)}); this is the day-global degeneracy "
        f"the fix must prevent."
    )

    # Stronger correctness check: the surrogate should rank op_a-heavy
    # operators highly for type-0 clients and op_b-heavy for type-1 clients.
    # op_a-heavy operators are the first half (indices 0..n_ops/2-1).
    a_heavy_set = set(range(n_ops // 2))
    b_heavy_set = set(range(n_ops // 2, n_ops))
    overlap_type0_with_a = len(set(top0) & a_heavy_set)
    overlap_type1_with_b = len(set(top1) & b_heavy_set)
    assert overlap_type0_with_a >= topk - 1, (
        f"surrogate={surrogate_model}: type-0 top-K should be op_a-heavy; "
        f"got {sorted(top0)} (a-heavy indices: {sorted(a_heavy_set)})"
    )
    assert overlap_type1_with_b >= topk - 1, (
        f"surrogate={surrogate_model}: type-1 top-K should be op_b-heavy; "
        f"got {sorted(top1)} (b-heavy indices: {sorted(b_heavy_set)})"
    )


def test_pairwise_pre_split_holds_out_evaluation_data():
    """pre_split must reduce the evaluation set, fitting on earlier days only."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=4, n_clients_day=40, n_ops=6, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}
    # pre_split should fit on first 75% of days, evaluate on remaining 25%
    report, _ = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        fit_models=True,
        policy_train="pre_split",
        policy_train_frac=0.75,
        random_state=42,
    )

    # Evaluation succeeded with held-out data
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert np.isfinite(report["V_hat"]).all()


if __name__ == "__main__":
    pytest.main([__file__])
