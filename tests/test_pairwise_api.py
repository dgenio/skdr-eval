"""Test pairwise evaluation API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from skdr_eval import evaluate_pairwise_models, make_pairwise_synth
from skdr_eval.pairwise import PairwiseDesign


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


if __name__ == "__main__":
    pytest.main([__file__])
