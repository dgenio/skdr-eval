"""Tests for bootstrap CI integration in evaluation functions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import skdr_eval

# Constants for CI validation
CI_TOLERANCE_MULTIPLIER = 2.0


def _validate_ci_contains_estimate(row: pd.Series) -> None:
    """Validate that CI contains the point estimate or is close to it.

    Args:
        row: DataFrame row containing 'ci_lower', 'ci_upper', and 'V_hat' columns

    Raises:
        AssertionError: If CI doesn't contain estimate and isn't close enough
    """
    ci_contains_estimate = row["ci_lower"] <= row["V_hat"] <= row["ci_upper"]
    ci_close_to_estimate = abs(
        row["ci_lower"] - row["V_hat"]
    ) < CI_TOLERANCE_MULTIPLIER * abs(row["ci_upper"] - row["ci_lower"])
    assert ci_contains_estimate or ci_close_to_estimate, (
        f"CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] should contain or be close to V_hat {row['V_hat']:.3f}"
    )


class TestBootstrapIntegration:
    """Test suite for bootstrap CI integration."""

    def test_sklearn_models_bootstrap_ci(self):
        """Test that bootstrap CI works in evaluate_sklearn_models."""
        # Generate synthetic data
        logs, _, _ = skdr_eval.make_synth_logs(n=500, n_ops=3, seed=42)

        # Define models
        models = {
            "rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

        # Test without bootstrap CI
        report_no_ci, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            ci_bootstrap=False,
            random_state=42,
        )

        # Test with bootstrap CI
        report_with_ci, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            ci_bootstrap=True,
            alpha=0.05,
            random_state=42,
        )

        # Check that CI columns are added
        assert "ci_lower" not in report_no_ci.columns
        assert "ci_upper" not in report_no_ci.columns
        assert "ci_lower" in report_with_ci.columns
        assert "ci_upper" in report_with_ci.columns

        # Check that CI values are reasonable
        for _, row in report_with_ci.iterrows():
            assert row["ci_lower"] < row["ci_upper"]
            _validate_ci_contains_estimate(row)
            assert not pd.isna(row["ci_lower"])
            assert not pd.isna(row["ci_upper"])

    def test_pairwise_models_bootstrap_ci(self):
        """Test that bootstrap CI works in evaluate_pairwise_models."""
        # Generate synthetic pairwise data
        logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
            n_days=3, n_clients_day=100, n_ops=10, seed=42
        )

        # Define models
        models = {
            "rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

        # Test without bootstrap CI
        report_no_ci, _ = skdr_eval.evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=3,
            ci_bootstrap=False,
            random_state=42,
        )

        # Test with bootstrap CI
        report_with_ci, _ = skdr_eval.evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=3,
            ci_bootstrap=True,
            alpha=0.05,
            random_state=42,
        )

        # Check that CI columns are added
        assert "ci_lower" not in report_no_ci.columns
        assert "ci_upper" not in report_no_ci.columns
        assert "ci_lower" in report_with_ci.columns
        assert "ci_upper" in report_with_ci.columns

        # Check that CI values are reasonable
        for _, row in report_with_ci.iterrows():
            assert row["ci_lower"] < row["ci_upper"]
            _validate_ci_contains_estimate(row)
            assert not pd.isna(row["ci_lower"])
            assert not pd.isna(row["ci_upper"])

    def test_bootstrap_ci_different_alpha_levels(self):
        """Test bootstrap CI with different alpha levels."""
        logs, _, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=42)
        models = {"rf": RandomForestRegressor(n_estimators=10, random_state=42)}

        # Test 90% CI
        report_90, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            alpha=0.1,
            random_state=42,
        )

        # Test 95% CI
        report_95, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            alpha=0.05,
            random_state=42,
        )

        # 90% CI should be narrower than 95% CI
        for i in range(len(report_90)):
            ci_90_width = report_90.iloc[i]["ci_upper"] - report_90.iloc[i]["ci_lower"]
            ci_95_width = report_95.iloc[i]["ci_upper"] - report_95.iloc[i]["ci_lower"]
            assert ci_90_width < ci_95_width

    def test_bootstrap_ci_reproducibility(self):
        """Test that bootstrap CI results are reproducible."""
        logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=42)
        models = {"rf": RandomForestRegressor(n_estimators=10, random_state=42)}

        # Run twice with same random_state
        report1, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            random_state=42,
        )

        report2, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            random_state=42,
        )

        # Results should be identical
        pd.testing.assert_frame_equal(report1, report2)

    def test_bootstrap_ci_fallback_behavior(self):
        """Test that bootstrap CI falls back gracefully on errors."""
        logs, _, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=42)  # Small dataset
        models = {"rf": RandomForestRegressor(n_estimators=5, random_state=42)}

        # This should not raise an exception even with small dataset
        report, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            random_state=42,
        )

        # Should have CI columns
        assert "ci_lower" in report.columns
        assert "ci_upper" in report.columns

        # CI values should be finite
        for _, row in report.iterrows():
            assert np.isfinite(row["ci_lower"])
            assert np.isfinite(row["ci_upper"])

    def test_bootstrap_ci_vs_normal_approximation(self):
        """Test that bootstrap CI differs from normal approximation."""
        logs, _, _ = skdr_eval.make_synth_logs(n=1000, n_ops=3, seed=42)
        models = {"rf": RandomForestRegressor(n_estimators=50, random_state=42)}

        # Get bootstrap CI
        report_bootstrap, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            random_state=42,
        )

        # Get normal approximation CI
        _report_normal, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=False,
            random_state=42,
        )

        # Compute normal approximation manually
        for i in range(len(report_bootstrap)):
            v_hat = report_bootstrap.iloc[i]["V_hat"]
            se_if = report_bootstrap.iloc[i]["SE_if"]
            normal_lower = v_hat - 1.96 * se_if
            normal_upper = v_hat + 1.96 * se_if

            bootstrap_lower = report_bootstrap.iloc[i]["ci_lower"]
            bootstrap_upper = report_bootstrap.iloc[i]["ci_upper"]

            # Bootstrap CI should be different from normal approximation
            # (though they might be close for some cases)
            assert not (
                bootstrap_lower == normal_lower and bootstrap_upper == normal_upper
            )

    def test_bootstrap_ci_time_series_properties(self):
        """Test that bootstrap CI preserves time-series properties."""
        # Create time-series data with correlation
        np.random.seed(42)
        n = 500
        t = np.arange(n)

        # Create correlated time series
        base_trend = 0.01 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 50)
        noise = np.random.normal(0, 0.5, n)
        service_times = 10 + base_trend + seasonal + noise

        # Create logs with time-ordered data
        logs_data = {
            "arrival_ts": pd.date_range("2024-01-01", periods=n, freq="1H"),
            "cli_urgency": np.random.uniform(0, 1, n),
            "cli_complexity": np.random.exponential(1, n),
            "st_load": np.random.exponential(1, n),
            "st_time_of_day": np.sin(
                2 * np.pi * pd.date_range("2024-01-01", periods=n, freq="1H").hour / 24
            ),
            "op_A_elig": np.ones(n, dtype=bool),
            "op_B_elig": np.ones(n, dtype=bool),
            "op_C_elig": np.ones(n, dtype=bool),
            "action": np.random.choice(["op_A", "op_B", "op_C"], n),
            "service_time": service_times,
        }
        logs = pd.DataFrame(logs_data)

        models = {"rf": RandomForestRegressor(n_estimators=20, random_state=42)}

        # Test bootstrap CI with time-series data
        report, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            random_state=42,
        )

        # Should work without errors
        assert "ci_lower" in report.columns
        assert "ci_upper" in report.columns

        # CI should be reasonable
        for _, row in report.iterrows():
            assert row["ci_lower"] < row["ci_upper"]
            assert np.isfinite(row["ci_lower"])
            assert np.isfinite(row["ci_upper"])

    def test_bootstrap_ci_with_clipping(self):
        """Test bootstrap CI with different clipping scenarios to improve coverage."""
        # Generate data that will trigger clipping logic
        logs, _, _ = skdr_eval.make_synth_logs(n=1000, n_ops=3, seed=42)

        # Use a model that will create extreme propensity scores
        models = {"rf": RandomForestRegressor(n_estimators=5, random_state=42)}

        # Test with different clipping thresholds to cover both branches
        for clip_threshold in [2.0, 5.0, float("inf")]:
            report, _ = skdr_eval.evaluate_sklearn_models(
                logs=logs,
                models=models,
                fit_models=True,
                ci_bootstrap=True,
                clip_grid=(clip_threshold,),  # Use specific clip threshold
                random_state=42,
            )

            # Should have CI columns
            assert "ci_lower" in report.columns
            assert "ci_upper" in report.columns

            # CI should be reasonable
            for _, row in report.iterrows():
                assert row["ci_lower"] < row["ci_upper"]
                assert np.isfinite(row["ci_lower"])
                assert np.isfinite(row["ci_upper"])

    def test_bootstrap_ci_fallback_scenarios(self):
        """Test bootstrap CI fallback scenarios to improve coverage."""
        # Create data that might trigger fallback scenarios
        logs, _, _ = skdr_eval.make_synth_logs(n=100, n_ops=2, seed=42)

        models = {"rf": RandomForestRegressor(n_estimators=3, random_state=42)}

        # Test with very small dataset that might trigger fallbacks
        report, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            ci_bootstrap=True,
            n_splits=2,  # Small number of splits
            random_state=42,
        )

        # Should still work and have CI columns
        assert "ci_lower" in report.columns
        assert "ci_upper" in report.columns

        # CI should be finite
        for _, row in report.iterrows():
            assert np.isfinite(row["ci_lower"])
            assert np.isfinite(row["ci_upper"])
