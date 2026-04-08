"""Tests for bootstrap CI integration in evaluation functions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import skdr_eval
from skdr_eval.core import block_bootstrap_ci

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
            "arrival_ts": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "cli_urgency": np.random.uniform(0, 1, n),
            "cli_complexity": np.random.exponential(1, n),
            "st_load": np.random.exponential(1, n),
            "st_time_of_day": np.sin(
                2 * np.pi * pd.date_range("2024-01-01", periods=n, freq="1h").hour / 24
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


class TestBootstrapCISimulationProof:
    """Simulation proof that the moving-block bootstrap CI machinery is
    properly calibrated.

    Required by AGENTS.md Section 6 and .github/copilot-instructions.md Rule 1:
    "Any change to statistical evaluation logic MUST include a simulation that
    proves the code recovers a known ground-truth parameter."

    Strategy overview
    -----------------
    We test at two levels:

    1. **Core CI function** (``block_bootstrap_ci``): Generate IID data from
       a known N(mu, sigma^2) distribution and verify the percentile bootstrap
       CI covers the true mean ``mu`` at approximately 95% rate across many
       replications.  This directly tests the statistical machinery.

    2. **Full pipeline sanity** (``evaluate_sklearn_models``): Verify that
       the CI contains the point estimate, has positive width, and is within
       a reasonable factor of the influence-function SE.  This confirms the
       pipeline wires the CI correctly without requiring an external oracle
       (which is complicated by the q_pi == q_hat design choice documented
       in core.py ``dr_value_with_clip``).
    """

    def test_block_bootstrap_ci_coverage(self):
        """Monte Carlo coverage of block_bootstrap_ci on known N(mu, sigma^2).

        Ground truth: mu = 25.0.
        We draw K=50 IID samples of size n=500 from N(mu, 3^2), compute the
        95% CI on each, and check coverage >= 85%.  The block bootstrap is
        slightly conservative for IID data (block_len = sqrt(n) introduces
        dependence structure where none exists), so 85% is a safe lower bound
        for the true 95% nominal rate.
        """
        mu = 25.0
        sigma = 3.0
        n = 500
        K = 50
        alpha = 0.05
        covered = 0

        for seed in range(K):
            rng = np.random.RandomState(seed)
            values = rng.normal(mu, sigma, size=n)

            ci_lo, ci_hi = block_bootstrap_ci(
                values_num=values,
                values_den=None,
                base_mean=np.array([values.mean()]),
                n_boot=400,
                alpha=alpha,
                random_state=seed,
            )

            if ci_lo <= mu <= ci_hi:
                covered += 1

        coverage = covered / K
        assert coverage >= 0.85, (
            f"block_bootstrap_ci coverage {coverage:.0%} ({covered}/{K}) "
            f"is below 85% threshold for 95% nominal CI on N({mu}, {sigma}^2)."
        )

    def test_pipeline_bootstrap_ci_sanity(self):
        """Sanity checks on bootstrap CI from the full evaluation pipeline.

        Verifies structural properties that any calibrated CI must satisfy:
        - The DR point estimate V_hat lies within [ci_lower, ci_upper].
        - The CI has positive width (ci_upper > ci_lower).
        - The CI width is within 10x of the normal-approximation CI
          (2 * z_{alpha/2} * SE_if), guarding against gross miscalibration.

        Note: Only DR rows are checked.  SNDR currently shares the DR
        bootstrap CI (a known limitation — see issue #58), so its point
        estimate may fall outside the shared CI.
        """
        logs, _, _ = skdr_eval.make_synth_logs(n=2000, n_ops=3, seed=42)
        model = RandomForestRegressor(n_estimators=30, random_state=42)

        report, _ = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"rf": model},
            fit_models=True,
            n_splits=3,
            ci_bootstrap=True,
            alpha=0.05,
            random_state=42,
        )

        dr_rows = report[report["estimator"] == "DR"]
        assert len(dr_rows) > 0, "No DR rows in report"

        for _, row in dr_rows.iterrows():
            v = row["V_hat"]
            lo = row["ci_lower"]
            hi = row["ci_upper"]
            se = row["SE_if"]

            # CI must contain point estimate
            assert lo <= v <= hi, f"DR: V_hat={v:.4f} outside CI ({lo:.4f}, {hi:.4f})"

            # CI must have positive width
            ci_width = hi - lo
            assert ci_width > 0, "DR: CI width is non-positive"

            # CI width should be within 10x of normal approximation
            se_width = 2 * 1.96 * se
            if se_width > 0:
                assert ci_width < 10 * se_width, (
                    f"DR: CI width {ci_width:.2f} is >10x the "
                    f"normal-approx width {se_width:.2f}"
                )
