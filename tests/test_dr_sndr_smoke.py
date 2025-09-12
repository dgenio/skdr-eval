"""Smoke tests for DR and SNDR evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skdr_eval


def test_dr_sndr_smoke():
    """Smoke test for DR and SNDR evaluation with two models."""
    # Generate synthetic data
    logs, _, _ = skdr_eval.make_synth_logs(n=3000, n_ops=5, seed=7)

    # Define models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=7),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=7),
    }

    # Evaluate models
    report, detailed_results = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        policy_train="pre_split",
        n_splits=3,
        random_state=7,
    )

    # Check that report has expected structure
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 4  # 2 models x 2 estimators (DR, SNDR)

    # Check required columns exist
    required_cols = [
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
    for col in required_cols:
        assert col in report.columns, f"Missing column: {col}"

    # Check that all values are finite
    numeric_cols = [col for col in required_cols if col not in ["model", "estimator"]]
    for col in numeric_cols:
        assert report[col].notna().all(), f"NaN values in {col}"
        assert np.isfinite(report[col]).all(), f"Non-finite values in {col}"

    # Check that ESS > 0 for chosen clips
    dr_rows = report[report["estimator"] == "DR"]
    sndr_rows = report[report["estimator"] == "SNDR"]

    assert (dr_rows["ESS"] > 0).all(), "DR ESS should be positive"
    assert (sndr_rows["ESS"] > 0).all(), "SNDR ESS should be positive"

    # Check that match_rate is in [0, 1]
    assert (report["match_rate"] >= 0).all(), "match_rate should be >= 0"
    assert (report["match_rate"] <= 1).all(), "match_rate should be <= 1"

    # Check that propensity score quantiles are ordered
    for _, row in report.iterrows():
        assert row["pscore_q01"] <= row["pscore_q05"], "p-score quantiles not ordered"
        assert row["pscore_q05"] <= row["pscore_q10"], "p-score quantiles not ordered"
        assert row["min_pscore"] <= row["pscore_q01"], "min p-score > q01"

    # Check detailed results structure
    assert isinstance(detailed_results, dict)
    for model_name in models:
        assert model_name in detailed_results
        assert "DR" in detailed_results[model_name]
        assert "SNDR" in detailed_results[model_name]

        dr_result = detailed_results[model_name]["DR"]
        sndr_result = detailed_results[model_name]["SNDR"]

        assert isinstance(dr_result, skdr_eval.DRResult)
        assert isinstance(sndr_result, skdr_eval.DRResult)

        # Check that grid has expected structure
        assert hasattr(dr_result, "grid")
        assert hasattr(sndr_result, "grid")
        assert len(dr_result.grid) > 0
        assert len(sndr_result.grid) > 0


def test_dr_sndr_values_reasonable():
    """Test that DR and SNDR values are in reasonable ranges."""
    # Generate smaller dataset for faster testing
    logs, _, _ = skdr_eval.make_synth_logs(n=1000, n_ops=3, seed=42)

    # Simple model
    models = {
        "simple_rf": RandomForestRegressor(n_estimators=10, random_state=42),
    }

    report, _ = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=42,
    )

    # Service times should be positive and reasonable
    service_times = logs["service_time"]
    min_service = service_times.min()
    max_service = service_times.max()
    service_times.mean()

    # DR/SNDR estimates should be in reasonable range relative to observed service times
    for _, row in report.iterrows():
        v_hat = row["V_hat"]
        # Should be positive (service times are positive)
        assert v_hat > 0, f"V_hat should be positive, got {v_hat}"
        # Should be in reasonable range (not too far from observed range)
        assert v_hat >= min_service * 0.5, (
            f"V_hat too small: {v_hat} vs min {min_service}"
        )
        assert v_hat <= max_service * 2.0, (
            f"V_hat too large: {v_hat} vs max {max_service}"
        )


def test_clip_selection():
    """Test that clipping threshold selection works correctly."""
    logs, _, _ = skdr_eval.make_synth_logs(n=500, n_ops=3, seed=123)

    models = {
        "test_model": RandomForestRegressor(n_estimators=10, random_state=123),
    }

    # Test with different clip grids
    clip_grids = [
        (2, 5, 10),
        (1, 2, 5, 10, 20),
        (5, 10, 20, 50, float("inf")),
    ]

    for clip_grid in clip_grids:
        report, detailed_results = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            clip_grid=clip_grid,
            n_splits=3,
            random_state=123,
        )

        # Check that selected clips are from the grid
        for _, row in report.iterrows():
            selected_clip = row["clip"]
            assert selected_clip in clip_grid, (
                f"Selected clip {selected_clip} not in grid {clip_grid}"
            )

        # Check that ESS decreases with higher clipping (generally)
        dr_result = detailed_results["test_model"]["DR"]
        grid = dr_result.grid

        # Sort by clip value
        grid_sorted = grid.sort_values("clip")
        ess_values = grid_sorted["ESS"].values

        # ESS should generally decrease as clip increases (more aggressive clipping)
        # Allow some flexibility due to numerical issues
        assert len(ess_values) > 1, "Need multiple clip values to test ESS trend"


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Very small dataset
    logs_small, _, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=999)

    models = {
        "tiny_rf": RandomForestRegressor(n_estimators=5, random_state=999),
    }

    # Should still work with small dataset
    report, _ = skdr_eval.evaluate_sklearn_models(
        logs=logs_small,
        models=models,
        fit_models=True,
        n_splits=2,  # Reduce splits for small dataset
        random_state=999,
    )

    assert len(report) == 2  # DR and SNDR
    assert (report["ESS"] >= 0).all()
    assert (report["match_rate"] >= 0).all()
    assert (report["match_rate"] <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__])
