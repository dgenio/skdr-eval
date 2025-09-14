"""Test propensity score diagnostics functionality."""

import numpy as np
import pandas as pd
import pytest

import skdr_eval
from skdr_eval.diagnostics import (
    PropensityDiagnostics,
    assess_propensity_calibration,
    assess_propensity_discrimination,
    check_propensity_balance,
    check_propensity_overlap,
    comprehensive_propensity_diagnostics,
    compute_balance_statistics,
    compute_propensity_log_loss,
    compute_propensity_statistics,
    generate_propensity_report,
)


def test_propensity_overlap():
    """Test propensity score overlap checking."""
    # Create test data with good overlap
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    # Create propensity scores with good overlap
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    overlap_ratio = check_propensity_overlap(propensities, actions)
    assert 0 <= overlap_ratio <= 1
    assert isinstance(overlap_ratio, float)
    
    # Test with poor overlap (extreme propensity scores)
    extreme_propensities = np.array([
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025],
        [0.025, 0.025, 0.95],
    ] * (n_samples // 3))
    
    poor_overlap = check_propensity_overlap(extreme_propensities, actions)
    assert poor_overlap < overlap_ratio  # Should have worse overlap


def test_propensity_balance():
    """Test propensity score balance checking."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    # Create balanced propensity scores
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    balance_ratio = check_propensity_balance(propensities, actions)
    assert 0 <= balance_ratio <= 1
    assert isinstance(balance_ratio, float)
    
    # Test with extreme scores
    extreme_propensities = np.array([
        [0.99, 0.005, 0.005],
        [0.005, 0.99, 0.005],
        [0.005, 0.005, 0.99],
    ] * (n_samples // 3))
    
    poor_balance = check_propensity_balance(extreme_propensities, actions)
    assert poor_balance < balance_ratio  # Should have worse balance


def test_propensity_calibration():
    """Test propensity score calibration assessment."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    # Create well-calibrated propensity scores
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    calibration_score, calibration_curve = assess_propensity_calibration(
        propensities, actions, n_bins=5
    )
    
    assert 0 <= calibration_score <= 1
    assert isinstance(calibration_score, float)
    assert len(calibration_curve) == 2
    assert len(calibration_curve[0]) <= 5  # Should have at most n_bins points


def test_propensity_discrimination():
    """Test propensity score discrimination assessment."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    # Create propensity scores with some discrimination
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    discrimination_score, roc_curve = assess_propensity_discrimination(
        propensities, actions
    )
    
    assert 0 <= discrimination_score <= 1
    assert isinstance(discrimination_score, float)
    assert len(roc_curve) == 3  # fpr, tpr, thresholds


def test_propensity_statistics():
    """Test propensity score statistics computation."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    stats = compute_propensity_statistics(propensities, actions)
    
    # Check required statistics
    required_stats = ["min_pscore", "max_pscore", "mean_pscore", "std_pscore", "median_pscore"]
    for stat in required_stats:
        assert stat in stats
        assert isinstance(stats[stat], float)
    
    # Check quantiles
    quantile_stats = [k for k in stats.keys() if k.startswith("pscore_q")]
    assert len(quantile_stats) > 0
    
    # Check logical relationships
    assert stats["min_pscore"] <= stats["mean_pscore"] <= stats["max_pscore"]
    assert stats["std_pscore"] >= 0


def test_balance_statistics():
    """Test balance statistics computation."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    balance_stats = compute_balance_statistics(propensities, actions)
    
    # Check that we have statistics for each action
    for action_idx in range(n_actions):
        assert f"action_{action_idx}_count" in balance_stats
        assert f"action_{action_idx}_mean_pscore" in balance_stats
        assert f"action_{action_idx}_std_pscore" in balance_stats
        
        assert balance_stats[f"action_{action_idx}_count"] >= 0
        assert 0 <= balance_stats[f"action_{action_idx}_mean_pscore"] <= 1
        assert balance_stats[f"action_{action_idx}_std_pscore"] >= 0


def test_propensity_log_loss():
    """Test propensity score log loss computation."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    log_loss_score = compute_propensity_log_loss(propensities, actions)
    
    assert log_loss_score >= 0
    assert isinstance(log_loss_score, float)


def test_comprehensive_diagnostics():
    """Test comprehensive propensity diagnostics."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    diagnostics = comprehensive_propensity_diagnostics(propensities, actions)
    
    assert isinstance(diagnostics, PropensityDiagnostics)
    
    # Check all required attributes
    assert hasattr(diagnostics, "overlap_ratio")
    assert hasattr(diagnostics, "balance_ratio")
    assert hasattr(diagnostics, "calibration_score")
    assert hasattr(diagnostics, "discrimination_score")
    assert hasattr(diagnostics, "log_loss_score")
    assert hasattr(diagnostics, "min_pscore")
    assert hasattr(diagnostics, "max_pscore")
    assert hasattr(diagnostics, "mean_pscore")
    assert hasattr(diagnostics, "std_pscore")
    assert hasattr(diagnostics, "quantiles")
    assert hasattr(diagnostics, "balance_stats")
    
    # Check value ranges
    assert 0 <= diagnostics.overlap_ratio <= 1
    assert 0 <= diagnostics.balance_ratio <= 1
    assert 0 <= diagnostics.calibration_score <= 1
    assert 0 <= diagnostics.discrimination_score <= 1
    assert diagnostics.log_loss_score >= 0
    assert 0 <= diagnostics.min_pscore <= diagnostics.max_pscore <= 1
    assert diagnostics.std_pscore >= 0


def test_generate_report():
    """Test report generation."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    diagnostics = comprehensive_propensity_diagnostics(propensities, actions)
    
    # Test text report
    text_report = generate_propensity_report(diagnostics, output_format="text")
    assert isinstance(text_report, str)
    assert len(text_report) > 0
    assert "PROPENSITY SCORE DIAGNOSTICS REPORT" in text_report
    
    # Test markdown report
    markdown_report = generate_propensity_report(diagnostics, output_format="markdown")
    assert isinstance(markdown_report, str)
    assert len(markdown_report) > 0
    assert "# Propensity Score Diagnostics Report" in markdown_report


def test_evaluate_propensity_diagnostics():
    """Test the main evaluation function."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3
    
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    diagnostics, report = skdr_eval.evaluate_propensity_diagnostics(
        propensities, actions, output_format="text"
    )
    
    assert isinstance(diagnostics, PropensityDiagnostics)
    assert isinstance(report, str)
    assert len(report) > 0


def test_diagnostics_error_handling():
    """Test error handling in diagnostics."""
    # Test with invalid input shapes
    propensities = np.random.rand(10, 3)
    actions = np.array([0, 1, 2])  # Wrong length
    
    with pytest.raises(skdr_eval.DataValidationError):
        check_propensity_overlap(propensities, actions)
    
    # Test with insufficient data
    small_propensities = np.random.rand(5, 3)
    small_actions = np.array([0, 1, 0, 1, 0])
    
    with pytest.raises(skdr_eval.InsufficientDataError):
        comprehensive_propensity_diagnostics(small_propensities, small_actions)


def test_diagnostics_integration():
    """Test diagnostics integration with main workflow."""
    # Generate synthetic data
    logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=42)
    design = skdr_eval.build_design(logs)
    
    # Fit propensity model
    propensities, _ = skdr_eval.fit_propensity_timecal(
        design.X_phi, design.A, design.ts, n_splits=3, random_state=42
    )
    
    # Run diagnostics
    diagnostics, report = skdr_eval.evaluate_propensity_diagnostics(
        propensities, design.A, output_format="text"
    )
    
    assert isinstance(diagnostics, PropensityDiagnostics)
    assert isinstance(report, str)
    assert len(report) > 0
    
    # Check that diagnostics make sense
    assert 0 <= diagnostics.overlap_ratio <= 1
    assert 0 <= diagnostics.balance_ratio <= 1
    assert diagnostics.calibration_score >= 0
    assert 0 <= diagnostics.discrimination_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])