"""Tests for visualization module."""

import numpy as np
import pytest

from skdr_eval.visualization import (
    create_dashboard,
    plot_calibration_curve,
    plot_diagnostics_summary,
    plot_dr_results,
    plot_propensity_distribution,
    plot_roc_curve,
)


def test_plot_propensity_distribution():
    """Test propensity distribution plotting."""
    n_samples = 100
    n_actions = 3
    
    # Create test data
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    # Test basic plotting
    fig = plot_propensity_distribution(propensities, actions)
    assert fig is not None
    
    # Test with action names
    action_names = ["Action A", "Action B", "Action C"]
    fig = plot_propensity_distribution(propensities, actions, action_names=action_names)
    assert fig is not None
    
    # Test with save path
    fig = plot_propensity_distribution(propensities, actions, save_path="/tmp/test_plot.png")
    assert fig is not None


def test_plot_dr_results():
    """Test DR results plotting."""
    # Create test results
    results = {
        "Model1": {"V_hat": 0.5, "SE_if": 0.1, "ESS": 50, "clip": 5},
        "Model2": {"V_hat": 0.6, "SE_if": 0.12, "ESS": 45, "clip": 10},
        "Model3": {"V_hat": 0.55, "SE_if": 0.11, "ESS": 48, "clip": 7},
    }
    
    fig = plot_dr_results(results)
    assert fig is not None
    
    # Test with save path
    fig = plot_dr_results(results, save_path="/tmp/test_dr_plot.png")
    assert fig is not None


def test_plot_calibration_curve():
    """Test calibration curve plotting."""
    # Create test calibration curve
    calibration_curve = [
        (0.1, 0.12),
        (0.3, 0.28),
        (0.5, 0.52),
        (0.7, 0.68),
        (0.9, 0.88),
    ]
    
    fig = plot_calibration_curve(calibration_curve)
    assert fig is not None
    
    # Test with save path
    fig = plot_calibration_curve(calibration_curve, save_path="/tmp/test_cal_plot.png")
    assert fig is not None


def test_plot_roc_curve():
    """Test ROC curve plotting."""
    # Create test ROC curve
    roc_curve = [
        (0.0, 0.0),
        (0.1, 0.3),
        (0.2, 0.5),
        (0.3, 0.7),
        (0.4, 0.8),
        (0.5, 0.85),
        (1.0, 1.0),
    ]
    
    fig = plot_roc_curve(roc_curve)
    assert fig is not None
    
    # Test with save path
    fig = plot_roc_curve(roc_curve, save_path="/tmp/test_roc_plot.png")
    assert fig is not None


def test_plot_diagnostics_summary():
    """Test diagnostics summary plotting."""
    from skdr_eval.diagnostics import PropensityDiagnostics
    
    # Create test diagnostics
    diagnostics = PropensityDiagnostics(
        overlap_ratio=0.8,
        balance_ratio=0.7,
        calibration_score=0.9,
        discrimination_score=0.85,
        log_loss_score=0.3,
        statistics={
            "min_pscore": 0.01,
            "max_pscore": 0.99,
            "mean_pscore": 0.5,
            "std_pscore": 0.2,
            "median_pscore": 0.48,
        },
        balance_stats={
            "action_0_count": 30,
            "action_0_mean_pscore": 0.4,
            "action_0_std_pscore": 0.15,
            "action_1_count": 35,
            "action_1_mean_pscore": 0.6,
            "action_1_std_pscore": 0.18,
        },
        calibration_curve=[(0.2, 0.22), (0.4, 0.38), (0.6, 0.62), (0.8, 0.78)],
        roc_curve=[(0.0, 0.0), (0.2, 0.4), (0.4, 0.7), (0.6, 0.85), (1.0, 1.0)],
        quantiles={"pscore_q25": 0.3, "pscore_q75": 0.7},
    )
    
    fig = plot_diagnostics_summary(diagnostics)
    assert fig is not None
    
    # Test with save path
    fig = plot_diagnostics_summary(diagnostics, save_path="/tmp/test_diag_plot.png")
    assert fig is not None


def test_create_dashboard():
    """Test dashboard creation."""
    n_samples = 100
    n_actions = 3
    
    # Create test data
    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)
    
    # Test basic dashboard
    fig = create_dashboard(propensities, actions)
    assert fig is not None
    
    # Test with results
    results = {
        "Model1": {"V_hat": 0.5, "SE_if": 0.1, "ESS": 50, "clip": 5},
        "Model2": {"V_hat": 0.6, "SE_if": 0.12, "ESS": 45, "clip": 10},
    }
    fig = create_dashboard(propensities, actions, results=results)
    assert fig is not None
    
    # Test with diagnostics
    from skdr_eval.diagnostics import PropensityDiagnostics
    
    diagnostics = PropensityDiagnostics(
        overlap_ratio=0.8,
        balance_ratio=0.7,
        calibration_score=0.9,
        discrimination_score=0.85,
        log_loss_score=0.3,
        statistics={
            "min_pscore": 0.01,
            "max_pscore": 0.99,
            "mean_pscore": 0.5,
            "std_pscore": 0.2,
            "median_pscore": 0.48,
        },
        balance_stats={
            "action_0_count": 30,
            "action_0_mean_pscore": 0.4,
            "action_0_std_pscore": 0.15,
        },
        calibration_curve=[(0.2, 0.22), (0.4, 0.38), (0.6, 0.62), (0.8, 0.78)],
        roc_curve=[(0.0, 0.0), (0.2, 0.4), (0.4, 0.7), (0.6, 0.85), (1.0, 1.0)],
        quantiles={"pscore_q25": 0.3, "pscore_q75": 0.7},
    )
    
    fig = create_dashboard(propensities, actions, diagnostics=diagnostics)
    assert fig is not None
    
    # Test with save path
    fig = create_dashboard(propensities, actions, save_path="/tmp/test_dashboard.png")
    assert fig is not None


def test_error_handling():
    """Test error handling in visualization functions."""
    # Test with mismatched lengths
    propensities = np.random.rand(10, 3)
    actions = np.array([0, 1, 2])  # Wrong length
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        plot_propensity_distribution(propensities, actions)
    
    # Test with insufficient data
    small_propensities = np.random.rand(5, 3)
    small_actions = np.array([0, 1, 0, 1, 0])
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        plot_propensity_distribution(small_propensities, small_actions)
    
    # Test with empty results
    with pytest.raises(Exception):  # Should raise DataValidationError
        plot_dr_results({})
    
    # Test with empty calibration curve
    with pytest.raises(Exception):  # Should raise DataValidationError
        plot_calibration_curve([])
    
    # Test with empty ROC curve
    with pytest.raises(Exception):  # Should raise DataValidationError
        plot_roc_curve([])


if __name__ == "__main__":
    pytest.main([__file__])