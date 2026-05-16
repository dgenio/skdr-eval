"""Test propensity score diagnostics functionality."""

import numpy as np
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
    compute_propensity_brier,
    compute_propensity_ece,
    compute_propensity_log_loss,
    compute_propensity_reliability_curve,
    compute_propensity_statistics,
    generate_propensity_report,
    psis_pareto_k,
)
from skdr_eval.exceptions import ConfigurationError, DataValidationError


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
    # Create a base pattern and repeat it to get exactly n_samples rows
    base_pattern = np.array(
        [
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95],
        ]
    )
    extreme_propensities = np.tile(base_pattern, (n_samples // 3 + 1, 1))[:n_samples]
    # Derive actions from propensity structure: each sample takes its highest-prob action
    extreme_actions = np.argmax(extreme_propensities, axis=1)

    poor_overlap = check_propensity_overlap(extreme_propensities, extreme_actions)
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
    # Create a base pattern and repeat it to get exactly n_samples rows
    base_pattern = np.array(
        [
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
        ]
    )
    extreme_propensities = np.tile(base_pattern, (n_samples // 3 + 1, 1))[:n_samples]
    # Derive actions from propensity structure: each sample takes its highest-prob action
    extreme_actions = np.argmax(extreme_propensities, axis=1)

    poor_balance = check_propensity_balance(extreme_propensities, extreme_actions)
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
    # calibration_curve is List[Tuple[mean_predicted, actual_fraction]], one tuple per bin
    assert len(calibration_curve) == 5
    assert all(len(pt) == 2 for pt in calibration_curve)


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
    # roc_curve is a list of (fpr, tpr) tuples
    assert len(roc_curve) > 0
    assert all(len(point) == 2 for point in roc_curve)


def test_propensity_statistics():
    """Test propensity score statistics computation."""
    np.random.seed(42)
    n_samples = 100
    n_actions = 3

    propensities = np.random.dirichlet([1, 1, 1], size=n_samples)
    actions = np.random.choice(n_actions, size=n_samples)

    stats = compute_propensity_statistics(propensities, actions)

    # Check required statistics
    required_stats = [
        "min_pscore",
        "max_pscore",
        "mean_pscore",
        "std_pscore",
        "median_pscore",
    ]
    for stat in required_stats:
        assert stat in stats
        assert isinstance(stats[stat], float)

    # Check quantiles
    quantile_stats = [k for k in stats if k.startswith("pscore_q")]
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
    assert hasattr(diagnostics, "statistics")
    assert hasattr(diagnostics, "balance_stats")

    # Statistics are stored in the statistics dict
    assert "min_pscore" in diagnostics.statistics
    assert "max_pscore" in diagnostics.statistics
    assert "mean_pscore" in diagnostics.statistics
    assert "std_pscore" in diagnostics.statistics

    # Check value ranges
    assert 0 <= diagnostics.overlap_ratio <= 1
    assert 0 <= diagnostics.balance_ratio <= 1
    assert 0 <= diagnostics.calibration_score <= 1
    assert 0 <= diagnostics.discrimination_score <= 1
    assert diagnostics.log_loss_score >= 0
    assert (
        0
        <= diagnostics.statistics["min_pscore"]
        <= diagnostics.statistics["max_pscore"]
        <= 1
    )
    assert diagnostics.statistics["std_pscore"] >= 0


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


# --------------------------------------------------------------------------- #
# Trust-additions (#80 PSIS Pareto-k, #84 ECE/Brier) — unit tests             #
# --------------------------------------------------------------------------- #


def test_psis_pareto_k_returns_nan_when_sample_too_small():
    """Below the per-tail sample floor, the diagnostic is statistically meaningless."""
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # n=5 < _MIN_SAMPLES_PARETO_K
    assert not np.isfinite(psis_pareto_k(weights))


def test_psis_pareto_k_rejects_non_finite_weights():
    """Inf / NaN weights are an upstream bug — we surface it rather than silently filter."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(1.0, 2.0, size=200)
    weights[10] = np.inf
    with pytest.raises(DataValidationError, match="non-finite"):
        psis_pareto_k(weights)


def test_psis_pareto_k_rejects_negative_weights():
    """Importance weights are 1/π and must be non-negative."""
    weights = np.array([1.0, 2.0, 3.0, -0.5] * 100)
    with pytest.raises(DataValidationError, match="non-negative"):
        psis_pareto_k(weights)


def test_psis_pareto_k_light_tail_below_threshold():
    """Bounded weights (uniform) have a very light tail; PSIS k should be < 0.5."""
    rng = np.random.default_rng(seed=12)
    weights = rng.uniform(1.0, 2.0, size=2000)
    k = psis_pareto_k(weights)
    assert np.isfinite(k)
    assert k < 0.5, f"light-tail Pareto-k {k:.3f} should sit well below 0.5"


def test_psis_pareto_k_finite_and_one_dimensional_acceptance():
    """Acceptable input: 1-D, finite, non-negative weights, sufficient sample."""
    rng = np.random.default_rng(seed=3)
    weights = rng.uniform(1.0, 5.0, size=500)
    k = psis_pareto_k(weights)
    assert np.isfinite(k)


def test_compute_propensity_ece_calibrated_low():
    """ECE on perfectly-calibrated synthetic propensities is at the noise floor."""
    rng = np.random.default_rng(seed=2024)
    n = 3000
    n_actions = 4
    propensities = rng.dirichlet([1.0] * n_actions, size=n)
    # Sample actions from the propensities themselves → perfectly calibrated.
    cdf = np.cumsum(propensities, axis=1)
    u = rng.uniform(size=(n, 1))
    actions = (u <= cdf).argmax(axis=1)

    ece = compute_propensity_ece(propensities, actions, n_bins=15)
    assert 0.0 <= ece <= 0.05, f"calibrated ECE should be small, got {ece:.4f}"


def test_compute_propensity_ece_rejects_invalid_n_bins():
    """``n_bins < 2`` is a misuse and must raise rather than silently return junk."""
    rng = np.random.default_rng(seed=0)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=50)
    a = rng.integers(0, 3, size=50)
    with pytest.raises(ConfigurationError, match="n_bins must be"):
        compute_propensity_ece(p, a, n_bins=1)


def test_compute_propensity_ece_returns_nan_for_tiny_samples():
    """Below ``_MIN_SAMPLES_RELIABILITY`` the diagnostic is meaningless."""
    rng = np.random.default_rng(seed=0)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=10)
    a = rng.integers(0, 3, size=10)
    assert not np.isfinite(compute_propensity_ece(p, a))


def test_compute_propensity_brier_hand_computed_value():
    """Brier score on a hand-computed fixture for byte-exact correctness.

    Two samples, two actions:
      sample 0: π = [0.8, 0.2], observed A=0 → (0.8-1)² + (0.2-0)² = 0.04 + 0.04 = 0.08
      sample 1: π = [0.3, 0.7], observed A=1 → (0.3-0)² + (0.7-1)² = 0.09 + 0.09 = 0.18
    Mean over 2 samples: (0.08 + 0.18) / 2 = 0.13.
    """
    propensities = np.array([[0.8, 0.2], [0.3, 0.7]])
    actions = np.array([0, 1])
    # Hand-computed Brier — Note: function raises on tiny samples too,
    # but min for Brier is _MIN_SAMPLES_SMALL=5; replicate 3 times for the floor.
    propensities = np.tile(propensities, (3, 1))
    actions = np.tile(actions, 3)
    brier = compute_propensity_brier(propensities, actions)
    assert brier == pytest.approx(0.13, abs=1e-12)


def test_compute_propensity_reliability_curve_shape():
    """Reliability curve returns exactly ``n_bins`` rows of (mean_pred, frac, count)."""
    rng = np.random.default_rng(seed=4)
    n_bins = 10
    p = rng.dirichlet([1.0, 1.0], size=200)
    a = rng.integers(0, 2, size=200)
    curve = compute_propensity_reliability_curve(p, a, n_bins=n_bins)
    assert len(curve) == n_bins
    for mean_pred, _frac, count in curve:
        assert 0.0 <= mean_pred <= 1.0 or np.isnan(mean_pred)
        assert isinstance(count, int)
        assert count >= 0


def test_comprehensive_diagnostics_populates_new_fields():
    """``comprehensive_propensity_diagnostics`` must populate ECE/Brier/reliability."""
    rng = np.random.default_rng(seed=5)
    n = 500
    n_actions = 3
    p = rng.dirichlet([1.0] * n_actions, size=n)
    cdf = np.cumsum(p, axis=1)
    u = rng.uniform(size=(n, 1))
    a = (u <= cdf).argmax(axis=1)

    diag = comprehensive_propensity_diagnostics(p, a)
    assert np.isfinite(diag.ece), "ECE should be finite for a reasonable sample"
    assert np.isfinite(diag.brier_score), "Brier should be finite"
    assert len(diag.reliability_curve) == diag.ece_n_bins
    assert diag.ece_n_bins == 15


def test_propensity_diagnostics_defaults_are_backward_compatible():
    """Direct construction without the new fields must still succeed (existing callers)."""
    diag = PropensityDiagnostics(
        overlap_ratio=0.8,
        balance_ratio=0.7,
        calibration_score=0.9,
        discrimination_score=0.85,
        log_loss_score=0.5,
        statistics={"min_pscore": 0.05},
        balance_stats={"action_0_count": 100.0},
        calibration_curve=[(0.5, 0.5)],
        roc_curve=[(0.0, 0.0), (1.0, 1.0)],
    )
    assert np.isnan(diag.ece)
    assert np.isnan(diag.brier_score)
    assert diag.reliability_curve == []
    assert diag.ece_n_bins == 15  # default


if __name__ == "__main__":
    pytest.main([__file__])
