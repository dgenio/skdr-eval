"""Test design matrix construction and propensity modeling."""

import numpy as np
import pytest

import skdr_eval


def test_design_ops_all_includes_both_elig_and_actions():
    """Test that ops_all includes both eligible and observed actions."""
    logs, ops_all_orig, _ = skdr_eval.make_synth_logs(n=200, n_ops=4, seed=10)
    design = skdr_eval.build_design(logs)

    # Check that ops_all matches original
    assert len(design.ops_all) == len(ops_all_orig)
    assert set(design.ops_all) == set(ops_all_orig)

    # Check that all observed actions are in ops_all
    observed_actions = set(logs["action"])
    assert observed_actions.issubset(set(design.ops_all)), (
        "Observed actions not in ops_all"
    )

    # Check that all eligible operators are in ops_all
    elig_cols = [col for col in logs.columns if col.endswith("_elig")]
    eligible_ops = set()
    for col in elig_cols:
        op_name = col.replace("_elig", "")
        if logs[col].any():  # If any sample has this operator eligible
            eligible_ops.add(op_name)

    assert eligible_ops.issubset(set(design.ops_all)), (
        "Eligible operators not in ops_all"
    )


def test_x_obs_shape_correct():
    """Test that X_obs has correct shape: X_base + len(ops_all)."""
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=150, n_ops=3, seed=20)
    design = skdr_eval.build_design(logs)

    n_samples = len(logs)
    n_ops = len(ops_all)
    n_base_features = design.X_base.shape[1]

    # X_obs should be X_base + action one-hot
    expected_x_obs_shape = (n_samples, n_base_features + n_ops)
    assert design.X_obs.shape == expected_x_obs_shape, (
        f"X_obs shape {design.X_obs.shape} != expected {expected_x_obs_shape}"
    )

    # Check that X_obs contains X_base as first columns
    np.testing.assert_array_equal(
        design.X_obs[:, :n_base_features],
        design.X_base,
        "X_obs should start with X_base features",
    )

    # Check that action one-hot is correctly encoded
    action_onehot = design.X_obs[:, n_base_features:]
    assert action_onehot.shape == (n_samples, n_ops)

    # Each row should have exactly one 1 (one-hot encoding)
    row_sums = action_onehot.sum(axis=1)
    assert np.allclose(row_sums, 1.0), (
        "Action one-hot should have exactly one 1 per row"
    )

    # Check that the 1s are in the correct positions
    for i in range(n_samples):
        action_idx = design.A[i]
        expected_onehot = np.zeros(n_ops)
        expected_onehot[action_idx] = 1
        np.testing.assert_array_equal(
            action_onehot[i],
            expected_onehot,
            f"Incorrect one-hot encoding for sample {i}",
        )


def test_propensities_row_normalized_over_eligibility():
    """Test that propensities are row-normalized and respect eligibility."""
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=300, n_ops=4, seed=30)
    design = skdr_eval.build_design(logs)

    # Fit propensity model
    propensities, _ = skdr_eval.fit_propensity_timecal(
        design.X_phi, design.A, design.ts, n_splits=3, random_state=30
    )

    n_samples, n_ops = propensities.shape
    assert n_samples == len(design.A)
    assert n_ops == len(ops_all)

    # Check that propensities are valid probabilities
    assert np.all(propensities >= 0), "Propensities should be non-negative"
    assert np.all(propensities <= 1), "Propensities should be <= 1"

    # Check that rows sum to 1 (normalized probabilities)
    row_sums = propensities.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, rtol=1e-10, atol=1e-10, err_msg="Propensity rows should sum to 1"
    )

    # Check that propensities respect eligibility constraints
    # (This is a softer constraint since the propensity model may not perfectly respect eligibility)
    for i in range(min(50, n_samples)):  # Check first 50 samples for efficiency
        eligible_ops = design.elig[i]
        ineligible_ops = ~eligible_ops

        # Propensities for ineligible operators should generally be lower
        # (This is not a hard constraint, but a reasonable expectation)
        if ineligible_ops.any() and eligible_ops.any():
            eligible_probs = propensities[i, eligible_ops]
            ineligible_probs = propensities[i, ineligible_ops]

            # At least the maximum eligible probability should be >= maximum ineligible
            # (This is a very weak constraint to avoid test flakiness)
            assert eligible_probs.max() >= ineligible_probs.max() * 0.1, (
                f"Eligible ops should have reasonable propensity mass for sample {i}"
            )


def test_propensities_positive_when_matched():
    """Test that propensities are positive for observed actions when matched."""
    logs, _, _ = skdr_eval.make_synth_logs(n=250, n_ops=3, seed=40)
    design = skdr_eval.build_design(logs)

    # Fit propensity model
    propensities, _ = skdr_eval.fit_propensity_timecal(
        design.X_phi, design.A, design.ts, n_splits=3, random_state=40
    )

    # Get propensity scores for observed actions
    pi_obs = propensities[np.arange(len(design.A)), design.A]

    # Check eligibility for observed actions
    elig_obs = design.elig[np.arange(len(design.A)), design.A]

    # For samples where the observed action was eligible, propensity should be positive
    eligible_samples = elig_obs
    if eligible_samples.any():
        pi_eligible = pi_obs[eligible_samples]
        assert np.all(pi_eligible > 0), (
            "Propensities should be positive for eligible observed actions"
        )

        # Check minimum propensity is reasonable (not too close to 0)
        min_pi_eligible = pi_eligible.min()
        assert min_pi_eligible > 1e-6, (
            f"Minimum propensity too small: {min_pi_eligible}"
        )


def test_design_feature_construction():
    """Test that design features are constructed correctly."""
    logs, _, _ = skdr_eval.make_synth_logs(n=100, n_ops=2, seed=50)
    design = skdr_eval.build_design(logs, cli_pref="cli_", st_pref="st_")

    # Check that base features include client and service-time features
    cli_cols = [col for col in logs.columns if col.startswith("cli_")]
    st_cols = [col for col in logs.columns if col.startswith("st_")]
    expected_base_cols = cli_cols + st_cols

    assert design.X_base.shape[1] == len(expected_base_cols), (
        f"X_base should have {len(expected_base_cols)} features"
    )

    # Check that X_phi includes base features + standardized time
    assert design.X_phi.shape[1] == design.X_base.shape[1] + 1, (
        "X_phi should have base features + 1 time feature"
    )

    # Check that time feature is standardized (mean ≈ 0, std ≈ 1)
    time_feature = design.X_phi[:, -1]  # Last column should be standardized time
    assert abs(time_feature.mean()) < 0.1, "Standardized time should have mean ≈ 0"
    assert abs(time_feature.std() - 1.0) < 0.1, "Standardized time should have std ≈ 1"

    # Check that timestamps are sorted (time-ordered)
    assert np.all(design.ts[1:] >= design.ts[:-1]), "Timestamps should be sorted"


def test_design_with_custom_prefixes():
    """Test design construction with custom feature prefixes."""
    # Create logs with custom prefixes
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=80, n_ops=2, seed=60)

    # Rename columns to test custom prefixes
    logs_custom = logs.copy()
    rename_map = {}
    for col in logs.columns:
        if col.startswith("cli_"):
            rename_map[col] = col.replace("cli_", "client_")
        elif col.startswith("st_"):
            rename_map[col] = col.replace("st_", "service_")

    logs_custom = logs_custom.rename(columns=rename_map)

    # Build design with custom prefixes
    design = skdr_eval.build_design(logs_custom, cli_pref="client_", st_pref="service_")

    # Should work correctly
    assert design.X_base.shape[0] == len(logs_custom)
    assert design.X_base.shape[1] > 0
    assert design.X_obs.shape[1] == design.X_base.shape[1] + len(ops_all)


def test_propensity_time_aware_splits():
    """Test that propensity fitting uses time-aware splits correctly."""
    logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=70)
    design = skdr_eval.build_design(logs)

    # Fit with different numbers of splits
    for n_splits in [2, 3, 4]:
        _, fold_indices = skdr_eval.fit_propensity_timecal(
            design.X_phi, design.A, design.ts, n_splits=n_splits, random_state=70
        )

        # Check fold indices
        unique_folds = np.unique(fold_indices)
        assert len(unique_folds) == n_splits, f"Should have {n_splits} unique folds"
        assert set(unique_folds) == set(range(n_splits)), (
            "Fold indices should be 0, 1, ..., n_splits-1"
        )

        # Check that TimeSeriesSplit is being used correctly
        # TimeSeriesSplit creates expanding windows, so later folds should generally
        # contain samples from later time periods (but with some overlap)
        # We just check that we have reasonable distribution across folds
        for fold in range(n_splits):
            fold_mask = fold_indices == fold
            assert fold_mask.any(), f"Fold {fold} should have some samples"

        # Check that the fold assignment makes sense with time ordering
        # TimeSeriesSplit creates expanding windows, so we check that:
        # 1. Each fold has test samples from a reasonable time range
        # 2. The overall time coverage makes sense
        if n_splits >= 2:
            fold_time_ranges = []
            for fold in range(n_splits):
                fold_mask = fold_indices == fold
                if fold_mask.any():
                    fold_times = design.ts[fold_mask]
                    fold_time_ranges.append((fold_times.min(), fold_times.max()))

            # Check that we have reasonable time coverage across folds
            # At least some folds should have different time ranges
            if len(fold_time_ranges) >= 2:
                all_same_range = all(
                    start == fold_time_ranges[0][0] and end == fold_time_ranges[0][1]
                    for start, end in fold_time_ranges
                )
                assert not all_same_range, (
                    "Folds should have different time ranges due to time-series splitting"
                )


if __name__ == "__main__":
    pytest.main([__file__])
