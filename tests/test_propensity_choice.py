"""Test propensity estimation and choice models."""

from unittest.mock import patch

import numpy as np
import pytest

from skdr_eval import make_pairwise_synth
from skdr_eval.core import estimate_propensity_pairwise
from skdr_eval.pairwise import PairwiseDesign

try:
    from skdr_eval.choice import (
        SCIPY_AVAILABLE,
        fit_conditional_logit_with_sampling,
        sample_negative_pairs,
    )
except ImportError:
    SCIPY_AVAILABLE = False
    fit_conditional_logit_with_sampling = None
    sample_negative_pairs = None


def test_propensity_multinomial_fallback():
    """Test that multinomial propensity works when scipy unavailable."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Force multinomial method
    propensities = estimate_propensity_pairwise(
        design,
        strategy="multinomial",
        method="multinomial",
        n_splits=2,
        random_state=42,
    )

    # Check shape
    n_decisions = len(logs_df)
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())
    assert propensities.shape == (n_decisions, max_ops)

    # Check that probabilities are non-negative
    assert (propensities >= 0).all()

    # Check that each row sums to approximately 1 (within eligible operators)
    for i in range(n_decisions):
        row_sum = np.sum(propensities[i, :])
        if row_sum > 0:  # Skip rows with no eligible operators
            assert abs(row_sum - 1.0) < 1e-6, (
                f"Row {i} probabilities don't sum to 1: {row_sum}"
            )


def test_propensity_auto_selection():
    """Test automatic propensity method selection."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=50, n_ops=8, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test auto selection (should work with small data)
    propensities = estimate_propensity_pairwise(
        design, strategy="auto", n_splits=2, random_state=42
    )

    assert propensities.shape[0] == len(logs_df)
    assert (propensities >= 0).all()


@pytest.mark.skipif(True, reason="SciPy conditional logit test - may not be available")
def test_conditional_logit_with_scipy():
    """Test conditional logit when scipy is available."""
    try:
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available")

        # Create simple test data
        np.random.seed(42)
        n_pairs = 1000
        n_features = 5
        n_choices = 100

        X = np.random.normal(0, 1, (n_pairs, n_features)).astype(np.float32)
        choice_ids = np.random.randint(0, n_choices, n_pairs)

        # Create realistic choice outcomes (one chosen per choice set)
        y = np.zeros(n_pairs)
        for choice_id in range(n_choices):
            mask = choice_ids == choice_id
            if np.sum(mask) > 0:
                # Randomly choose one option in each choice set
                choice_indices = np.where(mask)[0]
                chosen_idx = np.random.choice(choice_indices)
                y[chosen_idx] = 1

        # Fit conditional logit
        coef, intercept, temp = fit_conditional_logit_with_sampling(
            X, choice_ids, y, neg_per_pos=3, random_state=42
        )

        # Check that we get reasonable coefficients
        assert len(coef) == n_features
        assert isinstance(intercept, float)
        assert temp > 0

    except ImportError:
        pytest.skip("SciPy not available for conditional logit test")


def test_propensity_with_eligibility():
    """Test propensity estimation respects eligibility constraints."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=30, n_ops=6, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    propensities = estimate_propensity_pairwise(
        design, strategy="multinomial", n_splits=2, random_state=42
    )

    # Check that ineligible operators have zero probability
    for i, row in logs_df.iterrows():
        day = row["arrival_day"]
        elig_ops = row["elig_mask"]

        if isinstance(elig_ops, list) and day in design.ops_all_by_day:
            day_ops = design.ops_all_by_day[day]
            for j, op in enumerate(day_ops):
                if op not in elig_ops:
                    assert propensities[i, j] == 0, (
                        f"Ineligible operator {op} has non-zero probability"
                    )


def test_propensity_normalization():
    """Test that propensities are properly normalized."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    propensities = estimate_propensity_pairwise(
        design, strategy="multinomial", n_splits=2, random_state=42
    )

    # Check normalization for each decision
    for i in range(len(logs_df)):
        day = logs_df.iloc[i]["arrival_day"]
        if day in design.ops_all_by_day:
            n_ops_day = len(design.ops_all_by_day[day])
            row_probs = propensities[i, :n_ops_day]
            row_sum = np.sum(row_probs)

            if row_sum > 0:  # Skip if no eligible operators
                assert abs(row_sum - 1.0) < 1e-6, (
                    f"Row {i} probabilities sum to {row_sum}, not 1.0"
                )


def test_choice_model_sampling():
    """Test negative sampling in choice models."""
    try:
        # Create test data with multiple choice sets
        np.random.seed(42)
        n_pairs = 200
        n_features = 4

        X = np.random.normal(0, 1, (n_pairs, n_features)).astype(np.float32)
        choice_ids = np.repeat(np.arange(20), 10)  # 20 choice sets, 10 options each

        # Create outcomes (one positive per choice set)
        y = np.zeros(n_pairs)
        for i in range(20):
            start_idx = i * 10
            chosen_idx = start_idx + np.random.randint(0, 10)
            y[chosen_idx] = 1

        # Sample negatives
        X_sampled, choice_ids_sampled, y_sampled = sample_negative_pairs(
            X, choice_ids, y, neg_per_pos=3, random_state=42
        )

        # Check that we have fewer samples
        assert len(X_sampled) < len(X)
        assert len(X_sampled) == len(choice_ids_sampled) == len(y_sampled)

        # Check that all positives are kept
        assert np.sum(y_sampled) == np.sum(y)  # Same number of positives

        # Check that we have some negatives
        assert np.sum(y_sampled == 0) > 0

    except ImportError:
        pytest.skip("Choice module not available")


def test_propensity_error_handling():
    """Test error handling in propensity estimation."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=10, n_ops=3, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test with invalid strategy
    with pytest.raises((ValueError, KeyError)):
        estimate_propensity_pairwise(
            design, strategy="invalid_strategy", random_state=42
        )


def test_large_dataset_fallback():
    """Test that large datasets fall back to multinomial."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=100, n_ops=20, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Mock large dataset stats to force fallback
    with patch.object(design, "get_stats") as mock_stats:
        mock_stats.return_value = {
            "candidate_pairs": 100_000_000,  # Large number to force fallback
            "n_rows": 100,
            "n_days": 1,
        }

        propensities = estimate_propensity_pairwise(
            design, strategy="auto", n_splits=2, random_state=42
        )

        # Should still work (fallback to multinomial)
        assert propensities.shape[0] == len(logs_df)
        assert (propensities >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__])
