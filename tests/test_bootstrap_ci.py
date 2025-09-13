"""Tests for block bootstrap confidence intervals."""

import numpy as np
import pytest

import skdr_eval


class TestBlockBootstrapCI:
    """Test suite for block_bootstrap_ci function."""

    def test_basic_functionality(self):
        """Test basic bootstrap CI functionality."""
        # Generate test data
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        values_den = None
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=values_den,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        # Check that CI bounds are reasonable
        assert ci_lower < ci_upper
        assert ci_lower < np.mean(values_num)
        assert ci_upper > np.mean(values_num)

        # Check that CI is not too wide (should be reasonable for normal data)
        ci_width = ci_upper - ci_lower
        assert ci_width < 10.0  # Should be much smaller than data range

    def test_with_denominator(self):
        """Test bootstrap CI with denominator values."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        values_den = np.random.normal(1, 0.1, n)  # Denominator close to 1
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=values_den,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        # Check that CI bounds are reasonable
        assert ci_lower < ci_upper
        ratio_mean = np.mean(values_num) / np.mean(values_den)
        assert ci_lower < ratio_mean
        assert ci_upper > ratio_mean

    def test_different_alpha_levels(self):
        """Test bootstrap CI with different alpha levels."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        # Test 90% CI (alpha=0.1)
        ci_90_lower, ci_90_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            alpha=0.1,
            random_state=42,
        )

        # Test 95% CI (alpha=0.05)
        ci_95_lower, ci_95_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            alpha=0.05,
            random_state=42,
        )

        # 90% CI should be narrower than 95% CI
        ci_90_width = ci_90_upper - ci_90_lower
        ci_95_width = ci_95_upper - ci_95_lower
        assert ci_90_width < ci_95_width

    def test_different_block_lengths(self):
        """Test bootstrap CI with different block lengths."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        # Test with small block length
        ci_small_lower, ci_small_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            block_len=5,
            random_state=42,
        )

        # Test with large block length
        ci_large_lower, ci_large_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            block_len=20,
            random_state=42,
        )

        # Both should produce valid CIs
        assert ci_small_lower < ci_small_upper
        assert ci_large_lower < ci_large_upper

    def test_auto_block_length(self):
        """Test that auto block length works correctly."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        # Test with None block_len (should use sqrt(n))
        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            block_len=None,
            random_state=42,
        )

        assert ci_lower < ci_upper

    def test_small_dataset(self):
        """Test bootstrap CI with small dataset."""
        np.random.seed(42)
        n = 20  # Small dataset
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=50,  # Fewer bootstrap samples for speed
            random_state=42,
        )

        assert ci_lower < ci_upper

    def test_zero_denominator_handling(self):
        """Test handling of zero denominators."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        values_den = np.zeros(n)  # All zeros
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=values_den,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        # Should handle zero denominators gracefully
        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        # Run twice with same random_state
        ci1_lower, ci1_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        ci2_lower, ci2_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        # Results should be identical
        assert ci1_lower == ci2_lower
        assert ci1_upper == ci2_upper

    def test_different_random_states(self):
        """Test that different random states produce different results."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        # Run with different random states
        ci1_lower, ci1_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            random_state=42,
        )

        ci2_lower, ci2_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            random_state=123,
        )

        # Results should be different (very unlikely to be identical)
        assert not (ci1_lower == ci2_lower and ci1_upper == ci2_upper)

    def test_edge_case_single_value(self):
        """Test edge case with single value."""
        values_num = np.array([10.0])
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=10,
            random_state=42,
        )

        # With single value, CI should be the value itself
        assert ci_lower == ci_upper == 10.0

    def test_time_series_correlation(self):
        """Test that block bootstrap preserves time series correlation."""
        np.random.seed(42)
        n = 200
        # Create correlated time series
        t = np.arange(n)
        values_num = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.normal(0, 1, n)
        base_mean = np.array([0.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=100,
            block_len=10,  # Use block length appropriate for correlation
            random_state=42,
        )

        assert ci_lower < ci_upper
        # CI should contain the mean
        assert ci_lower < np.mean(values_num)
        assert ci_upper > np.mean(values_num)

    def test_large_bootstrap_samples(self):
        """Test with large number of bootstrap samples."""
        np.random.seed(42)
        n = 100
        values_num = np.random.normal(10, 2, n)
        base_mean = np.array([10.0])

        ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
            values_num=values_num,
            values_den=None,
            base_mean=base_mean,
            n_boot=1000,  # Large number of bootstrap samples
            random_state=42,
        )

        assert ci_lower < ci_upper

    def test_parameter_validation(self):
        """Test parameter validation."""
        values_num = np.array([1, 2, 3, 4, 5])
        base_mean = np.array([3.0])

        # Test with invalid alpha
        with pytest.raises(ValueError):
            skdr_eval.block_bootstrap_ci(
                values_num=values_num,
                values_den=None,
                base_mean=base_mean,
                alpha=1.5,  # Invalid alpha > 1
                random_state=42,
            )

        with pytest.raises(ValueError):
            skdr_eval.block_bootstrap_ci(
                values_num=values_num,
                values_den=None,
                base_mean=base_mean,
                alpha=-0.1,  # Invalid alpha < 0
                random_state=42,
            )

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        values_num = np.array([])
        base_mean = np.array([0.0])

        with pytest.raises(ValueError):
            skdr_eval.block_bootstrap_ci(
                values_num=values_num,
                values_den=None,
                base_mean=base_mean,
                random_state=42,
            )
