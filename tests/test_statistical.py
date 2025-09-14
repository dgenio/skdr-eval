"""Tests for statistical module."""

import numpy as np
import pytest

from skdr_eval.statistical import (
    bootstrap_confidence_interval,
    chi_square_test,
    kolmogorov_smirnov_test,
    mann_whitney_u_test,
    multiple_comparison_correction,
    permutation_test,
    power_analysis,
    sample_size_calculation,
    t_test,
)


def test_t_test():
    """Test t-test functionality."""
    # Create test samples
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 30)
    sample2 = np.random.normal(0.5, 1, 30)
    
    # Test two-sided t-test
    result = t_test(sample1, sample2, alternative="two-sided")
    assert result.test_name == "t-test"
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.degrees_of_freedom is not None
    assert result.confidence_interval is not None
    assert result.effect_size is not None
    assert isinstance(result.interpretation, str)
    
    # Test one-sided t-test
    result_less = t_test(sample1, sample2, alternative="less")
    assert result_less.p_value != result.p_value
    
    # Test with equal_var=False
    result_unequal = t_test(sample1, sample2, equal_var=False)
    assert result_unequal.test_name == "t-test"


def test_mann_whitney_u_test():
    """Test Mann-Whitney U test functionality."""
    # Create test samples
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 30)
    sample2 = np.random.normal(0.5, 1, 30)
    
    result = mann_whitney_u_test(sample1, sample2)
    assert result.test_name == "Mann-Whitney U test"
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.effect_size is not None
    assert isinstance(result.interpretation, str)


def test_chi_square_test():
    """Test chi-square test functionality."""
    # Test goodness of fit
    observed = np.array([10, 15, 20, 25, 30])
    result = chi_square_test(observed)
    assert result.test_name == "Chi-square test"
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.degrees_of_freedom is not None
    assert result.effect_size is not None
    
    # Test with expected frequencies
    expected = np.array([12, 18, 18, 18, 12])
    result_with_expected = chi_square_test(observed, expected)
    assert result_with_expected.test_name == "Chi-square test"


def test_kolmogorov_smirnov_test():
    """Test Kolmogorov-Smirnov test functionality."""
    # Create test sample
    np.random.seed(42)
    sample = np.random.normal(0, 1, 50)
    
    # Test against normal distribution
    result = kolmogorov_smirnov_test(sample, distribution="norm")
    assert result.test_name == "Kolmogorov-Smirnov test"
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.effect_size is not None
    
    # Test against uniform distribution
    result_uniform = kolmogorov_smirnov_test(sample, distribution="uniform")
    assert result_uniform.test_name == "Kolmogorov-Smirnov test"


def test_bootstrap_confidence_interval():
    """Test bootstrap confidence interval functionality."""
    # Create test data
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    
    # Test with mean
    ci_lower, ci_upper = bootstrap_confidence_interval(
        data, 
        statistic_func=np.mean,
        n_bootstrap=1000
    )
    assert ci_lower < ci_upper
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # Test with median
    ci_lower_med, ci_upper_med = bootstrap_confidence_interval(
        data,
        statistic_func=np.median,
        n_bootstrap=1000
    )
    assert ci_lower_med < ci_upper_med


def test_permutation_test():
    """Test permutation test functionality."""
    # Create test samples
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 20)
    sample2 = np.random.normal(0.5, 1, 20)
    
    # Test with default statistic (mean difference)
    result = permutation_test(sample1, sample2, n_permutations=1000)
    assert result.test_name == "Permutation test"
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.effect_size is not None
    
    # Test with custom statistic
    def custom_stat(x, y):
        return np.median(x) - np.median(y)
    
    result_custom = permutation_test(sample1, sample2, statistic_func=custom_stat, n_permutations=1000)
    assert result_custom.test_name == "Permutation test"


def test_multiple_comparison_correction():
    """Test multiple comparison correction functionality."""
    # Create test p-values
    p_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    # Test Bonferroni correction
    corrected_bonf = multiple_comparison_correction(p_values, method="bonferroni")
    assert len(corrected_bonf) == len(p_values)
    assert all(0 <= p <= 1 for p in corrected_bonf)
    assert all(corrected_bonf[i] >= p_values[i] for i in range(len(p_values)))
    
    # Test Holm correction
    corrected_holm = multiple_comparison_correction(p_values, method="holm")
    assert len(corrected_holm) == len(p_values)
    assert all(0 <= p <= 1 for p in corrected_holm)
    
    # Test FDR correction
    corrected_fdr = multiple_comparison_correction(p_values, method="fdr_bh")
    assert len(corrected_fdr) == len(p_values)
    assert all(0 <= p <= 1 for p in corrected_fdr)


def test_power_analysis():
    """Test power analysis functionality."""
    # Test two-sample t-test power
    power = power_analysis(effect_size=0.5, n=30, alpha=0.05, test_type="two_sample_t")
    assert 0 <= power <= 1
    assert isinstance(power, float)
    
    # Test one-sample t-test power
    power_one = power_analysis(effect_size=0.5, n=30, alpha=0.05, test_type="one_sample_t")
    assert 0 <= power_one <= 1
    
    # Test paired t-test power
    power_paired = power_analysis(effect_size=0.5, n=30, alpha=0.05, test_type="paired_t")
    assert 0 <= power_paired <= 1


def test_sample_size_calculation():
    """Test sample size calculation functionality."""
    # Test two-sample t-test sample size
    n = sample_size_calculation(effect_size=0.5, power=0.8, alpha=0.05, test_type="two_sample_t")
    assert n >= 2
    assert isinstance(n, int)
    
    # Test one-sample t-test sample size
    n_one = sample_size_calculation(effect_size=0.5, power=0.8, alpha=0.05, test_type="one_sample_t")
    assert n_one >= 2
    
    # Test paired t-test sample size
    n_paired = sample_size_calculation(effect_size=0.5, power=0.8, alpha=0.05, test_type="paired_t")
    assert n_paired >= 2


def test_error_handling():
    """Test error handling in statistical functions."""
    # Test with insufficient data
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        t_test(np.array([1]), np.array([2, 3]))
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        mann_whitney_u_test(np.array([1, 2]), np.array([3]))
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        chi_square_test(np.array([10]))
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        kolmogorov_smirnov_test(np.array([1, 2, 3, 4]))
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        bootstrap_confidence_interval(np.array([1]), np.mean)
    
    with pytest.raises(Exception):  # Should raise InsufficientDataError
        permutation_test(np.array([1]), np.array([2, 3]))
    
    # Test with invalid data
    with pytest.raises(Exception):  # Should raise DataValidationError
        t_test(np.array([1, 2, np.nan]), np.array([3, 4, 5]))
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        chi_square_test(np.array([-1, 2, 3]))
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        multiple_comparison_correction([1.5, 0.1])  # Invalid p-value
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        power_analysis(effect_size=-0.5, n=30)
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        sample_size_calculation(effect_size=-0.5, power=0.8)


def test_edge_cases():
    """Test edge cases in statistical functions."""
    # Test with identical samples
    sample1 = np.array([1, 2, 3, 4, 5])
    sample2 = np.array([1, 2, 3, 4, 5])
    
    result = t_test(sample1, sample2)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    
    # Test with very small effect size
    np.random.seed(42)
    sample1 = np.random.normal(0, 1, 100)
    sample2 = np.random.normal(0.01, 1, 100)
    
    result = t_test(sample1, sample2)
    assert isinstance(result.statistic, float)
    assert 0 <= result.p_value <= 1
    
    # Test with extreme values
    sample1 = np.array([1e10, 1e10, 1e10])
    sample2 = np.array([1e10, 1e10, 1e10])
    
    result = t_test(sample1, sample2)
    assert result.statistic == 0.0


if __name__ == "__main__":
    pytest.main([__file__])