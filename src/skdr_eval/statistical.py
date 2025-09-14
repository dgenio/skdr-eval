"""Statistical testing utilities for skdr-eval library."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, norm, t

from .exceptions import DataValidationError, InsufficientDataError

logger = logging.getLogger("skdr_eval")


@dataclass
class StatisticalTest:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""


def t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    equal_var: bool = True,
) -> StatisticalTest:
    """Perform t-test for comparing two samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample.
    sample2 : np.ndarray
        Second sample.
    alternative : str, default="two-sided"
        Alternative hypothesis ("two-sided", "less", "greater").
    alpha : float, default=0.05
        Significance level.
    equal_var : bool, default=True
        Whether to assume equal variances.

    Returns
    -------
    StatisticalTest
        Test results.
    """
    if len(sample1) < 2 or len(sample2) < 2:
        raise InsufficientDataError("Need at least 2 observations in each sample")

    if not np.all(np.isfinite(sample1)) or not np.all(np.isfinite(sample2)):
        raise DataValidationError("Samples must contain only finite values")

    # Perform t-test
    if equal_var:
        statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=True)
    else:
        statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)

    # Adjust p-value for one-sided tests
    if alternative == "less":
        p_value = p_value / 2 if statistic < 0 else 1 - p_value / 2
    elif alternative == "greater":
        p_value = p_value / 2 if statistic > 0 else 1 - p_value / 2

    # Calculate degrees of freedom
    if equal_var:
        df = len(sample1) + len(sample2) - 2
    else:
        # Welch's t-test degrees of freedom
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

    # Calculate critical value
    if alternative == "two-sided":
        critical_value = t.ppf(1 - alpha / 2, df)
    else:
        critical_value = t.ppf(1 - alpha, df)

    # Calculate confidence interval
    mean_diff = np.mean(sample1) - np.mean(sample2)
    se = np.sqrt(np.var(sample1, ddof=1) / len(sample1) + np.var(sample2, ddof=1) / len(sample2))
    if equal_var:
        se = se * np.sqrt(1 / len(sample1) + 1 / len(sample2))
    
    if alternative == "two-sided":
        ci_lower = mean_diff - critical_value * se
        ci_upper = mean_diff + critical_value * se
    elif alternative == "less":
        ci_lower = -np.inf
        ci_upper = mean_diff + critical_value * se
    else:  # greater
        ci_lower = mean_diff - critical_value * se
        ci_upper = np.inf

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) + 
                          (len(sample2) - 1) * np.var(sample2, ddof=1)) / 
                         (len(sample1) + len(sample2) - 2))
    effect_size = (np.mean(sample1) - np.mean(sample2)) / pooled_std

    # Interpretation
    if p_value < alpha:
        interpretation = f"Reject H0 at α={alpha} (p={p_value:.4f})"
    else:
        interpretation = f"Fail to reject H0 at α={alpha} (p={p_value:.4f})"

    return StatisticalTest(
        test_name="t-test",
        statistic=statistic,
        p_value=p_value,
        critical_value=critical_value,
        degrees_of_freedom=int(df),
        confidence_interval=(ci_lower, ci_upper),
        effect_size=effect_size,
        interpretation=interpretation,
    )


def mann_whitney_u_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTest:
    """Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Parameters
    ----------
    sample1 : np.ndarray
        First sample.
    sample2 : np.ndarray
        Second sample.
    alternative : str, default="two-sided"
        Alternative hypothesis ("two-sided", "less", "greater").
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    StatisticalTest
        Test results.
    """
    if len(sample1) < 3 or len(sample2) < 3:
        raise InsufficientDataError("Need at least 3 observations in each sample for Mann-Whitney U test")

    if not np.all(np.isfinite(sample1)) or not np.all(np.isfinite(sample2)):
        raise DataValidationError("Samples must contain only finite values")

    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)

    # Calculate critical value (approximate)
    n1, n2 = len(sample1), len(sample2)
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    
    if alternative == "two-sided":
        critical_value = norm.ppf(1 - alpha / 2) * std_u + mean_u
    else:
        critical_value = norm.ppf(1 - alpha) * std_u + mean_u

    # Calculate effect size (rank-biserial correlation)
    u1 = statistic
    u2 = n1 * n2 - u1
    effect_size = 2 * (u1 / (n1 * n2)) - 1

    # Interpretation
    if p_value < alpha:
        interpretation = f"Reject H0 at α={alpha} (p={p_value:.4f})"
    else:
        interpretation = f"Fail to reject H0 at α={alpha} (p={p_value:.4f})"

    return StatisticalTest(
        test_name="Mann-Whitney U test",
        statistic=statistic,
        p_value=p_value,
        critical_value=critical_value,
        degrees_of_freedom=None,
        confidence_interval=None,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> StatisticalTest:
    """Perform chi-square test for goodness of fit or independence.

    Parameters
    ----------
    observed : np.ndarray
        Observed frequencies.
    expected : np.ndarray, optional
        Expected frequencies. If None, assumes uniform distribution.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    StatisticalTest
        Test results.
    """
    if len(observed) < 2:
        raise InsufficientDataError("Need at least 2 categories for chi-square test")

    if np.any(observed < 0):
        raise DataValidationError("Observed frequencies must be non-negative")

    if expected is None:
        # Goodness of fit test with uniform distribution
        expected = np.full_like(observed, np.sum(observed) / len(observed), dtype=float)
    else:
        if len(expected) != len(observed):
            raise DataValidationError("Expected and observed arrays must have the same length")
        if np.any(expected < 0):
            raise DataValidationError("Expected frequencies must be non-negative")

    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    
    # Calculate p-value
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    # Calculate critical value
    critical_value = chi2.ppf(1 - alpha, df)

    # Calculate effect size (Cramér's V)
    n = np.sum(observed)
    effect_size = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))

    # Interpretation
    if p_value < alpha:
        interpretation = f"Reject H0 at α={alpha} (p={p_value:.4f})"
    else:
        interpretation = f"Fail to reject H0 at α={alpha} (p={p_value:.4f})"

    return StatisticalTest(
        test_name="Chi-square test",
        statistic=chi2_stat,
        p_value=p_value,
        critical_value=critical_value,
        degrees_of_freedom=df,
        confidence_interval=None,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def kolmogorov_smirnov_test(
    sample: np.ndarray,
    distribution: str = "norm",
    alpha: float = 0.05,
    **kwargs,
) -> StatisticalTest:
    """Perform Kolmogorov-Smirnov test for goodness of fit.

    Parameters
    ----------
    sample : np.ndarray
        Sample data.
    distribution : str, default="norm"
        Distribution to test against ("norm", "uniform", "expon").
    alpha : float, default=0.05
        Significance level.
    **kwargs
        Additional parameters for the distribution.

    Returns
    -------
    StatisticalTest
        Test results.
    """
    if len(sample) < 5:
        raise InsufficientDataError("Need at least 5 observations for KS test")

    if not np.all(np.isfinite(sample)):
        raise DataValidationError("Sample must contain only finite values")

    # Get distribution function
    if distribution == "norm":
        loc = kwargs.get("loc", np.mean(sample))
        scale = kwargs.get("scale", np.std(sample))
        dist_func = lambda x: stats.norm.cdf(x, loc=loc, scale=scale)
    elif distribution == "uniform":
        loc = kwargs.get("loc", np.min(sample))
        scale = kwargs.get("scale", np.max(sample) - np.min(sample))
        dist_func = lambda x: stats.uniform.cdf(x, loc=loc, scale=scale)
    elif distribution == "expon":
        loc = kwargs.get("loc", 0)
        scale = kwargs.get("scale", np.mean(sample))
        dist_func = lambda x: stats.expon.cdf(x, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Perform KS test
    ks_stat, p_value = stats.kstest(sample, dist_func)

    # Calculate critical value
    n = len(sample)
    critical_value = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)

    # Calculate effect size (approximate)
    effect_size = ks_stat * np.sqrt(n)

    # Interpretation
    if p_value < alpha:
        interpretation = f"Reject H0 at α={alpha} (p={p_value:.4f})"
    else:
        interpretation = f"Fail to reject H0 at α={alpha} (p={p_value:.4f})"

    return StatisticalTest(
        test_name="Kolmogorov-Smirnov test",
        statistic=ks_stat,
        p_value=p_value,
        critical_value=critical_value,
        degrees_of_freedom=None,
        confidence_interval=None,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    method: str = "percentile",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    statistic_func : callable
        Function to calculate the statistic.
    n_bootstrap : int, default=1000
        Number of bootstrap samples.
    alpha : float, default=0.05
        Significance level.
    method : str, default="percentile"
        Bootstrap method ("percentile", "bca", "basic").
    random_state : int, optional
        Random seed.

    Returns
    -------
    Tuple[float, float]
        Confidence interval bounds.
    """
    if len(data) < 2:
        raise InsufficientDataError("Need at least 2 observations for bootstrap")

    if not np.all(np.isfinite(data)):
        raise DataValidationError("Data must contain only finite values")

    if n_bootstrap < 100:
        raise DataValidationError("Need at least 100 bootstrap samples")

    rng = np.random.RandomState(random_state)
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)

    bootstrap_stats = np.array(bootstrap_stats)

    if method == "percentile":
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    elif method == "basic":
        original_stat = statistic_func(data)
        ci_lower = 2 * original_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        ci_upper = 2 * original_stat - np.percentile(bootstrap_stats, 100 * alpha / 2)
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")

    return ci_lower, ci_upper


def permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func: callable = lambda x, y: np.mean(x) - np.mean(y),
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> StatisticalTest:
    """Perform permutation test for comparing two samples.

    Parameters
    ----------
    sample1 : np.ndarray
        First sample.
    sample2 : np.ndarray
        Second sample.
    statistic_func : callable, default=lambda x, y: np.mean(x) - np.mean(y)
        Function to calculate the test statistic.
    n_permutations : int, default=1000
        Number of permutations.
    alpha : float, default=0.05
        Significance level.
    random_state : int, optional
        Random seed.

    Returns
    -------
    StatisticalTest
        Test results.
    """
    if len(sample1) < 2 or len(sample2) < 2:
        raise InsufficientDataError("Need at least 2 observations in each sample")

    if not np.all(np.isfinite(sample1)) or not np.all(np.isfinite(sample2)):
        raise DataValidationError("Samples must contain only finite values")

    if n_permutations < 100:
        raise DataValidationError("Need at least 100 permutations")

    rng = np.random.RandomState(random_state)
    
    # Calculate observed statistic
    observed_stat = statistic_func(sample1, sample2)
    
    # Combine samples
    combined = np.concatenate([sample1, sample2])
    n1, n2 = len(sample1), len(sample2)
    
    # Perform permutations
    permuted_stats = []
    for _ in range(n_permutations):
        # Shuffle combined sample
        shuffled = rng.permutation(combined)
        perm_sample1 = shuffled[:n1]
        perm_sample2 = shuffled[n1:]
        perm_stat = statistic_func(perm_sample1, perm_sample2)
        permuted_stats.append(perm_stat)
    
    permuted_stats = np.array(permuted_stats)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    
    # Calculate critical value
    critical_value = np.percentile(np.abs(permuted_stats), 100 * (1 - alpha))
    
    # Calculate effect size
    effect_size = observed_stat / np.std(combined)
    
    # Interpretation
    if p_value < alpha:
        interpretation = f"Reject H0 at α={alpha} (p={p_value:.4f})"
    else:
        interpretation = f"Fail to reject H0 at α={alpha} (p={p_value:.4f})"

    return StatisticalTest(
        test_name="Permutation test",
        statistic=observed_stat,
        p_value=p_value,
        critical_value=critical_value,
        degrees_of_freedom=None,
        confidence_interval=None,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> List[float]:
    """Apply multiple comparison correction to p-values.

    Parameters
    ----------
    p_values : List[float]
        List of p-values to correct.
    method : str, default="bonferroni"
        Correction method ("bonferroni", "holm", "fdr_bh").
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    List[float]
        Corrected p-values.
    """
    if not p_values:
        raise DataValidationError("P-values list cannot be empty")

    if not all(0 <= p <= 1 for p in p_values):
        raise DataValidationError("P-values must be between 0 and 1")

    p_values = np.array(p_values)
    n = len(p_values)

    if method == "bonferroni":
        corrected = p_values * n
        corrected = np.minimum(corrected, 1.0)
    elif method == "holm":
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        corrected = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            corrected[idx] = p_values[idx] * (n - i)
        corrected = np.minimum(corrected, 1.0)
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        corrected = np.zeros_like(p_values)
        for i in range(n):
            corrected[sorted_indices[i]] = sorted_p[i] * n / (i + 1)
        corrected = np.minimum(corrected, 1.0)
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist()


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    test_type: str = "two_sample_t",
) -> float:
    """Calculate statistical power for a given effect size and sample size.

    Parameters
    ----------
    effect_size : float
        Effect size (Cohen's d for t-tests).
    n : int
        Sample size per group.
    alpha : float, default=0.05
        Significance level.
    test_type : str, default="two_sample_t"
        Type of test ("two_sample_t", "one_sample_t", "paired_t").

    Returns
    -------
    float
        Statistical power (0-1).
    """
    if effect_size <= 0:
        raise DataValidationError("Effect size must be positive")

    if n < 2:
        raise DataValidationError("Sample size must be at least 2")

    if not 0 < alpha < 1:
        raise DataValidationError("Alpha must be between 0 and 1")

    if test_type == "two_sample_t":
        # Two-sample t-test
        df = 2 * n - 2
        ncp = effect_size * np.sqrt(n / 2)  # Non-centrality parameter
    elif test_type == "one_sample_t":
        # One-sample t-test
        df = n - 1
        ncp = effect_size * np.sqrt(n)
    elif test_type == "paired_t":
        # Paired t-test
        df = n - 1
        ncp = effect_size * np.sqrt(n)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Calculate critical value
    critical_value = t.ppf(1 - alpha / 2, df)
    
    # Calculate power
    power = 1 - t.cdf(critical_value, df, ncp) + t.cdf(-critical_value, df, ncp)
    
    return min(max(power, 0), 1)  # Ensure power is between 0 and 1


def sample_size_calculation(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = "two_sample_t",
) -> int:
    """Calculate required sample size for a given effect size and power.

    Parameters
    ----------
    effect_size : float
        Effect size (Cohen's d for t-tests).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    test_type : str, default="two_sample_t"
        Type of test ("two_sample_t", "one_sample_t", "paired_t").

    Returns
    -------
    int
        Required sample size per group.
    """
    if effect_size <= 0:
        raise DataValidationError("Effect size must be positive")

    if not 0 < power < 1:
        raise DataValidationError("Power must be between 0 and 1")

    if not 0 < alpha < 1:
        raise DataValidationError("Alpha must be between 0 and 1")

    # Use binary search to find required sample size
    n_min, n_max = 2, 10000
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        calculated_power = power_analysis(effect_size, n_mid, alpha, test_type)
        
        if calculated_power < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max