"""Tests targeting uncovered lines from Codecov patch coverage report on PR #57.

Each test class targets a specific source module to improve patch coverage.
"""

import json

import numpy as np
import pandas as pd
import pytest

from skdr_eval.config import (
    ConfigurationError,
    EvaluationConfig,
    ModelConfig,
    VisualizationConfig,
    load_config_from_file,
    merge_configs,
    save_config_to_file,
)
from skdr_eval.diagnostics import (
    assess_propensity_calibration,
    assess_propensity_discrimination,
    check_propensity_balance,
    check_propensity_overlap,
    compute_balance_statistics,
    compute_propensity_log_loss,
    compute_propensity_statistics,
)
from skdr_eval.exceptions import (
    DataValidationError,
    InsufficientDataError,
    SkdrEvalError,
)
from skdr_eval.exceptions import MemoryError as SkdrMemoryError
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
from skdr_eval.validation import (
    validate_dataframe,
    validate_finite_values,
    validate_memory_usage,
    validate_numpy_array,
    validate_parameter_range,
    validate_random_state,
    validate_string_choice,
)

# ── exceptions.py coverage ──────────────────────────────────────────────────


class TestExceptionStrWithDetails:
    """Cover SkdrEvalError.__str__ when details dict is non-empty (L16-17)."""

    def test_str_with_details(self):
        err = SkdrEvalError("something broke", details={"key": "val"})
        assert "key=val" in str(err)
        assert "something broke" in str(err)

    def test_str_without_details(self):
        err = SkdrEvalError("plain error")
        assert str(err) == "plain error"


# ── validation.py coverage ──────────────────────────────────────────────────


class TestValidationUncoveredBranches:
    """Cover missing branches in validation.py (patch 64%)."""

    def test_dataframe_missing_required_columns(self):
        """L62: raise when required columns are missing."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_dataframe(df, "test", required_columns=["a", "b", "c"])

    def test_numpy_array_wrong_ndim(self):
        """L106: raise when dimension count doesn't match."""
        arr = np.array([1, 2, 3])
        with pytest.raises(DataValidationError, match="dimensions"):
            validate_numpy_array(arr, "test", expected_shape=(3, 1))

    def test_numpy_array_wrong_dim_size(self):
        """L112: raise when a specific dimension size doesn't match."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(DataValidationError, match=r"dimension .* has size"):
            validate_numpy_array(arr, "test", expected_shape=(2, 3))

    def test_numpy_array_wrong_dtype(self):
        """L117-122: raise when dtype doesn't match."""
        arr = np.array([1, 2, 3], dtype=int)
        with pytest.raises(DataValidationError, match="dtype"):
            validate_numpy_array(arr, "test", expected_dtype=np.float64)

    def test_string_choice_case_insensitive(self):
        """L187: case_sensitive=False branch."""
        # Should pass when case differs
        validate_string_choice("YAML", "format", ["yaml", "json"], case_sensitive=False)

    def test_string_choice_case_insensitive_fail(self):
        """L324-325: raise when value not in choices (case-insensitive)."""
        with pytest.raises(DataValidationError, match="must be one of"):
            validate_string_choice(
                "xml", "format", ["yaml", "json"], case_sensitive=False
            )

    def test_finite_values_raises(self):
        """L218-219: raise for non-finite values."""
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_finite_values(np.array([1.0, np.nan, 3.0]), "test")

    def test_memory_usage_raises(self):
        """L262-263: raise MemoryError when exceeds limit."""
        with pytest.raises(SkdrMemoryError, match="exceeds limit"):
            validate_memory_usage(10.0, max_memory_gb=8.0)

    def test_parameter_range_max_val(self):
        """L295-296: raise when value > max_val."""
        with pytest.raises(DataValidationError, match="must be <="):
            validate_parameter_range(100, "test", max_val=50)

    def test_random_state_negative(self):
        """L382: raise for negative random_state integer."""
        with pytest.raises(DataValidationError, match="non-negative"):
            validate_random_state(-1)


# ── statistical.py coverage ─────────────────────────────────────────────────


class TestStatisticalUncoveredBranches:
    """Cover missing branches in statistical.py (patch 75%)."""

    def test_t_test_alternative_less(self):
        """L94-95: one-sided 'less' alternative."""
        rng = np.random.RandomState(42)
        s1 = rng.normal(0, 1, 30)
        s2 = rng.normal(2, 1, 30)
        result = t_test(s1, s2, alternative="less")
        assert result.p_value < 0.05

    def test_t_test_alternative_greater(self):
        """L101: one-sided 'greater' alternative."""
        rng = np.random.RandomState(42)
        s1 = rng.normal(5, 1, 30)
        s2 = rng.normal(0, 1, 30)
        result = t_test(s1, s2, alternative="greater")
        assert result.p_value < 0.05

    def test_t_test_one_sided_ci(self):
        """L133-134: CI for one-sided test."""
        rng = np.random.RandomState(42)
        s1 = rng.normal(0, 1, 30)
        s2 = rng.normal(0, 1, 30)
        result = t_test(s1, s2, alternative="less")
        ci = result.confidence_interval
        assert ci is not None
        assert ci[0] == -np.inf

    def test_mann_whitney_insufficient_data(self):
        """L193: raise for too few samples."""
        with pytest.raises(InsufficientDataError):
            mann_whitney_u_test(np.array([1.0, 2.0]), np.array([3.0]))

    def test_mann_whitney_non_finite(self):
        """L206: raise for non-finite values."""
        with pytest.raises(DataValidationError, match="finite"):
            mann_whitney_u_test(np.array([1.0, np.inf, 3.0]), np.array([4.0, 5.0, 6.0]))

    def test_chi_square_multidim(self):
        """L216: raise for multi-dimensional observed array."""
        with pytest.raises(DataValidationError, match="1D"):
            chi_square_test(np.array([[1, 2], [3, 4]]))

    def test_chi_square_expected_multidim(self):
        """L269: raise for multi-dimensional expected array."""
        with pytest.raises(DataValidationError, match="1D"):
            chi_square_test(np.array([10, 20, 30]), expected=np.array([[10, 20, 30]]))

    def test_chi_square_length_mismatch(self):
        """L273: raise when expected/observed lengths differ."""
        with pytest.raises(DataValidationError, match="same length"):
            chi_square_test(np.array([10, 20, 30]), expected=np.array([10.0, 20.0]))

    def test_chi_square_negative_expected(self):
        """L277: raise for negative expected frequencies."""
        with pytest.raises(DataValidationError, match="non-negative"):
            chi_square_test(
                np.array([10, 20, 30]), expected=np.array([-1.0, 20.0, 30.0])
            )

    def test_chi_square_zero_expected(self):
        """L299: raise for zero expected frequencies."""
        with pytest.raises(DataValidationError, match="strictly positive"):
            chi_square_test(
                np.array([10, 20, 30]), expected=np.array([0.0, 20.0, 30.0])
            )

    def test_ks_test_insufficient_data(self):
        """L341: raise for too few samples."""
        with pytest.raises(InsufficientDataError):
            kolmogorov_smirnov_test(np.array([1.0, 2.0]))

    def test_ks_test_uniform(self):
        """L363-373: uniform distribution branch."""
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, 100)
        result = kolmogorov_smirnov_test(data, distribution="uniform")
        assert result.p_value > 0.05

    def test_ks_test_expon(self):
        """L363-373: exponential distribution branch."""
        rng = np.random.RandomState(42)
        data = rng.exponential(2.0, 100)
        result = kolmogorov_smirnov_test(data, distribution="expon")
        assert result.test_name == "Kolmogorov-Smirnov test"

    def test_ks_test_unknown_distribution(self):
        """L387: raise for unknown distribution."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            kolmogorov_smirnov_test(np.array([1.0] * 10), distribution="gamma")

    def test_permutation_insufficient_data(self):
        """L437: raise for too few samples."""
        with pytest.raises(InsufficientDataError):
            permutation_test(np.array([1.0]), np.array([2.0, 3.0]))

    def test_permutation_non_finite(self):
        """L440: raise for non-finite values."""
        with pytest.raises(DataValidationError, match="finite"):
            permutation_test(np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, 6.0]))

    def test_permutation_test_runs(self):
        """L457-466: cover permutation loop."""
        rng = np.random.RandomState(42)
        s1 = rng.normal(0, 1, 20)
        s2 = rng.normal(2, 1, 20)
        result = permutation_test(s1, s2, n_permutations=200, random_state=42)
        assert result.p_value < 0.05

    def test_multiple_comparison_empty(self):
        """L509: raise for empty p-values."""
        with pytest.raises(DataValidationError, match="empty"):
            multiple_comparison_correction([])

    def test_multiple_comparison_invalid_range(self):
        """L512: raise for p-values out of [0,1]."""
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            multiple_comparison_correction([0.05, 1.5])

    def test_multiple_comparison_holm(self):
        """L546-549: holm correction method."""
        result = multiple_comparison_correction([0.01, 0.04, 0.03], method="holm")
        assert len(result) == 3
        assert all(0 <= p <= 1 for p in result)

    def test_multiple_comparison_fdr_bh(self):
        """L555: fdr_bh correction method."""
        result = multiple_comparison_correction([0.01, 0.04, 0.03], method="fdr_bh")
        assert len(result) == 3
        assert all(0 <= p <= 1 for p in result)

    def test_power_analysis_invalid_effect_size(self):
        """L590: raise for effect_size <= 0."""
        with pytest.raises(DataValidationError, match="positive"):
            power_analysis(effect_size=0, n=30)

    def test_power_analysis_one_sample_t(self):
        """L620: one_sample_t test type."""
        power = power_analysis(effect_size=0.5, n=30, test_type="one_sample_t")
        assert 0 < power < 1

    def test_sample_size_invalid_effect_size(self):
        """L653: raise for effect_size <= 0."""
        with pytest.raises(DataValidationError, match="positive"):
            sample_size_calculation(effect_size=-0.5)

    def test_sample_size_invalid_power(self):
        """L656: raise for power out of (0,1)."""
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            sample_size_calculation(effect_size=0.5, power=1.5)

    def test_sample_size_calculation_runs(self):
        """L671: binary search loop executes."""
        n = sample_size_calculation(effect_size=0.5, power=0.8)
        assert n > 2

    def test_bootstrap_ci_basic_method(self):
        """L710: 'basic' bootstrap CI method."""
        data = np.random.RandomState(42).normal(0, 1, 50)
        lo, hi = bootstrap_confidence_interval(
            data, np.mean, method="basic", random_state=42
        )
        assert lo < hi

    def test_bootstrap_ci_unknown_method(self):
        """L713: raise for unknown bootstrap method."""
        data = np.random.RandomState(42).normal(0, 1, 50)
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            bootstrap_confidence_interval(data, np.mean, method="invalid")


# ── diagnostics.py coverage ─────────────────────────────────────────────────


class TestDiagnosticsUncoveredBranches:
    """Cover missing validation/edge-case branches in diagnostics.py (patch 81%)."""

    def _make_props_and_actions(self, n=50, n_actions=3, seed=42):
        """Helper: return propensities (n, n_actions) and actions (n,)."""
        rng = np.random.RandomState(seed)
        raw = rng.dirichlet(np.ones(n_actions), size=n)
        actions = rng.randint(0, n_actions, size=n)
        return raw, actions

    # ── check_propensity_overlap ──

    def test_overlap_length_mismatch(self):
        """L60: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            check_propensity_overlap(np.ones((10, 3)), np.zeros(5))

    def test_overlap_insufficient_data(self):
        """L68: raise for <10 samples."""
        with pytest.raises(InsufficientDataError):
            check_propensity_overlap(np.ones((5, 3)), np.zeros(5, dtype=int))

    def test_overlap_skips_rare_action(self):
        """L74: continue when action_mask.sum() < _MIN_ACTION_COUNT."""
        # All samples have action=0, so actions 1,2 have 0 samples -> skip
        props = np.random.RandomState(42).dirichlet([1, 1, 1], size=20)
        actions = np.zeros(20, dtype=int)
        result = check_propensity_overlap(props, actions)
        assert isinstance(result, float)

    def test_overlap_skips_empty_other(self):
        """L84: continue when other_props is empty (single-action data)."""
        # Only one action type, all same
        props = np.random.RandomState(42).dirichlet([1, 1], size=20)
        actions = np.zeros(20, dtype=int)
        result = check_propensity_overlap(props, actions)
        assert isinstance(result, float)

    # ── check_propensity_balance ──

    def test_balance_length_mismatch(self):
        """L109: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            check_propensity_balance(np.ones((10, 3)), np.zeros(5))

    def test_balance_insufficient_data(self):
        """L114: raise for <10 samples."""
        with pytest.raises(InsufficientDataError):
            check_propensity_balance(np.ones((5, 3)), np.zeros(5, dtype=int))

    def test_balance_skips_rare_action(self):
        """L122: continue when action_mask.sum() < _MIN_ACTION_COUNT."""
        props = np.random.RandomState(42).dirichlet([1, 1, 1], size=20)
        actions = np.zeros(20, dtype=int)
        result = check_propensity_balance(props, actions)
        assert isinstance(result, float)

    def test_balance_skips_empty_other(self):
        """L128: continue when other_props is empty."""
        props = np.random.RandomState(42).dirichlet([1, 1], size=20)
        actions = np.zeros(20, dtype=int)
        result = check_propensity_balance(props, actions)
        assert isinstance(result, float)

    def test_balance_zero_variance(self):
        """L143: pooled_std == 0 branch."""
        # All propensities identical -> zero variance
        props = np.full((20, 2), 0.5)
        actions = np.array([0] * 10 + [1] * 10)
        result = check_propensity_balance(props, actions)
        assert result == 1.0  # Means are same, variance is zero -> balance=1

    # ── assess_propensity_calibration ──

    def test_calibration_length_mismatch(self):
        """L177: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            assess_propensity_calibration(np.ones((10, 3)), np.zeros(5))

    def test_calibration_insufficient_data(self):
        """L182: raise for <10 samples."""
        with pytest.raises(InsufficientDataError):
            assess_propensity_calibration(np.ones((5, 3)), np.zeros(5, dtype=int))

    # ── assess_propensity_discrimination ──

    def test_discrimination_length_mismatch(self):
        """L264: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            assess_propensity_discrimination(np.ones((10, 3)), np.zeros(5))

    def test_discrimination_insufficient_data(self):
        """L271: raise for <10 samples."""
        with pytest.raises(InsufficientDataError):
            assess_propensity_discrimination(np.ones((5, 3)), np.zeros(5, dtype=int))

    def test_discrimination_skips_rare_action(self):
        """L280-281: continue when action_mask.sum() < threshold."""
        props = np.random.RandomState(42).dirichlet([1, 1, 1], size=20)
        actions = np.zeros(20, dtype=int)
        result = assess_propensity_discrimination(props, actions)
        assert result is not None  # Returns a tuple or scalar

    # ── compute_propensity_statistics ──

    def test_statistics_length_mismatch(self):
        """L308: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            compute_propensity_statistics(np.ones((10, 3)), np.zeros(5))

    def test_statistics_insufficient_data(self):
        """L313: raise for <5 samples."""
        with pytest.raises(InsufficientDataError):
            compute_propensity_statistics(np.ones((3, 3)), np.zeros(3, dtype=int))

    # ── compute_balance_statistics ──

    def test_balance_stats_length_mismatch(self):
        """L352: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            compute_balance_statistics(np.ones((10, 3)), np.zeros(5))

    def test_balance_stats_insufficient_data(self):
        """L357: raise for <5 samples."""
        with pytest.raises(InsufficientDataError):
            compute_balance_statistics(np.ones((3, 3)), np.zeros(3, dtype=int))

    # ── compute_propensity_log_loss ──

    def test_log_loss_length_mismatch(self):
        """L393: raise for length mismatch."""
        with pytest.raises(DataValidationError, match="length"):
            compute_propensity_log_loss(np.ones((10, 3)), np.zeros(5))

    def test_log_loss_insufficient_data(self):
        """L398: raise for <5 samples."""
        with pytest.raises(InsufficientDataError):
            compute_propensity_log_loss(np.ones((3, 3)), np.zeros(3, dtype=int))


# ── config.py coverage ──────────────────────────────────────────────────────


class TestConfigUncoveredBranches:
    """Cover missing branches in config.py (patch 92%)."""

    def test_evaluation_config_invalid_min_ess_frac_high(self):
        """L171/174: min_ess_frac > 1.0 should raise."""
        with pytest.raises(ConfigurationError):
            EvaluationConfig(min_ess_frac=1.5)

    def test_model_config_invalid_test_size(self):
        """L193: test_size out of range."""
        with pytest.raises(ConfigurationError):
            ModelConfig(test_size=1.5)

    def test_visualization_config_invalid_font_size(self):
        """Cover label_size < 8 path if not already tested."""
        with pytest.raises(ConfigurationError):
            VisualizationConfig(label_size=5)

    def test_load_config_json(self, tmp_path):
        """L399-400: .json load branch."""
        cfg = {"key": "value"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(cfg))
        loaded = load_config_from_file(p)
        assert loaded == cfg

    def test_load_config_unsupported_format(self, tmp_path):
        """L407-408: raise for unsupported file format."""
        p = tmp_path / "test.xml"
        p.write_text("<xml/>")
        with pytest.raises(ConfigurationError, match="Unsupported"):
            load_config_from_file(p)

    def test_save_config_json(self, tmp_path):
        """L453-454: .json save branch."""
        cfg = {"hello": "world"}
        p = tmp_path / "out.json"
        save_config_to_file(cfg, p)
        loaded = json.loads(p.read_text())
        assert loaded == cfg

    def test_save_config_unsupported_format(self, tmp_path):
        """L459-460: raise for unsupported file format on save."""
        p = tmp_path / "out.xml"
        with pytest.raises(ConfigurationError, match="Unsupported"):
            save_config_to_file({"a": 1}, p)

    def test_merge_configs_nested(self):
        """L481-482/487-488: recursive dict merge."""
        a = {"x": {"a": 1, "b": 2}}
        b = {"x": {"b": 3, "c": 4}}
        merged = merge_configs(a, b)
        assert merged == {"x": {"a": 1, "b": 3, "c": 4}}
