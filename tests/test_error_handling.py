"""Test error handling and validation functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import skdr_eval
from skdr_eval.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ModelValidationError,
    OutcomeModelError,
)
from skdr_eval.validation import (
    validate_dataframe,
    validate_finite_values,
    validate_numpy_array,
    validate_parameter_range,
    validate_positive_integer,
    validate_positive_values,
    validate_probabilities,
    validate_random_state,
    validate_string_choice,
)


def test_build_design_validation():
    """Test build_design input validation."""
    # Test empty DataFrame
    with pytest.raises(InsufficientDataError):
        skdr_eval.build_design(pd.DataFrame())

    # Test missing required columns
    with pytest.raises(DataValidationError):
        skdr_eval.build_design(pd.DataFrame({"col1": [1, 2, 3]}))

    # Test invalid actions
    logs = pd.DataFrame(
        {
            "arrival_ts": [1, 2, 3],
            "action": ["op1", "op2", "invalid_op"],
            "service_time": [1.0, 2.0, 3.0],
            "op1_elig": [1, 1, 1],
            "op2_elig": [1, 1, 1],
        }
    )
    with pytest.raises(DataValidationError):
        skdr_eval.build_design(logs)

    # Test invalid eligibility values
    logs = pd.DataFrame(
        {
            "arrival_ts": [1, 2, 3],
            "action": ["op1", "op2", "op1"],
            "service_time": [1.0, 2.0, 3.0],
            "op1_elig": [1, 0, 1],
            "op2_elig": [1, 2, 0],  # Invalid: should be 0 or 1
        }
    )
    with pytest.raises(DataValidationError):
        skdr_eval.build_design(logs)


def test_fit_propensity_timecal_validation():
    """Test fit_propensity_timecal input validation."""
    # Test mismatched array lengths
    X_phi = np.random.randn(10, 3)
    A = np.array([0, 1, 2])  # Wrong length
    with pytest.raises(DataValidationError):
        skdr_eval.fit_propensity_timecal(X_phi, A)

    # Test insufficient data
    X_phi = np.random.randn(5, 3)
    A = np.array([0, 1, 0, 1, 0])
    with pytest.raises(InsufficientDataError):
        skdr_eval.fit_propensity_timecal(X_phi, A, n_splits=3)

    # Test only one action
    X_phi = np.random.randn(10, 3)
    A = np.zeros(10)  # All same action
    with pytest.raises(InsufficientDataError):
        skdr_eval.fit_propensity_timecal(X_phi, A)


def test_fit_outcome_crossfit_validation():
    """Test fit_outcome_crossfit input validation."""
    # Test mismatched array lengths
    X_obs = np.random.randn(10, 5)
    Y = np.random.randn(5)  # Wrong length
    with pytest.raises(DataValidationError):
        skdr_eval.fit_outcome_crossfit(X_obs, Y)

    # Test invalid estimator
    X_obs = np.random.randn(10, 5)
    Y = np.random.randn(10)
    with pytest.raises(OutcomeModelError):
        skdr_eval.fit_outcome_crossfit(X_obs, Y, estimator="invalid")

    # Test callable estimator without required methods
    class BadEstimator:
        def fit(self, X, y):
            pass

        # Missing predict method

    def bad_est_factory():
        return BadEstimator()

    with pytest.raises(ModelValidationError):
        skdr_eval.fit_outcome_crossfit(X_obs, Y, estimator=bad_est_factory)


def test_induce_policy_from_sklearn_validation():
    """Test induce_policy_from_sklearn input validation."""

    # Test invalid model
    class BadModel:
        # Missing predict method
        pass

    X_base = np.random.randn(5, 3)
    ops_all = ["op1", "op2"]
    elig = np.array([[1, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
    idx = {"op1": 0, "op2": 1}

    with pytest.raises(ModelValidationError):
        skdr_eval.induce_policy_from_sklearn(BadModel(), X_base, ops_all, elig, idx)

    # Test mismatched array dimensions
    model = RandomForestRegressor(random_state=42)
    X_base = np.random.randn(5, 3)
    elig = np.array([[1, 1], [1, 0]])  # Wrong shape
    with pytest.raises(DataValidationError):
        skdr_eval.induce_policy_from_sklearn(model, X_base, ops_all, elig, idx)

    # Test invalid eligibility values
    elig = np.array([[1, 1], [1, 2], [0, 1], [1, 1], [0, 0]])  # Contains 2
    with pytest.raises(DataValidationError):
        skdr_eval.induce_policy_from_sklearn(model, X_base, ops_all, elig, idx)


def test_validation_utilities():
    """Test validation utility functions."""
    # Test DataFrame validation
    with pytest.raises(DataValidationError):
        validate_dataframe("not_a_dataframe", "test")

    with pytest.raises(InsufficientDataError):
        validate_dataframe(pd.DataFrame(), "test")

    # Test numpy array validation
    with pytest.raises(DataValidationError):
        validate_numpy_array("not_an_array", "test")

    with pytest.raises(InsufficientDataError):
        validate_numpy_array(np.array([]), "test")

    # Test probability validation
    with pytest.raises(DataValidationError):
        validate_probabilities(np.array([[-0.1, 0.5, 0.6]]), "test")

    with pytest.raises(DataValidationError):
        validate_probabilities(np.array([[0.3, 0.4, 0.2]]), "test")  # Doesn't sum to 1

    # Test positive values validation
    with pytest.raises(DataValidationError):
        validate_positive_values(np.array([-1, 2, 3]), "test")

    # Test finite values validation
    with pytest.raises(DataValidationError):
        validate_finite_values(np.array([1, np.inf, 3]), "test")

    # Test parameter range validation
    with pytest.raises(DataValidationError):
        validate_parameter_range(5, "test", min_val=10)

    # Test string choice validation
    with pytest.raises(DataValidationError):
        validate_string_choice("invalid", "test", ["valid1", "valid2"])

    # Test positive integer validation
    with pytest.raises(DataValidationError):
        validate_positive_integer(-1, "test")

    with pytest.raises(DataValidationError):
        validate_positive_integer(1.5, "test")

    # Test random state validation
    with pytest.raises(DataValidationError):
        validate_random_state("invalid", "test")


def test_error_handling_integration():
    """Test error handling in integrated workflow."""
    # Create valid test data
    logs = pd.DataFrame(
        {
            "arrival_ts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "action": [
                "op1",
                "op2",
                "op1",
                "op2",
                "op1",
                "op2",
                "op1",
                "op2",
                "op1",
                "op2",
            ],
            "service_time": [1.0, 2.0, 1.5, 2.5, 1.2, 1.8, 2.1, 1.7, 1.9, 2.3],
            "op1_elig": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "op2_elig": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "cli_feat1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "cli_feat2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        }
    )

    # This should work without errors
    design = skdr_eval.build_design(logs)
    assert design is not None

    # Test with invalid data that should raise appropriate errors
    invalid_logs = logs.copy()
    invalid_logs["service_time"] = [
        np.inf,
        2.0,
        1.5,
        2.5,
        1.2,
        1.8,
        2.1,
        1.7,
        1.9,
        2.3,
    ]  # Contains inf

    with pytest.raises(DataValidationError):
        skdr_eval.build_design(invalid_logs)


if __name__ == "__main__":
    pytest.main([__file__])
