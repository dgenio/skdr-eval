"""Tests for the validated logged-behavior-propensity contract (#167)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.exceptions import DataValidationError
from skdr_eval.propensity_contract import (
    LoggedActionPropensity,
    observed_importance_ratio,
    validate_logged_action_propensity,
)


def test_valid_logged_propensity_records_provenance() -> None:
    raw = np.array([0.2, 0.5, 1.0])
    result = validate_logged_action_propensity(
        raw,
        n_rows=3,
        field_name="propensity",
        policy_id="epsilon-greedy-v7",
    )
    assert isinstance(result, LoggedActionPropensity)
    assert result.source == "logged"
    assert result.field_name == "propensity"
    assert result.policy_id == "epsilon-greedy-v7"
    np.testing.assert_array_equal(result.values, raw)
    assert result.n_rows == 3


def test_validated_values_are_immutable_copy() -> None:
    raw = np.array([0.25, 0.75])
    result = validate_logged_action_propensity(raw)
    raw[0] = 0.9
    assert result.values[0] == 0.25
    with pytest.raises(ValueError):
        result.values[0] = 0.3


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.000001])
def test_probability_bounds_fail_closed(bad: float) -> None:
    with pytest.raises(DataValidationError, match=r"\(0, 1\]"):
        validate_logged_action_propensity(np.array([0.5, bad]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_fail_closed(bad: float) -> None:
    with pytest.raises(DataValidationError, match="finite"):
        validate_logged_action_propensity(np.array([0.5, bad]))


def test_dense_matrix_is_rejected_instead_of_reinterpreted() -> None:
    with pytest.raises(DataValidationError, match="one-dimensional"):
        validate_logged_action_propensity(np.array([[0.4, 0.6], [0.7, 0.3]]))


def test_expected_row_count_is_enforced() -> None:
    with pytest.raises(DataValidationError, match="shape"):
        validate_logged_action_propensity(np.array([0.5, 0.5]), n_rows=3)


def test_blank_provenance_fields_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="field_name"):
        validate_logged_action_propensity(np.array([0.5]), field_name="   ")
    with pytest.raises(DataValidationError, match="policy_id"):
        validate_logged_action_propensity(np.array([0.5]), policy_id="")


def test_observed_importance_ratio_uses_exact_logged_probability() -> None:
    behavior = validate_logged_action_propensity(np.array([0.25, 0.5, 1.0]))
    target = np.array([0.5, 0.25, 0.0])
    ratio = observed_importance_ratio(target, behavior)
    np.testing.assert_allclose(ratio, np.array([2.0, 0.5, 0.0]))


def test_importance_ratio_requires_row_alignment() -> None:
    behavior = validate_logged_action_propensity(np.array([0.5, 0.5]))
    with pytest.raises(DataValidationError, match="identical shape"):
        observed_importance_ratio(np.array([0.5]), behavior)


@pytest.mark.parametrize(
    "target",
    [
        np.array([np.nan, 0.5]),
        np.array([-0.1, 0.5]),
        np.array([1.1, 0.5]),
    ],
)
def test_importance_ratio_defensively_validates_target_probability(
    target: np.ndarray,
) -> None:
    behavior = validate_logged_action_propensity(np.array([0.5, 0.5]))
    with pytest.raises(DataValidationError):
        observed_importance_ratio(target, behavior)
