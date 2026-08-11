"""Tests for the corrected multi-action q_obs / q_pi contract (#58)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.estimand import compute_action_value_terms
from skdr_eval.exceptions import DataValidationError


def test_q_obs_and_q_pi_are_distinct_and_correct() -> None:
    q = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    pi = np.array(
        [
            [0.25, 0.75],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    observed = np.array([0, 1, 0], dtype=np.int64)

    terms = compute_action_value_terms(q, pi, observed)

    np.testing.assert_allclose(terms.q_obs, np.array([1.0, 20.0, 3.0]))
    np.testing.assert_allclose(
        terms.q_pi,
        np.array([0.25 * 1.0 + 0.75 * 10.0, 2.0, 30.0]),
    )
    assert terms.q_pi[0] != terms.q_obs[0]


def test_one_dimensional_q_hat_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="1-D q_hat"):
        compute_action_value_terms(
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0, 1], dtype=np.int64),
        )


def test_policy_shape_must_equal_action_value_shape() -> None:
    with pytest.raises(DataValidationError, match="same shape"):
        compute_action_value_terms(
            np.ones((2, 3)),
            np.ones((2, 2)) / 2,
            np.array([0, 1], dtype=np.int64),
        )


def test_observed_actions_must_be_integer_indices() -> None:
    with pytest.raises(DataValidationError, match="integer"):
        compute_action_value_terms(
            np.ones((2, 2)),
            np.ones((2, 2)) / 2,
            np.array([0.0, 1.0]),
        )


def test_observed_action_bounds_are_checked() -> None:
    with pytest.raises(DataValidationError, match="out of bounds"):
        compute_action_value_terms(
            np.ones((2, 2)),
            np.ones((2, 2)) / 2,
            np.array([0, 2], dtype=np.int64),
        )


@pytest.mark.parametrize("which", ["q", "pi"])
def test_non_finite_inputs_fail_closed(which: str) -> None:
    q = np.ones((2, 2))
    pi = np.ones((2, 2)) / 2
    if which == "q":
        q[1, 0] = np.nan
    else:
        pi[1, 0] = np.inf
    with pytest.raises(DataValidationError, match="finite"):
        compute_action_value_terms(q, pi, np.array([0, 1], dtype=np.int64))


def test_non_normalized_policy_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="sum to 1"):
        compute_action_value_terms(
            np.ones((1, 2)),
            np.array([[0.4, 0.4]]),
            np.array([0], dtype=np.int64),
        )


def test_negative_policy_mass_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        compute_action_value_terms(
            np.ones((1, 2)),
            np.array([[1.1, -0.1]]),
            np.array([0], dtype=np.int64),
        )


def test_logged_action_must_be_eligible() -> None:
    with pytest.raises(DataValidationError, match="observed action must be eligible"):
        compute_action_value_terms(
            np.array([[1.0, 2.0]]),
            np.array([[0.0, 1.0]]),
            np.array([0], dtype=np.int64),
            eligible_actions=np.array([[0, 1]]),
        )


def test_target_policy_mass_on_ineligible_action_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="ineligible"):
        compute_action_value_terms(
            np.array([[1.0, 2.0]]),
            np.array([[0.5, 0.5]]),
            np.array([0], dtype=np.int64),
            eligible_actions=np.array([[1, 0]]),
        )


def test_output_is_stable_after_input_mutation() -> None:
    q = np.array([[1.0, 2.0]])
    pi = np.array([[0.25, 0.75]])
    terms = compute_action_value_terms(q, pi, np.array([0], dtype=np.int64))
    q[0, 0] = 999.0
    pi[0, 0] = 1.0
    np.testing.assert_array_equal(terms.q_by_action, np.array([[1.0, 2.0]]))
    assert terms.q_obs[0] == 1.0
    assert terms.q_pi[0] == pytest.approx(1.75)
    with pytest.raises(ValueError):
        terms.q_pi[0] = 0.0
