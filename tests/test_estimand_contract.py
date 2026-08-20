"""Tests for corrected multi-action q_obs / q_pi construction (#58)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.estimand import compute_action_value_terms
from skdr_eval.exceptions import DataValidationError


ACTIONS = ("left", "right")


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

    terms = compute_action_value_terms(
        q,
        pi,
        observed,
        actions=ACTIONS,
    )

    assert terms.actions == ACTIONS
    np.testing.assert_allclose(terms.q_obs, np.array([1.0, 20.0, 3.0]))
    np.testing.assert_allclose(
        terms.q_pi,
        np.array([0.25 * 1.0 + 0.75 * 10.0, 2.0, 30.0]),
    )
    assert terms.q_pi[0] != terms.q_obs[0]


def test_two_action_dgp_recovers_known_target_value_and_rejects_old_broadcast_value() -> None:
    """Simulation proof for the corrected target-policy outcome term (#58)."""
    rng = np.random.default_rng(58)
    n_rows = 50_000
    context = rng.integers(0, 2, size=n_rows)

    # Heterogeneous action values with E[q_left]=2 and E[q_right]=3.5.
    q = np.column_stack((1.0 + 2.0 * context, 4.0 - context))

    # Behavior strongly favors left; target strongly favors right.
    observed = (rng.random(n_rows) < 0.15).astype(np.int64)
    pi = np.tile(np.array([0.2, 0.8]), (n_rows, 1))

    terms = compute_action_value_terms(q, pi, observed, actions=ACTIONS)

    # V(pi) = 0.2 * 2 + 0.8 * 3.5 = 3.2.
    assert float(np.mean(terms.q_pi)) == pytest.approx(3.2, abs=0.01)

    # The old 1-D broadcast construction collapses q_pi to q_obs and therefore
    # tracks the behavior-policy value instead of the target-policy value.
    old_broadcast_value = float(np.mean(terms.q_obs))
    assert abs(old_broadcast_value - 3.2) > 0.5


def test_one_dimensional_q_hat_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="1-D q_hat"):
        compute_action_value_terms(
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0, 1], dtype=np.int64),
            actions=ACTIONS,
        )


def test_q_columns_must_match_explicit_action_vocabulary() -> None:
    with pytest.raises(DataValidationError, match="action vocabulary"):
        compute_action_value_terms(
            np.ones((2, 3)),
            np.ones((2, 3)) / 3,
            np.array([0, 1], dtype=np.int64),
            actions=ACTIONS,
        )


def test_policy_validation_uses_canonical_contract() -> None:
    with pytest.raises(DataValidationError, match="sum to 1"):
        compute_action_value_terms(
            np.ones((1, 2)),
            np.array([[0.4, 0.4]]),
            np.array([0], dtype=np.int64),
            actions=ACTIONS,
        )
    with pytest.raises(DataValidationError, match="non-negative"):
        compute_action_value_terms(
            np.ones((1, 2)),
            np.array([[1.1, -0.1]]),
            np.array([0], dtype=np.int64),
            actions=ACTIONS,
        )


def test_observed_actions_must_be_integer_indices() -> None:
    with pytest.raises(DataValidationError, match="integer"):
        compute_action_value_terms(
            np.ones((2, 2)),
            np.ones((2, 2)) / 2,
            np.array([0.0, 1.0]),
            actions=ACTIONS,
        )


def test_observed_action_bounds_are_checked() -> None:
    with pytest.raises(DataValidationError, match="out of bounds"):
        compute_action_value_terms(
            np.ones((2, 2)),
            np.ones((2, 2)) / 2,
            np.array([0, 2], dtype=np.int64),
            actions=ACTIONS,
        )


def test_non_finite_action_values_fail_closed() -> None:
    q = np.ones((2, 2))
    q[1, 0] = np.nan
    with pytest.raises(DataValidationError, match="finite"):
        compute_action_value_terms(
            q,
            np.ones((2, 2)) / 2,
            np.array([0, 1], dtype=np.int64),
            actions=ACTIONS,
        )


def test_logged_action_must_be_eligible() -> None:
    with pytest.raises(DataValidationError, match="observed action must be eligible"):
        compute_action_value_terms(
            np.array([[1.0, 2.0]]),
            np.array([[0.0, 1.0]]),
            np.array([0], dtype=np.int64),
            actions=ACTIONS,
            eligible_actions=np.array([[0, 1]]),
        )


def test_target_policy_mass_on_ineligible_action_is_rejected_by_policy_contract() -> None:
    with pytest.raises(DataValidationError, match="ineligible"):
        compute_action_value_terms(
            np.array([[1.0, 2.0]]),
            np.array([[0.5, 0.5]]),
            np.array([0], dtype=np.int64),
            actions=ACTIONS,
            eligible_actions=np.array([[1, 0]]),
        )


def test_output_is_stable_after_input_mutation() -> None:
    q = np.array([[1.0, 2.0]])
    pi = np.array([[0.25, 0.75]])
    terms = compute_action_value_terms(
        q,
        pi,
        np.array([0], dtype=np.int64),
        actions=ACTIONS,
    )
    q[0, 0] = 999.0
    pi[0, 0] = 1.0
    np.testing.assert_array_equal(terms.q_by_action, np.array([[1.0, 2.0]]))
    assert terms.q_obs[0] == 1.0
    assert terms.q_pi[0] == pytest.approx(1.75)
    with pytest.raises(ValueError):
        terms.q_pi[0] = 0.0
