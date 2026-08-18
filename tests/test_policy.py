"""Contract tests for explicit target-policy action distributions (#279)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.exceptions import DataValidationError
from skdr_eval.policy import (
    ExplicitPolicy,
    Policy,
    resolve_action_distribution,
    validate_action_distribution,
)


ACTIONS = ("a", "b", "c")


def test_explicit_policy_satisfies_protocol_and_preserves_distribution():
    probs = np.array([[1.0, 0.0, 0.0], [0.1, 0.2, 0.7]])
    policy = ExplicitPolicy(probs, actions=ACTIONS, name="candidate-v1")

    assert isinstance(policy, Policy)
    resolved = policy.action_distribution(np.zeros((2, 4)), actions=ACTIONS)
    np.testing.assert_allclose(resolved, probs)
    assert policy.name == "candidate-v1"


def test_explicit_policy_probability_state_is_read_only():
    source = np.array([[0.2, 0.8]])
    policy = ExplicitPolicy(source, actions=("control", "candidate"))

    # Construction copies the caller's array, so mutating the source cannot
    # change the policy that was bound to this action vocabulary.
    source[0, 0] = 1.0
    np.testing.assert_allclose(policy.probabilities, [[0.2, 0.8]])
    assert policy.probabilities.flags.writeable is False

    with pytest.raises(ValueError):
        policy.probabilities[0, 0] = 1.0

    # Resolved output is a fresh validated copy; callers may manipulate their
    # local result without mutating the immutable policy state.
    resolved = policy.action_distribution(
        np.zeros((1, 1)), actions=("control", "candidate")
    )
    assert resolved.flags.writeable is True
    resolved[0, 0] = 0.3
    np.testing.assert_allclose(policy.probabilities, [[0.2, 0.8]])


def test_deterministic_one_hot_is_same_contract_as_stochastic_policy():
    deterministic = np.array([[0.0, 1.0], [1.0, 0.0]])
    resolved = validate_action_distribution(deterministic, actions=("left", "right"))
    np.testing.assert_array_equal(resolved, deterministic)


def test_reordered_actions_fail_closed():
    policy = ExplicitPolicy([[0.2, 0.8]], actions=("control", "candidate"))
    with pytest.raises(DataValidationError, match="vocabulary/order"):
        policy.action_distribution(
            np.zeros((1, 2)), actions=("candidate", "control")
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([0.5, 0.5], "2D"),
        ([[0.5, 0.5, 0.0]], "columns"),
        ([[0.5, np.nan]], "finite"),
        ([[-0.1, 1.1]], "non-negative"),
        ([[0.4, 0.4]], "sum to 1"),
    ],
)
def test_invalid_probability_matrices_fail(probabilities, message):
    with pytest.raises(DataValidationError, match=message):
        validate_action_distribution(probabilities, actions=("a", "b"))


def test_action_vocabulary_must_be_non_empty_unique_strings():
    with pytest.raises(DataValidationError, match="at least one"):
        validate_action_distribution(np.empty((1, 0)), actions=())
    with pytest.raises(DataValidationError, match="unique"):
        validate_action_distribution([[0.5, 0.5]], actions=("a", "a"))
    with pytest.raises(DataValidationError, match="non-empty strings"):
        validate_action_distribution([[1.0]], actions=("",))


def test_context_row_count_must_match_probability_rows():
    policy = ExplicitPolicy([[0.5, 0.5], [0.2, 0.8]], actions=("a", "b"))
    with pytest.raises(DataValidationError, match="rows must match contexts"):
        policy.action_distribution(np.zeros((3, 2)), actions=("a", "b"))


def test_ineligible_actions_must_receive_zero_probability():
    probs = np.array([[0.8, 0.2], [0.0, 1.0]])
    eligibility = np.array([[1, 0], [0, 1]])
    with pytest.raises(DataValidationError, match="ineligible action"):
        validate_action_distribution(
            probs, actions=("a", "b"), eligible_actions=eligibility
        )


def test_valid_variable_eligibility_is_preserved():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    eligibility = np.array([[True, False, False], [False, True, True]])
    resolved = validate_action_distribution(
        probs, actions=ACTIONS, eligible_actions=eligibility
    )
    np.testing.assert_array_equal(resolved, probs)


def test_rows_with_no_eligible_actions_fail():
    with pytest.raises(DataValidationError, match="at least one eligible"):
        validate_action_distribution(
            [[1.0, 0.0]],
            actions=("a", "b"),
            eligible_actions=np.array([[0, 0]]),
        )


def test_malformed_eligibility_values_fail():
    with pytest.raises(DataValidationError, match="boolean or contain only 0/1"):
        validate_action_distribution(
            [[0.5, 0.5]],
            actions=("a", "b"),
            eligible_actions=np.array([[1, 2]]),
        )


def test_resolver_accepts_raw_numpy_matrix_through_same_validation():
    contexts = np.zeros((2, 4))
    probs = np.array([[0.7, 0.3], [0.1, 0.9]])
    resolved = resolve_action_distribution(probs, contexts, actions=("a", "b"))
    np.testing.assert_allclose(resolved, probs)


def test_resolver_accepts_custom_policy_and_revalidates_output():
    class CustomPolicy:
        def action_distribution(self, contexts, *, actions, eligible_actions=None):
            del actions, eligible_actions
            return np.tile([0.25, 0.75], (len(contexts), 1))

    resolved = resolve_action_distribution(
        CustomPolicy(), np.zeros((3, 1)), actions=("a", "b")
    )
    np.testing.assert_allclose(resolved, np.tile([0.25, 0.75], (3, 1)))


def test_resolver_rejects_object_without_policy_contract():
    with pytest.raises(DataValidationError, match="must implement action_distribution"):
        resolve_action_distribution(object(), np.zeros((1, 1)), actions=("a", "b"))


def test_policy_output_cannot_bypass_eligibility_validation():
    class BadPolicy:
        def action_distribution(self, contexts, *, actions, eligible_actions=None):
            del contexts, actions, eligible_actions
            return np.array([[0.5, 0.5]])

    with pytest.raises(DataValidationError, match="ineligible action"):
        resolve_action_distribution(
            BadPolicy(),
            np.zeros((1, 1)),
            actions=("a", "b"),
            eligible_actions=np.array([[True, False]]),
        )
