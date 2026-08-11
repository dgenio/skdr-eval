"""Tests for the explicit target-policy contract (#279)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.exceptions import DataValidationError
from skdr_eval.policy import (
    CallablePolicy,
    Policy,
    resolve_action_distribution,
    validate_action_distribution,
)


def _contexts(n: int = 3) -> np.ndarray:
    return np.arange(n * 2, dtype=np.float64).reshape(n, 2)


def test_explicit_matrix_round_trips_without_renormalization() -> None:
    probs = np.array(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
    )
    resolved = resolve_action_distribution(
        probs,
        _contexts(),
        actions=("a", "b"),
    )
    np.testing.assert_array_equal(resolved, probs)


def test_deterministic_policy_is_one_hot_distribution() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = validate_action_distribution(
        probs,
        n_rows=2,
        actions=("a", "b"),
    )
    np.testing.assert_array_equal(out, probs)


def test_callable_policy_satisfies_protocol_and_receives_vocabulary() -> None:
    seen: dict[str, object] = {}

    def fn(
        contexts: np.ndarray,
        actions,
        eligible_actions: np.ndarray | None,
    ) -> np.ndarray:
        seen["shape"] = contexts.shape
        seen["actions"] = tuple(actions)
        seen["elig"] = eligible_actions
        return np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])

    policy = CallablePolicy(fn=fn, name="candidate")
    assert isinstance(policy, Policy)

    elig = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
    out = resolve_action_distribution(
        policy,
        _contexts(),
        actions=("left", "right"),
        eligible_actions=elig,
    )
    assert seen["shape"] == (3, 2)
    assert seen["actions"] == ("left", "right")
    assert seen["elig"] is elig
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_duplicate_action_labels_rejected() -> None:
    with pytest.raises(DataValidationError, match="unique"):
        validate_action_distribution(
            np.array([[0.5, 0.5]]),
            n_rows=1,
            actions=("a", "a"),
        )


def test_wrong_shape_rejected() -> None:
    with pytest.raises(DataValidationError, match="shape"):
        validate_action_distribution(
            np.array([[1.0, 0.0, 0.0]]),
            n_rows=1,
            actions=("a", "b"),
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_probability_rejected(bad: float) -> None:
    probs = np.array([[1.0, 0.0], [bad, 0.0]])
    with pytest.raises(DataValidationError, match="finite"):
        validate_action_distribution(
            probs,
            n_rows=2,
            actions=("a", "b"),
        )


def test_material_negative_probability_rejected() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_action_distribution(
            np.array([[1.1, -0.1]]),
            n_rows=1,
            actions=("a", "b"),
        )


def test_tiny_float_noise_is_zeroed_but_not_renormalized() -> None:
    probs = np.array([[1.0 + 1e-12, -1e-12]])
    out = validate_action_distribution(
        probs,
        n_rows=1,
        actions=("a", "b"),
        atol=1e-10,
    )
    np.testing.assert_array_equal(out, np.array([[1.0 + 1e-12, 0.0]]))


def test_non_normalized_row_rejected_instead_of_repaired() -> None:
    with pytest.raises(DataValidationError, match="sum to 1"):
        validate_action_distribution(
            np.array([[0.4, 0.4]]),
            n_rows=1,
            actions=("a", "b"),
        )


def test_positive_mass_on_ineligible_action_rejected() -> None:
    with pytest.raises(DataValidationError, match="ineligible"):
        validate_action_distribution(
            np.array([[0.8, 0.2]]),
            n_rows=1,
            actions=("a", "b"),
            eligible_actions=np.array([[1, 0]]),
        )


def test_numeric_zero_one_eligibility_is_accepted() -> None:
    out = validate_action_distribution(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        n_rows=2,
        actions=("a", "b"),
        eligible_actions=np.array([[1, 0], [0, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(out, np.eye(2))


def test_non_binary_eligibility_rejected() -> None:
    with pytest.raises(DataValidationError, match="0/1"):
        validate_action_distribution(
            np.array([[1.0, 0.0]]),
            n_rows=1,
            actions=("a", "b"),
            eligible_actions=np.array([[1.0, 0.5]]),
        )


def test_empty_eligibility_row_fails_closed() -> None:
    with pytest.raises(DataValidationError, match="at least one eligible"):
        validate_action_distribution(
            np.array([[1.0, 0.0]]),
            n_rows=1,
            actions=("a", "b"),
            eligible_actions=np.array([[0, 0]]),
        )


def test_policy_without_action_distribution_rejected() -> None:
    class NotAPolicy:
        pass

    with pytest.raises(DataValidationError, match="action_distribution"):
        resolve_action_distribution(
            NotAPolicy(),  # type: ignore[arg-type]
            _contexts(1),
            actions=("a", "b"),
        )


def test_policy_failure_is_wrapped_as_data_validation_error() -> None:
    class BrokenPolicy:
        def action_distribution(
            self,
            contexts: np.ndarray,
            *,
            actions,
            eligible_actions=None,
        ) -> np.ndarray:
            del contexts, actions, eligible_actions
            raise RuntimeError("boom")

    with pytest.raises(DataValidationError, match="boom"):
        resolve_action_distribution(
            BrokenPolicy(),
            _contexts(1),
            actions=("a", "b"),
        )
