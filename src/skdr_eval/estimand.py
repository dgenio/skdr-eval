"""Action-specific outcome terms for the corrected multi-action DR estimand.

For a finite-discrete target policy, doubly robust evaluation needs two distinct
outcome-model quantities (#58):

``q_obs[i] = q_hat(x_i, a_i)``
    Prediction for the action actually observed in the log.

``q_pi[i] = sum_a pi(a | x_i) * q_hat(x_i, a)``
    Prediction averaged under the explicit target action distribution.

A one-dimensional ``q_hat`` cannot generally represent both quantities. This
module therefore requires an explicit ``(n_rows, n_actions)`` action-value
matrix and consumes target probabilities through the canonical policy validator
introduced by #279.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import DataValidationError
from .policy import validate_action_distribution


@dataclass(frozen=True)
class ActionValueTerms:
    """Computed action-specific outcome terms for a validated action vocabulary."""

    actions: tuple[str, ...]
    q_by_action: np.ndarray
    q_obs: np.ndarray
    q_pi: np.ndarray


def compute_action_value_terms(
    q_by_action: np.ndarray,
    policy_probabilities: np.ndarray,
    observed_actions: np.ndarray,
    *,
    actions: list[str] | tuple[str, ...],
    eligible_actions: np.ndarray | None = None,
) -> ActionValueTerms:
    """Compute ``q_obs`` and ``q_pi`` from action-specific predictions.

    Parameters
    ----------
    q_by_action:
        Outcome predictions ``q_hat(x_i, a)`` with exact shape
        ``(n_rows, n_actions)``.
    policy_probabilities:
        Explicit target-policy probabilities. Validation is delegated to
        :func:`skdr_eval.policy.validate_action_distribution`, keeping one
        source of truth for action vocabulary, normalization, finiteness, and
        eligibility semantics.
    observed_actions:
        Integer column indices into ``actions`` with shape ``(n_rows,)``.
    actions:
        Ordered action vocabulary shared by ``q_by_action`` and the target
        policy matrix.
    eligible_actions:
        Optional boolean/0-1 eligibility matrix. In addition to the canonical
        policy checks, the logged action itself must be eligible on every row.

    Notes
    -----
    This helper deliberately rejects one-dimensional outcome predictions. The
    corrected kernel must construct action-specific ``q_hat(x,a)`` explicitly
    rather than relying on numpy broadcasting.
    """

    action_tuple = tuple(actions)
    if not action_tuple:
        raise DataValidationError("actions must contain at least one action")

    try:
        q = np.asarray(q_by_action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("q_by_action must be numeric") from exc

    if q.ndim != 2:
        raise DataValidationError(
            "q_by_action must be a 2D (n_rows, n_actions) matrix; a 1-D q_hat "
            "cannot define q_pi for a multi-action validated evaluation"
        )
    n_rows, n_actions = q.shape
    if n_actions != len(action_tuple):
        raise DataValidationError(
            "q_by_action columns must match the explicit action vocabulary: "
            f"got {n_actions} columns for {len(action_tuple)} actions"
        )
    if not np.all(np.isfinite(q)):
        bad = np.argwhere(~np.isfinite(q))[0]
        raise DataValidationError(
            "q_by_action must be finite; first invalid value at "
            f"row={int(bad[0])}, action={action_tuple[int(bad[1])]!r}"
        )

    pi = validate_action_distribution(
        policy_probabilities,
        actions=action_tuple,
        n_rows=n_rows,
        eligible_actions=eligible_actions,
    )

    observed = np.asarray(observed_actions)
    if observed.shape != (n_rows,):
        raise DataValidationError(
            f"observed_actions must have shape ({n_rows},); got {observed.shape}"
        )
    if not np.issubdtype(observed.dtype, np.integer):
        raise DataValidationError("observed_actions must contain integer action indices")
    action_idx = observed.astype(np.int64, copy=False)
    invalid = (action_idx < 0) | (action_idx >= n_actions)
    if np.any(invalid):
        row = int(np.flatnonzero(invalid)[0])
        raise DataValidationError(
            f"observed action index out of bounds at row {row}: "
            f"{int(action_idx[row])} not in [0, {n_actions})"
        )

    if eligible_actions is not None:
        # ``validate_action_distribution`` has already validated shape and
        # boolean/0-1 semantics, so this cast only supports the additional
        # logged-action eligibility invariant.
        elig = np.asarray(eligible_actions, dtype=bool)
        logged_eligible = elig[np.arange(n_rows), action_idx]
        if not np.all(logged_eligible):
            row = int(np.flatnonzero(~logged_eligible)[0])
            raise DataValidationError(
                "observed action must be eligible on every evaluated row; "
                f"row {row} action {action_tuple[int(action_idx[row])]!r} is not"
            )

    q_obs = q[np.arange(n_rows), action_idx]
    q_pi = np.sum(pi * q, axis=1)

    q_stable = q.copy()
    q_obs_stable = q_obs.copy()
    q_pi_stable = q_pi.copy()
    for array in (q_stable, q_obs_stable, q_pi_stable):
        array.setflags(write=False)

    return ActionValueTerms(
        actions=action_tuple,
        q_by_action=q_stable,
        q_obs=q_obs_stable,
        q_pi=q_pi_stable,
    )


__all__ = ["ActionValueTerms", "compute_action_value_terms"]
