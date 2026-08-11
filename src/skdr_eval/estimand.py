"""Validated multi-action outcome/estimand helpers for one-step DR OPE.

The validated multi-action DR estimand requires two distinct quantities (#58):

``q_obs[i] = q_hat(x_i, a_i)``
    Outcome-model prediction for the action actually observed in the log.

``q_pi[i] = sum_a pi(a | x_i) * q_hat(x_i, a)``
    Outcome-model prediction averaged under the explicit target policy.

A one-dimensional outcome prediction cannot generally represent both quantities
for a multi-action problem.  This module therefore requires an explicit
``(n_rows, n_actions)`` action-value matrix and refuses silent broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import DataValidationError


@dataclass(frozen=True)
class ActionValueTerms:
    """Action-specific outcome terms used by the DR contribution."""

    q_by_action: np.ndarray
    q_obs: np.ndarray
    q_pi: np.ndarray


def compute_action_value_terms(
    q_by_action: np.ndarray,
    policy_probabilities: np.ndarray,
    observed_actions: np.ndarray,
    *,
    eligible_actions: np.ndarray | None = None,
    atol: float = 1e-10,
) -> ActionValueTerms:
    """Validate action-specific predictions and compute ``q_obs`` / ``q_pi``.

    Parameters
    ----------
    q_by_action:
        Outcome-model predictions ``q_hat(x_i, a)`` with exact shape
        ``(n_rows, n_actions)``.
    policy_probabilities:
        Explicit target-policy probabilities with the same shape. Rows must be
        finite, non-negative and sum to one. Positive mass on ineligible actions
        is rejected when ``eligible_actions`` is supplied.
    observed_actions:
        Integer action indices with shape ``(n_rows,)``.
    eligible_actions:
        Optional boolean/0-1 matrix with the same shape. The logged action must
        be eligible on every row and target-policy mass on ineligible actions
        must be zero.
    atol:
        Floating-point tolerance for row-normalization / zero-mass checks.

    Returns
    -------
    ActionValueTerms
        Immutable bundle containing the validated action-value matrix plus
        ``q_obs`` and ``q_pi`` vectors.

    Notes
    -----
    This helper deliberately does not accept a one-dimensional ``q_hat`` for a
    multi-action problem.  Callers must construct action-specific predictions
    explicitly rather than relying on numpy broadcasting.
    """

    if not np.isfinite(atol) or atol < 0:
        raise DataValidationError(f"atol must be finite and non-negative; got {atol!r}")

    q = np.asarray(q_by_action, dtype=np.float64)
    pi = np.asarray(policy_probabilities, dtype=np.float64)
    actions = np.asarray(observed_actions)

    if q.ndim != 2:
        raise DataValidationError(
            "q_by_action must be a two-dimensional (n_rows, n_actions) matrix; "
            f"got shape {q.shape}. A 1-D q_hat cannot define q_pi for a "
            "multi-action validated evaluation."
        )
    if pi.shape != q.shape:
        raise DataValidationError(
            "policy_probabilities must have the same shape as q_by_action; "
            f"got {pi.shape} and {q.shape}"
        )
    n_rows, n_actions = q.shape
    if n_actions < 1:
        raise DataValidationError("q_by_action must contain at least one action column")
    if actions.shape != (n_rows,):
        raise DataValidationError(
            f"observed_actions must have shape ({n_rows},); got {actions.shape}"
        )
    if not np.issubdtype(actions.dtype, np.integer):
        # Floats such as 1.0 are ambiguous action identifiers; require the
        # caller to map the explicit action vocabulary to integer positions.
        raise DataValidationError("observed_actions must contain integer action indices")
    action_idx = actions.astype(np.int64, copy=False)
    if np.any((action_idx < 0) | (action_idx >= n_actions)):
        row = int(np.flatnonzero((action_idx < 0) | (action_idx >= n_actions))[0])
        raise DataValidationError(
            f"observed action index out of bounds at row {row}: "
            f"{int(action_idx[row])} not in [0, {n_actions})"
        )

    if not np.all(np.isfinite(q)):
        bad = np.argwhere(~np.isfinite(q))[0]
        raise DataValidationError(
            "q_by_action must be finite; first invalid value at "
            f"row={int(bad[0])}, action_index={int(bad[1])}"
        )
    if not np.all(np.isfinite(pi)):
        bad = np.argwhere(~np.isfinite(pi))[0]
        raise DataValidationError(
            "policy_probabilities must be finite; first invalid value at "
            f"row={int(bad[0])}, action_index={int(bad[1])}"
        )
    if np.any(pi < -atol):
        bad = np.argwhere(pi < -atol)[0]
        raise DataValidationError(
            "policy_probabilities must be non-negative; first invalid value at "
            f"row={int(bad[0])}, action_index={int(bad[1])}"
        )
    pi = np.where(np.abs(pi) <= atol, 0.0, pi)

    row_sums = pi.sum(axis=1)
    bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=0.0, atol=atol))
    if bad_rows.size:
        row = int(bad_rows[0])
        raise DataValidationError(
            "policy_probabilities must sum to 1 on every row; "
            f"row {row} sums to {float(row_sums[row])}"
        )

    if eligible_actions is not None:
        elig = np.asarray(eligible_actions)
        if elig.shape != q.shape:
            raise DataValidationError(
                "eligible_actions must have the same shape as q_by_action; "
                f"got {elig.shape} and {q.shape}"
            )
        if elig.dtype != np.bool_:
            if not np.all(np.isin(elig, (0, 1))):
                raise DataValidationError(
                    "eligible_actions must contain only booleans or 0/1 values"
                )
            elig = elig.astype(bool)
        else:
            elig = elig.astype(bool, copy=False)

        logged_eligible = elig[np.arange(n_rows), action_idx]
        if not np.all(logged_eligible):
            row = int(np.flatnonzero(~logged_eligible)[0])
            raise DataValidationError(
                "observed action must be eligible on every evaluated row; "
                f"row {row} is not"
            )
        invalid_mass = np.argwhere((~elig) & (pi > atol))
        if invalid_mass.size:
            bad = invalid_mass[0]
            raise DataValidationError(
                "target policy assigns positive mass to an ineligible action; "
                f"row={int(bad[0])}, action_index={int(bad[1])}"
            )

    q_obs = q[np.arange(n_rows), action_idx]
    q_pi = np.sum(pi * q, axis=1)

    # Copy/lock outputs so the returned terms remain consistent after
    # validation even if callers mutate their original arrays.
    q_stable = q.copy()
    q_obs_stable = q_obs.copy()
    q_pi_stable = q_pi.copy()
    for arr in (q_stable, q_obs_stable, q_pi_stable):
        arr.setflags(write=False)

    return ActionValueTerms(
        q_by_action=q_stable,
        q_obs=q_obs_stable,
        q_pi=q_pi_stable,
    )


__all__ = ["ActionValueTerms", "compute_action_value_terms"]
