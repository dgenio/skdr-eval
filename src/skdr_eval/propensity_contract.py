"""Behavior-propensity contracts for the validated one-step OPE path.

The first validated-v1 path requires the probability of the *observed action*
to have been recorded by the logging policy at decision time (#167 / #297).
This module deliberately models that quantity as a one-dimensional vector
rather than fabricating a dense behavior-policy matrix that the logs may never
have contained.

Estimated behavior propensities remain a separate, validation-pending workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exceptions import DataValidationError


@dataclass(frozen=True)
class LoggedActionPropensity:
    """Validated logged probability of the observed action.

    Parameters
    ----------
    values:
        Probability assigned by the behavior/logging policy to the action that
        was actually taken on each row. Shape ``(n_rows,)`` with values in
        ``(0, 1]``.
    source:
        Provenance category. ``"logged"`` means captured at decision time and
        is the only source eligible for the initial validated-v1 path.
    field_name:
        Optional column/input identifier such as ``"propensity"``. This is
        metadata only; raw data are not stored here.
    policy_id:
        Optional safe identifier for the logging-policy version/configuration.
    """

    values: np.ndarray
    source: Literal["logged"] = "logged"
    field_name: str | None = None
    policy_id: str | None = None

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])


def validate_logged_action_propensity(
    values: np.ndarray,
    *,
    n_rows: int | None = None,
    field_name: str | None = None,
    policy_id: str | None = None,
) -> LoggedActionPropensity:
    """Validate logged probabilities for the observed actions.

    This function performs **no clipping, filling, smoothing, estimation, or
    renormalization**. Missing or invalid behavior probabilities are a data
    contract failure for the validated path.

    Parameters
    ----------
    values:
        One probability per evaluated decision: ``P_behavior(A_i | X_i)``.
    n_rows:
        Optional expected row count. When supplied, shape must be exactly
        ``(n_rows,)``.
    field_name:
        Optional source column/input name for provenance.
    policy_id:
        Optional safe logging-policy version identifier.

    Returns
    -------
    LoggedActionPropensity
        Immutable validated propensity object.

    Raises
    ------
    DataValidationError
        If dimensionality, row count, finiteness, missingness, or probability
        bounds violate the logged-propensity contract.
    """

    if n_rows is not None and (not isinstance(n_rows, int) or n_rows < 0):
        raise DataValidationError(
            f"n_rows must be a non-negative integer or None; got {n_rows!r}"
        )

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise DataValidationError(
            "logged action propensity must be a one-dimensional vector with "
            f"one value per decision; got shape {arr.shape}"
        )
    if n_rows is not None and arr.shape != (n_rows,):
        raise DataValidationError(
            f"logged action propensity must have shape ({n_rows},); got {arr.shape}"
        )

    if not np.all(np.isfinite(arr)):
        bad = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise DataValidationError(
            "logged action propensity must contain only finite values; "
            f"first invalid row is {bad}"
        )

    invalid = np.flatnonzero((arr <= 0.0) | (arr > 1.0))
    if invalid.size:
        row = int(invalid[0])
        raise DataValidationError(
            "logged action propensity must lie in (0, 1] for every evaluated "
            f"decision; row {row} has {float(arr[row])}"
        )

    if field_name is not None and not str(field_name).strip():
        raise DataValidationError("field_name must be non-empty when supplied")
    if policy_id is not None and not str(policy_id).strip():
        raise DataValidationError("policy_id must be non-empty when supplied")

    # Copy so caller mutation cannot change an already-validated contract.
    stable = arr.copy()
    stable.setflags(write=False)
    return LoggedActionPropensity(
        values=stable,
        field_name=None if field_name is None else str(field_name),
        policy_id=None if policy_id is None else str(policy_id),
    )


def observed_importance_ratio(
    target_action_probability: np.ndarray,
    behavior: LoggedActionPropensity,
) -> np.ndarray:
    """Compute the unclipped observed-action ratio ``pi(A|x) / e(A|x)``.

    This small helper makes the validated quantity explicit without requiring a
    dense behavior-policy matrix. Target probabilities are still validated at
    the policy boundary (#279); here we only enforce row alignment and finite
    ``[0, 1]`` values defensively.
    """

    target = np.asarray(target_action_probability, dtype=np.float64)
    if target.shape != behavior.values.shape:
        raise DataValidationError(
            "target observed-action probabilities and logged behavior "
            f"propensities must have identical shape; got {target.shape} and "
            f"{behavior.values.shape}"
        )
    if not np.all(np.isfinite(target)):
        raise DataValidationError(
            "target observed-action probabilities must be finite"
        )
    if np.any((target < 0.0) | (target > 1.0)):
        raise DataValidationError(
            "target observed-action probabilities must lie in [0, 1]"
        )
    return target / behavior.values


__all__ = [
    "LoggedActionPropensity",
    "observed_importance_ratio",
    "validate_logged_action_propensity",
]
