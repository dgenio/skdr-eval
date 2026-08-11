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

    Construction itself is fail-closed: callers cannot bypass validation by
    instantiating this dataclass directly instead of using
    :func:`validate_logged_action_propensity`.

    Parameters
    ----------
    values:
        Probability assigned by the behavior/logging policy to the action that
        was actually taken on each row. Shape ``(n_rows,)`` with values in
        ``(0, 1]``.
    source:
        Provenance category. Only ``"logged"`` is accepted by this class. An
        estimated propensity needs a separate contract and validation status.
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

    def __post_init__(self) -> None:
        if self.source != "logged":
            raise DataValidationError(
                "LoggedActionPropensity.source must be exactly 'logged'; "
                "estimated propensities require a separate contract"
            )

        try:
            arr = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "logged action propensity must be numeric"
            ) from exc

        if arr.ndim != 1:
            raise DataValidationError(
                "logged action propensity must be a one-dimensional vector with "
                f"one value per decision; got shape {arr.shape}"
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

        field_name = self.field_name
        policy_id = self.policy_id
        if field_name is not None:
            field_name = str(field_name).strip()
            if not field_name:
                raise DataValidationError("field_name must be non-empty when supplied")
        if policy_id is not None:
            policy_id = str(policy_id).strip()
            if not policy_id:
                raise DataValidationError("policy_id must be non-empty when supplied")

        # Copy so caller mutation cannot alter an already-validated contract.
        stable = arr.copy()
        stable.setflags(write=False)
        object.__setattr__(self, "values", stable)
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "policy_id", policy_id)

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
    """

    if n_rows is not None and (not isinstance(n_rows, int) or n_rows < 0):
        raise DataValidationError(
            f"n_rows must be a non-negative integer or None; got {n_rows!r}"
        )

    result = LoggedActionPropensity(
        values=np.asarray(values),
        field_name=field_name,
        policy_id=policy_id,
    )
    if n_rows is not None and result.values.shape != (n_rows,):
        raise DataValidationError(
            f"logged action propensity must have shape ({n_rows},); "
            f"got {result.values.shape}"
        )
    return result


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
