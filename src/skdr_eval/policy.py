"""Explicit target-policy contract for one-step offline policy evaluation.

The validated OPE path must evaluate the policy the caller actually intends to
study, not infer a policy implicitly from arbitrary model scores. This module
provides the small representation/validation seam tracked by issue #279 without
changing any estimator mathematics.

The contract is intentionally backend-neutral so native estimators and external
reference implementations can consume the exact same action distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .exceptions import DataValidationError


@runtime_checkable
class Policy(Protocol):
    """One-step target policy represented as action probabilities.

    Implementations return a matrix with shape ``(n_rows, n_actions)`` in the
    exact order supplied by ``actions``. Deterministic policies use one-hot
    rows; stochastic policies use any valid probability simplex.

    ``eligible_actions`` is an optional boolean matrix with the same shape.
    Policies must assign zero probability to ineligible actions.
    """

    def action_distribution(
        self,
        contexts: Any,
        *,
        actions: list[str] | tuple[str, ...],
        eligible_actions: np.ndarray | None = None,
    ) -> np.ndarray: ...


def _n_rows(contexts: Any) -> int:
    """Return a context row count without constraining the frame backend."""
    try:
        return len(contexts)
    except TypeError as exc:  # pragma: no cover - defensive protocol boundary
        raise DataValidationError("contexts must be a sized row collection") from exc


def validate_action_distribution(
    probabilities: Any,
    *,
    actions: list[str] | tuple[str, ...],
    n_rows: int | None = None,
    eligible_actions: np.ndarray | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """Validate the target-policy representation without silently repairing it.

    Parameters
    ----------
    probabilities:
        Candidate ``(n_rows, n_actions)`` probability matrix.
    actions:
        Ordered action vocabulary that defines the matrix columns.
    n_rows:
        Optional expected row count.
    eligible_actions:
        Optional boolean matrix. Ineligible actions must receive probability
        zero (within ``atol``), and every row must contain at least one eligible
        action.
    atol:
        Absolute tolerance for row normalization and eligibility-zero checks.

    Returns
    -------
    np.ndarray
        A float64 probability matrix after validation. Values are not clipped or
        renormalized: an invalid target policy fails closed rather than being
        silently repaired.
    """
    action_tuple = tuple(actions)
    if not action_tuple:
        raise DataValidationError("actions must contain at least one action")
    if len(set(action_tuple)) != len(action_tuple):
        raise DataValidationError("actions must be unique and order-stable")
    if any(not isinstance(action, str) or not action for action in action_tuple):
        raise DataValidationError("actions must be non-empty strings")

    try:
        probs = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("policy probabilities must be numeric") from exc

    if probs.ndim != 2:
        raise DataValidationError(
            "policy probabilities must be a 2D (n_rows, n_actions) matrix"
        )
    if probs.shape[1] != len(action_tuple):
        raise DataValidationError(
            "policy probability columns do not match the supplied action vocabulary: "
            f"got {probs.shape[1]} columns for {len(action_tuple)} actions"
        )
    if n_rows is not None and probs.shape[0] != n_rows:
        raise DataValidationError(
            f"policy probability rows must match contexts: got {probs.shape[0]}, "
            f"expected {n_rows}"
        )
    if not np.all(np.isfinite(probs)):
        raise DataValidationError("policy probabilities must all be finite")
    if np.any(probs < -atol):
        raise DataValidationError("policy probabilities must be non-negative")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        bad = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=atol, rtol=0.0))
        preview = bad[:5].tolist()
        raise DataValidationError(
            "each policy probability row must sum to 1; "
            f"invalid row indices include {preview}"
        )

    if eligible_actions is not None:
        elig = np.asarray(eligible_actions)
        if elig.shape != probs.shape:
            raise DataValidationError(
                "eligible_actions must have the same shape as policy probabilities"
            )
        if elig.dtype != np.bool_:
            # Accept clean 0/1 masks, but reject arbitrary truthy numeric values.
            if not np.all(np.isin(elig, [0, 1])):
                raise DataValidationError(
                    "eligible_actions must be boolean or contain only 0/1 values"
                )
            elig = elig.astype(bool)
        if np.any(elig.sum(axis=1) == 0):
            raise DataValidationError("every row must have at least one eligible action")
        if np.any(np.abs(probs[~elig]) > atol):
            raise DataValidationError(
                "target policy assigns non-zero probability to an ineligible action"
            )

    # Avoid propagating harmless negative signed-zero / tiny round-off values
    # without silently correcting genuinely invalid rows (caught above).
    result = probs.copy()
    result[np.abs(result) <= atol] = 0.0
    return result


@dataclass(frozen=True)
class ExplicitPolicy:
    """An explicit target policy backed by a fixed probability matrix.

    This class is primarily useful when candidate action probabilities have
    already been computed by another model/service. It binds the probability
    columns to an explicit action vocabulary so action reordering cannot be
    silently accepted. The stored probability matrix is copied and marked
    read-only so policy state cannot change after construction.
    """

    probabilities: np.ndarray
    actions: tuple[str, ...]
    name: str | None = None

    def __init__(
        self,
        probabilities: Any,
        *,
        actions: list[str] | tuple[str, ...],
        name: str | None = None,
    ) -> None:
        action_tuple = tuple(actions)
        validated = validate_action_distribution(probabilities, actions=action_tuple)
        validated.setflags(write=False)
        object.__setattr__(self, "probabilities", validated)
        object.__setattr__(self, "actions", action_tuple)
        object.__setattr__(self, "name", name)

    def action_distribution(
        self,
        contexts: Any,
        *,
        actions: list[str] | tuple[str, ...],
        eligible_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        requested = tuple(actions)
        if requested != self.actions:
            raise DataValidationError(
                "requested action vocabulary/order does not match ExplicitPolicy: "
                f"expected {self.actions!r}, got {requested!r}"
            )
        return validate_action_distribution(
            self.probabilities,
            actions=requested,
            n_rows=_n_rows(contexts),
            eligible_actions=eligible_actions,
        )


def resolve_action_distribution(
    policy: Policy | np.ndarray,
    contexts: Any,
    *,
    actions: list[str] | tuple[str, ...],
    eligible_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve a Policy object or explicit matrix through one validation seam.

    Future native/reference evaluators should call this helper rather than
    interpreting model scores independently. Passing a raw matrix is supported
    as a convenience but receives the same validation as a ``Policy`` object.
    """
    n_rows = _n_rows(contexts)
    if isinstance(policy, np.ndarray):
        return validate_action_distribution(
            policy,
            actions=actions,
            n_rows=n_rows,
            eligible_actions=eligible_actions,
        )

    action_distribution = getattr(policy, "action_distribution", None)
    if action_distribution is None or not callable(action_distribution):
        raise DataValidationError(
            "policy must implement action_distribution(...) or be a numpy matrix"
        )
    probabilities = action_distribution(
        contexts,
        actions=actions,
        eligible_actions=eligible_actions,
    )
    return validate_action_distribution(
        probabilities,
        actions=actions,
        n_rows=n_rows,
        eligible_actions=eligible_actions,
    )


__all__ = [
    "ExplicitPolicy",
    "Policy",
    "resolve_action_distribution",
    "validate_action_distribution",
]
