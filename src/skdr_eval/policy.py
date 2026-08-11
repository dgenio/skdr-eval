"""Explicit target-policy contracts for one-step offline policy evaluation.

This module defines the policy boundary required by the validated-v1 roadmap
(#279 / #297).  A target policy is represented by the probability distribution
it would use over the *explicit* action vocabulary for each evaluated context.

The contract is deliberately independent of outcome-model scores.  A regression
or ranking model may be adapted into a :class:`Policy`, but the validated OPE
kernel must consume the resulting action distribution rather than infer a policy
from arbitrary prediction semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .exceptions import DataValidationError


@runtime_checkable
class Policy(Protocol):
    """Protocol for an explicit finite-discrete target policy.

    Implementations return one probability row per context and one column per
    action in the supplied ``actions`` vocabulary.  Deterministic policies use
    one-hot rows; stochastic policies return ordinary categorical
    distributions.

    ``eligible_actions`` is an optional boolean matrix with shape
    ``(n_rows, n_actions)``.  When supplied, implementations must put exactly
    zero probability on ineligible actions.
    """

    def action_distribution(
        self,
        contexts: np.ndarray,
        *,
        actions: Sequence[str],
        eligible_actions: np.ndarray | None = None,
    ) -> np.ndarray: ...


PolicyCallable = Callable[
    [np.ndarray, Sequence[str], np.ndarray | None],
    np.ndarray,
]


@dataclass(frozen=True)
class CallablePolicy:
    """Small adapter for a callable that already produces action probabilities.

    The callable receives ``(contexts, actions, eligible_actions)`` and must
    return an ``(n_rows, n_actions)`` array.  Output validation is performed by
    :func:`resolve_action_distribution`, not by this adapter, so every policy
    implementation is held to the same contract.
    """

    fn: PolicyCallable
    name: str | None = None

    def action_distribution(
        self,
        contexts: np.ndarray,
        *,
        actions: Sequence[str],
        eligible_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.asarray(self.fn(contexts, actions, eligible_actions), dtype=np.float64)


def _coerce_eligibility(
    eligible_actions: np.ndarray | None,
    *,
    n_rows: int,
    n_actions: int,
) -> np.ndarray:
    """Return a validated boolean eligibility matrix.

    ``None`` means every action is eligible.  A row with no eligible actions is
    invalid for the explicit-policy contract: there is no categorical
    distribution that can both sum to one and assign zero probability to every
    ineligible action.
    """

    if eligible_actions is None:
        return np.ones((n_rows, n_actions), dtype=bool)

    eligibility = np.asarray(eligible_actions)
    expected = (n_rows, n_actions)
    if eligibility.shape != expected:
        raise DataValidationError(
            "eligible_actions must have shape "
            f"{expected}; got {eligibility.shape}"
        )

    if eligibility.dtype == np.bool_:
        eligibility_bool = eligibility.astype(bool, copy=False)
    else:
        if not np.all(np.isin(eligibility, (0, 1))):
            raise DataValidationError(
                "eligible_actions must contain only booleans or 0/1 values"
            )
        eligibility_bool = eligibility.astype(bool)

    empty_rows = np.flatnonzero(~eligibility_bool.any(axis=1))
    if empty_rows.size:
        preview = empty_rows[:5].tolist()
        raise DataValidationError(
            "Every evaluated row must have at least one eligible action; "
            f"rows without eligible actions include {preview}"
        )

    return eligibility_bool


def validate_action_distribution(
    probabilities: np.ndarray,
    *,
    n_rows: int,
    actions: Sequence[str],
    eligible_actions: np.ndarray | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """Validate and return a target-policy action distribution.

    Parameters
    ----------
    probabilities:
        Candidate probability matrix with shape ``(n_rows, n_actions)``.
    n_rows:
        Number of evaluated contexts.
    actions:
        Explicit action vocabulary.  Column ``j`` in ``probabilities`` refers
        to ``actions[j]``; duplicate action labels are rejected.
    eligible_actions:
        Optional boolean/0-1 eligibility matrix.  Probability on an ineligible
        action is a hard error rather than being silently renormalized away.
    atol:
        Absolute tolerance used only for floating-point normalization and
        effectively-zero eligibility checks.

    Returns
    -------
    np.ndarray
        A float64 matrix satisfying the explicit policy contract.  The function
        does **not** silently clip, fill, or renormalize invalid input.

    Raises
    ------
    DataValidationError
        If shape, action vocabulary, finiteness, non-negativity,
        normalization, or eligibility constraints are violated.
    """

    if not isinstance(n_rows, int) or n_rows < 0:
        raise DataValidationError(f"n_rows must be a non-negative integer; got {n_rows!r}")
    if not np.isfinite(atol) or atol < 0:
        raise DataValidationError(f"atol must be finite and non-negative; got {atol!r}")

    action_tuple = tuple(actions)
    if not action_tuple:
        raise DataValidationError("actions must contain at least one action")
    if len(set(action_tuple)) != len(action_tuple):
        raise DataValidationError("actions must be unique and ordered explicitly")

    n_actions = len(action_tuple)
    probs = np.asarray(probabilities, dtype=np.float64)
    expected = (n_rows, n_actions)
    if probs.shape != expected:
        raise DataValidationError(
            f"policy probabilities must have shape {expected}; got {probs.shape}"
        )

    if not np.all(np.isfinite(probs)):
        bad = np.argwhere(~np.isfinite(probs))[0]
        raise DataValidationError(
            "policy probabilities must be finite; first non-finite value at "
            f"row={int(bad[0])}, action={action_tuple[int(bad[1])]!r}"
        )

    negative = np.argwhere(probs < -atol)
    if negative.size:
        bad = negative[0]
        raise DataValidationError(
            "policy probabilities must be non-negative; first negative value at "
            f"row={int(bad[0])}, action={action_tuple[int(bad[1])]!r}"
        )

    # Tiny negative floating-point noise inside tolerance is accepted as zero,
    # but material invalidity is never repaired silently.
    probs = np.where(np.abs(probs) <= atol, 0.0, probs)

    eligibility = _coerce_eligibility(
        eligible_actions,
        n_rows=n_rows,
        n_actions=n_actions,
    )
    ineligible_mass = np.argwhere((~eligibility) & (probs > atol))
    if ineligible_mass.size:
        bad = ineligible_mass[0]
        raise DataValidationError(
            "target policy assigns positive probability to an ineligible action; "
            f"row={int(bad[0])}, action={action_tuple[int(bad[1])]!r}, "
            f"probability={float(probs[tuple(bad)])}"
        )

    row_sums = probs.sum(axis=1)
    bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=0.0, atol=atol))
    if bad_rows.size:
        row = int(bad_rows[0])
        raise DataValidationError(
            "target-policy probabilities must sum to 1 on every row; "
            f"row {row} sums to {float(row_sums[row])}"
        )

    return probs


def resolve_action_distribution(
    policy: Policy | np.ndarray,
    contexts: np.ndarray,
    *,
    actions: Sequence[str],
    eligible_actions: np.ndarray | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """Resolve a policy object or explicit matrix into validated probabilities.

    This is the intended seam between policy construction and the statistical
    kernel.  The kernel should consume only the returned matrix and explicit
    action vocabulary; it should not reinterpret model scores.
    """

    contexts_array = np.asarray(contexts)
    if contexts_array.ndim == 0:
        raise DataValidationError("contexts must have a row dimension")
    n_rows = int(contexts_array.shape[0])

    if isinstance(policy, np.ndarray):
        raw = policy
    else:
        action_distribution = getattr(policy, "action_distribution", None)
        if action_distribution is None or not callable(action_distribution):
            raise DataValidationError(
                "policy must be an explicit probability matrix or implement "
                "action_distribution(contexts, *, actions, eligible_actions)"
            )
        try:
            raw = action_distribution(
                contexts_array,
                actions=tuple(actions),
                eligible_actions=eligible_actions,
            )
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataValidationError(
                f"policy.action_distribution failed: {exc}"
            ) from exc

    return validate_action_distribution(
        np.asarray(raw),
        n_rows=n_rows,
        actions=actions,
        eligible_actions=eligible_actions,
        atol=atol,
    )


__all__ = [
    "CallablePolicy",
    "Policy",
    "PolicyCallable",
    "resolve_action_distribution",
    "validate_action_distribution",
]
