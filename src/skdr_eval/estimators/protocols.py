"""Strategy protocols for the DR/SNDR estimator family (#86).

Two pluggable seams replace the previously hard-coded clip + MSE pair inside
:func:`skdr_eval.core.dr_value_with_clip`:

* :class:`WeightTransform` — maps the raw inverse-propensity vector
  ``1 / pi_obs`` into the working importance weights used inside the DR
  pseudo-outcome. ``ClipTransform`` is the original behaviour; ``SwitchTau``
  forces the weight to ``0`` above a threshold (SWITCH-DR); ``DRosShrink``
  applies optimistic shrinkage ``w * lam / (w**2 + lam)``; ``MIPSTransform``
  marginalises over an action-embedding sufficient statistic (#85).
* :class:`OutcomeLoss` — emits a sample-weight vector for the outcome model's
  cross-fitted regression. ``MSEOutcomeLoss`` is the historical default;
  ``MRDRWeightedLoss`` returns ``w**2`` to recover the variance-minimising
  MRDR fit.

The protocols are deliberately minimal so callers can implement custom
strategies without depending on ``skdr_eval`` internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "EstimatorStrategy",
    "OutcomeLoss",
    "TransformContext",
    "WeightTransform",
]


@dataclass(frozen=True)
class TransformContext:
    """Per-call context handed to a :class:`WeightTransform`.

    Attributes
    ----------
    pi_obs : np.ndarray
        Logging propensity for the observed action, shape ``(n,)``. Strictly
        positive on the matched subset; zero entries flag unmatched rows.
    matched : np.ndarray
        Boolean mask of length ``n`` selecting rows with positive logging
        propensity and an eligible observed action.
    policy_probs : np.ndarray
        Target-policy probability matrix, shape ``(n, n_actions)``.
    A : np.ndarray
        Observed action indices, shape ``(n,)``.
    elig : np.ndarray
        Eligibility matrix, shape ``(n, n_actions)``.
    propensities : np.ndarray
        Full logging-policy propensity matrix, shape ``(n, n_actions)``.
        Required by MIPS to compute the embedding-marginal logging density.
    action_embedding : np.ndarray or None
        Optional action embedding for MIPS, shape ``(n_actions, embed_dim)``
        or ``(n, embed_dim)`` for per-decision embeddings. ``None`` when the
        caller did not provide one.
    """

    pi_obs: np.ndarray
    matched: np.ndarray
    policy_probs: np.ndarray
    A: np.ndarray
    elig: np.ndarray
    propensities: np.ndarray
    action_embedding: np.ndarray | None = None


class WeightTransform(Protocol):
    """Maps logging propensity into a working importance weight.

    A ``WeightTransform`` must return a non-negative array of shape ``(n,)``
    with zeros on the unmatched rows. The DR pseudo-outcome then becomes
    ``q_pi + w * (Y - q_hat)`` with the transformed ``w``. Concrete
    implementations typically expose a ``name`` attribute for diagnostics,
    but the protocol intentionally does not require it so dataclass
    implementations can supply ``name`` as a defaulted field without mypy
    treating it as a writable Protocol member.
    """

    def __call__(self, context: TransformContext) -> np.ndarray:
        """Return the working weight vector."""
        ...


class OutcomeLoss(Protocol):
    """Emits per-sample weights for the outcome-model cross-fit.

    The cross-fit minimises ``Σ_i sample_weight_i · (Y_i - q̂(X_i))^2``.
    Returning a constant array recovers ordinary MSE; returning ``w_i^2``
    recovers MRDR (Farajtabar et al. 2018).
    """

    def __call__(
        self, *, pi_obs: np.ndarray, policy_probs: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        """Return per-sample weights, shape ``(n,)``, non-negative."""
        ...


@dataclass(frozen=True)
class EstimatorStrategy:
    """A named pairing of a :class:`WeightTransform` and an :class:`OutcomeLoss`.

    The estimator name is what appears in the report ``estimator`` column.
    Built-in strategies map to:

    * ``"DR"`` — ``ClipTransform(grid)`` + ``MSEOutcomeLoss``
    * ``"SNDR"`` — ``ClipTransform(grid)`` + ``MSEOutcomeLoss`` + self-normalised pool
    * ``"MRDR"`` — ``ClipTransform(grid)`` + ``MRDRWeightedLoss``
    * ``"SWITCH-DR"`` — ``SwitchTauTransform(tau)`` + ``MSEOutcomeLoss``
    * ``"DRos"`` — ``DRosShrinkTransform(lam)`` + ``MSEOutcomeLoss``
    * ``"MIPS"`` — ``MIPSTransform(embedding)`` + ``MSEOutcomeLoss``
    """

    name: str
    weight_transform: WeightTransform
    outcome_loss: OutcomeLoss
    self_normalised: bool = False
