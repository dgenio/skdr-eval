"""Built-in :class:`OutcomeLoss` strategies (#86).

An ``OutcomeLoss`` returns the per-sample weights used inside the outcome
model's cross-fit. ``MSEOutcomeLoss`` is the ordinary unweighted least
squares loss (this matches the historical default); ``MRDRWeightedLoss``
returns ``w^2``, which under DR yields the variance-minimising More-Robust
Doubly Robust (MRDR) estimator (Farajtabar, Chow & Ghavamzadeh 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .protocols import OutcomeLoss

__all__ = [
    "MRDRWeightedLoss",
    "MSEOutcomeLoss",
]


@dataclass(frozen=True)
class MSEOutcomeLoss:
    """Unweighted MSE — historical default; recovers the DR fit."""

    name: str = "mse"

    def __call__(
        self,
        *,
        pi_obs: np.ndarray,
        policy_probs: np.ndarray,  # noqa: ARG002 — kept for protocol parity
        A: np.ndarray,  # noqa: ARG002 — kept for protocol parity
    ) -> np.ndarray:
        return np.ones_like(pi_obs, dtype=np.float64)


@dataclass(frozen=True)
class MRDRWeightedLoss:
    """MRDR sample weights ``(π(A|x) / e(A|x))^2``.

    Farajtabar, Chow & Ghavamzadeh (2018) show that minimising the weighted
    MSE under these sample weights produces the variance-minimising outcome
    model for the DR estimator. The factor of ``π / e`` is squared because the
    DR estimator's variance is proportional to ``w^2 · Var(Y - q̂ | X, A)``.

    Notes
    -----
    Weights are clipped against ``clip_floor`` to keep small-propensity rows
    from dominating the fit numerically. The default floor mirrors the
    weakest finite entry in the DR clip grid.
    """

    clip_floor: float = 1e-3
    name: str = "mrdr"

    def __call__(
        self, *, pi_obs: np.ndarray, policy_probs: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        n = pi_obs.shape[0]
        rows = np.arange(n)
        a_int = A.astype(int)
        pi_target = policy_probs[rows, a_int]
        denom = np.maximum(pi_obs, self.clip_floor)
        weights = (pi_target / denom) ** 2
        # MRDR weights are positive; clip to a finite ceiling to avoid
        # numeric instability if the user supplies very small propensities.
        weights = np.minimum(weights, 1.0 / (self.clip_floor**2))
        return weights.astype(np.float64)


def _assert_protocols() -> None:  # pragma: no cover - type-check only
    _: OutcomeLoss = MSEOutcomeLoss()
    _ = MRDRWeightedLoss()
