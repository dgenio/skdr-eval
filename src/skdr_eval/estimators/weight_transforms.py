"""Built-in :class:`WeightTransform` strategies (#86, #85).

Each transform is a pure function of the logging propensity (and optionally
an action embedding) that returns the per-decision working weight ``w``.
The DR pseudo-outcome is then ``q_pi + w * (Y - q_hat)`` and the SNDR pool is
``Σ w * (Y - q_hat) / Σ w``.

All transforms emit ``0`` on the unmatched subset so callers can sum without
masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .protocols import TransformContext, WeightTransform

__all__ = [
    "ClipTransform",
    "DRosShrinkTransform",
    "IdentityTransform",
    "MIPSTransform",
    "SwitchTauTransform",
]


def _raw_inverse(pi_obs: np.ndarray, matched: np.ndarray) -> np.ndarray:
    """Return ``1 / pi_obs`` with zeros on the unmatched rows."""
    w = np.zeros_like(pi_obs, dtype=np.float64)
    safe = matched & (pi_obs > 0)
    w[safe] = 1.0 / pi_obs[safe]
    return w


@dataclass(frozen=True)
class IdentityTransform:
    """Unclipped inverse-propensity weight.

    Useful as a numerical baseline (matches ``ClipTransform(clip=inf)``) but
    not recommended for production — variance explodes on small propensity
    tails.
    """

    name: str = "identity"

    def __call__(self, context: TransformContext) -> np.ndarray:
        return _raw_inverse(context.pi_obs, context.matched)


@dataclass(frozen=True)
class ClipTransform:
    """Hard-clipped inverse-propensity weight ``min(1/pi, clip)`` (DR/SNDR).

    Parameters
    ----------
    clip : float
        Upper bound on the working weight. ``float("inf")`` disables clipping
        and recovers :class:`IdentityTransform`.
    """

    clip: float
    name: str = "clip"

    def __call__(self, context: TransformContext) -> np.ndarray:
        w = _raw_inverse(context.pi_obs, context.matched)
        if np.isfinite(self.clip):
            w = np.minimum(w, self.clip)
        return w


@dataclass(frozen=True)
class SwitchTauTransform:
    """SWITCH-DR weight (Wang, Agarwal & Dudík 2017).

    When the raw IPS weight ``1/pi_obs`` exceeds ``tau`` the importance term
    is zeroed, so the DR pseudo-outcome collapses to the direct method
    ``q_pi``. This trades bias for variance on heavy propensity tails.
    """

    tau: float
    name: str = "switch_tau"

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError(f"tau must be a positive finite float, got {self.tau!r}")

    def __call__(self, context: TransformContext) -> np.ndarray:
        w_raw = _raw_inverse(context.pi_obs, context.matched)
        # Fall back to direct-method on rows where the raw weight blows past tau.
        w = np.where(w_raw <= self.tau, w_raw, 0.0)
        return w


@dataclass(frozen=True)
class DRosShrinkTransform:
    """DRos (Doubly Robust w/ Optimistic Shrinkage) — Su et al. 2020.

    Replaces ``w`` with ``w * lam / (w^2 + lam)``. ``lam → 0`` collapses to
    the direct method; ``lam → ∞`` recovers raw IPS. The optimal-bias choice
    is data-dependent and is left to the user (sweep ``lam`` over a grid).
    """

    lam: float
    name: str = "dros_shrink"

    def __post_init__(self) -> None:
        if self.lam < 0 or not np.isfinite(self.lam):
            raise ValueError(
                f"lam must be a non-negative finite float, got {self.lam!r}"
            )

    def __call__(self, context: TransformContext) -> np.ndarray:
        w_raw = _raw_inverse(context.pi_obs, context.matched)
        if self.lam == 0:
            return np.zeros_like(w_raw)
        denom = w_raw**2 + self.lam
        return np.where(denom > 0, w_raw * self.lam / denom, 0.0)


@dataclass(frozen=True)
class MIPSTransform:
    """MIPS (Marginalized IPS) weight — Saito & Joachims 2022.

    Replaces the per-action propensity with a per-embedding propensity, so
    common support need only hold over the action *embedding* rather than the
    raw action index. This is critical for large action spaces (operator
    pools, candidate-set rerankers).

    The current implementation assumes an action-level embedding
    ``E in R^(n_actions x d)`` and computes the embedding-marginal propensity

        ``p_e(x_i) = Σ_a π_log(a | x_i) · k(E_a, E_{A_i})``

    where ``k`` is a row-normalised Gaussian kernel over embeddings. The
    working weight is the inverse logging embedding-marginal ``1 / p_e``; no
    target-policy embedding-marginal numerator is formed here because the
    target policy enters separately through the DR core's ``q_pi`` term. When
    the kernel is the identity matrix this reduces to skdr-eval's per-action
    IPS weight ``1 / π_log(A_i | x_i)``.

    Parameters
    ----------
    action_embedding : np.ndarray, shape (n_actions, embed_dim)
        Per-action embedding. The embedding must encode the actions' relevant
        sufficient statistics (skill vectors, capacity, role one-hots).
    bandwidth : float, default 1.0
        Gaussian-kernel bandwidth. ``inf`` collapses to a uniform kernel
        (every action equally relevant — pure marginalisation). Small values
        collapse back to per-action IPS.
    clip : float, default ``float("inf")``
        Optional top-side clip on the marginalised weight.

    Notes
    -----
    MIPS is *biased* unless the embedding is a sufficient statistic for the
    reward distribution; the bias scales with how much information about the
    action is lost in the embedding. The companion
    :func:`skdr_eval.estimators.mips.embedding_sufficiency_diagnostic`
    estimates this loss empirically.
    """

    action_embedding: np.ndarray
    bandwidth: float = 1.0
    clip: float = float("inf")
    name: str = "mips"

    _EXPECTED_NDIM = 2  # (n_actions, embed_dim)

    def __post_init__(self) -> None:
        if self.action_embedding.ndim != self._EXPECTED_NDIM:
            raise ValueError(
                f"action_embedding must be 2D, got ndim={self.action_embedding.ndim}"
            )
        if self.bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be > 0, got {self.bandwidth!r} "
                "(use float('inf') for a uniform kernel)"
            )

    def _embedding_kernel(self) -> np.ndarray:
        """Pairwise row-normalised kernel matrix ``K[i, j] = k(E_i, E_j)``."""
        emb = self.action_embedding.astype(np.float64)
        if not np.isfinite(self.bandwidth):
            # Uniform kernel — every action equally relevant; collapses to
            # marginalising over the entire action set.
            n_actions = emb.shape[0]
            kernel = np.full((n_actions, n_actions), 1.0 / n_actions)
            return kernel
        # Squared Euclidean distance, exp-decayed.
        sq = np.sum(emb * emb, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * emb @ emb.T
        d2 = np.maximum(d2, 0.0)
        # `(self.bandwidth ** 2)` is a scalar, so 2 * b**2 cannot underflow
        # for the practical bandwidths users will pass; clip for safety.
        denom = max(2.0 * self.bandwidth**2, np.finfo(np.float64).tiny)
        k = np.exp(-d2 / denom)
        # Row-normalise so each row sums to 1 — embedding-marginal probability.
        row_sum = k.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        return k / row_sum

    def __call__(self, context: TransformContext) -> np.ndarray:
        n, n_actions = context.policy_probs.shape
        kernel = self._embedding_kernel()
        if kernel.shape != (n_actions, n_actions):
            raise ValueError(
                f"action_embedding rows {kernel.shape[0]} != policy_probs "
                f"actions {n_actions}; the embedding must be supplied per "
                "action in the same order as the design matrix."
            )
        if context.propensities.shape != (n, n_actions):
            raise ValueError(
                "MIPSTransform requires the full logging-policy propensity "
                f"matrix in context.propensities, expected shape ({n}, "
                f"{n_actions}), got {context.propensities.shape!r}"
            )

        # Embedding-marginal logging density at the observed action's
        # embedding neighbourhood:
        #   p_e_log[i] = Σ_{a'} π_log(a' | x_i) · k(E_{a'}, E_{A_i})
        # When the kernel is the identity matrix this collapses to
        # pi_obs[i] = π_log(A_i | x_i) and MIPS reduces to skdr-eval's IPS.
        # When the kernel is uniform (every action equivalent in embedding
        # space) p_e_log[i] = 1 / n_actions and the MIPS weight becomes a
        # constant ``n_actions`` — the action propensity is fully ignored.
        a_idx = context.A.astype(int)
        # Column of kernel at A_i for each row.
        kernel_at_A = kernel[:, a_idx]  # (n_actions, n)
        # Weighted sum over actions of the logging propensity times kernel.
        p_e_log = np.einsum("na,an->n", context.propensities, kernel_at_A)
        safe = context.matched & (p_e_log > 0)
        w = np.zeros(n, dtype=np.float64)
        w[safe] = 1.0 / p_e_log[safe]
        if np.isfinite(self.clip):
            w = np.minimum(w, self.clip)
        return w


# Static-type confirmation: each concrete dataclass is structurally a
# :class:`WeightTransform`.
def _assert_protocols() -> None:  # pragma: no cover - type-check only
    _: WeightTransform = ClipTransform(clip=10.0)
    _ = IdentityTransform()
    _ = SwitchTauTransform(tau=5.0)
    _ = DRosShrinkTransform(lam=1.0)
    _ = MIPSTransform(action_embedding=np.eye(3))
