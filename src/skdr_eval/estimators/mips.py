"""MIPS estimator wiring and embedding-sufficiency diagnostic (#85).

The ``MIPSTransform`` itself lives in :mod:`weight_transforms`; this module
provides the higher-level :func:`mips_value` convenience wrapper and the
:func:`embedding_sufficiency_diagnostic` probe that flags when the action
embedding has lost too much information about the reward to be used as a
marginalisation sufficient statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core import dr_value_with_strategy
from .outcome_losses import MSEOutcomeLoss
from .protocols import EstimatorStrategy
from .weight_transforms import MIPSTransform

if TYPE_CHECKING:
    from ..core import DRResult

__all__ = [
    "EmbeddingSufficiencyReport",
    "embedding_sufficiency_diagnostic",
    "mips_value",
]


@dataclass(frozen=True)
class EmbeddingSufficiencyReport:
    """Output of :func:`embedding_sufficiency_diagnostic`.

    Attributes
    ----------
    r2_action : float
        ``R^2`` of a one-vs-rest regression of the per-row reward residual on
        the action one-hot, *after* projecting out the embedding. Values
        close to ``0`` indicate the embedding captures most of the
        action-driven reward signal; large positive values mean the
        embedding is an *insufficient* marginalisation target and MIPS will
        be biased.
    n_actions : int
        Number of distinct actions in the embedding.
    embed_dim : int
        Embedding dimension.
    notes : str
        Human-readable interpretation banner.
    """

    r2_action: float
    n_actions: int
    embed_dim: int
    notes: str


def mips_value(
    *,
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    action_embedding: np.ndarray,
    bandwidth: float = 1.0,
    clip: float = float("inf"),
) -> DRResult:
    """Convenience wrapper around :func:`dr_value_with_strategy` for MIPS.

    Parameters mirror ``dr_value_with_clip``; ``action_embedding`` is the
    extra MIPS-specific input.
    """
    strategy = EstimatorStrategy(
        name="MIPS",
        weight_transform=MIPSTransform(
            action_embedding=action_embedding, bandwidth=bandwidth, clip=clip
        ),
        outcome_loss=MSEOutcomeLoss(),
        self_normalised=False,
    )
    return dr_value_with_strategy(
        propensities=propensities,
        policy_probs=policy_probs,
        Y=Y,
        q_hat=q_hat,
        A=A,
        elig=elig,
        strategy=strategy,
        action_embedding=action_embedding,
    )


def embedding_sufficiency_diagnostic(
    *,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    action_embedding: np.ndarray,
) -> EmbeddingSufficiencyReport:
    """Estimate how much action-specific signal the embedding misses.

    Implementation
    --------------
    Compute the residual ``r_i = Y_i - q̂_i``. Regress ``r`` on the embedding
    of the observed action with OLS to obtain a "kept" R²; regress on the
    action one-hot to obtain an "upper-bound" R². The diagnostic returns the
    *gap*: large gaps mean the action carries reward signal that the
    embedding throws away, biasing MIPS.

    The OLS regression is intentionally simple (closed-form) so the
    diagnostic stays dependency-light and deterministic.
    """
    _EXPECTED_NDIM = 2  # (n_actions, embed_dim)
    if action_embedding.ndim != _EXPECTED_NDIM:
        raise ValueError(
            f"action_embedding must be 2D, got ndim={action_embedding.ndim}"
        )
    n = Y.shape[0]
    if not (q_hat.shape == (n,) and A.shape == (n,)):
        raise ValueError("Y, q_hat, A must all have shape (n,)")

    r = (Y - q_hat).astype(np.float64)
    r_centered = r - r.mean()
    total_ss = float(np.sum(r_centered**2))
    if total_ss == 0:
        return EmbeddingSufficiencyReport(
            r2_action=0.0,
            n_actions=int(action_embedding.shape[0]),
            embed_dim=int(action_embedding.shape[1]),
            notes="residual variance is zero; outcome model is perfect",
        )

    # Embedding regression: design matrix is per-row embedding of A_i.
    a_int = A.astype(int)
    X_emb = action_embedding[a_int]
    # Add intercept for both designs.
    ones = np.ones((n, 1))
    X_emb_full = np.concatenate([ones, X_emb], axis=1)
    # One-hot action design — encodes the supremum signal the embedding
    # could plausibly recover.
    n_actions = int(action_embedding.shape[0])
    X_action = np.zeros((n, n_actions))
    X_action[np.arange(n), a_int] = 1.0
    X_action_full = np.concatenate([ones, X_action], axis=1)

    def _r2(X: np.ndarray) -> float:
        # Closed-form OLS R².
        try:
            beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        except np.linalg.LinAlgError:
            return float("nan")
        pred = X @ beta
        ss_res = float(np.sum((r - pred) ** 2))
        return float(1.0 - ss_res / total_ss)

    r2_emb = _r2(X_emb_full)
    r2_act = _r2(X_action_full)
    # Action R² is the supremum; the gap is what the embedding misses.
    gap = max(0.0, r2_act - r2_emb)

    _GAP_LOW = 0.01
    _GAP_MID = 0.05
    if gap < _GAP_LOW:
        notes = "embedding is approximately sufficient (gap < 1%)"
    elif gap < _GAP_MID:
        notes = "embedding loses some action signal (gap 1-5%) — MIPS lightly biased"
    else:
        notes = (
            "embedding loses substantial action signal (gap >= 5%) — MIPS may "
            "be heavily biased; consider enriching the embedding or falling "
            "back to standard DR/SNDR on action-level common support."
        )

    return EmbeddingSufficiencyReport(
        r2_action=float(gap),
        n_actions=n_actions,
        embed_dim=int(action_embedding.shape[1]),
        notes=notes,
    )
