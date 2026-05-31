"""Strategy-aware DR core (#86).

Generalises :func:`skdr_eval.core.dr_value_with_clip` around the
``WeightTransform`` / ``OutcomeLoss`` protocols defined in
:mod:`skdr_eval.estimators.protocols`.

The public entry-point :func:`dr_value_with_strategy` is what
:func:`skdr_eval.core.evaluate_sklearn_models` and
:func:`skdr_eval.core.evaluate_pairwise_models` call when the user requests
an estimator outside the ``("DR", "SNDR")`` pair. It returns a single
:class:`~skdr_eval.core.DRResult` per strategy, mirroring the shape of the
historical API so the rest of the reporting pipeline does not need to
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from .protocols import EstimatorStrategy, TransformContext

if TYPE_CHECKING:
    from ..core import DRResult


__all__ = [
    "StrategyResult",
    "dr_value_with_strategy",
]

# q_hat may be 1D (marginal) or 2D (per-action). 2D arrays carry one column
# per action, so the observed-action prediction is a gather along axis 1.
_PER_ACTION_NDIM = 2


@dataclass
class StrategyResult:
    """Internal aggregate returned by :func:`dr_value_with_strategy`.

    Mirrors the fields of :class:`~skdr_eval.core.DRResult` so callers can
    project either directly. ``grid`` is a single-row DataFrame keyed by the
    strategy name to keep the public API uniform with the clip-grid path.
    """

    name: str
    V_hat: float
    SE_if: float
    ESS: float
    tail_mass: float
    MSE_est: float
    match_rate: float
    min_pscore: float
    pscore_q10: float
    pscore_q05: float
    pscore_q01: float
    grid: pd.DataFrame
    pareto_k: float


def _matched_mask(pi_obs: np.ndarray, A: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Standard ``matched`` mask shared with the clip-grid path."""
    A_int = A.astype(int)
    elig_bool = elig.astype(bool)
    return cast(
        "np.ndarray", (pi_obs > 0) & elig_bool[np.arange(pi_obs.shape[0]), A_int]
    )


def dr_value_with_strategy(
    *,
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    strategy: EstimatorStrategy,
    action_embedding: np.ndarray | None = None,
) -> DRResult:
    """Compute V̂ under an arbitrary ``WeightTransform`` + ``OutcomeLoss``.

    Parameters
    ----------
    propensities : np.ndarray
        Logging-policy propensity matrix, shape ``(n, n_actions)``.
    policy_probs : np.ndarray
        Target-policy probability matrix, shape ``(n, n_actions)``.
    Y, q_hat, A, elig
        Observed outcomes, cross-fitted outcome predictions, observed action
        indices, and the eligibility matrix. All shapes match the
        ``dr_value_with_clip`` convention.
    strategy : EstimatorStrategy
        The pre-built strategy (weight transform + outcome loss + optional
        self-normalisation flag).
    action_embedding : np.ndarray, optional
        Forwarded into the :class:`TransformContext` so transforms like MIPS
        can read it without being re-instantiated per call.

    Returns
    -------
    DRResult
        With ``clip`` set to ``nan`` (the strategy supersedes the clip-grid
        selection) and ``grid`` a single-row DataFrame summarising the run.
    """
    # Imported lazily to avoid a circular import with ``core``.
    from ..core import DRResult  # noqa: PLC0415
    from ..diagnostics import psis_pareto_k  # noqa: PLC0415

    n_samples = Y.shape[0]
    rows = np.arange(n_samples)
    A_int = A.astype(int)
    pi_obs = propensities[rows, A_int]
    pi_target_obs = policy_probs[rows, A_int]
    # Behavioural overlap: positive logging propensity on an eligible observed
    # action. Target support is *not* pre-filtered into the context mask — each
    # transform layers its own target-policy numerator. Exact-action ratios
    # zero out automatically where π(A|x) == 0, while MIPS keeps rows whose
    # target has support in the observed action's embedding *neighbourhood*; an
    # exact-action ``π(A|x) > 0`` pre-filter would wrongly drop those rows (and
    # then zero the weight MIPS just computed for them) — see #142.
    behaviour_matched = _matched_mask(pi_obs, A, elig)
    if not behaviour_matched.any():
        raise ValueError("No matched samples found")

    context = TransformContext(
        pi_obs=pi_obs,
        matched=behaviour_matched,
        policy_probs=policy_probs,
        A=A,
        elig=elig,
        propensities=propensities,
        action_embedding=action_embedding,
    )
    w = strategy.weight_transform(context).astype(np.float64, copy=False)
    if w.shape != (n_samples,):
        raise ValueError(
            f"WeightTransform {strategy.weight_transform!r} returned shape "
            f"{w.shape}; expected ({n_samples},)"
        )
    if np.any(w < 0):
        raise ValueError(
            f"WeightTransform {strategy.weight_transform!r} returned negative "
            "weights; transforms must be non-negative."
        )
    w[~behaviour_matched] = 0.0

    # Strategy-aware overlap & tail diagnostics (#142). MIPS marginalises over
    # an embedding neighbourhood, so for the embedding path the overlap set and
    # heavy-tailed quantity are the *realised MIPS weight* — not the
    # exact-action π(A|x)/e(A|x) ratio, which would mislabel neighbourhood-
    # supported rows as unmatched and diagnose the wrong tail. For
    # clip/SWITCH/DRos the exact-action ratio is correct and retained (#106),
    # keeping those estimators' diagnostics byte-identical.
    if action_embedding is not None:
        overlap = w > 0
        diag_weights = w[overlap]
    else:
        overlap = behaviour_matched & (pi_target_obs > 0)
        diag_weights = pi_target_obs[overlap] / pi_obs[overlap]
    if not overlap.any():
        raise ValueError("No matched samples found")
    # ``min_pscore`` and the pscore quantiles summarise the observed-action
    # *logging propensity* e(A|x) over the overlap set — a data-overlap
    # descriptor. On the MIPS/embedding path this is only loosely tied to the
    # working weight (which is the embedding-marginal density ratio); ESS,
    # tail_mass and Pareto-k describe that weight directly.
    pi_overlap = pi_obs[overlap]
    match_rate = float(overlap.mean())
    min_pscore = float(pi_overlap.min())
    pscore_q01 = float(np.percentile(pi_overlap, 1))
    pscore_q05 = float(np.percentile(pi_overlap, 5))
    pscore_q10 = float(np.percentile(pi_overlap, 10))
    # PSIS Pareto-k on the strategy's own importance-weight tail.
    pareto_k = float(psis_pareto_k(diag_weights))

    # Policy-weighted outcome (q_pi). Same broadcasting rules as the
    # clip-grid path: q_hat may be 1D (marginal) or 2D (per-action).
    q_pi = np.sum(policy_probs * q_hat.reshape(n_samples, -1), axis=1)

    # Observed-action prediction for the DR residual. A 2D (per-action) q_hat
    # must be indexed at the logged action; a 1D q_hat already is the
    # observed-action prediction. Without this slice, (Y - q_hat) would
    # mis-broadcast (n,) against (n, n_actions) on the 2D path.
    q_hat_obs = q_hat[rows, A_int] if q_hat.ndim == _PER_ACTION_NDIM else q_hat

    if strategy.self_normalised:
        denom = w.sum()
        if denom > 0:
            V_hat = float(q_pi.mean() + (w * (Y - q_hat_obs)).sum() / denom)
        else:
            V_hat = float(q_pi.mean())
    else:
        V_hat = float((q_pi + w * (Y - q_hat_obs)).mean())

    ess = float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0
    tail_mass = float((w == 0).mean())  # zero-weight fraction
    se_if = float(np.std(q_pi + w * (Y - q_hat_obs)) / np.sqrt(n_samples))
    mse_est = float(se_if**2)

    grid_df = pd.DataFrame(
        [
            {
                "strategy": strategy.name,
                "V_hat": V_hat,
                "SE_if": se_if,
                "ESS": ess,
                "tail_mass": tail_mass,
                "MSE_est": mse_est,
            }
        ]
    )

    return DRResult(
        clip=float("nan"),
        V_hat=V_hat,
        SE_if=se_if,
        ESS=ess,
        tail_mass=tail_mass,
        MSE_est=mse_est,
        match_rate=match_rate,
        min_pscore=min_pscore,
        pscore_q10=pscore_q10,
        pscore_q05=pscore_q05,
        pscore_q01=pscore_q01,
        grid=grid_df,
        pareto_k=pareto_k,
    )
