"""Slate / top-K off-policy estimators (#75).

Three estimators ship in this module:

* :func:`slate_standard_ips` — vanilla IPS over the full slate. Unbiased
  when ``π_target(slate) / π_logging(slate)`` is bounded; collapses to
  ``0`` when the target picks any slate the logging never showed.
* :func:`reward_interaction_ips` — RIPS (Vardasbi, Sarvi & de Rijke 2020).
  Decomposes the slate-level weight into per-position weights and uses
  per-rank reward attributions.
* :func:`pseudo_inverse_ips` — PI-IPS (Swaminathan et al. 2017). Builds the
  pseudo-inverse of the per-rank examination matrix and uses it as a
  per-position weight.
* :func:`slate_cascade_dr` — Cascade-DR (Kiyohara et al. 2022) for ranking
  policies under the cascade click model. Combines a per-rank IPS weight
  with a position-conditional outcome model.

All estimators consume the DataFrame returned by
:func:`skdr_eval.slate.make_slate_synth` (or any input matching the same
schema: ``slate``, ``clicks``, ``reward``, ``logging_prob``) and return a
:class:`SlateResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SlateResult",
    "pseudo_inverse_ips",
    "reward_interaction_ips",
    "slate_cascade_dr",
    "slate_standard_ips",
]


@dataclass
class SlateResult:
    """Per-estimator slate policy value with simple diagnostics.

    Attributes
    ----------
    name : str
        Estimator name (``"SlateStandardIPS"``, ``"RIPS"``, ``"PI-IPS"``,
        ``"SlateCascadeDR"``).
    V_hat : float
        Estimated policy value.
    SE : float
        Sample standard error of the per-impression contribution.
    ESS : float
        Effective sample size of the importance weights.
    n : int
        Number of impressions evaluated.
    """

    name: str
    V_hat: float
    SE: float
    ESS: float
    n: int


# Target policy is supplied as a callable ``slate -> probability`` (the
# logged slate probability under the *target* policy). Callers wishing to
# evaluate "always recommend slate S" can pass ``lambda s: 1.0 if s == S
# else 0.0``.
TargetPolicy = Callable[[list[int]], float]


def _matched_slate_indicator(slate: list[int], target_slate: list[int]) -> int:
    """Return 1 iff the logged and target slates match elementwise."""
    if len(slate) != len(target_slate):
        return 0
    return int(all(int(a) == int(b) for a, b in zip(slate, target_slate, strict=False)))


def _impression_iter(
    logs: pd.DataFrame,
) -> tuple[list[list[int]], list[list[int]], np.ndarray, np.ndarray]:
    """Project the slate DataFrame into typed arrays."""
    slates = [list(map(int, s)) for s in logs["slate"]]
    clicks = [list(map(int, c)) for c in logs["clicks"]]
    rewards = logs["reward"].to_numpy(dtype=np.float64)
    logging_probs = logs["logging_prob"].to_numpy(dtype=np.float64)
    return slates, clicks, rewards, logging_probs


def _ess_from_weights(w: np.ndarray) -> float:
    if (w**2).sum() == 0:
        return 0.0
    return float(w.sum() ** 2 / (w**2).sum())


def _stacked_slates(
    slates: list[list[int]], clicks: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged-but-fixed-width slate / click rows into ``(n, K)`` arrays."""
    slate_arr = np.asarray(slates, dtype=np.int64)
    click_arr = np.asarray(clicks, dtype=np.float64)
    return slate_arr, click_arr


def _per_rank_matrix(
    policy_per_rank: Callable[[int, int], float], slate_size: int, n_items: int
) -> np.ndarray:
    """Materialise a ``(slate_size, n_items)`` per-rank probability matrix.

    The Python callable is evaluated exactly ``slate_size * n_items`` times —
    once per (rank, item) cell — *outside* the per-impression loop, which is
    the whole point of the vectorised estimators (#137): the catalogue grid is
    built once and reused across every impression instead of being re-queried
    ``n_impressions`` times.
    """
    mat = np.empty((slate_size, n_items), dtype=np.float64)
    for k in range(slate_size):
        for j in range(n_items):
            mat[k, j] = float(policy_per_rank(k, j))
    return mat


def _vec_se(contribs: np.ndarray) -> float:
    """Standard error of the per-impression contribution mean."""
    return (
        float(contribs.std(ddof=1) / np.sqrt(contribs.size))
        if contribs.size > 1
        else 0.0
    )


def slate_standard_ips(logs: pd.DataFrame, target_policy: TargetPolicy) -> SlateResult:
    """Vanilla IPS at the slate level: ``E[π_T(s)/π_L(s) · R]``.

    Parameters
    ----------
    logs : pd.DataFrame
        DataFrame with ``slate``, ``reward``, ``logging_prob``.
    target_policy : callable
        Maps a slate (``list[int]``) to its probability under the target
        policy. Must be normalised over the slate space implied by the
        logging policy.
    """
    slates, _clicks, rewards, logging_probs = _impression_iter(logs)
    target_probs = np.array([float(target_policy(s)) for s in slates], dtype=np.float64)
    safe = logging_probs > 0
    w = np.zeros_like(rewards)
    w[safe] = target_probs[safe] / logging_probs[safe]
    contribs = w * rewards
    v_hat = float(contribs.mean()) if contribs.size > 0 else 0.0
    se = (
        float(contribs.std(ddof=1) / np.sqrt(contribs.size))
        if contribs.size > 1
        else 0.0
    )
    return SlateResult(
        name="SlateStandardIPS",
        V_hat=v_hat,
        SE=se,
        ESS=_ess_from_weights(w),
        n=int(rewards.size),
    )


def reward_interaction_ips(
    logs: pd.DataFrame,
    target_policy_per_rank: Callable[[int, int], float],
    logging_policy_per_rank: Callable[[int, int], float] | None = None,
) -> SlateResult:
    """Reward-Interaction IPS (Vardasbi, Sarvi & de Rijke 2020).

    Decomposes the slate-level weight into per-rank weights — assumes
    item-position factorised target / logging policies. The reward is the
    sum of per-position click rewards and the per-rank IPS weight is
    ``π_T(item | rank) / π_L(item | rank)``.

    Parameters
    ----------
    target_policy_per_rank : callable
        ``(rank, item) -> probability``. Should sum to 1 over items at each
        rank.
    logging_policy_per_rank : callable, optional
        If ``None`` (default), use uniform ``1 / n_items`` derived from the
        slate / catalogue size.
    """
    slates, clicks, _rewards, _logging_probs = _impression_iter(logs)
    if not slates:
        return SlateResult(name="RIPS", V_hat=0.0, SE=0.0, ESS=0.0, n=0)
    n_items = int(max(max(s) for s in slates)) + 1
    slate_size = len(slates[0])
    slate_arr, click_arr = _stacked_slates(slates, clicks)

    # Per-rank target / logging matrices, built once over the catalogue grid.
    pi_t = _per_rank_matrix(target_policy_per_rank, slate_size, n_items)
    if logging_policy_per_rank is None:
        # Uniform-over-items prior (matches make_slate_synth's default).
        pi_l = np.full((slate_size, n_items), 1.0 / n_items, dtype=np.float64)
    else:
        pi_l = _per_rank_matrix(logging_policy_per_rank, slate_size, n_items)

    rank_idx = np.arange(slate_size)
    # Gather the per-(impression, rank) logged item's probabilities.
    p_t = pi_t[rank_idx[None, :], slate_arr]  # (n, K)
    p_l = pi_l[rank_idx[None, :], slate_arr]  # (n, K)

    # A non-positive logging probability anywhere in the slate breaks the
    # factorised weight (matches the loop's ``break`` → discarded contribution).
    valid = np.all(p_l > 0, axis=1)
    safe_p_l = np.where(p_l > 0, p_l, 1.0)
    ratios = p_t / safe_p_l  # (n, K)
    slate_weight = np.where(valid, np.prod(ratios, axis=1), 0.0)
    contribs = np.where(valid, np.sum(ratios * click_arr, axis=1), 0.0)

    v_hat = float(contribs.mean())
    return SlateResult(
        name="RIPS",
        V_hat=v_hat,
        SE=_vec_se(contribs),
        ESS=_ess_from_weights(slate_weight),
        n=int(contribs.size),
    )


def pseudo_inverse_ips(
    logs: pd.DataFrame,
    target_policy_per_rank: Callable[[int, int], float],
    n_items: int | None = None,
) -> SlateResult:
    """Pseudo-Inverse IPS (Swaminathan et al. 2017).

    Builds a per-rank target probability matrix ``pi_T in R^(K x |items|)``,
    takes its Moore-Penrose pseudo-inverse, and uses each column as the
    per-position importance weight. Lower variance than RIPS when the
    examination model is well-conditioned.
    """
    slates, clicks, _rewards, _logging_probs = _impression_iter(logs)
    if not slates:
        return SlateResult(name="PI-IPS", V_hat=0.0, SE=0.0, ESS=0.0, n=0)
    slate_size = len(slates[0])
    if n_items is None:
        n_items = int(max(max(s) for s in slates)) + 1
    slate_arr, click_arr = _stacked_slates(slates, clicks)

    # Build the K x |items| target probability matrix (per-rank marginals)
    # once, then take the Moore-Penrose pseudo-inverse (SVD-stable).
    pi_t = _per_rank_matrix(target_policy_per_rank, slate_size, n_items)
    pi_t_pinv = np.linalg.pinv(pi_t)  # (n_items, K)

    rank_idx = np.arange(slate_size)
    # Per-(impression, rank) weight: the (logged-item, rank) pseudo-inverse cell.
    w_kr = pi_t_pinv[slate_arr, rank_idx[None, :]]  # (n, K)
    contribs = np.sum(w_kr * click_arr, axis=1)
    weights = np.sum(np.abs(w_kr), axis=1)

    v_hat = float(contribs.mean())
    return SlateResult(
        name="PI-IPS",
        V_hat=v_hat,
        SE=_vec_se(contribs),
        ESS=_ess_from_weights(weights),
        n=int(contribs.size),
    )


def slate_cascade_dr(
    logs: pd.DataFrame,
    target_policy_per_rank: Callable[[int, int], float],
    q_hat_per_rank: np.ndarray,
    logging_policy_per_rank: Callable[[int, int], float] | None = None,
) -> SlateResult:
    """Cascade-DR (Kiyohara et al. WSDM 2022).

    Combines a per-rank IPS weight with a position-conditional outcome
    model. Cascade-DR is unbiased under the cascade click model and is
    expected to have strictly lower variance than RIPS when the outcome
    model is informative.

    Parameters
    ----------
    q_hat_per_rank : np.ndarray, shape (n_impressions, slate_size, n_items)
        Position- and item-conditional reward predictions.
    logging_policy_per_rank : callable, optional
        Defaults to uniform (matches :func:`make_slate_synth`).
    """
    slates, clicks, _rewards, _log_probs = _impression_iter(logs)
    if not slates:
        return SlateResult(name="SlateCascadeDR", V_hat=0.0, SE=0.0, ESS=0.0, n=0)
    slate_size = len(slates[0])
    n_items = int(max(max(s) for s in slates)) + 1

    if q_hat_per_rank.shape != (len(slates), slate_size, n_items):
        raise ValueError(
            "q_hat_per_rank must have shape (n_impressions, slate_size, "
            f"n_items), got {q_hat_per_rank.shape!r}"
        )

    slate_arr, click_arr = _stacked_slates(slates, clicks)
    pi_t = _per_rank_matrix(target_policy_per_rank, slate_size, n_items)
    if logging_policy_per_rank is None:
        pi_l = np.full((slate_size, n_items), 1.0 / n_items, dtype=np.float64)
    else:
        pi_l = _per_rank_matrix(logging_policy_per_rank, slate_size, n_items)

    rank_idx = np.arange(slate_size)
    # Direct-method term per (impression, rank): Σ_j π_T(k, j) · q̂[i, k, j].
    q_pi_rank = np.einsum("kj,ikj->ik", pi_t, q_hat_per_rank)  # (n, K)
    # Observed-item outcome prediction q̂[i, k, item_{i,k}].
    q_hat_obs = np.take_along_axis(q_hat_per_rank, slate_arr[:, :, None], axis=2)[
        :, :, 0
    ]  # (n, K)

    p_t = pi_t[rank_idx[None, :], slate_arr]  # (n, K)
    p_l = pi_l[rank_idx[None, :], slate_arr]  # (n, K)
    w = np.where(p_l > 0, p_t / np.where(p_l > 0, p_l, 1.0), 0.0)  # (n, K)

    per_rank_contrib = q_pi_rank + w * (click_arr - q_hat_obs)
    contribs = np.sum(per_rank_contrib, axis=1)
    weights = np.max(w, axis=1)

    v_hat = float(contribs.mean())
    return SlateResult(
        name="SlateCascadeDR",
        V_hat=v_hat,
        SE=_vec_se(contribs),
        ESS=_ess_from_weights(weights),
        n=int(contribs.size),
    )
