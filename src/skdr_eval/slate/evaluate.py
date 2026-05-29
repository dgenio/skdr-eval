"""Top-level slate-OPE entry point returning an ``EvaluationArtifact`` (#135).

The estimators in :mod:`skdr_eval.slate.estimators` each return a standalone
:class:`SlateResult`. Practitioners, however, expect the same bundled
:class:`~skdr_eval.reporting.EvaluationArtifact` surface they get from
:func:`skdr_eval.evaluate_sklearn_models` — a report table, support-health
warnings, a clip-grid-style sensitivity summary, and a renderable HTML card.

:func:`evaluate_slate_models` wires the slate estimators into that surface:

* Each ``(model, estimator)`` pair becomes one report row.
* Support diagnostics (``ESS``, ``match_rate``, ``min_pscore``, ``pareto_k``,
  ``tail_mass``) are computed from the *slate-level* importance weight
  ``π_target(slate) / π_logging(slate)`` — an estimator-agnostic measure of how
  well the logged slates overlap the target policy, which is exactly what the
  support-health warnings (#22) are designed to read.
* The per-estimator value / SE come from the underlying :class:`SlateResult`.

See ``docs/slate-vs-pairwise-vs-standard.md`` for when to reach for this entry
point instead of :func:`evaluate_sklearn_models` /
:func:`evaluate_pairwise_models`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .estimators import (
    SlateResult,
    pseudo_inverse_ips,
    reward_interaction_ips,
    slate_cascade_dr,
    slate_standard_ips,
)

if TYPE_CHECKING:
    from ..reporting import EvaluationArtifact, SupportHealthThresholds

__all__ = ["evaluate_slate_models"]

# Per-rank target policy: ``(rank, item) -> probability``.
SlatePerRankPolicy = Callable[[int, int], float]

_SUPPORTED_ESTIMATORS = ("SlateStandardIPS", "RIPS", "PI-IPS", "SlateCascadeDR")


def _slate_level_weights(
    logs: pd.DataFrame,
    target_policy_per_rank: SlatePerRankPolicy,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slate-level IPS weight per impression and the logging slate probability.

    Returns ``(weights, logging_probs)`` where
    ``weights[i] = Π_k π_target(k, slate_{i,k}) / π_logging(slate_i)`` — the
    standard-IPS slate weight, used purely as an estimator-agnostic support
    diagnostic for the report's warning columns.
    """
    slates = [list(map(int, s)) for s in logs["slate"]]
    logging_probs = logs["logging_prob"].to_numpy(dtype=np.float64)
    if not slates:
        return np.zeros(0, dtype=np.float64), logging_probs
    slate_size = len(slates[0])
    slate_arr = np.asarray(slates, dtype=np.int64)
    pi_t = np.empty((slate_size, n_items), dtype=np.float64)
    for k in range(slate_size):
        for j in range(n_items):
            pi_t[k, j] = float(target_policy_per_rank(k, j))
    rank_idx = np.arange(slate_size)
    target_slate_prob = np.prod(pi_t[rank_idx[None, :], slate_arr], axis=1)
    weights = np.zeros_like(logging_probs)
    safe = logging_probs > 0
    weights[safe] = target_slate_prob[safe] / logging_probs[safe]
    return weights, logging_probs


def _empirical_q_hat_per_rank(
    logs: pd.DataFrame, slate_size: int, n_items: int
) -> np.ndarray:
    """Position-by-item empirical click-rate table broadcast over impressions.

    Builds ``q̂[i, k, j] = mean click at rank k for item j`` (a simple direct
    method) so Cascade-DR has an informative outcome model without requiring
    the caller to supply one.
    """
    slates = np.asarray([list(map(int, s)) for s in logs["slate"]], dtype=np.int64)
    clicks = np.asarray([list(map(float, c)) for c in logs["clicks"]], dtype=np.float64)
    click_sum = np.zeros((slate_size, n_items), dtype=np.float64)
    click_cnt = np.zeros((slate_size, n_items), dtype=np.float64)
    for k in range(slate_size):
        np.add.at(click_sum[k], slates[:, k], clicks[:, k])
        np.add.at(click_cnt[k], slates[:, k], 1.0)
    rate = np.divide(
        click_sum, click_cnt, out=np.zeros_like(click_sum), where=click_cnt > 0
    )
    n = slates.shape[0]
    return np.broadcast_to(rate[None, :, :], (n, slate_size, n_items)).copy()


def _slate_result_to_drresult(
    result: SlateResult,
    weights: np.ndarray,
    logging_probs: np.ndarray,
) -> object:
    """Map a :class:`SlateResult` + slate-level weights into a ``DRResult``.

    The estimator-specific value/SE come from ``result``; the support
    diagnostics come from the slate-level weights so the warning columns are
    populated consistently across estimators.
    """
    from ..core import DRResult  # noqa: PLC0415
    from ..diagnostics import psis_pareto_k  # noqa: PLC0415

    matched = (logging_probs > 0) & (weights > 0)
    match_rate = float(matched.mean()) if logging_probs.size else 0.0
    if matched.any():
        # Support diagnostics use the importance-weight-implied *effective
        # propensity* ``1 / w`` rather than the raw slate-permutation
        # ``logging_prob``. The slate-joint probability lives on a different
        # scale than the per-action propensity the ``1/n`` POOR_OVERLAP floor
        # was designed for, so comparing them directly would fire POOR_OVERLAP
        # for any non-tiny catalogue (#106). With ``1/w``, POOR_OVERLAP fires
        # only when a single slate weight exceeds the sample size — the
        # genuine slate-overlap failure mode.
        eff_prop = 1.0 / weights[matched]
        min_pscore = float(eff_prop.min())
        pscore_q01 = float(np.percentile(eff_prop, 1))
        pscore_q05 = float(np.percentile(eff_prop, 5))
        pscore_q10 = float(np.percentile(eff_prop, 10))
        pareto_k = float(psis_pareto_k(weights[matched]))
    else:
        min_pscore = 0.0
        pscore_q01 = pscore_q05 = pscore_q10 = 0.0
        pareto_k = float("nan")
    # ``tail_mass`` drives the EXTREME_CLIP warning, which measures the fraction
    # of *clipped* weight mass. The slate path applies no clipping, so the
    # clipped mass is always zero; reporting the zero-weight (no-overlap)
    # fraction here would spuriously fire EXTREME_CLIP. Zero-weight mass is
    # already captured by ``match_rate`` above.
    tail_mass = 0.0
    ess = (
        float(weights.sum() ** 2 / (weights**2).sum())
        if (weights**2).sum() > 0
        else 0.0
    )
    grid = pd.DataFrame(
        [
            {
                "clip": float("nan"),
                f"V_{result.name}": result.V_hat,
                f"MSE_{result.name}": float(result.SE**2),
            }
        ]
    )
    return DRResult(
        clip=float("nan"),
        V_hat=float(result.V_hat),
        SE_if=float(result.SE),
        ESS=ess,
        tail_mass=tail_mass,
        MSE_est=float(result.SE**2),
        match_rate=match_rate,
        min_pscore=min_pscore,
        pscore_q10=pscore_q10,
        pscore_q05=pscore_q05,
        pscore_q01=pscore_q01,
        grid=grid,
        pareto_k=pareto_k,
    )


def evaluate_slate_models(
    logs: pd.DataFrame,
    models: dict[str, SlatePerRankPolicy],
    slate_size: int | None = None,
    *,
    click_model: Literal["cascade", "position_bias", "linear"] = "cascade",
    n_items: int | None = None,
    estimators: tuple[str, ...] = ("RIPS", "PI-IPS", "SlateCascadeDR"),
    logging_policy_per_rank: SlatePerRankPolicy | None = None,
    support_thresholds: SupportHealthThresholds | None = None,
    random_state: int = 0,
    baseline: float | str | None = None,
) -> EvaluationArtifact:
    """Evaluate slate / top-K ranking policies and bundle an ``EvaluationArtifact``.

    Parameters
    ----------
    logs : pd.DataFrame
        Slate logs with columns ``slate``, ``clicks``, ``reward``,
        ``logging_prob`` — the schema produced by
        :func:`skdr_eval.slate.make_slate_synth`.
    models : dict[str, callable]
        Map of policy name to a per-rank target policy ``(rank, item) ->
        probability``. The slate-level probability is taken as the per-rank
        product, so the same callable drives every estimator.
    slate_size : int, optional
        Slate length. Inferred from the first logged slate when omitted.
    click_model : {"cascade", "position_bias", "linear"}, default "cascade"
        Recorded in the artifact metadata; documents which estimator is
        unbiased for these logs (Cascade-DR under ``"cascade"``).
    n_items : int, optional
        Item-catalogue size. Inferred as ``max(item) + 1`` when omitted.
    estimators : tuple[str, ...], default ("RIPS", "PI-IPS", "SlateCascadeDR")
        Subset of ``{"SlateStandardIPS", "RIPS", "PI-IPS", "SlateCascadeDR"}``.
    logging_policy_per_rank : callable, optional
        Per-rank logging policy ``(rank, item) -> probability``. Defaults to
        uniform-over-items (matches :func:`make_slate_synth`).
    support_thresholds : SupportHealthThresholds, optional
        Forwarded to the warning computation.
    random_state : int, default 0
        Persisted to the artifact metadata for reproducibility.
    baseline : float or {"logged"} or None
        Baseline policy value for the report's ``delta_V_hat`` column.
        ``"logged"`` uses the mean observed reward (the same sentinel the
        other evaluators accept; unknown strings raise ``DataValidationError``).

    Returns
    -------
    EvaluationArtifact
        With ``report``, ``warnings``, ``sensitivity``, and a renderable card
        populated for the slate estimators.
    """
    from ..reporting import build_evaluation_artifact  # noqa: PLC0415

    if not isinstance(models, dict) or not models:
        from ..exceptions import DataValidationError  # noqa: PLC0415

        raise DataValidationError("models must be a non-empty {name: policy} dict")

    unknown = [e for e in estimators if e not in _SUPPORTED_ESTIMATORS]
    if unknown:
        from ..exceptions import DataValidationError  # noqa: PLC0415

        raise DataValidationError(
            f"unknown slate estimators {unknown}; supported: "
            f"{list(_SUPPORTED_ESTIMATORS)}"
        )

    slates = [list(map(int, s)) for s in logs["slate"]]
    if not slates:
        from ..exceptions import DataValidationError  # noqa: PLC0415

        raise DataValidationError("logs is empty; nothing to evaluate")
    if slate_size is None:
        slate_size = len(slates[0])
    if n_items is None:
        n_items = int(max(max(s) for s in slates)) + 1

    q_hat_per_rank = (
        _empirical_q_hat_per_rank(logs, slate_size, n_items)
        if "SlateCascadeDR" in estimators
        else None
    )

    report_rows: list[dict[str, object]] = []
    detailed: dict[str, dict[str, object]] = {}

    for model_name, per_rank in models.items():
        weights, logging_probs = _slate_level_weights(logs, per_rank, n_items)
        per_model: dict[str, object] = {}
        for est in estimators:
            if est == "SlateStandardIPS":

                def _slate_prob(
                    slate: list[int], _p: SlatePerRankPolicy = per_rank
                ) -> float:
                    return float(
                        np.prod([_p(k, int(item)) for k, item in enumerate(slate)])
                    )

                result = slate_standard_ips(logs, target_policy=_slate_prob)
            elif est == "RIPS":
                result = reward_interaction_ips(logs, per_rank, logging_policy_per_rank)
            elif est == "PI-IPS":
                result = pseudo_inverse_ips(logs, per_rank, n_items=n_items)
            else:  # SlateCascadeDR
                assert q_hat_per_rank is not None
                result = slate_cascade_dr(
                    logs, per_rank, q_hat_per_rank, logging_policy_per_rank
                )

            dr_result = _slate_result_to_drresult(result, weights, logging_probs)
            per_model[result.name] = dr_result
            report_rows.append(
                {
                    "model": model_name,
                    "estimator": result.name,
                    "V_hat": dr_result.V_hat,  # type: ignore[attr-defined]
                    "SE_if": dr_result.SE_if,  # type: ignore[attr-defined]
                    "clip": float("nan"),
                    "ESS": dr_result.ESS,  # type: ignore[attr-defined]
                    "tail_mass": dr_result.tail_mass,  # type: ignore[attr-defined]
                    "MSE_est": dr_result.MSE_est,  # type: ignore[attr-defined]
                    "match_rate": dr_result.match_rate,  # type: ignore[attr-defined]
                    "min_pscore": dr_result.min_pscore,  # type: ignore[attr-defined]
                    "pscore_q10": dr_result.pscore_q10,  # type: ignore[attr-defined]
                    "pscore_q05": dr_result.pscore_q05,  # type: ignore[attr-defined]
                    "pscore_q01": dr_result.pscore_q01,  # type: ignore[attr-defined]
                    "pareto_k": dr_result.pareto_k,  # type: ignore[attr-defined]
                }
            )
        detailed[model_name] = per_model

    report = pd.DataFrame(report_rows)

    # Reuse the canonical resolver so the slate evaluator shares one baseline
    # contract with the other evaluators: ``"logged"`` (not ``"logging"``) is
    # the string sentinel, unknown strings and ``bool`` raise rather than being
    # silently ignored, and a numeric value is a fixed scalar baseline.
    from ..core import _resolve_baseline  # noqa: PLC0415

    baseline_kind, baseline_value = _resolve_baseline(
        baseline, logs["reward"].to_numpy(dtype=np.float64)
    )

    return build_evaluation_artifact(
        report=report,
        detailed=detailed,  # type: ignore[arg-type]
        n_samples=len(logs),
        propensities=None,
        actions=None,
        thresholds=support_thresholds,
        evaluator="evaluate_slate_models",
        random_state=random_state,
        extra_metadata={
            "slate_size": int(slate_size),
            "n_items": int(n_items),
            "click_model": click_model,
            "estimators": list(estimators),
        },
        baseline_kind=baseline_kind,
        baseline_value=baseline_value,
    )
