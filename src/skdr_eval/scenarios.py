"""What-if autoscaling scenario simulator for pairwise evaluation (issue #34).

This is a thin, **clearly-documented** what-if layer on top of
:func:`skdr_eval.evaluate_pairwise_models`. It transforms the *operational
constraints* a target policy faces — how many operators are available, and which
ones a client may be routed to — then evaluates the resulting policy against the
**unchanged** logged outcomes with the usual DR/SNDR machinery and trust
diagnostics.

Statistical honesty
-------------------
The scenario only narrows **eligibility** (the set of operators the policy may
choose from). The logged actions, logged outcomes and propensity model are left
untouched, so no counterfactual outcomes are invented: a scenario answers
"if the policy had been forced to route within this reduced operator pool, what
does the logged data say its value would have been — and is there enough support
to trust that?". Reducing capacity typically lowers ``match_rate`` and ``ESS``
(fewer logged decisions overlap the constrained policy), and the support-health
diagnostics will say so. The first supported knobs are deliberately narrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

if TYPE_CHECKING:
    from .reporting import EvaluationArtifact

_SUPPORTED_KNOBS = {"capacity_multiplier", "eligibility_mode"}
_ELIGIBILITY_MODES = ("as_logged", "restricted")


def _kept_operators_by_day(
    op_daily_df: pd.DataFrame,
    day_col: str,
    operator_id_col: str,
    capacity_multiplier: float,
    random_state: int,
) -> dict[Any, list[Any]]:
    """Pick a reproducible kept-operator subset per day under reduced capacity.

    For each day ``K = max(1, round(m * n_day_ops))`` operators are retained,
    drawn without replacement from that day's operators. Selection is seeded per
    day so results are reproducible across runs and independent of day ordering.
    """
    kept: dict[Any, list[Any]] = {}
    for day_idx, (day, group) in enumerate(op_daily_df.groupby(day_col, sort=True)):
        ops = group[operator_id_col].tolist()
        n_day = len(ops)
        if capacity_multiplier >= 1.0:
            kept[day] = ops
            continue
        k = max(1, round(capacity_multiplier * n_day))
        # Deterministic per-day stream: a list seed feeds NumPy's SeedSequence,
        # which is reproducible across processes/interpreters (unlike Python's
        # per-process-salted built-in ``hash()``) and gives an independent draw
        # per day regardless of day ordering.
        rng = np.random.default_rng([int(random_state), day_idx])
        chosen_idx = rng.choice(n_day, size=k, replace=False)
        # Preserve original operator order for stable, comparable output.
        kept[day] = [ops[i] for i in sorted(chosen_idx)]
    return kept


def simulate_autoscaling_scenario(
    logs_df: pd.DataFrame,
    op_daily_df: pd.DataFrame,
    models: dict[str, Any],
    scenario: dict[str, Any],
    *,
    metric_col: str,
    task_type: Literal["regression", "binary"],
    direction: Literal["min", "max"],
    day_col: str = "arrival_day",
    operator_id_col: str = "operator_id",
    elig_col: str = "elig_mask",
    random_state: int = 0,
    **eval_kwargs: Any,
) -> EvaluationArtifact:
    """Evaluate a target policy under a what-if operational scenario (issue #34).

    Supported scenario knobs (first version)
    ----------------------------------------
    ``capacity_multiplier`` : float in ``(0, 1]``, default ``1.0``
        Fraction of each day's operators that remain available (a "fewer staff"
        knob). ``K = max(1, round(m * n_day_ops))`` operators are kept per day,
        chosen reproducibly. Eligibility is intersected with this kept set.
    ``eligibility_mode`` : ``"as_logged"`` | ``"restricted"``, default
        ``"as_logged"``
        - ``"as_logged"``: keep each decision's logged eligibility, intersected
          with the capacity-reduced kept set (falling back to the kept set if
          the intersection is empty so no decision is left with zero options).
        - ``"restricted"``: ignore the logged eligibility and let the policy
          route from the full capacity-reduced pool for that day — models a hard
          capacity constraint that supersedes prior routing rules.

    With ``capacity_multiplier=1.0`` and ``eligibility_mode="as_logged"`` the
    scenario is a no-op and returns exactly what
    :func:`evaluate_pairwise_models` would (regression-tested).

    Parameters
    ----------
    logs_df, op_daily_df, models, metric_col, task_type, direction
        As in :func:`evaluate_pairwise_models`.
    scenario : dict[str, Any]
        Scenario knobs (see above). Unknown keys raise ``DataValidationError``.
    day_col, operator_id_col, elig_col : str
        Column-name overrides, forwarded to :func:`evaluate_pairwise_models`.
    random_state : int, default=0
        Seeds both the capacity sampling and the underlying evaluation.
    **eval_kwargs : Any
        Forwarded to :func:`evaluate_pairwise_models` (e.g. ``n_splits``,
        ``strategy``, ``propensity``, ``ci_bootstrap``, ``estimators``,
        ``execution_mode``).

    Returns
    -------
    EvaluationArtifact
        The evaluation artifact, with the applied scenario and its assumptions
        recorded under ``artifact.metadata["scenario"]``.

    Raises
    ------
    DataValidationError
        If ``scenario`` contains unknown knobs or out-of-range values, or the
        logs lack the eligibility column.
    """
    # Imported here to avoid a circular import (core imports reporting which is
    # heavy); scenarios is a thin layer over core.
    from .core import evaluate_pairwise_models  # noqa: PLC0415

    unknown = set(scenario) - _SUPPORTED_KNOBS
    if unknown:
        raise DataValidationError(
            f"Unknown scenario knob(s): {sorted(unknown)}. "
            f"Supported: {sorted(_SUPPORTED_KNOBS)}."
        )
    capacity_multiplier = float(scenario.get("capacity_multiplier", 1.0))
    eligibility_mode = scenario.get("eligibility_mode", "as_logged")
    if not 0.0 < capacity_multiplier <= 1.0:
        raise DataValidationError(
            f"capacity_multiplier must be in (0, 1], got {capacity_multiplier}."
        )
    if eligibility_mode not in _ELIGIBILITY_MODES:
        raise DataValidationError(
            f"eligibility_mode must be one of {_ELIGIBILITY_MODES}, "
            f"got {eligibility_mode!r}."
        )
    if elig_col not in logs_df.columns:
        raise DataValidationError(
            f"simulate_autoscaling_scenario requires the eligibility column "
            f"{elig_col!r} in logs_df."
        )
    if "elig_col" in eval_kwargs:
        raise TypeError("pass elig_col as a keyword argument, not inside eval_kwargs")

    kept_by_day = _kept_operators_by_day(
        op_daily_df, day_col, operator_id_col, capacity_multiplier, random_state
    )

    scenario_logs = logs_df.copy()
    days = scenario_logs[day_col].to_numpy()
    logged_elig = scenario_logs[elig_col].to_numpy()
    new_elig: list[list[Any]] = []
    for day, elig in zip(days, logged_elig, strict=False):
        kept = kept_by_day.get(day, [])
        kept_set = set(kept)
        if eligibility_mode == "restricted":
            new_elig.append(list(kept))
            continue
        # Mode as_logged keeps the logged eligibility intersected with the kept
        # set; fall back to the kept set if empty. ``validate_pairwise_inputs``
        # blesses list/tuple/set eligibility values, so all three are honored.
        if isinstance(elig, (list, tuple, set, np.ndarray)):
            filtered = [op for op in elig if op in kept_set]
        else:
            filtered = list(kept)
        new_elig.append(filtered if filtered else list(kept))
    scenario_logs[elig_col] = pd.Series(new_elig, index=scenario_logs.index)

    artifact = evaluate_pairwise_models(
        logs_df=scenario_logs,
        op_daily_df=op_daily_df,
        models=models,
        metric_col=metric_col,
        task_type=task_type,
        direction=direction,
        day_col=day_col,
        operator_id_col=operator_id_col,
        elig_col=elig_col,
        random_state=random_state,
        **eval_kwargs,
    )

    artifact.metadata["scenario"] = {
        "capacity_multiplier": capacity_multiplier,
        "eligibility_mode": eligibility_mode,
        "assumptions": [
            "Only eligibility (the policy's allowed operators) is changed; "
            "logged actions, outcomes and propensities are untouched.",
            "Reduced capacity narrows eligibility and typically lowers "
            "match_rate and ESS — read support_health before trusting V_hat.",
            "Operator capacity subsets are sampled reproducibly from "
            "random_state and do not model queueing or shift dynamics.",
        ],
    }
    return artifact
