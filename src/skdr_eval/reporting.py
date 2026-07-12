"""Evaluation reporting artifacts for ``skdr-eval``.

This module bundles the existing ``evaluate_*_models`` outputs into a single
:class:`EvaluationArtifact` that adds:

- Support-health warnings (issue #22).
- First-class propensity diagnostics on the artifact (issue #23).
- Clip-grid sensitivity summary (issue #27).
- JSON and HTML export with a versioned Pydantic schema (issue #28).
- Stakeholder-ready evaluation cards (issue #30).

Statistical scope (read me)
---------------------------
The artifact reads existing per-clip grids and per-decision diagnostics. It
does **not** change the DR/SNDR estimator math, the variance formulas, or the
bootstrap. Known limitations of the underlying metrics — the simplified
``SE_SNDR``, the conditional bootstrap caveat, and the marginal-q̂ design —
are tracked separately in issues #58, #60, #62. Warnings and sensitivity
summaries here are therefore honest about what they reflect: heuristics
computed from currently-reported metrics, not new statistical guarantees.

Default warning thresholds are grounded in published guidance (see
:class:`SupportHealthThresholds`). Tune via ``support_thresholds=...``.
"""

from __future__ import annotations

import base64
import copy
import importlib.resources as _resources
import io
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

from .diagnostics import (
    PropensityDiagnostics,
    comprehensive_propensity_diagnostics,
)
from .exceptions import (
    ConfigurationError,
    DataValidationError,
    OptionalDependencyError,
)

if TYPE_CHECKING:
    from .core import DRResult

logger = logging.getLogger("skdr_eval")

# Bump SCHEMA_VERSION when the artifact JSON layout changes incompatibly.
# 1.0.0 → 1.1.0: additive trust diagnostics (#80 pareto_k, #84 ECE/Brier).
# Old readers that pinned the 1.0.0 schema can still load 1.1.0 payloads
# because every new field has a ``None``/default fallback in the Pydantic
# models (``extra="allow"`` on _ReportRowSchema makes the new column
# transparent; PropensityDiagnostics payload fields default to ``None``).
SCHEMA_VERSION = "1.1.0"

# Canonical assumption tags surfaced on EvaluationCard.estimand (#128). The
# prose for each tag lives in docs/concepts/estimands-and-assumptions.md;
# downstream tooling should treat unknown tags as forward-compatible
# additions and ignore them.
DEFAULT_ASSUMPTION_TAGS: tuple[str, ...] = (
    "unconfoundedness",
    "overlap",
    "sutva",
    "double_robustness",
    "stochastic_logging",
    "bounded_weight_variance",
    "time_structure_respected",
)
DEFAULT_ESTIMAND_TEX = r"V(\pi) = E_X [ E_{A \sim \pi(\cdot|X)} [ Y(X, A) ] ]"
DEFAULT_ESTIMAND_SUMMARY = (
    "Policy value of the target policy over the held-out evaluation slice"
    " of the logs, under the assumptions listed in"
    " docs/concepts/estimands-and-assumptions.md."
)

# Machine-readable warning codes. Do not localize.
WARN_LOW_ESS = "LOW_ESS"
WARN_EXTREME_CLIP = "EXTREME_CLIP"
WARN_POOR_OVERLAP = "POOR_OVERLAP"
WARN_LOW_MATCH_RATE = "LOW_MATCH_RATE"
WARN_HIGH_PARETO_K = "HIGH_PARETO_K"  # #80 — PSIS Pareto-k > threshold
WARN_MISCAL_PROP = "MISCAL_PROP"  # #84 — ECE above threshold
WARN_PER_ACTION_MISCAL = (
    "PER_ACTION_MISCAL"  # #131 — max per-action ECE above threshold
)
WARN_RARE_ACTION_NO_SUPPORT = (
    "RARE_ACTION_NO_SUPPORT"  # #131 — rare-and-insufficient action
)

# Severity labels.
SUPPORT_OK = "ok"
SUPPORT_CAUTION = "caution"
SUPPORT_HIGH_RISK = "high_risk"

_VALID_SEVERITIES = (SUPPORT_OK, SUPPORT_CAUTION, SUPPORT_HIGH_RISK)

# Two or more caution-level warnings escalate to high_risk.
_CAUTION_ESCALATE_COUNT = 2


# --------------------------------------------------------------------------- #
# Thresholds                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportHealthThresholds:
    """Configurable thresholds for support-health warnings.

    Defaults are grounded in published guidance:

    - ``low_ess_frac = 0.10`` — Owen (2013) *Monte Carlo theory, methods and
      examples*, §9.4: an effective-sample-size fraction below 10% means a
      small number of weights dominate the estimator. A value below half this
      floor (5%) is escalated from ``caution`` to ``high_risk``.
    - ``extreme_clip_tail_mass = 0.05`` — Kang & Schafer (2007). When more
      than 5% of observations have their importance weights truncated, the
      estimator is regression-driven rather than weight-driven. A value above
      double this floor (10%) is escalated.
    - ``low_match_rate = 0.50`` — Reflects half the sample having zero
      propensity for the policy-chosen action. Below half this floor (25%) is
      escalated.
    - ``poor_overlap_min_pscore`` — Default ``None`` resolves at runtime to
      ``1 / n_samples`` (no realistic support at all). ``POOR_OVERLAP`` is
      always ``high_risk`` since DR validity does not survive zero overlap.
    - ``high_pareto_k = 0.7`` — Vehtari et al. (2024) PSIS thresholds (#80).
      ``k`` is the GPD shape parameter of the importance-weight tail.
      ``0.5 ≤ k < 0.7`` issues ``HIGH_PARETO_K`` as ``caution``;
      ``k ≥ 0.7`` escalates to ``high_risk`` (variance does not exist).
    - ``miscal_ece = 0.10`` — 10-point Expected Calibration Error gate (#84).
      Above ``miscal_ece`` fires ``MISCAL_PROP`` as ``caution``;
      above ``2 · miscal_ece`` escalates to ``high_risk`` (predictions are
      decoupled enough from outcomes that IPW estimates are biased).

    Attributes
    ----------
    low_ess_frac : float
        Floor on ``ESS / n_samples`` below which ``LOW_ESS`` fires (caution).
    extreme_clip_tail_mass : float
        Ceiling on ``tail_mass`` above which ``EXTREME_CLIP`` fires (caution).
    low_match_rate : float
        Floor on ``match_rate`` below which ``LOW_MATCH_RATE`` fires (caution).
    poor_overlap_min_pscore : float, optional
        Floor on ``min_pscore``. ``None`` means ``1 / n_samples``.
    high_pareto_k_caution : float
        Floor on Pareto-k above which ``HIGH_PARETO_K`` fires (caution).
        Default 0.5 (Vehtari et al. 2024 "convergence slow" threshold).
    high_pareto_k : float
        Floor on Pareto-k above which ``HIGH_PARETO_K`` escalates to
        ``high_risk``. Default 0.7 (PSIS "do not trust" threshold).
    miscal_ece : float
        Ceiling on Expected Calibration Error above which ``MISCAL_PROP``
        fires (caution). Default 0.10.
    """

    low_ess_frac: float = 0.10
    extreme_clip_tail_mass: float = 0.05
    low_match_rate: float = 0.50
    poor_overlap_min_pscore: float | None = None
    high_pareto_k_caution: float = 0.5
    high_pareto_k: float = 0.7
    miscal_ece: float = 0.10


# --------------------------------------------------------------------------- #
# Warning computation                                                         #
# --------------------------------------------------------------------------- #


def _compute_row_warnings(
    row: dict[str, Any],
    thresholds: SupportHealthThresholds,
    n_samples: int,
    *,
    model_ece: float | None = None,
    model_max_per_action_ece: float | None = None,
    model_n_rare_actions: int | None = None,
    model_n_insufficient_actions: int | None = None,
    model_n_rare_and_insufficient_actions: int | None = None,
) -> tuple[list[str], str]:
    """Compute warning codes and severity for a single report row.

    Parameters
    ----------
    row : dict
        A report row (must contain ``ESS, tail_mass, match_rate, min_pscore``;
        ``pareto_k`` is read if present).
    thresholds : SupportHealthThresholds
        Threshold set.
    n_samples : int
        Evaluation sample size (for ESS-fraction).
    model_ece : float, optional
        Per-model propensity Expected Calibration Error (#84). When provided
        and above ``thresholds.miscal_ece``, ``MISCAL_PROP`` is emitted.
        ``nan`` and ``None`` are treated identically and skip the check.

    Returns
    -------
    codes : list of str
        Stable-ordered warning codes.
    severity : str
        One of ``"ok"``, ``"caution"``, ``"high_risk"``.
    """
    n = max(int(n_samples), 1)
    overlap_floor = thresholds.poor_overlap_min_pscore
    if overlap_floor is None:
        overlap_floor = 1.0 / n

    caution: list[str] = []
    high: list[str] = []

    ess_frac = float(row["ESS"]) / n
    if ess_frac < thresholds.low_ess_frac / 2:
        high.append(WARN_LOW_ESS)
    elif ess_frac < thresholds.low_ess_frac:
        caution.append(WARN_LOW_ESS)

    tail_mass = float(row["tail_mass"])
    if tail_mass > thresholds.extreme_clip_tail_mass * 2:
        high.append(WARN_EXTREME_CLIP)
    elif tail_mass > thresholds.extreme_clip_tail_mass:
        caution.append(WARN_EXTREME_CLIP)

    match_rate = float(row["match_rate"])
    if match_rate < thresholds.low_match_rate / 2:
        high.append(WARN_LOW_MATCH_RATE)
    elif match_rate < thresholds.low_match_rate:
        caution.append(WARN_LOW_MATCH_RATE)

    min_pscore = float(row["min_pscore"])
    if min_pscore < overlap_floor:
        high.append(WARN_POOR_OVERLAP)

    # HIGH_PARETO_K (#80): Pareto-k of the unclipped importance-weight tail.
    # ``row['pareto_k']`` may be missing on legacy callers that build their own
    # report rows; treat missing/nan as "no signal" rather than fail loudly.
    pk_raw = row.get("pareto_k")
    if pk_raw is not None:
        pk = float(pk_raw)
        if np.isfinite(pk):
            if pk >= thresholds.high_pareto_k:
                high.append(WARN_HIGH_PARETO_K)
            elif pk >= thresholds.high_pareto_k_caution:
                caution.append(WARN_HIGH_PARETO_K)

    # MISCAL_PROP (#84): per-model propensity ECE.  Same nan-as-no-signal rule.
    if model_ece is not None and np.isfinite(float(model_ece)):
        ece = float(model_ece)
        if ece > thresholds.miscal_ece * 2:
            high.append(WARN_MISCAL_PROP)
        elif ece > thresholds.miscal_ece:
            caution.append(WARN_MISCAL_PROP)

    # PER_ACTION_MISCAL (#131): a single action's ECE blows past the global
    # threshold even when the global ECE looks healthy. Threshold is the
    # same ``miscal_ece``; per-action evidence is strictly *more* sensitive
    # than the global signal so this fires when MISCAL_PROP would not.
    if model_max_per_action_ece is not None and np.isfinite(
        float(model_max_per_action_ece)
    ):
        per_ece = float(model_max_per_action_ece)
        if per_ece > thresholds.miscal_ece * 2:
            high.append(WARN_PER_ACTION_MISCAL)
        elif per_ece > thresholds.miscal_ece:
            caution.append(WARN_PER_ACTION_MISCAL)

    # RARE_ACTION_NO_SUPPORT (#131): at least one target-support action is
    # *both* rare (logged frequency below ``rare_action_floor``) AND
    # insufficient (fewer than ``_MIN_ACTION_COUNT_DISC`` samples). Rare and
    # insufficient must be the *same* action for this to be a high_risk
    # signal — disjoint rare and insufficient actions don't compose.
    # ``model_n_rare_and_insufficient_actions`` carries that intersection.
    # Falls back to the legacy disjoint check only when the new count is
    # unavailable (older callers / pre-#131 diagnostics).
    rare_and_insuff = model_n_rare_and_insufficient_actions
    if rare_and_insuff is None:
        rare = model_n_rare_actions or 0
        insuff = model_n_insufficient_actions or 0
        if rare > 0 and insuff > 0:
            high.append(WARN_RARE_ACTION_NO_SUPPORT)
    elif rare_and_insuff > 0:
        high.append(WARN_RARE_ACTION_NO_SUPPORT)

    # Stable order: LOW_ESS, EXTREME_CLIP, LOW_MATCH_RATE, POOR_OVERLAP,
    # HIGH_PARETO_K, MISCAL_PROP.  New codes append rather than interleave so
    # historical artifact diffs remain readable.
    code_order = [
        WARN_LOW_ESS,
        WARN_EXTREME_CLIP,
        WARN_LOW_MATCH_RATE,
        WARN_POOR_OVERLAP,
        WARN_HIGH_PARETO_K,
        WARN_MISCAL_PROP,
        WARN_PER_ACTION_MISCAL,
        WARN_RARE_ACTION_NO_SUPPORT,
    ]
    codes = [c for c in code_order if c in caution or c in high]

    if high or len(caution) >= _CAUTION_ESCALATE_COUNT:
        severity = SUPPORT_HIGH_RISK
    elif caution:
        severity = SUPPORT_CAUTION
    else:
        severity = SUPPORT_OK
    return codes, severity


def attach_warnings(
    report: pd.DataFrame,
    n_samples: int,
    thresholds: SupportHealthThresholds | None = None,
    *,
    model_ece: dict[str, float] | None = None,
    model_per_action_ece: dict[str, float] | None = None,
    model_n_rare_actions: dict[str, int] | None = None,
    model_n_insufficient_actions: dict[str, int] | None = None,
    model_n_rare_and_insufficient_actions: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``support_health`` and ``diagnostic_warnings`` columns.

    Parameters
    ----------
    report : pd.DataFrame
        Report from ``evaluate_*_models``. Must contain at least the columns
        ``model, estimator, ESS, tail_mass, match_rate, min_pscore``.
        ``pareto_k`` is read if present (drives ``HIGH_PARETO_K``; #80).
    n_samples : int
        Number of evaluation samples (used for ESS-fraction and default
        overlap floor). Must be positive.
    thresholds : SupportHealthThresholds, optional
        Threshold set. Defaults to :class:`SupportHealthThresholds`.
    model_ece : dict[str, float], optional
        Map ``model_name -> Expected Calibration Error`` (#84). When supplied,
        ``MISCAL_PROP`` may be emitted on rows whose model has ECE above
        ``thresholds.miscal_ece``. Models missing from this dict (or values
        of ``nan``) are treated as "no signal" and skip the check.

    Returns
    -------
    enriched_report : pd.DataFrame
        Copy of ``report`` with two new columns:

        - ``support_health`` : str — ``"ok"`` | ``"caution"`` | ``"high_risk"``.
        - ``diagnostic_warnings`` : str — comma-joined warning codes.
    warnings_df : pd.DataFrame
        Per-row warning records with columns
        ``model, estimator, support_health, warning_codes`` where
        ``warning_codes`` is a Python list (not a string).
    """
    if thresholds is None:
        thresholds = SupportHealthThresholds()
    if n_samples <= 0:
        raise DataValidationError(
            f"n_samples must be positive, got {n_samples}",
        )

    required = {"model", "estimator", "ESS", "tail_mass", "match_rate", "min_pscore"}
    missing = required - set(report.columns)
    if missing:
        raise DataValidationError(
            f"report is missing required columns for warnings: {sorted(missing)}",
        )

    ece_lookup = model_ece or {}
    per_action_ece_lookup = model_per_action_ece or {}
    rare_lookup = model_n_rare_actions or {}
    insuff_lookup = model_n_insufficient_actions or {}
    rare_and_insuff_lookup = model_n_rare_and_insufficient_actions or {}
    severities: list[str] = []
    code_strings: list[str] = []
    warning_records: list[dict[str, Any]] = []
    for _, row in report.iterrows():
        model_key = str(row["model"])
        ece_val = ece_lookup.get(model_key)
        codes, severity = _compute_row_warnings(
            row.to_dict(),
            thresholds,
            n_samples,
            model_ece=ece_val,
            model_max_per_action_ece=per_action_ece_lookup.get(model_key),
            model_n_rare_actions=rare_lookup.get(model_key),
            model_n_insufficient_actions=insuff_lookup.get(model_key),
            model_n_rare_and_insufficient_actions=rare_and_insuff_lookup.get(model_key),
        )
        severities.append(severity)
        code_strings.append(",".join(codes))
        warning_records.append(
            {
                "model": str(row["model"]),
                "estimator": str(row["estimator"]),
                "support_health": severity,
                "warning_codes": codes,
            }
        )

    enriched = report.copy()
    enriched["support_health"] = severities
    enriched["diagnostic_warnings"] = code_strings
    warnings_df = pd.DataFrame(warning_records)
    return enriched, warnings_df


# --------------------------------------------------------------------------- #
# Clip-grid sensitivity                                                       #
# --------------------------------------------------------------------------- #


_STABILITY_RANGE_FRAC = 0.10  # range/|chosen_V| must stay below this to be "stable"

# #133 — stability-grade boundaries on V_range / |chosen_V|. The "sensitive"
# band is wider than the original ``_STABILITY_RANGE_FRAC`` cliff so the
# grade carries genuine triage signal: stable < 10% < sensitive < 25% < unstable.
_STABILITY_GRADE_SENSITIVE_FRAC = 0.10
_STABILITY_GRADE_UNSTABLE_FRAC = 0.25

STABILITY_GRADES = ("stable", "sensitive", "unstable")


def _stability_grade(v_range_frac: float, dr_sndr_agree: bool) -> str:
    """Map a clip-grid range fraction + DR/SNDR agreement to a grade.

    Used by :func:`summarize_sensitivity` (#133). The grade is intended for
    triage, not as a substitute for a confidence-interval overlap test.

    The three bands collapse two distinct signals — clip-grid spread
    (``v_range_frac``) and DR↔SNDR consensus (``dr_sndr_agree``) — into a
    single label. ``"sensitive"`` therefore covers two semantically different
    cases:

    1. ``v_range_frac < 10%`` but DR and SNDR disagree (tight range, hidden
       estimator-strategy disagreement), AND
    2. ``10% ≤ v_range_frac < 25%`` regardless of DR↔SNDR agreement (wider
       range, possibly self-consistent).

    Consumers that need to distinguish the two should read ``dr_sndr_agree``
    directly from the sensitivity row alongside ``stability_grade``.
    ``"unstable"`` is also assigned on non-finite ``v_range_frac`` (e.g. when
    ``chosen_V`` is zero) so the grade stays defined on every row.
    """
    if not math.isfinite(v_range_frac):
        return "unstable"
    if v_range_frac < _STABILITY_GRADE_SENSITIVE_FRAC and dr_sndr_agree:
        return "stable"
    if v_range_frac < _STABILITY_GRADE_UNSTABLE_FRAC:
        return "sensitive"
    return "unstable"


def summarize_sensitivity(
    detailed_results: dict[str, dict[str, DRResult]],
) -> pd.DataFrame:
    """Summarize per-(model, estimator) value stability across the clip grid.

    For each ``DRResult`` in ``detailed_results``, reads ``DRResult.grid`` and
    computes:

    - ``V_min``, ``V_max``, ``V_range`` over the relevant per-estimator column
      (``V_DR`` for DR rows; ``V_SNDR`` for SNDR rows).
    - ``chosen_clip``, ``chosen_V`` from the selected operating point.
    - ``argmin_MSE_clip`` — the clip that minimizes the per-estimator MSE proxy.
    - ``dr_sndr_agree`` — whether DR and SNDR at ``chosen_clip`` are within one
      ``SE_if`` of each other.
    - ``stable`` — ``V_range / max(|chosen_V|, eps) < 0.10`` **and**
      ``dr_sndr_agree``.

    Caveats
    -------
    This is a heuristic stability summary, **not** a confidence-interval
    overlap test across clips. Issue #62 tracks a CI-based replacement once
    nuisance-resampled bootstrap is implemented.
    """
    rows: list[dict[str, Any]] = []
    for model_name, est_to_result in detailed_results.items():
        # Build a quick (clip -> V_DR, V_SNDR) lookup once per model.
        any_result = next(iter(est_to_result.values()))
        grid = any_result.grid
        if "clip" not in grid.columns:
            raise DataValidationError(
                f"DRResult.grid for {model_name!r} is missing 'clip' column",
            )
        clip_to_dr = dict(
            zip(
                grid["clip"],
                grid.get("V_DR", pd.Series([np.nan] * len(grid))),
                strict=False,
            )
        )
        clip_to_sndr = dict(
            zip(
                grid["clip"],
                grid.get("V_SNDR", pd.Series([np.nan] * len(grid))),
                strict=False,
            )
        )

        for est_name, res in est_to_result.items():
            v_col = f"V_{est_name}"
            mse_col = f"MSE_{est_name}"
            g = res.grid
            if v_col not in g.columns:
                # Estimator's column not present; skip rather than fabricate.
                continue

            v_series = g[v_col].astype(float)
            v_min = float(v_series.min())
            v_max = float(v_series.max())
            v_range = v_max - v_min
            chosen_clip = float(res.clip)
            chosen_v = float(res.V_hat)
            chosen_se = float(res.SE_if)

            if mse_col in g.columns:
                argmin_idx = int(g[mse_col].astype(float).idxmin())
                argmin_clip = float(g.loc[argmin_idx, "clip"])
            else:
                argmin_clip = chosen_clip

            dr_at = float(clip_to_dr.get(chosen_clip, np.nan))
            sndr_at = float(clip_to_sndr.get(chosen_clip, np.nan))
            if np.isnan(dr_at) or np.isnan(sndr_at):
                dr_sndr_agree = False
            else:
                tol = max(chosen_se, 1e-12)
                dr_sndr_agree = bool(abs(dr_at - sndr_at) <= tol)

            scale = max(abs(chosen_v), 1e-12)
            v_range_frac = v_range / scale
            stable = bool(v_range_frac < _STABILITY_RANGE_FRAC and dr_sndr_agree)
            # #133 — three-band stability grade and a normalized range
            # fraction. This is a triage signal, not a CI overlap test.
            grade = _stability_grade(v_range_frac, dr_sndr_agree)

            rows.append(
                {
                    "model": model_name,
                    "estimator": est_name,
                    "V_min": v_min,
                    "V_max": v_max,
                    "V_range": v_range,
                    "chosen_clip": chosen_clip,
                    "chosen_V": chosen_v,
                    "argmin_MSE_clip": argmin_clip,
                    "dr_sndr_agree": dr_sndr_agree,
                    "stable": stable,
                    "v_range_frac": float(v_range_frac),
                    "stability_grade": grade,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pydantic schema (used only for JSON I/O)                                    #
# --------------------------------------------------------------------------- #


class _ReportRowSchema(BaseModel):
    # Numeric fields are ``float | None`` because :func:`_jsonable` serializes
    # non-finite floats (NaN / ±inf) as ``null`` for RFC-8259 compliance. The
    # ``clip = inf`` "no clipping" sentinel is therefore wired as ``null`` in
    # the JSON; downstream consumers should treat null clips as unbounded.
    model_config = ConfigDict(extra="allow")
    model: str
    estimator: str
    V_hat: float | None
    SE_if: float | None
    clip: float | None
    ESS: float | None
    tail_mass: float | None
    MSE_est: float | None
    match_rate: float | None
    min_pscore: float | None
    pscore_q10: float | None
    pscore_q05: float | None
    pscore_q01: float | None
    ci_lower: float | None = None
    ci_upper: float | None = None
    support_health: str | None = None
    diagnostic_warnings: str | None = None
    # Trust-additions field (#80). Optional default keeps 1.0.0 schemas
    # loadable as-is — old payloads simply lack this key.
    pareto_k: float | None = None


class ReportRow(BaseModel):
    """Typed, public view of one ``EvaluationArtifact.report`` row (#234).

    A thin, generated projection of the headline report DataFrame so downstream
    code can read results by attribute (``row.V_hat``) instead of by string
    column name (``report["V_hat"]``). Built by
    :meth:`EvaluationArtifact.rows` / :meth:`EvaluationArtifact.row`; the
    DataFrame remains the source of truth and stays available unchanged.

    ``verdict`` / ``confidence`` mirror the :class:`Recommendation` for the row
    (``None`` when a recommendation could not be computed — e.g. no CI). Every
    numeric field is ``float | None`` for the same non-finite-→-``null`` reason
    documented on :class:`_ReportRowSchema`. ``extra="allow"`` keeps
    forward-added report columns (e.g. ``delta_V_hat``) accessible.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    estimator: str
    V_hat: float | None = None
    SE_if: float | None = None
    clip: float | None = None
    ESS: float | None = None
    tail_mass: float | None = None
    MSE_est: float | None = None
    match_rate: float | None = None
    min_pscore: float | None = None
    pscore_q10: float | None = None
    pscore_q05: float | None = None
    pscore_q01: float | None = None
    pareto_k: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    support_health: str | None = None
    diagnostic_warnings: str | None = None
    verdict: str | None = None
    confidence: str | None = None


class _WarningRowSchema(BaseModel):
    model: str
    estimator: str
    support_health: str
    warning_codes: list[str]


class _SensitivityRowSchema(BaseModel):
    # See _ReportRowSchema: clip / chosen_clip / argmin_MSE_clip can be the
    # ``inf`` "no clipping" sentinel, serialized as ``null``.
    model: str
    estimator: str
    V_min: float | None
    V_max: float | None
    V_range: float | None
    chosen_clip: float | None
    chosen_V: float | None
    argmin_MSE_clip: float | None
    dr_sndr_agree: bool
    stable: bool
    # #133 — Three-band decision-stability grade. Optional defaults keep 1.0.0
    # payloads loadable; new artifacts populate both fields.
    v_range_frac: float | None = None
    stability_grade: str | None = None


class _DiagnosticsPayloadSchema(BaseModel):
    # Non-finite diagnostic values are coerced to ``None`` by ``_jsonable``
    # (matches the report / sensitivity schemas) so that the emitted JSON
    # stays RFC-8259 compliant even if a diagnostic field is ill-defined.
    overlap_ratio: float | None
    balance_ratio: float | None
    calibration_score: float | None
    discrimination_score: float | None
    log_loss_score: float | None
    statistics: dict[str, float | None]
    balance_stats: dict[str, float | None]
    # Trust additions (#84). Optional defaults keep 1.0.0 payloads loadable;
    # old artifacts simply lack the ECE/Brier keys, which Pydantic v2 accepts
    # because the fields have ``None`` defaults.
    ece: float | None = None
    brier_score: float | None = None
    reliability_curve: list[tuple[float | None, float | None, int]] | None = None
    ece_n_bins: int | None = None


class ArtifactSchema(BaseModel):
    """Versioned, JSON-serializable schema for :class:`EvaluationArtifact`.

    This is the canonical wire-format. See :meth:`EvaluationArtifact.to_json`
    and :func:`load_artifact_json` for round-trip behaviour.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    skdr_eval_version: str
    timestamp: str
    metadata: dict[str, Any]
    report: list[_ReportRowSchema]
    warnings: list[_WarningRowSchema]
    sensitivity: list[_SensitivityRowSchema]
    diagnostics: dict[str, _DiagnosticsPayloadSchema]
    # #128 — Statistical contract on the wire. Defaults keep 1.0.0 payloads
    # loadable; new artifacts always populate the three fields.
    estimand_tex: str | None = None
    estimand_summary: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    # #132 — Baseline configuration so delta_V_hat / delta_ci_* columns on
    # the report row remain interpretable after a round trip.
    baseline_kind: str | None = None
    baseline_value: float | None = None

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the Pydantic-generated JSON Schema for the artifact (#205).

        The published, downloadable contract for the serialized
        ``artifact.json`` — the sibling of :meth:`EvaluationCard.json_schema`.
        Lets downstream tooling validate ``skdr-eval`` outputs without
        importing the library.
        """
        schema: dict[str, Any] = cls.model_json_schema()
        return schema


# --------------------------------------------------------------------------- #
# Jinja2 environment                                                          #
# --------------------------------------------------------------------------- #


def _fmt_num(value: Any, spec: str = "%.4g", default: str = "—") -> str:
    """Jinja-friendly numeric formatter that handles ``None`` gracefully.

    ``_jsonable`` coerces non-finite floats (NaN, ±inf) to ``None`` so the
    artifact JSON stays RFC-8259 compliant. Template cells still need a
    readable representation: this filter renders ``None`` as ``default``
    (use ``"∞"`` on known clip columns) and otherwise applies the printf
    ``spec`` to the value.
    """
    if value is None:
        return default
    return str(spec % value)


def _template_env() -> Environment:
    """Build the Jinja2 environment, resolving template path via the package."""
    template_dir = _resources.files("skdr_eval").joinpath("templates")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        keep_trailing_newline=True,
    )
    env.filters["fmt_num"] = _fmt_num
    return env


# --------------------------------------------------------------------------- #
# Helpers: JSON-safe conversion + diagnostics packing                         #
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars to JSON-friendly Python primitives.

    Non-finite floats (``NaN``, ``+inf``, ``-inf``) become ``None`` so the
    emitted JSON stays RFC-8259 compliant. ``clip = inf`` (the "no clipping"
    sentinel) is therefore serialized as ``null``; downstream consumers
    should treat null clips as the unbounded operating point.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        coerced = value.item()
        # Continue to the float / finiteness branch with the coerced primitive.
        value = coerced
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


_OPTIONAL_REPORT_KEYS = (
    "ci_lower",
    "ci_upper",
    "support_health",
    "diagnostic_warnings",
    "pareto_k",
)


def _normalize_report_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure templates can access optional report keys without ``UndefinedError``."""
    for key in _OPTIONAL_REPORT_KEYS:
        row.setdefault(key, None)
    return row


def _diag_to_payload(diag: PropensityDiagnostics) -> dict[str, Any]:
    """Convert a :class:`PropensityDiagnostics` to a JSON-safe payload dict.

    Every numeric field flows through :func:`_jsonable` so non-finite values
    (NaN, ±inf) are coerced to ``None`` and the resulting payload round-trips
    through RFC-8259 strict JSON parsers.

    The ``reliability_curve`` is emitted as a list of 3-tuples
    ``(bin_mean_predicted, bin_empirical_frac, bin_count)``; non-finite
    numerics again become ``None``, and the per-bin counts are kept as
    integers for downstream histogram code.
    """
    reliability_payload = [
        (_jsonable(float(mean_pred)), _jsonable(float(frac)), int(count))
        for (mean_pred, frac, count) in diag.reliability_curve
    ]
    return {
        "overlap_ratio": _jsonable(float(diag.overlap_ratio)),
        "balance_ratio": _jsonable(float(diag.balance_ratio)),
        "calibration_score": _jsonable(float(diag.calibration_score)),
        "discrimination_score": _jsonable(float(diag.discrimination_score)),
        "log_loss_score": _jsonable(float(diag.log_loss_score)),
        "statistics": {str(k): _jsonable(float(v)) for k, v in diag.statistics.items()},
        "balance_stats": {
            str(k): _jsonable(float(v)) for k, v in diag.balance_stats.items()
        },
        # Trust additions (#84). nan ECE / Brier / counts flow through
        # ``_jsonable`` so the JSON stays RFC-8259 compliant.
        "ece": _jsonable(float(diag.ece)),
        "brier_score": _jsonable(float(diag.brier_score)),
        "reliability_curve": reliability_payload,
        "ece_n_bins": int(diag.ece_n_bins),
    }


def _compute_diagnostics(
    propensities: np.ndarray | None,
    actions: np.ndarray | None,
    model_names: list[str],
    *,
    target_actions: np.ndarray | None = None,
) -> dict[str, PropensityDiagnostics]:
    """Compute per-model propensity diagnostics, sharing one fit across models.

    The propensity model is fitted once per evaluation run (not per model
    being evaluated), so we compute it once and then deep-copy it per model
    so callers mutating ``artifact.diagnostics[m].statistics`` for one model
    do not silently affect every other model in the run. Returns an empty
    dict if propensities are missing or diagnostics fail.

    ``target_actions`` (#131) is forwarded so the per-action ``rare`` flag
    respects target-policy support.
    """
    if propensities is None or actions is None or len(model_names) == 0:
        return {}
    try:
        shared = comprehensive_propensity_diagnostics(
            propensities, actions, target_actions=target_actions
        )
    except (DataValidationError, ConfigurationError, ValueError) as exc:
        logger.warning("Propensity diagnostics failed (%s); omitting.", exc)
        return {}
    return {name: copy.deepcopy(shared) for name in model_names}


def _detailed_to_jsonable(
    detailed: dict[str, dict[str, DRResult]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert detailed results to JSON-safe nested dicts (excluding grids)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, est_to_res in detailed.items():
        out[model_name] = {}
        for est_name, res in est_to_res.items():
            out[model_name][est_name] = {
                "clip": _jsonable(res.clip),
                "V_hat": _jsonable(res.V_hat),
                "SE_if": _jsonable(res.SE_if),
                "ESS": _jsonable(res.ESS),
                "tail_mass": _jsonable(res.tail_mass),
                "MSE_est": _jsonable(res.MSE_est),
                "match_rate": _jsonable(res.match_rate),
                "min_pscore": _jsonable(res.min_pscore),
                "pscore_q10": _jsonable(res.pscore_q10),
                "pscore_q05": _jsonable(res.pscore_q05),
                "pscore_q01": _jsonable(res.pscore_q01),
                "grid": [
                    {str(k): _jsonable(v) for k, v in row.items()}
                    for row in res.grid.to_dict(orient="records")
                ],
            }
    return out


# --------------------------------------------------------------------------- #
# EvaluationArtifact                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class EvaluationArtifact:
    """Bundled evaluation result returned by ``evaluate_*_models``.

    Combines the report DataFrame, per-model ``DRResult`` map, support-health
    warnings (#22), clip-grid sensitivity (#27), and propensity diagnostics
    (#23) under a single versioned object. Exports to JSON / HTML and renders
    a stakeholder-ready card (#28, #30).

    Attributes
    ----------
    report : pd.DataFrame
        Per-(model, estimator) headline metrics, plus ``support_health`` and
        ``diagnostic_warnings`` columns appended by :func:`attach_warnings`.
    detailed : dict[str, dict[str, DRResult]]
        Same as the legacy second-return-value of ``evaluate_*_models``.
    warnings : pd.DataFrame
        Per-row warning records (``model, estimator, support_health,
        warning_codes``).
    sensitivity : pd.DataFrame
        Per-(model, estimator) clip-grid sensitivity summary; see
        :func:`summarize_sensitivity`.
    diagnostics : dict[str, PropensityDiagnostics]
        Propensity diagnostics keyed by model name. May be empty if
        propensities are unavailable.
    metadata : dict[str, Any]
        Run metadata (``schema_version``, ``skdr_eval_version``, ``timestamp``,
        ``n_samples``, ``random_state``, ``evaluator``, …).
    """

    report: pd.DataFrame
    detailed: dict[str, dict[str, DRResult]]
    warnings: pd.DataFrame
    sensitivity: pd.DataFrame
    diagnostics: dict[str, PropensityDiagnostics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # #128 — Statistical contract carried with the artifact and serialized
    # on every card. Defaults are the standard contextual-bandit estimand
    # and the seven canonical assumption tags.
    estimand_tex: str = DEFAULT_ESTIMAND_TEX
    estimand_summary: str = DEFAULT_ESTIMAND_SUMMARY
    assumptions: list[str] = field(
        default_factory=lambda: list(DEFAULT_ASSUMPTION_TAGS)
    )
    # #132 — Baseline configuration (kind, value) so the artifact carries
    # how delta_V_hat columns on the report were computed.
    baseline_kind: str | None = None
    baseline_value: float | None = None

    # ------------------------------------------------------------------ #
    # Pydantic-backed payload                                            #
    # ------------------------------------------------------------------ #

    def to_schema(self) -> ArtifactSchema:
        """Materialize the JSON-serializable :class:`ArtifactSchema` view."""
        report_rows = [
            _ReportRowSchema(**{k: _jsonable(v) for k, v in row.items()})
            for row in self.report.to_dict(orient="records")
        ]
        warning_rows = [
            _WarningRowSchema(**{k: _jsonable(v) for k, v in row.items()})
            for row in self.warnings.to_dict(orient="records")
        ]
        sensitivity_rows = [
            _SensitivityRowSchema(**{k: _jsonable(v) for k, v in row.items()})
            for row in self.sensitivity.to_dict(orient="records")
        ]
        diag_payload = {
            name: _DiagnosticsPayloadSchema(**_diag_to_payload(diag))
            for name, diag in self.diagnostics.items()
        }
        return ArtifactSchema(
            schema_version=str(self.metadata.get("schema_version", SCHEMA_VERSION)),
            skdr_eval_version=str(self.metadata.get("skdr_eval_version", "unknown")),
            timestamp=str(self.metadata.get("timestamp", "")),
            metadata={
                k: _jsonable(v)
                for k, v in self.metadata.items()
                if k not in {"schema_version", "skdr_eval_version", "timestamp"}
            },
            report=report_rows,
            warnings=warning_rows,
            sensitivity=sensitivity_rows,
            diagnostics=diag_payload,
            estimand_tex=self.estimand_tex,
            estimand_summary=self.estimand_summary,
            assumptions=list(self.assumptions),
            baseline_kind=self.baseline_kind,
            baseline_value=self.baseline_value,
        )

    # ------------------------------------------------------------------ #
    # JSON                                                               #
    # ------------------------------------------------------------------ #

    def to_json_str(self, *, indent: int | None = 2) -> str:
        """Serialize the artifact to a JSON string via the Pydantic schema."""
        return str(self.to_schema().model_dump_json(indent=indent))

    def to_json(
        self, path: str | Path | None = None, *, indent: int | None = 2
    ) -> str | Path:
        """Serialize the artifact to JSON.

        Follows the pandas-style convention: with ``path=None`` (the default)
        the JSON **string** is returned; with a ``path`` the JSON is written
        there (parent dirs created) and the written :class:`~pathlib.Path` is
        returned.
        """
        text = self.to_json_str(indent=indent)
        if path is None:
            return text
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        logger.info("Wrote evaluation artifact JSON to %s", p)
        return p

    # ------------------------------------------------------------------ #
    # HTML                                                               #
    # ------------------------------------------------------------------ #

    def to_html_str(self) -> str:
        """Render the full artifact as a self-contained HTML page."""
        env = _template_env()
        template = env.get_template("report.html")
        report_rows = [
            _normalize_report_row({k: _jsonable(v) for k, v in row.items()})
            for row in self.report.to_dict(orient="records")
        ]
        warnings_rows = [
            {k: _jsonable(v) for k, v in row.items()}
            for row in self.warnings.to_dict(orient="records")
        ]
        sensitivity_rows = [
            {k: _jsonable(v) for k, v in row.items()}
            for row in self.sensitivity.to_dict(orient="records")
        ]
        diagnostics_payload = {
            name: _diag_to_payload(diag) for name, diag in self.diagnostics.items()
        }
        return str(
            template.render(
                schema_version=self.metadata.get("schema_version", SCHEMA_VERSION),
                skdr_eval_version=self.metadata.get("skdr_eval_version", "unknown"),
                timestamp=self.metadata.get("timestamp", ""),
                report_rows=report_rows,
                warnings_rows=warnings_rows,
                sensitivity_rows=sensitivity_rows,
                diagnostics=diagnostics_payload,
                metadata_json=json.dumps(
                    {k: _jsonable(v) for k, v in self.metadata.items()},
                    indent=2,
                    sort_keys=True,
                ),
            )
        )

    def to_html(self, path: str | Path | None = None) -> str | Path:
        """Render the artifact as HTML.

        Follows the pandas-style convention: with ``path=None`` (the default)
        the HTML **string** is returned; with a ``path`` the HTML is written
        there (parent dirs created) and the written :class:`~pathlib.Path` is
        returned.
        """
        html = self.to_html_str()
        if path is None:
            return html
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        logger.info("Wrote evaluation artifact HTML to %s", p)
        return p

    # ------------------------------------------------------------------ #
    # Polars / Arrow accessors (#72)                                     #
    # ------------------------------------------------------------------ #

    def to_polars(self) -> Any:
        """Return the headline :attr:`report` as a Polars ``DataFrame``.

        Convenience accessor for the ``[speed]`` extra. The underlying
        :attr:`report` pandas frame is unchanged; this only converts the
        public-facing table on demand so Polars-native pipelines avoid a
        manual ``pl.from_pandas`` round-trip.

        Raises
        ------
        OptionalDependencyError
            If ``polars`` is not installed.
        """
        try:
            import polars as pl  # noqa: PLC0415
        except ImportError as exc:
            raise OptionalDependencyError(
                "EvaluationArtifact.to_polars", "polars", extra="speed"
            ) from exc
        return pl.from_pandas(self.report)

    def to_arrow(self) -> Any:
        """Return the headline :attr:`report` as a PyArrow ``Table``.

        Convenience accessor for the ``[speed]`` extra; see :meth:`to_polars`.

        Raises
        ------
        OptionalDependencyError
            If ``pyarrow`` is not installed.
        """
        try:
            import pyarrow as pa  # noqa: PLC0415
        except ImportError as exc:
            raise OptionalDependencyError(
                "EvaluationArtifact.to_arrow", "pyarrow", extra="speed"
            ) from exc
        return pa.Table.from_pandas(self.report)

    # ------------------------------------------------------------------ #
    # Card                                                               #
    # ------------------------------------------------------------------ #

    def _build_card_context(
        self,
        model_name: str,
        *,
        headline_estimator: str,
    ) -> dict[str, Any]:
        if model_name not in self.detailed:
            raise DataValidationError(
                f"model {model_name!r} not in artifact "
                f"(known: {sorted(self.detailed)})",
            )
        est_map = self.detailed[model_name]
        if headline_estimator not in est_map:
            raise DataValidationError(
                f"estimator {headline_estimator!r} not in detailed[{model_name!r}] "
                f"(known: {sorted(est_map)})",
            )

        rep = self.report[self.report["model"] == model_name]
        head = rep[rep["estimator"] == headline_estimator].iloc[0]
        head_dict = {k: _jsonable(v) for k, v in head.to_dict().items()}

        ci_lower = head_dict.get("ci_lower")
        ci_upper = head_dict.get("ci_upper")
        warn_codes_str = head_dict.get("diagnostic_warnings") or ""
        warn_codes = [c for c in warn_codes_str.split(",") if c]
        support = head_dict.get("support_health") or SUPPORT_OK
        interpretation = _build_interpretation(
            head_dict["V_hat"],
            head_dict["SE_if"],
            support,
            warn_codes,
            ci_lower,
            ci_upper,
        )
        plot_b64 = _render_sensitivity_plot(
            est_map[headline_estimator], headline_estimator
        )

        estimator_rows = [
            {k: _jsonable(v) for k, v in row.items()}
            for row in rep.to_dict(orient="records")
        ]

        diag = self.diagnostics.get(model_name)
        diag_payload = _diag_to_payload(diag) if diag is not None else None

        # Top contributors / detractors block (#92). Available when the
        # evaluator was called with keep_contributions=True; otherwise the
        # lists stay empty.
        top_contributors, bottom_detractors = self._card_contribution_rows(
            model_name, headline_estimator
        )

        return {
            "model_name": model_name,
            "task_summary": self.metadata.get("evaluator", "skdr_eval"),
            "headline_estimator": headline_estimator,
            "headline_v": head_dict["V_hat"],
            "headline_se": head_dict["SE_if"],
            "headline_ci_lower": ci_lower,
            "headline_ci_upper": ci_upper,
            "ci_label": _ci_label(self.metadata.get("alpha", 0.05)),
            "support_health": support,
            "warning_codes": warn_codes,
            # Trust diagnostics (#80): the Pareto-k for the headline row is
            # the same value carried on every (model, estimator) row of the
            # report, but the card surfaces only the headline slot.
            "headline_pareto_k": head_dict.get("pareto_k"),
            "interpretation": interpretation,
            "plot_b64": plot_b64,
            "estimator_rows": estimator_rows,
            "diagnostics": diag_payload,
            "top_contributors": top_contributors,
            "bottom_detractors": bottom_detractors,
            "schema_version": self.metadata.get("schema_version", SCHEMA_VERSION),
            "skdr_eval_version": self.metadata.get("skdr_eval_version", "unknown"),
            "timestamp": self.metadata.get("timestamp", ""),
        }

    def _card_contribution_rows(
        self,
        model_name: str,
        headline_estimator: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (top contributors, bottom detractors) for the card.

        Empty lists when contributions are not captured for this run.
        """
        result = self.detailed[model_name][headline_estimator]
        if result.contributions is None:
            return [], []
        contrib = result.contributions["contribution_to_V"]
        decision_id = result.contributions["decision_id"]
        n_contrib = len(contrib)
        if n_contrib == 0:
            return [], []
        order = np.argsort(contrib)
        if n_contrib == 1:
            # Only one decision: show it as a single top contributor, no
            # bottom block — otherwise the same id would render in both.
            return [
                {
                    "decision_id": int(decision_id[order[0]]),
                    "contribution_to_V": float(contrib[order[0]]),
                }
            ], []
        # Partition into two non-overlapping halves so the card never shows the
        # same decision in both "top contributors" and "bottom detractors" when
        # n_contrib < 10. Each block caps at 5.
        n_show = min(5, n_contrib // 2)
        bottom_idx = order[:n_show]
        top_idx = order[-n_show:][::-1]
        top = [
            {
                "decision_id": int(decision_id[i]),
                "contribution_to_V": float(contrib[i]),
            }
            for i in top_idx
        ]
        bottom = [
            {
                "decision_id": int(decision_id[i]),
                "contribution_to_V": float(contrib[i]),
            }
            for i in bottom_idx
        ]
        return top, bottom

    def card(
        self,
        model_name: str,
        *,
        headline_estimator: str = "SNDR",
        format: str = "html",
    ) -> str:
        """Render a stakeholder-ready evaluation card for one model.

        Parameters
        ----------
        model_name : str
            Model to render. Must be present in :attr:`detailed`.
        headline_estimator : str, default ``"SNDR"``
            Which estimator's value to feature in the headline.
        format : ``"html"``, default ``"html"``
            Currently HTML only. Reserved for future formats.

        Returns
        -------
        str
            The rendered card markup.
        """
        if format != "html":
            raise ConfigurationError(
                f"Unknown card format: {format!r} (supported: 'html')",
            )
        ctx = self._build_card_context(
            model_name, headline_estimator=headline_estimator
        )
        env = _template_env()
        template = env.get_template("card.html")
        return str(template.render(**ctx))

    def save_card(
        self,
        path: str | Path,
        model_name: str,
        *,
        headline_estimator: str = "SNDR",
    ) -> Path:
        """Render and write a card to ``path``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            self.card(model_name, headline_estimator=headline_estimator),
            encoding="utf-8",
        )
        logger.info("Wrote evaluation card to %s", p)
        return p

    # ------------------------------------------------------------------ #
    # Recommendation (#83)                                               #
    # ------------------------------------------------------------------ #

    def recommendation(
        self,
        model_name: str,
        *,
        estimator: str = "SNDR",
        baseline: float | None = None,
        policy: RecommendationPolicy | None = None,
    ) -> Recommendation:
        """Generate a structured deployment recommendation for one model.

        Aggregates support-health warnings, CI position relative to a
        ``baseline``, and diagnostic flags into a single
        :class:`Recommendation` object.

        Parameters
        ----------
        model_name : str
            Model in :attr:`detailed`.
        estimator : str, default ``"SNDR"``
            Estimator row to base the recommendation on (``"DR"`` or
            ``"SNDR"``).
        baseline : float or None, default None
            Minimum policy value threshold.  The CI gate checks
            ``ci_lower > baseline``.  Takes precedence over
            ``policy.baseline`` when provided (even if 0.0).
            When ``None``, falls back to ``policy.baseline`` if a policy
            is given, otherwise defaults to 0.0.
        policy : RecommendationPolicy, optional
            Advanced policy object.  ``baseline`` kwarg takes precedence
            when supplied.

        Returns
        -------
        Recommendation
            Structured verdict with explanatory :class:`Reason` objects.

        Raises
        ------
        DataValidationError
            If ``model_name`` or ``estimator`` is not found in the artifact.
        """
        if baseline is not None:
            _policy = RecommendationPolicy(baseline=baseline)
        elif policy is not None:
            _policy = policy
        else:
            _policy = RecommendationPolicy(baseline=0.0)
        return _build_recommendation(self, model_name, estimator, _policy)

    def explain(
        self,
        model_name: str,
        *,
        estimator: str = "SNDR",
        baseline: float | None = None,
    ) -> Explanation:
        """Narrate *why* one (model, estimator) row received its verdict (#201).

        Combines the structured :class:`Recommendation` (verdict, confidence,
        reasons) with the :func:`gate_diagnostics` pass/warn/fail gate and the
        headline estimate/CI into a single :class:`Explanation`. This is a pure
        presentation layer over already-computed fields — it never recomputes
        the evaluation, so the narrative cannot drift from the card/report.

        Parameters
        ----------
        model_name : str
            Model present in :attr:`detailed`.
        estimator : str, default ``"SNDR"``
            Estimator row to explain (any first-class estimator in the artifact).
        baseline : float or None, default None
            Baseline the CI is compared against; defaults to ``0.0``.

        Returns
        -------
        Explanation
            Structured, renderable narrative.

        Raises
        ------
        DataValidationError
            If ``model_name`` or ``estimator`` is not found in the artifact.
        """
        rec = self.recommendation(model_name, estimator=estimator, baseline=baseline)
        # The gate is best-effort context: model/estimator membership was already
        # validated by ``recommendation`` above, so the only expected failures here
        # are missing diagnostic columns / metadata in a thinly reconstructed
        # artifact. Catch those narrowly (not bare ``Exception``) so a genuine bug
        # in ``gate_diagnostics`` still surfaces, and record why the gate was dropped.
        try:
            gate: DiagnosticGate | None = gate_diagnostics(self, model_name, estimator)
        except (
            DataValidationError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:  # pragma: no cover - gate is best-effort context
            logger.debug(
                "explain: gate omitted for %s/%s (best-effort): %s",
                model_name,
                estimator,
                exc,
            )
            gate = None

        row_mask = (self.report["model"] == model_name) & (
            self.report["estimator"] == estimator
        )
        rows = self.report[row_mask]
        row = rows.iloc[0] if not rows.empty else None
        return Explanation(
            model_name=model_name,
            estimator=estimator,
            verdict=rec.verdict,
            confidence=rec.confidence,
            primary_blocker=rec.primary_blocker,
            V_hat=_coerce_optional_float(row.get("V_hat")) if row is not None else None,
            ci_lower=(
                _coerce_optional_float(row.get("ci_lower")) if row is not None else None
            ),
            ci_upper=(
                _coerce_optional_float(row.get("ci_upper")) if row is not None else None
            ),
            baseline=baseline if baseline is not None else 0.0,
            reasons=rec.reasons,
            gate=gate,
        )

    # ------------------------------------------------------------------ #
    # Card schema (#88)                                                   #
    # ------------------------------------------------------------------ #

    def card_schema(
        self,
        model_name: str,
        *,
        estimator: str = "SNDR",
        baseline: float | None = None,
        include_gate: bool = True,
        include_recommendation: bool = True,
    ) -> EvaluationCard:
        """Build a machine-readable :class:`EvaluationCard` for one model row.

        The card is the typed sibling of :meth:`card` (HTML) and is intended
        to be pinned in Git, posted to a tracker (#93), or gated in CI (e.g.,
        ``if card.trust.recommendation['verdict'] == 'do_not_deploy': exit(1)``).

        Parameters
        ----------
        model_name : str
            Model present in :attr:`detailed`.
        estimator : str, default ``"SNDR"``
            ``"DR"`` or ``"SNDR"``.
        baseline : float or None
            Baseline policy value used in :class:`HeadlineBlock` and in the
            recommendation. Defaults to ``0.0`` for the recommendation when
            ``None`` is passed.
        include_gate : bool, default True
            Whether to embed the :func:`gate_diagnostics` result in the
            diagnostics block. Failures are swallowed (rendered as ``None``).
        include_recommendation : bool, default True
            Whether to embed the :class:`Recommendation` dict in the trust
            block. Failures are swallowed (rendered as ``None``).

        Returns
        -------
        EvaluationCard
            A validated card view of the row.

        Raises
        ------
        DataValidationError
            If the ``(model_name, estimator)`` row is not present.
        """
        return _build_card_from_row(
            self,
            model_name,
            estimator,
            baseline=baseline,
            include_gate=include_gate,
            include_recommendation=include_recommendation,
        )

    # ------------------------------------------------------------------ #
    # Per-decision contributions (#92)                                    #
    # ------------------------------------------------------------------ #

    _CONTRIBUTION_COLUMNS = (
        "decision_id",
        "q_pi",
        "q_hat",
        "weight",
        "reward",
        "contribution_to_V",
    )

    def contributions(
        self,
        model_name: str,
        *,
        estimator: str = "DR",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Per-decision contributions to ``V_hat`` (issue #92).

        Returns a DataFrame with columns ``decision_id, q_pi, q_hat, weight,
        reward, contribution_to_V``. By construction
        ``contribution_to_V.mean() == V_hat`` to float64 precision for the
        selected ``(model_name, estimator)`` row in :attr:`report`.

        Only available when the evaluator was called with
        ``keep_contributions=True``.

        Parameters
        ----------
        model_name : str
            Model in :attr:`detailed`.
        estimator : ``"DR"`` or ``"SNDR"``, default ``"DR"``
            Which estimator's contributions to return.
        top_k : int, optional
            If set, return only the ``top_k`` largest-magnitude
            contributions (``contribution_to_V.abs().nlargest(top_k)``).
            ``None`` returns all rows.

        Returns
        -------
        pd.DataFrame
            One row per evaluated decision (or ``top_k`` rows).

        Raises
        ------
        DataValidationError
            If ``model_name``/``estimator`` is unknown, or if contributions
            were not captured for this run.
        """
        if model_name not in self.detailed:
            raise DataValidationError(
                f"model {model_name!r} not in artifact "
                f"(known: {sorted(self.detailed)})",
            )
        est_map = self.detailed[model_name]
        if estimator not in est_map:
            raise DataValidationError(
                f"estimator {estimator!r} not in detailed[{model_name!r}] "
                f"(known: {sorted(est_map)})",
            )

        result = est_map[estimator]
        payload = result.contributions
        if payload is None:
            raise DataValidationError(
                "per-decision contributions are not available; re-run with "
                "keep_contributions=True.",
            )

        frame = pd.DataFrame({col: payload[col] for col in self._CONTRIBUTION_COLUMNS})
        if top_k is not None:
            if top_k <= 0:
                raise DataValidationError(
                    f"top_k must be a positive integer, got {top_k!r}",
                )
            ordering = frame["contribution_to_V"].abs().sort_values(ascending=False)
            frame = frame.loc[ordering.index[:top_k]].reset_index(drop=True)
        return frame

    # ------------------------------------------------------------------ #
    # Typed results façade (#234)                                         #
    # ------------------------------------------------------------------ #

    def _row_to_report_row(self, row: dict[str, Any]) -> ReportRow:
        """Build one :class:`ReportRow`, filling verdict/confidence if possible."""
        model_name = str(row["model"])
        estimator = str(row["estimator"])
        verdict: str | None = None
        confidence: str | None = None
        try:
            rec = self.recommendation(model_name, estimator=estimator)
            verdict = rec.verdict
            confidence = rec.confidence
        except (DataValidationError, KeyError, ValueError):
            # A row whose verdict cannot be computed (e.g. no CI, or a thin
            # artifact) still yields a typed row — verdict simply stays None.
            pass
        payload = {k: _jsonable(v) for k, v in row.items()}
        payload["verdict"] = verdict
        payload["confidence"] = confidence
        return ReportRow(**payload)

    def rows(self) -> list[ReportRow]:
        """Return the headline report as a list of typed :class:`ReportRow` (#234).

        A typed, attribute-addressable view of :attr:`report`; the DataFrame is
        left untouched and remains the source of truth. Each row carries the
        report columns plus the computed ``verdict``/``confidence`` (``None``
        when a recommendation cannot be produced for that row).
        """
        return [
            self._row_to_report_row(row)
            for row in self.report.to_dict(orient="records")
        ]

    def row(self, model_name: str, estimator: str = "SNDR") -> ReportRow:
        """Return the typed :class:`ReportRow` for one ``(model, estimator)`` (#234).

        Raises
        ------
        DataValidationError
            If no report row matches ``(model_name, estimator)``.
        """
        report = self.report
        mask = (report["model"] == model_name) & (report["estimator"] == estimator)
        matched = report[mask]
        if matched.empty:
            raise DataValidationError(
                f"No report row for model={model_name!r}, estimator={estimator!r} "
                f"(known estimators for this model: "
                f"{sorted(report.loc[report['model'] == model_name, 'estimator'])}).",
            )
        return self._row_to_report_row(matched.iloc[0].to_dict())

    # ------------------------------------------------------------------ #
    # Bulk export                                                         #
    # ------------------------------------------------------------------ #

    def export(
        self,
        path: str | Path,
        *,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Write artifact in one or more formats.

        Parameters
        ----------
        path : str or Path
            Path semantics:

            - If ``path`` already has a suffix (e.g. ``run.json``), the
              suffix is replaced per format: ``run.json`` becomes
              ``run.json`` and ``run.html``.
            - If ``path`` is an existing directory **or** ends in a path
              separator, files are written as ``<path>/artifact.<fmt>``.
            - Otherwise ``path`` is treated as a stem and ``.<fmt>`` is
              appended directly: ``"artifacts/run"`` produces
              ``artifacts/run.json`` and ``artifacts/run.html``.
        formats : list of {"json", "html", "markdown"}, optional
            Defaults to ``["json", "html"]``. ``"markdown"`` writes the
            compact :meth:`to_markdown` summary with a ``.md`` suffix (#237).

        Returns
        -------
        dict[str, Path]
            Map of format name to written path.
        """
        if formats is None:
            formats = ["json", "html"]
        supported = {"json", "html", "markdown"}
        unknown = set(formats) - supported
        if unknown:
            raise ConfigurationError(
                f"Unknown export format(s): {sorted(unknown)} "
                f"(supported: {sorted(supported)})",
            )
        # ``markdown`` maps to a ``.md`` file suffix; other formats use the
        # format name directly.
        _suffix = {"markdown": "md"}

        p = Path(path)
        # Trailing-separator hint must be checked on the original string,
        # since Path() normalizes the trailing slash away.
        explicit_dir = isinstance(path, str) and (
            path.endswith("/") or path.endswith("\\")
        )
        treat_as_dir = explicit_dir or (p.exists() and p.is_dir())

        written: dict[str, Path] = {}
        for fmt in formats:
            ext = _suffix.get(fmt, fmt)
            target = p / f"artifact.{ext}" if treat_as_dir else p.with_suffix(f".{ext}")
            if fmt == "json":
                # ``target`` is always a concrete path here, so to_json returns
                # the written Path (never the str branch).
                written["json"] = cast("Path", self.to_json(target))
            elif fmt == "html":
                written["html"] = cast("Path", self.to_html(target))
            elif fmt == "markdown":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.to_markdown(), encoding="utf-8")
                logger.info("Wrote evaluation artifact Markdown to %s", target)
                written["markdown"] = target
        return written

    # ------------------------------------------------------------------ #
    # Markdown summary (#237)                                             #
    # ------------------------------------------------------------------ #

    def to_markdown(
        self, model: str | None = None, *, estimator: str | None = None
    ) -> str:
        """Render a compact, paste-ready Markdown summary (#237).

        Designed for PR descriptions, tickets, and chat, alongside the JSON /
        HTML / card outputs. Deterministic and free of non-reproducible
        content. Follows the same escaped-pipe table convention as
        :meth:`skdr_eval.doctor.DoctorReport.to_markdown`.

        Parameters
        ----------
        model : str, optional
            Restrict to one model. ``None`` (default) summarizes every row.
        estimator : str, optional
            Restrict to one estimator (e.g. ``"SNDR"``). ``None`` includes all.

        Returns
        -------
        str
            A Markdown document: one headline table row per
            ``(model, estimator)`` with V̂, CI, and the trust diagnostics
            (support-health, ESS, Pareto-k). It intentionally reports the
            stable trust layer, not the deployment verdict.
        """
        selected = self.rows()
        if model is not None:
            selected = [r for r in selected if r.model == model]
        if estimator is not None:
            selected = [r for r in selected if r.estimator == estimator]

        def _md(v: Any) -> str:
            return _fmt_num(v).replace("|", "\\|")

        def _ci(r: ReportRow) -> str:
            if r.ci_lower is None or r.ci_upper is None:
                return "—"
            return f"[{_fmt_num(r.ci_lower)}, {_fmt_num(r.ci_upper)}]"

        lines = [
            "# skdr-eval evaluation summary",
            "",
            "| Model | Estimator | V̂ | CI | Support | ESS | Pareto-k |",
            "| ----- | --------- | -- | -- | ------- | --- | -------- |",
        ]
        for r in selected:
            lines.append(
                f"| {str(r.model).replace('|', chr(92) + '|')} "
                f"| {r.estimator} | {_md(r.V_hat)} | {_ci(r)} "
                f"| {r.support_health or '—'} "
                f"| {_md(r.ESS)} | {_md(r.pareto_k)} |"
            )
        lines.append("")
        lines.append(
            "_Generated by [skdr-eval](https://github.com/dgenio/skdr-eval); "
            "V̂ is the estimated policy value under the logged data. Read the "
            "support column before acting on V̂._"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Card helpers                                                                #
# --------------------------------------------------------------------------- #


def _ci_label(alpha: float) -> str:
    pct = round((1.0 - float(alpha)) * 100)
    return f"{pct}% CI"


def _build_interpretation(
    v_hat: float,
    se: float,
    support_health: str,
    warning_codes: list[str],
    ci_lower: float | None,
    ci_upper: float | None,
) -> str:
    """Construct a short, factual interpretation paragraph for the card.

    Stakeholder-facing: state the headline number, the uncertainty band, and
    the trust status. No editorializing — just facts grounded in the metrics
    that produced the warning codes.
    """
    parts: list[str] = []
    if ci_lower is not None and ci_upper is not None:
        parts.append(
            f"Estimated policy value V̂ = {v_hat:.4g} "
            f"with band [{ci_lower:.4g}, {ci_upper:.4g}]."
        )
    else:
        parts.append(
            f"Estimated policy value V̂ = {v_hat:.4g} (SE = {se:.4g}; "
            "no bootstrap CI computed)."
        )

    if support_health == SUPPORT_OK:
        parts.append("Support diagnostics are healthy.")
    elif support_health == SUPPORT_CAUTION:
        parts.append(
            "Support diagnostics show one elevated risk indicator; interpret with care."
        )
    else:
        parts.append(
            "Support diagnostics show multiple elevated risk indicators; "
            "the estimate is statistically fragile and should not be acted on "
            "without follow-up analysis."
        )

    if WARN_LOW_ESS in warning_codes:
        parts.append("Low ESS: a few weights dominate the estimator.")
    if WARN_EXTREME_CLIP in warning_codes:
        parts.append(
            "Extreme clipping: tail-mass above threshold means the result is regression-driven."
        )
    if WARN_LOW_MATCH_RATE in warning_codes:
        parts.append("Low match rate: the policy frequently picks unobserved actions.")
    if WARN_POOR_OVERLAP in warning_codes:
        parts.append(
            "Poor overlap: minimum propensity is at or below 1/n; DR validity does not hold."
        )
    if WARN_HIGH_PARETO_K in warning_codes:
        parts.append(
            "High Pareto-k: importance-weight tail is heavy (PSIS), so V̂ is "
            "driven by a few decisions and the bootstrap CI under-states uncertainty."
        )
    if WARN_MISCAL_PROP in warning_codes:
        parts.append(
            "Miscalibrated propensities: ECE above threshold; IPW weights are "
            "systematically biased and the DR correction may not fully recover."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Recommendation / DiagnosticGate engine (#83, #99)                           #
# --------------------------------------------------------------------------- #
#
# The decision layer was extracted into ``skdr_eval.recommendation`` (#235).
# It is re-exported here so that every historical import path
# (``from skdr_eval.reporting import gate_diagnostics`` etc.) keeps working
# and ``EvaluationArtifact`` methods can reference these names as module
# globals. The verdict/gate logic is unchanged by the move.
from .recommendation import (  # noqa: E402 - re-export after artifact definitions
    DiagnosticGate,
    Explanation,
    GateResult,  # noqa: F401 - re-exported for skdr_eval.reporting compatibility
    Reason,  # noqa: F401 - re-exported for skdr_eval.reporting compatibility
    Recommendation,
    RecommendationPolicy,
    _build_recommendation,
    gate_diagnostics,
)

# --------------------------------------------------------------------------- #
# EvaluationCard schema (#88)                                                  #
# --------------------------------------------------------------------------- #

# Bump CARD_SCHEMA_VERSION when the card layout changes incompatibly. The card
# is the machine-readable sibling of the HTML stakeholder card and is intended
# to be CI-gateable.
CARD_SCHEMA_VERSION = "1.1.0"

_CARD_MODEL_CONFIG = ConfigDict(extra="allow", protected_namespaces=())


class HeadlineBlock(BaseModel):
    """Headline metrics for a card."""

    model_config = _CARD_MODEL_CONFIG

    estimator: str
    V_hat: float | None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_alpha: float | None = None
    baseline: float | None = None
    delta_vs_baseline: float | None = None
    clip: float | None = None


class TrustBlock(BaseModel):
    """Support-health / recommendation summary."""

    model_config = _CARD_MODEL_CONFIG

    support_health: str | None = None
    warning_codes: list[str] = Field(default_factory=list)
    recommendation: dict[str, Any] | None = None
    primary_blocker: str | None = None


class DiagnosticsBlock(BaseModel):
    """Diagnostic metrics carried on the report row."""

    model_config = _CARD_MODEL_CONFIG

    ESS: float | None = None
    ess_frac: float | None = None
    match_rate: float | None = None
    pareto_k: float | None = None
    min_pscore: float | None = None
    tail_mass: float | None = None
    ece: float | None = None
    brier_score: float | None = None
    gate: dict[str, Any] | None = None


class SensitivityBlock(BaseModel):
    """Clip-grid sensitivity summary for the card row."""

    model_config = _CARD_MODEL_CONFIG

    V_min: float | None = None
    V_max: float | None = None
    V_range: float | None = None
    chosen_clip: float | None = None
    chosen_V: float | None = None
    argmin_MSE_clip: float | None = None
    dr_sndr_agree: bool | None = None
    stable: bool | None = None
    # #133 — three-band decision-stability grade.
    v_range_frac: float | None = None
    stability_grade: str | None = None


class ProvenanceBlock(BaseModel):
    """Run provenance metadata."""

    model_config = _CARD_MODEL_CONFIG

    skdr_eval_version: str = "unknown"
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    n_samples: int | None = None
    n_splits: int | None = None
    random_state: int | None = None
    evaluator: str | None = None


class CoverageSimBlock(BaseModel):
    """Reference to the last coverage simulation run, if any."""

    model_config = _CARD_MODEL_CONFIG

    dgp: str | None = None
    n_reps: int | None = None
    empirical_coverage: float | None = None
    passes_nominal: bool | None = None


class EstimandBlock(BaseModel):
    """Target estimand and assumption tags carried with every card (#128).

    The block makes the *statistical contract* of the report explicit on
    the artifact and the YAML/JSON card so it travels with the headline.
    Defaults match :data:`DEFAULT_ESTIMAND_TEX`,
    :data:`DEFAULT_ESTIMAND_SUMMARY`, and :data:`DEFAULT_ASSUMPTION_TAGS`.
    See ``docs/concepts/estimands-and-assumptions.md`` for the prose.
    """

    model_config = _CARD_MODEL_CONFIG

    estimand_tex: str = DEFAULT_ESTIMAND_TEX
    summary: str = DEFAULT_ESTIMAND_SUMMARY
    assumptions: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ASSUMPTION_TAGS)
    )
    docs_url: str | None = "docs/concepts/estimands-and-assumptions.md"


class BaselineBlock(BaseModel):
    """Baseline policy value and delta-vs-baseline summary (#132)."""

    model_config = _CARD_MODEL_CONFIG

    kind: str | None = None  # "scalar" | "logged" | "column" | None
    value: float | None = None
    delta_V_hat: float | None = None
    delta_ci_lower: float | None = None
    delta_ci_upper: float | None = None


class EvaluationCard(BaseModel):
    """Machine-readable sibling of the HTML evaluation card (#88).

    Bundles the headline result, trust signals, diagnostics, sensitivity, and
    provenance for a single ``(model, estimator)`` row. Designed to be:

    - YAML/JSON serializable for pinning in Git.
    - JSON-Schema exportable for downstream tooling.
    - CI-gateable (``card.trust.recommendation['verdict'] == 'do_not_deploy'``).

    See Also
    --------
    EvaluationArtifact.card_schema : Build a card from a finished artifact.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    card_schema_version: str = Field(default=CARD_SCHEMA_VERSION)
    model_name: str
    headline: HeadlineBlock
    trust: TrustBlock
    diagnostics: DiagnosticsBlock
    sensitivity: SensitivityBlock = Field(default_factory=SensitivityBlock)
    provenance: ProvenanceBlock = Field(default_factory=ProvenanceBlock)
    coverage_sim: CoverageSimBlock | None = None
    estimand: EstimandBlock = Field(default_factory=EstimandBlock)
    baseline: BaselineBlock | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dict."""
        result: dict[str, Any] = self.model_dump(mode="json")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCard:
        """Build from a plain dict (validates)."""
        loaded: EvaluationCard = cls.model_validate(data)
        return loaded

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize to JSON, optionally writing to ``path``. Returns the JSON text."""
        text: str = self.model_dump_json(indent=indent)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> EvaluationCard:
        """Load from a JSON file path or JSON string.

        Distinguishes a ``Path`` (or a short string that resolves to an
        existing file) from inline JSON content. Long JSON strings are never
        interpreted as paths even on filesystems that would error on the
        length check.
        """
        text = _read_path_or_string(path_or_str)
        loaded: EvaluationCard = cls.model_validate_json(text)
        return loaded

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML, optionally writing to ``path``. Returns the YAML text."""
        import yaml as _yaml  # noqa: PLC0415

        text: str = _yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path_or_str: str | Path) -> EvaluationCard:
        """Load from a YAML file path or YAML string.

        A ``Path`` is always treated as a file. A plain string is treated as
        a file path only when it is short (< 4096 chars), contains no
        newlines, and the file exists; otherwise it is parsed as inline YAML.
        """
        import yaml as _yaml  # noqa: PLC0415

        text = _read_path_or_string(path_or_str)
        data = _yaml.safe_load(text)
        if not isinstance(data, dict):
            raise DataValidationError(
                f"EvaluationCard.from_yaml expected a YAML mapping, got {type(data).__name__}"
            )
        loaded: EvaluationCard = cls.model_validate(data)
        return loaded

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the Pydantic-generated JSON Schema for this card."""
        schema: dict[str, Any] = cls.model_json_schema()
        return schema


# NAME_MAX is 255 on most filesystems; 4096 leaves room for full paths
# without triggering ``OSError: File name too long`` when a long YAML/JSON
# blob is mistakenly passed in place of a file path.
_PATH_MAX_LIKELY = 4096


def _read_path_or_string(path_or_str: str | Path) -> str:
    """Return file content if ``path_or_str`` is a readable file, else the string.

    A :class:`Path` instance is always treated as a path. A plain string is
    treated as a path only when (a) it is short enough to be a valid filename
    on the host filesystem and (b) the file exists. Otherwise the string is
    returned verbatim. This lets ``from_yaml``/``from_json`` accept either
    inline document text or a path without falling over on long inline text.
    """
    if isinstance(path_or_str, Path):
        return path_or_str.read_text(encoding="utf-8")
    s = str(path_or_str)
    if len(s) > _PATH_MAX_LIKELY or "\n" in s:
        return s
    try:
        p = Path(s)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return s


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce numeric values to ``float | None``; non-finite → None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _build_card_from_row(
    artifact: EvaluationArtifact,
    model_name: str,
    estimator: str,
    *,
    baseline: float | None = None,
    include_gate: bool = True,
    include_recommendation: bool = True,
) -> EvaluationCard:
    """Internal: assemble an EvaluationCard from one ``(model, estimator)`` row."""
    if model_name not in artifact.detailed:
        raise DataValidationError(
            f"model {model_name!r} not in artifact "
            f"(known: {sorted(artifact.detailed)})",
        )
    est_map = artifact.detailed[model_name]
    if estimator not in est_map:
        raise DataValidationError(
            f"estimator {estimator!r} not in detailed[{model_name!r}] "
            f"(known: {sorted(est_map)})",
        )

    report = artifact.report
    row_mask = (report["model"] == model_name) & (report["estimator"] == estimator)
    rows = report[row_mask]
    if rows.empty:
        raise DataValidationError(
            f"No report row for model={model_name!r}, estimator={estimator!r}.",
        )
    row = rows.iloc[0]

    v_hat = _coerce_optional_float(row.get("V_hat"))
    ci_lower = _coerce_optional_float(row.get("ci_lower"))
    ci_upper = _coerce_optional_float(row.get("ci_upper"))
    delta = (v_hat - baseline) if (v_hat is not None and baseline is not None) else None
    alpha = artifact.metadata.get("alpha")

    headline = HeadlineBlock(
        estimator=estimator,
        V_hat=v_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_alpha=_coerce_optional_float(alpha),
        baseline=baseline,
        delta_vs_baseline=delta,
        clip=_coerce_optional_float(row.get("clip")),
    )

    support_health = row.get("support_health")
    warn_codes_raw = row.get("diagnostic_warnings", "")
    warning_codes = (
        [c.strip() for c in str(warn_codes_raw).split(",") if c.strip()]
        if warn_codes_raw
        else []
    )
    rec_dict: dict[str, Any] | None = None
    primary_blocker: str | None = None
    if include_recommendation:
        try:
            rec = artifact.recommendation(
                model_name,
                estimator=estimator,
                baseline=baseline if baseline is not None else 0.0,
            )
            rec_dict = rec.to_dict()
            primary_blocker = rec.primary_blocker
        except (DataValidationError, KeyError, ValueError):
            rec_dict = None

    trust = TrustBlock(
        support_health=(str(support_health) if support_health is not None else None),
        warning_codes=warning_codes,
        recommendation=rec_dict,
        primary_blocker=primary_blocker,
    )

    n = int(artifact.metadata.get("n_samples", 0)) or None
    ess_val = _coerce_optional_float(row.get("ESS"))
    ess_frac = (ess_val / n) if (ess_val is not None and n) else None

    diag = artifact.diagnostics.get(model_name)
    diag_gate: dict[str, Any] | None = None
    if include_gate:
        try:
            gate = gate_diagnostics(artifact, model_name, estimator=estimator)
            diag_gate = gate.to_dict()
        except (DataValidationError, KeyError, ValueError):
            diag_gate = None

    diagnostics = DiagnosticsBlock(
        ESS=ess_val,
        ess_frac=ess_frac,
        match_rate=_coerce_optional_float(row.get("match_rate")),
        pareto_k=_coerce_optional_float(row.get("pareto_k")),
        min_pscore=_coerce_optional_float(row.get("min_pscore")),
        tail_mass=_coerce_optional_float(row.get("tail_mass")),
        ece=_coerce_optional_float(getattr(diag, "ece", None)) if diag else None,
        brier_score=(
            _coerce_optional_float(getattr(diag, "brier_score", None)) if diag else None
        ),
        gate=diag_gate,
    )

    sens_mask = (artifact.sensitivity["model"] == model_name) & (
        artifact.sensitivity["estimator"] == estimator
    )
    sens_rows = artifact.sensitivity[sens_mask]
    if sens_rows.empty:
        sensitivity = SensitivityBlock()
    else:
        s = sens_rows.iloc[0]
        grade = s.get("stability_grade", None)
        sensitivity = SensitivityBlock(
            V_min=_coerce_optional_float(s.get("V_min")),
            V_max=_coerce_optional_float(s.get("V_max")),
            V_range=_coerce_optional_float(s.get("V_range")),
            chosen_clip=_coerce_optional_float(s.get("chosen_clip")),
            chosen_V=_coerce_optional_float(s.get("chosen_V")),
            argmin_MSE_clip=_coerce_optional_float(s.get("argmin_MSE_clip")),
            dr_sndr_agree=bool(s["dr_sndr_agree"])
            if "dr_sndr_agree" in s and s["dr_sndr_agree"] is not None
            else None,
            stable=bool(s["stable"])
            if "stable" in s and s["stable"] is not None
            else None,
            v_range_frac=_coerce_optional_float(s.get("v_range_frac")),
            stability_grade=str(grade) if grade is not None else None,
        )

    provenance = ProvenanceBlock(
        skdr_eval_version=str(artifact.metadata.get("skdr_eval_version", "unknown")),
        schema_version=str(artifact.metadata.get("schema_version", SCHEMA_VERSION)),
        timestamp=str(artifact.metadata.get("timestamp", "")),
        n_samples=(int(n) if n is not None else None),
        n_splits=(
            int(artifact.metadata["n_splits"])
            if "n_splits" in artifact.metadata
            and artifact.metadata["n_splits"] is not None
            else None
        ),
        random_state=(
            int(artifact.metadata["random_state"])
            if "random_state" in artifact.metadata
            and artifact.metadata["random_state"] is not None
            else None
        ),
        evaluator=(
            str(artifact.metadata["evaluator"])
            if artifact.metadata.get("evaluator") is not None
            else None
        ),
    )

    estimand_block = EstimandBlock(
        estimand_tex=artifact.estimand_tex,
        summary=artifact.estimand_summary,
        assumptions=list(artifact.assumptions),
    )

    baseline_block: BaselineBlock | None = None
    if artifact.baseline_kind is not None or baseline is not None:
        b_kind = artifact.baseline_kind or ("scalar" if baseline is not None else None)
        b_val = (
            artifact.baseline_value if artifact.baseline_value is not None else baseline
        )
        delta_v = _coerce_optional_float(row.get("delta_V_hat"))
        d_lo = _coerce_optional_float(row.get("delta_ci_lower"))
        d_hi = _coerce_optional_float(row.get("delta_ci_upper"))
        if delta_v is None and v_hat is not None and b_val is not None:
            delta_v = v_hat - float(b_val)
        if d_lo is None and ci_lower is not None and b_val is not None:
            d_lo = ci_lower - float(b_val)
        if d_hi is None and ci_upper is not None and b_val is not None:
            d_hi = ci_upper - float(b_val)
        baseline_block = BaselineBlock(
            kind=b_kind,
            value=_coerce_optional_float(b_val),
            delta_V_hat=delta_v,
            delta_ci_lower=d_lo,
            delta_ci_upper=d_hi,
        )

    return EvaluationCard(
        model_name=model_name,
        headline=headline,
        trust=trust,
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        provenance=provenance,
        estimand=estimand_block,
        baseline=baseline_block,
    )


def _render_sensitivity_plot(result: DRResult, estimator: str) -> str | None:
    """Render a tiny inline PNG of V vs clip and return base64. ``None`` on failure."""
    try:
        import matplotlib  # noqa: PLC0415  (defer import + headless guard)

        matplotlib.use("Agg", force=False)
        from matplotlib import pyplot as plt  # noqa: PLC0415

        v_col = f"V_{estimator}"
        grid = result.grid
        if v_col not in grid.columns or "clip" not in grid.columns:
            return None
        fig, ax = plt.subplots(figsize=(4.0, 1.8), dpi=120)
        clips = grid["clip"].astype(float).to_numpy()
        vs = grid[v_col].astype(float).to_numpy()
        # Replace inf clip with a visual sentinel (max finite + 1 step).
        finite_mask = np.isfinite(clips)
        if finite_mask.any():
            inf_value = (
                float(clips[finite_mask].max()) * 2 if finite_mask.any() else 1.0
            )
        else:
            inf_value = 1.0
        x = np.where(finite_mask, clips, inf_value)
        ax.plot(x, vs, marker="o", linewidth=1.2, color="#0969da")
        ax.axvline(
            float(result.clip) if np.isfinite(result.clip) else inf_value,
            color="#cf222e",
            linestyle="--",
            linewidth=0.8,
            label="chosen",
        )
        ax.set_xlabel("clip")
        ax.set_ylabel(f"V_{estimator}")
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.debug("Card plot rendering failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Factory + public helpers                                                    #
# --------------------------------------------------------------------------- #


def build_evaluation_artifact(
    *,
    report: pd.DataFrame,
    detailed: dict[str, dict[str, DRResult]],
    n_samples: int,
    propensities: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    thresholds: SupportHealthThresholds | None = None,
    evaluator: str = "skdr_eval",
    random_state: int | None = None,
    alpha: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
    baseline_kind: str | None = None,
    baseline_value: float | None = None,
) -> EvaluationArtifact:
    """Build an :class:`EvaluationArtifact` from raw ``evaluate_*_models`` outputs.

    This is the single factory used by ``evaluate_sklearn_models`` and
    ``evaluate_pairwise_models``; tests and adapters can call it directly.

    Parameters
    ----------
    report : pd.DataFrame
        Per-(model, estimator) report rows (pre-warnings).
    detailed : dict[str, dict[str, DRResult]]
        Per-(model, estimator) detailed result objects.
    n_samples : int
        Evaluation sample size (used for warnings ESS-fraction).
    propensities, actions : np.ndarray, optional
        Inputs to :func:`comprehensive_propensity_diagnostics`. Omit to skip
        diagnostics.
    thresholds : SupportHealthThresholds, optional
        Threshold set passed to :func:`attach_warnings`.
    evaluator : str
        Free-form name written to ``metadata['evaluator']`` for downstream
        bookkeeping (e.g. ``"evaluate_sklearn_models"``).
    random_state, alpha : optional
        Persisted to ``metadata`` for reproducibility/labeling.
    extra_metadata : dict, optional
        Additional metadata entries (e.g. evaluator-specific config).
    """
    from . import __version__ as _pkg_version  # noqa: PLC0415

    # Compute diagnostics first so MISCAL_PROP (#84) and PER_ACTION_MISCAL /
    # RARE_ACTION_NO_SUPPORT (#131) can read the per-model maps.
    sensitivity_df = summarize_sensitivity(detailed)
    diagnostics = _compute_diagnostics(propensities, actions, list(detailed.keys()))
    model_ece = {name: float(diag.ece) for name, diag in diagnostics.items()}
    model_per_action_ece = {
        name: float(diag.max_per_action_ece) for name, diag in diagnostics.items()
    }
    model_n_rare = {
        name: int(diag.n_rare_actions) for name, diag in diagnostics.items()
    }
    model_n_insuff = {
        name: int(diag.n_insufficient_actions) for name, diag in diagnostics.items()
    }
    model_n_rare_and_insuff = {
        name: int(diag.n_rare_and_insufficient_actions)
        for name, diag in diagnostics.items()
    }
    enriched_report, warnings_df = attach_warnings(
        report,
        n_samples,
        thresholds,
        model_ece=model_ece,
        model_per_action_ece=model_per_action_ece,
        model_n_rare_actions=model_n_rare,
        model_n_insufficient_actions=model_n_insuff,
        model_n_rare_and_insufficient_actions=model_n_rare_and_insuff,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "skdr_eval_version": _pkg_version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator": evaluator,
        "n_samples": int(n_samples),
    }
    if random_state is not None:
        metadata["random_state"] = int(random_state)
    if alpha is not None:
        metadata["alpha"] = float(alpha)
    if extra_metadata:
        metadata.update(extra_metadata)

    # #132 — baseline + delta-vs-baseline first-class columns.
    if baseline_value is not None and "V_hat" in enriched_report.columns:
        b = float(baseline_value)
        enriched_report = enriched_report.copy()
        enriched_report["delta_V_hat"] = enriched_report["V_hat"].astype(float) - b
        if "ci_lower" in enriched_report.columns:
            enriched_report["delta_ci_lower"] = (
                enriched_report["ci_lower"].astype(float) - b
            )
        if "ci_upper" in enriched_report.columns:
            enriched_report["delta_ci_upper"] = (
                enriched_report["ci_upper"].astype(float) - b
            )

    return EvaluationArtifact(
        report=enriched_report,
        detailed=detailed,
        warnings=warnings_df,
        sensitivity=sensitivity_df,
        diagnostics=diagnostics,
        metadata=metadata,
        baseline_kind=baseline_kind,
        baseline_value=float(baseline_value) if baseline_value is not None else None,
    )


def load_artifact_json(path: str | Path) -> ArtifactSchema:
    """Load and validate an artifact JSON file. Returns the Pydantic schema."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    schema: ArtifactSchema = ArtifactSchema.model_validate(data)
    return schema


def _thin_artifact_from_schema(schema: ArtifactSchema) -> EvaluationArtifact:
    """Reconstruct a minimal :class:`EvaluationArtifact` from a saved schema.

    The verdict (:meth:`EvaluationArtifact.recommendation`) and gate
    (:func:`gate_diagnostics`) read **only** the report DataFrame, the
    ``(model, estimator)`` membership of ``detailed``, and ``metadata`` —
    none of which require the original ``DRResult`` objects. So an explanation
    can be rebuilt faithfully from a saved ``artifact.json`` without
    re-running the evaluation (#201). The ``detailed`` map therefore carries
    ``None`` sentinels: it is sufficient for membership checks but must not be
    used for anything that dereferences a ``DRResult``.
    """
    report_df = pd.DataFrame([row.model_dump() for row in schema.report])
    detailed: dict[str, dict[str, Any]] = {}
    for row in schema.report:
        detailed.setdefault(row.model, {})[row.estimator] = None
    return EvaluationArtifact(
        report=report_df,
        detailed=detailed,  # sentinels only; see docstring
        warnings=pd.DataFrame(),
        sensitivity=pd.DataFrame(),
        diagnostics={},
        metadata=dict(schema.metadata),
    )


def explain_artifact_schema(
    schema: ArtifactSchema,
    model_name: str,
    *,
    estimator: str = "SNDR",
    baseline: float | None = None,
) -> Explanation:
    """Explain a verdict from a saved :class:`ArtifactSchema` (#201).

    Companion to :meth:`EvaluationArtifact.explain` for the common case where
    only a saved ``artifact.json`` (loaded via :func:`load_artifact_json`) is
    available — e.g. the ``skdr-eval explain`` CLI command. Reuses the existing
    recommendation/gate logic, so the narrative matches the live artifact.

    Raises
    ------
    DataValidationError
        If ``model_name`` or ``estimator`` is not present in the schema.
    """
    artifact = _thin_artifact_from_schema(schema)
    return artifact.explain(model_name, estimator=estimator, baseline=baseline)


def export_results(
    artifact: EvaluationArtifact,
    path: str | Path,
    *,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Convenience wrapper around :meth:`EvaluationArtifact.export`.

    Provided so issue #28's documented API (``skdr_eval.export_results``) is
    available as a top-level function.
    """
    return artifact.export(path, formats=formats)


def render_evaluation_card(
    artifact: EvaluationArtifact,
    model_name: str,
    *,
    headline_estimator: str = "SNDR",
    format: str = "html",
) -> str:
    """Convenience wrapper around :meth:`EvaluationArtifact.card` (issue #30)."""
    return artifact.card(
        model_name, headline_estimator=headline_estimator, format=format
    )
