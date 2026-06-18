"""Deployment recommendation and diagnostic-gate engine for ``skdr-eval``.

This module isolates the *decision layer* — the part of the pipeline that turns
per-row diagnostics and confidence intervals into a deployment verdict — out of
the much larger :mod:`skdr_eval.reporting` module (#235). It is the
safety-relevant core of the product, so keeping it in a small, focused module
makes the gating thresholds easier to read, test, and protect.

Public surface (all re-exported from :mod:`skdr_eval` and, for backwards
compatibility, from :mod:`skdr_eval.reporting`):

- :class:`RecommendationPolicy`, :class:`Reason`, :class:`Recommendation`
- :class:`GateResult`, :class:`DiagnosticGate`
- :func:`gate_diagnostics`

The verdict/gate **logic is unchanged** from its previous home in
``reporting.py``; this is a behaviour-preserving extraction.

The shared warning-code constants and :class:`~skdr_eval.reporting.SupportHealthThresholds`
still live in :mod:`skdr_eval.reporting` (they are used by the warning logic as
well as the gate). They are imported lazily inside the two functions that need
them so that importing this module never depends on ``reporting`` having
finished executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import DataValidationError

if TYPE_CHECKING:
    from .reporting import EvaluationArtifact, SupportHealthThresholds


# --------------------------------------------------------------------------- #
# Recommendation API (#83)                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecommendationPolicy:
    """Configurable thresholds / baseline for the recommendation heuristic.

    Parameters
    ----------
    baseline : float, default 0.0
        Minimum policy value that constitutes "beating the baseline". The CI
        gate checks whether ``ci_lower > baseline`` (directional win).
    """

    baseline: float = 0.0


@dataclass
class Reason:
    """A single human-readable reason contributing to a :class:`Recommendation`.

    Attributes
    ----------
    code : str
        Stable machine-readable code (e.g. ``"HIGH_RISK_OVERLAP"``).
    message : str
        Human-readable explanation.
    severity : str
        ``"high_risk"``, ``"caution"``, or ``"info"``.
    """

    code: str
    message: str
    severity: str


@dataclass
class Recommendation:
    """Structured deployment recommendation for one (model, estimator) row.

    Attributes
    ----------
    verdict : str
        One of ``"deploy"``, ``"ab_test"``, ``"do_not_deploy"``,
        ``"insufficient_evidence"``.
    confidence : str
        ``"high"`` | ``"medium"`` | ``"low"``.
    primary_blocker : str or None
        Warning code that single-handedly prevents deployment, or ``None``.
    reasons : list[Reason]
        Ordered reasons (most severe first).
    recommended_estimator : str
        ``"DR"`` or ``"SNDR"`` — the estimator this recommendation is based on.
    model_name : str
        Name of the model.
    """

    verdict: str
    confidence: str
    primary_blocker: str | None
    reasons: list[Reason]
    recommended_estimator: str
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON export."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "primary_blocker": self.primary_blocker,
            "reasons": [
                {"code": r.code, "message": r.message, "severity": r.severity}
                for r in self.reasons
            ],
            "recommended_estimator": self.recommended_estimator,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recommendation:
        """Deserialize from a plain dict."""
        reasons = [
            Reason(
                code=r["code"],
                message=r["message"],
                severity=r["severity"],
            )
            for r in data.get("reasons", [])
        ]
        return cls(
            verdict=data["verdict"],
            confidence=data["confidence"],
            primary_blocker=data.get("primary_blocker"),
            reasons=reasons,
            recommended_estimator=data.get("recommended_estimator", "DR"),
            model_name=data.get("model_name", ""),
        )


def _build_recommendation(
    artifact: EvaluationArtifact,
    model_name: str,
    estimator: str,
    policy: RecommendationPolicy,
) -> Recommendation:
    """Internal logic for :meth:`EvaluationArtifact.recommendation`."""
    from .reporting import (  # noqa: PLC0415 - lazy to avoid an import cycle
        WARN_EXTREME_CLIP,
        WARN_HIGH_PARETO_K,
        WARN_LOW_ESS,
        WARN_LOW_MATCH_RATE,
        WARN_MISCAL_PROP,
        WARN_POOR_OVERLAP,
        _coerce_optional_float,
    )

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

    # Pull the row from the report for this (model, estimator).
    row_mask = (artifact.report["model"] == model_name) & (
        artifact.report["estimator"] == estimator
    )
    rows = artifact.report[row_mask]
    if rows.empty:
        raise DataValidationError(
            f"No report row for model={model_name!r}, estimator={estimator!r}.",
        )
    row = rows.iloc[0]
    warn_str = str(row.get("diagnostic_warnings", "") or "")
    warn_codes = [c for c in warn_str.split(",") if c]
    # Coerce both CI bounds via the shared helper so non-finite values
    # (NumPy/Pandas NaN, ±inf) are treated as "missing" — consistent with
    # the report/card rendering path in ``reporting.py``.
    ci_lower = _coerce_optional_float(row.get("ci_lower"))
    ci_upper = _coerce_optional_float(row.get("ci_upper"))

    reasons: list[Reason] = []

    # --- Collect reasons based on warning codes ---
    _HIGH_RISK_MSG = {
        WARN_POOR_OVERLAP: "Minimum propensity at/below 1/n; DR validity does not hold.",
        WARN_HIGH_PARETO_K: "Pareto-k > threshold; importance-weight tail is heavy.",
        WARN_EXTREME_CLIP: "Tail-mass above threshold; result is regression-driven.",
    }
    _CAUTION_MSG = {
        WARN_LOW_ESS: "Effective sample size is low; a few weights dominate.",
        WARN_LOW_MATCH_RATE: "Policy frequently picks unobserved actions.",
        WARN_MISCAL_PROP: "ECE above threshold; IPW weights are systematically biased.",
    }

    high_risk_blockers: list[str] = []
    for code in warn_codes:
        if code in _HIGH_RISK_MSG:
            reasons.append(
                Reason(code=code, message=_HIGH_RISK_MSG[code], severity="high_risk")
            )
            high_risk_blockers.append(code)
        elif code in _CAUTION_MSG:
            reasons.append(
                Reason(code=code, message=_CAUTION_MSG[code], severity="caution")
            )

    # --- Determine verdict ---
    if high_risk_blockers:
        primary_blocker = high_risk_blockers[0]
        verdict = "do_not_deploy"
        confidence = "high"
        reasons.append(
            Reason(
                code="DO_NOT_DEPLOY_HIGH_RISK",
                message=f"High-risk diagnostic flag(s) present: {', '.join(high_risk_blockers)}.",
                severity="high_risk",
            )
        )
    elif ci_lower is not None and ci_upper is not None:
        primary_blocker = None
        ci_beats_baseline = ci_lower > policy.baseline
        if ci_beats_baseline:
            caution_present = any(r.severity == "caution" for r in reasons)
            if caution_present:
                verdict = "ab_test"
                confidence = "medium"
                reasons.append(
                    Reason(
                        code="CI_ABOVE_BASELINE_WITH_CAUTION",
                        message=(
                            f"CI [{ci_lower:.4g}, {ci_upper:.4g}] is above "
                            f"baseline {policy.baseline:.4g} but caution diagnostics present."
                        ),
                        severity="caution",
                    )
                )
            else:
                verdict = "deploy"
                confidence = "high"
                reasons.append(
                    Reason(
                        code="CI_ABOVE_BASELINE_CLEAN",
                        message=(
                            f"CI [{ci_lower:.4g}, {ci_upper:.4g}] fully exceeds "
                            f"baseline {policy.baseline:.4g} with no caution flags."
                        ),
                        severity="info",
                    )
                )
        else:
            verdict = "ab_test"
            confidence = "low"
            reasons.append(
                Reason(
                    code="CI_OVERLAPS_BASELINE",
                    message=(
                        f"CI [{ci_lower:.4g}, {ci_upper:.4g}] overlaps or is below "
                        f"baseline {policy.baseline:.4g}."
                    ),
                    severity="caution",
                )
            )
    else:
        # No CI available — without a confidence interval the gate cannot make a
        # deploy/don't-deploy call, so the verdict is always insufficient_evidence.
        primary_blocker = None
        verdict = "insufficient_evidence"
        confidence = "low"
        reasons.append(
            Reason(
                code="NO_CI",
                message=(
                    "No bootstrap CI available. Re-run with ci_bootstrap=True "
                    "for a more confident recommendation."
                ),
                severity="info",
            )
        )

    # Sort: high_risk first, then caution, then info.
    _sev_order = {"high_risk": 0, "caution": 1, "info": 2}
    reasons.sort(key=lambda r: _sev_order.get(r.severity, 3))

    return Recommendation(
        verdict=verdict,
        confidence=confidence,
        primary_blocker=primary_blocker,
        reasons=reasons,
        recommended_estimator=estimator,
        model_name=model_name,
    )


# --------------------------------------------------------------------------- #
# DiagnosticGate API (#99)                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    """Single-check outcome from :func:`gate_diagnostics`.

    Attributes
    ----------
    check : str
        The check name: ``"overlap"``, ``"ess"``, or ``"calibration"``.
    state : str
        ``"pass"``, ``"warn"``, or ``"fail"``.
    code : str
        Stable machine-readable reason code.
    message : str
        Human-readable explanation.
    value : float or None
        The actual metric value checked.
    threshold : float or None
        The threshold used.
    """

    check: str
    state: str
    code: str
    message: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "state": self.state,
            "code": self.code,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class DiagnosticGate:
    """Structured pass/warn/fail gate across key diagnostic checks.

    Returned by :func:`gate_diagnostics`. Use ``overall`` for a single-signal
    GO / CAUTION / STOP check before deploying a policy.

    Attributes
    ----------
    overlap : GateResult
        Overlap / support check (min propensity vs 1/n threshold).
    ess : GateResult
        Effective sample size check (ESS / n).
    calibration : GateResult
        Calibration / Pareto-k check.
    overall : str
        ``"pass"`` if all checks pass; ``"warn"`` if any is ``"warn"``
        (but none is ``"fail"``); ``"fail"`` if any check fails.
    """

    overlap: GateResult
    ess: GateResult
    calibration: GateResult
    overall: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap": self.overlap.to_dict(),
            "ess": self.ess.to_dict(),
            "calibration": self.calibration.to_dict(),
            "overall": self.overall,
        }

    def to_text(self) -> str:
        """Short human-readable summary."""
        lines = [f"DiagnosticGate overall: {self.overall.upper()}"]
        for check in (self.overlap, self.ess, self.calibration):
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(check.state, "?")
            val = f" (value={check.value:.4g})" if check.value is not None else ""
            lines.append(
                f"  {icon} {check.check}: {check.state} — {check.message}{val}"
            )
        return "\n".join(lines)


def gate_diagnostics(
    artifact: EvaluationArtifact,
    model_name: str,
    estimator: str = "DR",
    *,
    thresholds: SupportHealthThresholds | None = None,
) -> DiagnosticGate:
    """Run pass/warn/fail gates across key diagnostics for one (model, estimator) pair.

    Parameters
    ----------
    artifact : EvaluationArtifact
        Output of ``evaluate_sklearn_models`` or ``evaluate_pairwise_models``.
    model_name : str
        Model name present in ``artifact.detailed``.
    estimator : str, default ``"DR"``
        ``"DR"`` or ``"SNDR"``.
    thresholds : SupportHealthThresholds, optional
        Custom thresholds. Defaults to :class:`SupportHealthThresholds` defaults.

    Returns
    -------
    DiagnosticGate
        Three individual :class:`GateResult` objects plus an ``overall`` verdict.

    Raises
    ------
    DataValidationError
        If ``model_name`` or ``estimator`` is not found in the artifact.
    """
    from .reporting import (  # noqa: PLC0415 - lazy to avoid an import cycle
        WARN_HIGH_PARETO_K,
        WARN_LOW_ESS,
        WARN_LOW_MATCH_RATE,
        WARN_POOR_OVERLAP,
        SupportHealthThresholds,
    )

    thr = thresholds or SupportHealthThresholds()

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

    row_mask = (artifact.report["model"] == model_name) & (
        artifact.report["estimator"] == estimator
    )
    rows = artifact.report[row_mask]
    if rows.empty:
        raise DataValidationError(
            f"No report row for model={model_name!r}, estimator={estimator!r}.",
        )
    row = rows.iloc[0]
    n = int(artifact.metadata.get("n_samples", 1))

    # --- Overlap gate ---
    min_ps = row.get("min_pscore")
    match_rate = row.get("match_rate")
    overlap_threshold = 1.0 / max(n, 1)
    min_ps_val = (
        float(min_ps) if min_ps is not None and not np.isnan(float(min_ps)) else None
    )
    match_rate_val = (
        float(match_rate)
        if match_rate is not None and not np.isnan(float(match_rate))
        else None
    )

    if min_ps_val is not None and min_ps_val <= overlap_threshold:
        overlap = GateResult(
            check="overlap",
            state="fail",
            code=WARN_POOR_OVERLAP,
            message=f"min_pscore {min_ps_val:.4g} ≤ 1/n={overlap_threshold:.4g}; DR validity does not hold.",
            value=min_ps_val,
            threshold=overlap_threshold,
        )
    elif match_rate_val is not None and match_rate_val < thr.low_match_rate:
        overlap = GateResult(
            check="overlap",
            state="warn",
            code=WARN_LOW_MATCH_RATE,
            message=f"match_rate {match_rate_val:.4g} < {thr.low_match_rate:.4g}; policy picks unobserved actions.",
            value=match_rate_val,
            threshold=thr.low_match_rate,
        )
    else:
        overlap = GateResult(
            check="overlap",
            state="pass",
            code="OVERLAP_OK",
            message="Overlap diagnostics are healthy.",
            value=min_ps_val,
            threshold=overlap_threshold,
        )

    # --- ESS gate ---
    ess_val_raw = row.get("ESS")
    ess_val = (
        float(ess_val_raw)
        if ess_val_raw is not None and not np.isnan(float(ess_val_raw))
        else None
    )
    ess_frac = (ess_val / n) if ess_val is not None else None
    # low_ess_frac is a fraction (e.g. 0.10); absolute threshold = frac * n
    ess_abs_threshold = thr.low_ess_frac * n

    if ess_val is not None and ess_val < ess_abs_threshold:
        ess = GateResult(
            check="ess",
            state="fail" if ess_frac is not None and ess_frac < 0.05 else "warn",  # noqa: PLR2004
            code=WARN_LOW_ESS,
            message=f"ESS={ess_val:.1f} (n={n}) is below threshold {ess_abs_threshold:.1f} ({thr.low_ess_frac:.0%}).",
            value=ess_val,
            threshold=ess_abs_threshold,
        )
    else:
        ess = GateResult(
            check="ess",
            state="pass",
            code="ESS_OK",
            message=f"ESS={ess_val:.1f} is healthy."
            if ess_val is not None
            else "ESS not available.",
            value=ess_val,
            threshold=ess_abs_threshold,
        )

    # --- Calibration gate: prefer Pareto-k ---
    pareto_k_raw = row.get("pareto_k")
    pareto_k = (
        float(pareto_k_raw)
        if pareto_k_raw is not None and not np.isnan(float(pareto_k_raw))
        else None
    )

    if pareto_k is not None and pareto_k > thr.high_pareto_k:
        cal = GateResult(
            check="calibration",
            state="fail" if pareto_k > 1.0 else "warn",
            code=WARN_HIGH_PARETO_K,
            message=f"Pareto-k={pareto_k:.3f} > {thr.high_pareto_k:.3f}; importance-weight tail is heavy.",
            value=pareto_k,
            threshold=thr.high_pareto_k,
        )
    else:
        cal_msg = (
            f"Pareto-k={pareto_k:.3f} is healthy."
            if pareto_k is not None
            else "Pareto-k not available."
        )
        cal = GateResult(
            check="calibration",
            state="pass",
            code="CALIBRATION_OK",
            message=cal_msg,
            value=pareto_k,
            threshold=thr.high_pareto_k if pareto_k is not None else None,
        )

    # --- Overall ---
    states = {g.state for g in (overlap, ess, cal)}
    if "fail" in states:
        overall = "fail"
    elif "warn" in states:
        overall = "warn"
    else:
        overall = "pass"

    return DiagnosticGate(overlap=overlap, ess=ess, calibration=cal, overall=overall)
