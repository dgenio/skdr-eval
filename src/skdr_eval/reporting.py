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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

from .diagnostics import (
    PropensityDiagnostics,
    comprehensive_propensity_diagnostics,
)
from .exceptions import ConfigurationError, DataValidationError

if TYPE_CHECKING:
    from .core import DRResult

logger = logging.getLogger("skdr_eval")

# Bump SCHEMA_VERSION when the artifact JSON layout changes incompatibly.
SCHEMA_VERSION = "1.0.0"

# Machine-readable warning codes. Do not localize.
WARN_LOW_ESS = "LOW_ESS"
WARN_EXTREME_CLIP = "EXTREME_CLIP"
WARN_POOR_OVERLAP = "POOR_OVERLAP"
WARN_LOW_MATCH_RATE = "LOW_MATCH_RATE"

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
    """

    low_ess_frac: float = 0.10
    extreme_clip_tail_mass: float = 0.05
    low_match_rate: float = 0.50
    poor_overlap_min_pscore: float | None = None


# --------------------------------------------------------------------------- #
# Warning computation                                                         #
# --------------------------------------------------------------------------- #


def _compute_row_warnings(
    row: dict[str, Any],
    thresholds: SupportHealthThresholds,
    n_samples: int,
) -> tuple[list[str], str]:
    """Compute warning codes and severity for a single report row.

    Returns
    -------
    codes : list of str
        Stable-ordered warning codes (``LOW_ESS`` before ``EXTREME_CLIP``
        before ``LOW_MATCH_RATE`` before ``POOR_OVERLAP``).
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

    # Stable order: LOW_ESS, EXTREME_CLIP, LOW_MATCH_RATE, POOR_OVERLAP.
    code_order = [
        WARN_LOW_ESS,
        WARN_EXTREME_CLIP,
        WARN_LOW_MATCH_RATE,
        WARN_POOR_OVERLAP,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``support_health`` and ``diagnostic_warnings`` columns.

    Parameters
    ----------
    report : pd.DataFrame
        Report from ``evaluate_*_models``. Must contain at least the columns
        ``model, estimator, ESS, tail_mass, match_rate, min_pscore``.
    n_samples : int
        Number of evaluation samples (used for ESS-fraction and default
        overlap floor). Must be positive.
    thresholds : SupportHealthThresholds, optional
        Threshold set. Defaults to :class:`SupportHealthThresholds`.

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

    severities: list[str] = []
    code_strings: list[str] = []
    warning_records: list[dict[str, Any]] = []
    for _, row in report.iterrows():
        codes, severity = _compute_row_warnings(row.to_dict(), thresholds, n_samples)
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
            stable = bool((v_range / scale) < _STABILITY_RANGE_FRAC and dr_sndr_agree)

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
    """
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
    }


def _compute_diagnostics(
    propensities: np.ndarray | None,
    actions: np.ndarray | None,
    model_names: list[str],
) -> dict[str, PropensityDiagnostics]:
    """Compute per-model propensity diagnostics, sharing one fit across models.

    The propensity model is fitted once per evaluation run (not per model
    being evaluated), so we compute it once and then deep-copy it per model
    so callers mutating ``artifact.diagnostics[m].statistics`` for one model
    do not silently affect every other model in the run. Returns an empty
    dict if propensities are missing or diagnostics fail.
    """
    if propensities is None or actions is None or len(model_names) == 0:
        return {}
    try:
        shared = comprehensive_propensity_diagnostics(propensities, actions)
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
        )

    # ------------------------------------------------------------------ #
    # JSON                                                               #
    # ------------------------------------------------------------------ #

    def to_json_str(self, *, indent: int | None = 2) -> str:
        """Serialize the artifact to a JSON string via the Pydantic schema."""
        return str(self.to_schema().model_dump_json(indent=indent))

    def to_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write the artifact JSON to ``path``. Creates parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json_str(indent=indent), encoding="utf-8")
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

    def to_html(self, path: str | Path) -> Path:
        """Write the artifact HTML to ``path``. Creates parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html_str(), encoding="utf-8")
        logger.info("Wrote evaluation artifact HTML to %s", p)
        return p

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
        n_show = min(5, len(contrib))
        if n_show == 0:
            return [], []
        order = np.argsort(contrib)
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
        formats : list of {"json", "html"}, optional
            Defaults to both.

        Returns
        -------
        dict[str, Path]
            Map of format name to written path.
        """
        if formats is None:
            formats = ["json", "html"]
        supported = {"json", "html"}
        unknown = set(formats) - supported
        if unknown:
            raise ConfigurationError(
                f"Unknown export format(s): {sorted(unknown)} "
                f"(supported: {sorted(supported)})",
            )

        p = Path(path)
        # Trailing-separator hint must be checked on the original string,
        # since Path() normalizes the trailing slash away.
        explicit_dir = isinstance(path, str) and (
            path.endswith("/") or path.endswith("\\")
        )
        treat_as_dir = explicit_dir or (p.exists() and p.is_dir())

        written: dict[str, Path] = {}
        for fmt in formats:
            if treat_as_dir:
                target = p / f"artifact.{fmt}"
            elif p.suffix:
                target = p.with_suffix(f".{fmt}")
            else:
                target = p.with_suffix(f".{fmt}")
            if fmt == "json":
                written["json"] = self.to_json(target)
            elif fmt == "html":
                written["html"] = self.to_html(target)
        return written


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
    return " ".join(parts)


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

    enriched_report, warnings_df = attach_warnings(report, n_samples, thresholds)
    sensitivity_df = summarize_sensitivity(detailed)
    diagnostics = _compute_diagnostics(propensities, actions, list(detailed.keys()))

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "skdr_eval_version": _pkg_version,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluator": evaluator,
        "n_samples": int(n_samples),
    }
    if random_state is not None:
        metadata["random_state"] = int(random_state)
    if alpha is not None:
        metadata["alpha"] = float(alpha)
    if extra_metadata:
        metadata.update(extra_metadata)

    return EvaluationArtifact(
        report=enriched_report,
        detailed=detailed,
        warnings=warnings_df,
        sensitivity=sensitivity_df,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def load_artifact_json(path: str | Path) -> ArtifactSchema:
    """Load and validate an artifact JSON file. Returns the Pydantic schema."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    schema: ArtifactSchema = ArtifactSchema.model_validate(data)
    return schema


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
