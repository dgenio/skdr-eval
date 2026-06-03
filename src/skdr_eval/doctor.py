"""Preflight diagnostics for ``skdr-eval`` inputs and environment (#91).

The :func:`doctor` entry point composes the existing public validators
(:func:`skdr_eval.validate_logs`, :func:`skdr_eval.validate_pairwise_inputs`)
and capability probe (:func:`skdr_eval.get_capabilities`) into a single
non-raising report with concrete fix hints. The report is what turns the
"cryptic stack trace" first-run failure mode into actionable diagnostics.

The doctor never raises on a malformed DataFrame: it surfaces failures as
:class:`Check` records. It does not modify the input DataFrames.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .capabilities import get_capabilities
from .exceptions import DataValidationError, InsufficientDataError
from .validation import validate_logs, validate_pairwise_inputs

if TYPE_CHECKING:
    from collections.abc import Iterable

CheckStatus = Literal["pass", "warn", "fail"]

_STATUS_ICONS: dict[str, str] = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}
_STATUS_GLYPHS: dict[str, str] = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


@dataclass(frozen=True)
class Check:
    """One row in a :class:`DoctorReport`.

    Attributes
    ----------
    name : str
        Short identifier of the check (``"environment"``, ``"schema"``, …).
    status : str
        ``"pass"``, ``"warn"``, or ``"fail"``.
    message : str
        Human-readable summary.
    fix_hint : str
        Suggested fix (empty string when none applies).
    category : str
        High-level grouping: ``"environment"``, ``"schema"``,
        ``"statistical"``, ``"sample_size"``.
    """

    name: str
    status: CheckStatus
    message: str
    fix_hint: str = ""
    category: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    """Result of running :func:`doctor`.

    Attributes
    ----------
    ok : bool
        ``True`` iff no check has status ``"fail"``.
    checks : list[Check]
        Ordered list of all executed checks.
    summary : str
        ``"pass"``, ``"warn"``, or ``"fail"`` — the worst per-check status.
    """

    ok: bool
    checks: list[Check] = field(default_factory=list)
    summary: CheckStatus = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_markdown(self) -> str:
        """Render as a copy-pasteable Markdown table."""
        lines = [
            "| Status | Check | Message | Fix hint |",
            "| ------ | ----- | ------- | -------- |",
        ]
        for c in self.checks:
            icon = _STATUS_ICONS.get(c.status, c.status)
            fix = c.fix_hint.replace("|", "\\|") if c.fix_hint else ""
            msg = c.message.replace("|", "\\|")
            lines.append(f"| {icon} | {c.name} | {msg} | {fix} |")
        lines.append("")
        lines.append(f"**Overall:** {_STATUS_ICONS.get(self.summary, self.summary)}")
        return "\n".join(lines)

    def to_text(self, *, color: bool = False) -> str:
        """Render as a terminal-friendly text block."""
        glyph_map = _STATUS_GLYPHS if color else _STATUS_ICONS
        lines = ["skdr-eval doctor", "─" * 40]
        for c in self.checks:
            icon = glyph_map.get(c.status, c.status)
            line = f"{icon}  {c.name}: {c.message}"
            lines.append(line)
            if c.fix_hint:
                lines.append(f"    Fix: {c.fix_hint}")
        lines.append("─" * 40)
        overall_icon = glyph_map.get(self.summary, self.summary)
        lines.append(f"Overall: {overall_icon} ({self.summary})")
        return "\n".join(lines)

    def print(self, *, color: bool = True) -> None:
        """Print the report to stdout. Color glyphs auto-disabled off-TTY."""
        # Honor explicit ``color=False`` but auto-fall back when stdout is not a TTY.
        use_color = color and sys.stdout.isatty()
        print(self.to_text(color=use_color))


_PYTHON_MIN: tuple[int, int] = (3, 10)


def _check_environment() -> Check:
    """Verify Python version meets the project requirement."""
    if sys.version_info < _PYTHON_MIN:
        return Check(
            name="environment",
            status="fail",
            message=(
                f"Python {platform.python_version()} is below skdr-eval's "
                f"minimum of 3.10."
            ),
            fix_hint="Upgrade to Python 3.10+ (CI matrix covers 3.10-3.14).",
            category="environment",
        )
    return Check(
        name="environment",
        status="pass",
        message=(
            f"Python {platform.python_version()} on {platform.system()} "
            f"({platform.machine()})."
        ),
        category="environment",
    )


def _check_optional_extras() -> Check:
    """Report which optional ``[extra]`` capabilities are available."""
    caps = get_capabilities()
    missing_raw = caps.get("missing_extras") or []
    missing = [str(x) for x in missing_raw] if isinstance(missing_raw, list) else []
    if not missing:
        return Check(
            name="optional_extras",
            status="pass",
            message="All optional extras installed.",
            category="environment",
        )
    hint = " ".join(f"pip install 'skdr-eval[{e}]'" for e in missing)
    return Check(
        name="optional_extras",
        status="warn",
        message=f"Optional extras not installed: {', '.join(missing)}.",
        fix_hint=hint,
        category="environment",
    )


def _check_logs_schema(logs: pd.DataFrame, *, metric_col: str, strict: bool) -> Check:
    try:
        validate_logs(logs, y_col=metric_col, strict=strict)
    except (DataValidationError, InsufficientDataError) as exc:
        return Check(
            name="schema",
            status="fail",
            message=str(exc),
            fix_hint=(
                "Repair the logs schema: action must reference a *_elig column, "
                "include cli_*/st_* feature columns, and no non-finite features. "
                "See validate_logs() in skdr_eval.validation."
            ),
            category="schema",
        )
    return Check(
        name="schema",
        status="pass",
        message="validate_logs() passed.",
        category="schema",
    )


def _check_pairwise_schema(
    logs_df: pd.DataFrame,
    op_daily_df: pd.DataFrame | None,
    metric_col: str,
    *,
    strict: bool,
) -> Check:
    if op_daily_df is None:
        return Check(
            name="schema",
            status="fail",
            message=(
                "Pairwise doctor requires an op_daily_df argument; received None."
            ),
            fix_hint="Pass the daily-operator snapshot DataFrame as op_daily_df=.",
            category="schema",
        )
    try:
        validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col=metric_col, strict=strict
        )
    except (DataValidationError, InsufficientDataError) as exc:
        return Check(
            name="schema",
            status="fail",
            message=str(exc),
            fix_hint=(
                "Repair the pairwise schema: logs_df needs client_id/operator_id"
                "/arrival_day, op_daily_df needs op_* feature columns and the "
                "metric column."
            ),
            category="schema",
        )
    return Check(
        name="schema",
        status="pass",
        message="validate_pairwise_inputs() passed.",
        category="schema",
    )


def _check_no_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
) -> Check:
    available = [c for c in key_cols if c in df.columns]
    if not available:
        return Check(
            name="duplicates",
            status="warn",
            message=(f"No duplicate-key columns to check (looked for {key_cols})."),
            category="schema",
        )
    n_dup = int(df.duplicated(subset=available, keep=False).sum())
    if n_dup == 0:
        return Check(
            name="duplicates",
            status="pass",
            message=f"No duplicate {tuple(available)} rows.",
            category="schema",
        )
    return Check(
        name="duplicates",
        status="warn",
        message=f"{n_dup} duplicate rows on key {tuple(available)}.",
        fix_hint=(
            "Deduplicate before evaluation — duplicate keys collapse propensity "
            "rows and bias DR estimates."
        ),
        category="schema",
    )


def _check_finite_outcomes(
    df: pd.DataFrame,
    outcome_cols: Iterable[str],
) -> Check:
    cols = [c for c in outcome_cols if c in df.columns]
    if not cols:
        return Check(
            name="finite_outcomes",
            status="warn",
            message=(
                f"None of the candidate outcome columns ({list(outcome_cols)}) "
                f"were found."
            ),
            category="statistical",
        )
    nonfinite = {}
    for c in cols:
        try:
            series = pd.to_numeric(df[c], errors="coerce")
        except (TypeError, ValueError):
            nonfinite[c] = len(df)
            continue
        n_bad = int((~np.isfinite(series.to_numpy(dtype=float))).sum())
        if n_bad:
            nonfinite[c] = n_bad
    if not nonfinite:
        return Check(
            name="finite_outcomes",
            status="pass",
            message=f"All outcome columns {tuple(cols)} are finite.",
            category="statistical",
        )
    return Check(
        name="finite_outcomes",
        status="fail",
        message=(
            "Non-finite outcome values: "
            + ", ".join(f"{c}={n}" for c, n in sorted(nonfinite.items()))
        ),
        fix_hint=(
            "Drop or impute non-finite outcome rows before calling evaluate_*; "
            "DR estimators propagate NaN."
        ),
        category="statistical",
    )


def _check_positivity(
    df: pd.DataFrame,
    *,
    epsilon: float = 0.01,
) -> Check:
    """Approximate positivity: fraction of rows where the observed-action's
    eligibility column equals 1 but per-action propensities (if present) are
    near zero."""
    pscore_col = next((c for c in df.columns if c.lower() == "propensity"), None)
    if pscore_col is None:
        return Check(
            name="positivity",
            status="warn",
            message=(
                "No 'propensity' column; positivity check skipped. evaluate_* "
                "will estimate propensities internally."
            ),
            category="statistical",
        )
    try:
        series = pd.to_numeric(df[pscore_col], errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError):
        return Check(
            name="positivity",
            status="warn",
            message="Could not coerce 'propensity' column to numeric.",
            category="statistical",
        )
    n_total = series.size
    if n_total == 0:
        return Check(
            name="positivity",
            status="warn",
            message="Empty 'propensity' column.",
            category="statistical",
        )
    bad_mask = (series < epsilon) | (~np.isfinite(series))
    n_bad = int(bad_mask.sum())
    frac_bad = n_bad / n_total if n_total else 0.0
    if frac_bad >= 0.05:  # noqa: PLR2004
        return Check(
            name="positivity",
            status="warn",
            message=(
                f"{n_bad}/{n_total} ({frac_bad:.1%}) rows have propensity < "
                f"{epsilon:.3g}; positivity is marginal."
            ),
            fix_hint=(
                "Trim or reweight rows with near-zero propensity; consider a "
                "stricter clip grid in evaluate_*."
            ),
            category="statistical",
        )
    return Check(
        name="positivity",
        status="pass",
        message=(f"{frac_bad:.1%} of rows below ε={epsilon:.3g} — positivity ok."),
        category="statistical",
    )


def _check_sample_size(
    df: pd.DataFrame, *, min_rows: int = 500, n_splits: int = 3
) -> Check:
    n = len(df)
    fold_floor = min_rows * n_splits
    if n < min_rows:
        return Check(
            name="sample_size",
            status="fail",
            message=f"Only {n} rows; below absolute minimum of {min_rows}.",
            fix_hint="Collect more data or lower n_splits when calling evaluate_*.",
            category="sample_size",
        )
    if n < fold_floor:
        return Check(
            name="sample_size",
            status="warn",
            message=(
                f"{n} rows is below the {fold_floor}-row floor for {n_splits} "
                f"CV folds; per-fold sample sizes will be tight."
            ),
            fix_hint=f"Either collect more data or reduce n_splits below {n_splits}.",
            category="sample_size",
        )
    return Check(
        name="sample_size",
        status="pass",
        message=f"{n} rows is sufficient for {n_splits}-fold CV.",
        category="sample_size",
    )


def _check_input_is_dataframe(name: str, value: Any) -> Check | None:
    if not isinstance(value, pd.DataFrame):
        return Check(
            name="input_type",
            status="fail",
            message=f"{name} is not a pandas DataFrame (got {type(value).__name__}).",
            fix_hint=f"Pass a pandas DataFrame as {name}.",
            category="schema",
        )
    return None


def doctor(
    logs: pd.DataFrame,
    *,
    kind: Literal["standard", "pairwise"] = "standard",
    op_daily_df: pd.DataFrame | None = None,
    metric_col: str = "service_time",
    n_splits: int = 3,
    strict: bool = False,
) -> DoctorReport:
    """Run a preflight diagnostic battery on ``logs`` (and optional
    ``op_daily_df`` for pairwise mode).

    Parameters
    ----------
    logs : pd.DataFrame
        Standard-mode logs or pairwise logs (interpretation depends on
        ``kind``).
    kind : ``"standard"`` or ``"pairwise"``
        Which validator family to call.
    op_daily_df : pd.DataFrame, optional
        Required when ``kind="pairwise"``; ignored otherwise.
    metric_col : str, default ``"service_time"``
        Name of the reward/outcome column. Used by the standard and pairwise
        schema validators (as ``validate_logs(y_col=...)`` in standard mode) and
        by the finite-outcomes check. Pass your own column name for
        general-purpose OPE logs whose reward is not ``"service_time"``.
    n_splits : int, default 3
        CV split count used by the sample-size floor heuristic.
    strict : bool, default False
        Forwarded to ``validate_logs`` / ``validate_pairwise_inputs``.

    Returns
    -------
    DoctorReport
        Always returned. The function is non-raising — even on a malformed
        ``logs`` argument the report surfaces the failure as a
        ``status="fail"`` :class:`Check`.
    """
    checks: list[Check] = [_check_environment(), _check_optional_extras()]
    input_err = _check_input_is_dataframe("logs", logs)
    if input_err is not None:
        checks.append(input_err)
        return _finalize_report(checks)

    if kind not in ("standard", "pairwise"):
        checks.append(
            Check(
                name="kind",
                status="fail",
                message=f"Unknown kind={kind!r}. Expected 'standard' or 'pairwise'.",
                fix_hint="Pass kind='standard' or kind='pairwise'.",
                category="input",
            )
        )
        return _finalize_report(checks)

    if kind == "pairwise":
        checks.append(
            _check_pairwise_schema(logs, op_daily_df, metric_col, strict=strict)
        )
        checks.append(
            _check_no_duplicates(
                logs, key_cols=["arrival_day", "client_id", "operator_id"]
            )
        )
        outcome_cols: tuple[str, ...] = (metric_col,)
    else:
        checks.append(_check_logs_schema(logs, metric_col=metric_col, strict=strict))
        checks.append(_check_no_duplicates(logs, key_cols=["client_id", "arrival_ts"]))
        outcome_cols = tuple(dict.fromkeys((metric_col, "service_time", "reward", "y")))

    checks.append(_check_finite_outcomes(logs, outcome_cols=outcome_cols))
    checks.append(_check_positivity(logs))
    checks.append(_check_sample_size(logs, n_splits=n_splits))

    return _finalize_report(checks)


def _finalize_report(checks: list[Check]) -> DoctorReport:
    summary: CheckStatus = "pass"
    for c in checks:
        if c.status == "fail":
            summary = "fail"
            break
        if c.status == "warn":
            summary = "warn"
    ok = summary != "fail"
    return DoctorReport(ok=ok, checks=checks, summary=summary)


__all__ = ["Check", "DoctorReport", "doctor"]
