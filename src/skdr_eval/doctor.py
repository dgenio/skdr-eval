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
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .capabilities import Capability, get_capabilities, get_capability_matrix
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


@dataclass(frozen=True)
class DataProfile:
    """Privacy-safe shape/schema fingerprint of the inspected ``logs`` (#246).

    Captures only column names, dtypes, row count, and mode — **never any cell
    values** — so it can seed a copy-paste minimal reproduction
    (:meth:`DoctorReport.to_repro`) without leaking user data.

    Attributes
    ----------
    kind : str
        ``"standard"`` or ``"pairwise"``.
    metric_col : str
        The reward/outcome column name passed to ``doctor``.
    n_rows : int
        Row count of the inspected frame.
    columns : list[tuple[str, str]]
        ``(column_name, dtype_str)`` pairs, in column order.
    """

    kind: str
    metric_col: str
    n_rows: int
    columns: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric_col": self.metric_col,
            "n_rows": self.n_rows,
            "columns": [{"name": n, "dtype": d} for n, d in self.columns],
        }


def _profile_dataframe(df: pd.DataFrame, *, kind: str, metric_col: str) -> DataProfile:
    """Build a :class:`DataProfile` from ``df`` (names/dtypes/shape only)."""
    return DataProfile(
        kind=kind,
        metric_col=metric_col,
        n_rows=len(df),
        columns=[(str(c), str(df[c].dtype)) for c in df.columns],
    )


def _repro_column_expr(dtype: str) -> str:
    """Map a dtype string to a runnable, data-free column generator expression.

    Best-effort: pandas extension/annotated dtypes (e.g. ``Int64``, ``Float64``,
    ``int64[pyarrow]``, ``boolean``, ``string``, ``category``) are normalized to
    a NumPy-constructible placeholder so the generated snippet always runs. The
    column's broad dtype family is preserved, not necessarily the exact
    extension dtype.
    """
    # Drop any storage annotation (e.g. ``int64[pyarrow]`` → ``int64``) and
    # case-fold so nullable dtypes (``Int64`` → ``int64``) map to a NumPy name.
    base = dtype.split("[", 1)[0].strip().lower()
    if base.startswith("datetime"):
        return 'pd.date_range("2024-01-01", periods=n, freq="s")'
    if base in {"bool", "boolean"}:
        return "np.zeros(n, dtype=bool)"
    if base.startswith(("int", "uint", "float")):
        return f'np.zeros(n, dtype="{base}")'
    # object / string / category / everything else: data-free placeholder.
    return '[""] * n'


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
    capabilities: list[Capability] = field(default_factory=list)
    profile: DataProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "capabilities": [c.to_dict() for c in self.capabilities],
            "profile": self.profile.to_dict() if self.profile is not None else None,
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
        if self.capabilities:
            lines.append("─" * 40)
            lines.append("Optional extras (capability matrix):")
            on = "✓" if color else "yes"
            off = "✗" if color else "no "
            for cap in self.capabilities:
                mark = on if cap.installed else off
                hint = "" if cap.installed else f"  → {cap.install_hint}"
                lines.append(f"  [{mark}] {cap.extra}: {cap.feature}{hint}")
        lines.append("─" * 40)
        overall_icon = glyph_map.get(self.summary, self.summary)
        lines.append(f"Overall: {overall_icon} ({self.summary})")
        return "\n".join(lines)

    def to_repro(self) -> str:
        """Generate a copy-paste, **data-free** minimal reproduction (#246).

        Emits a runnable Python snippet that rebuilds a synthetic frame with the
        *same column names and row count* as the inspected logs, and best-effort
        dtypes (pandas extension/categorical dtypes are normalized to a
        NumPy-constructible placeholder, so the snippet always runs even if it
        does not reproduce the exact extension dtype). Columns are filled with
        placeholder zeros/empties — never any real values — and the snippet
        re-runs :func:`doctor`. Paste it into a bug report so maintainers can
        reproduce a schema/setup issue without access to your data.

        Returns
        -------
        str
            A self-contained Python snippet. If no :class:`DataProfile` was
            captured (e.g. the input was not a DataFrame), returns a short
            comment explaining why no reproduction could be generated.
        """
        if self.profile is None:
            return (
                "# No reproduction available: the doctor could not profile the "
                "input\n# (it was not a pandas DataFrame)."
            )
        p = self.profile
        col_lines = [
            f"        {name!r}: {_repro_column_expr(dtype)},  # dtype: {dtype}"
            for name, dtype in p.columns
        ]
        cols_block = "\n".join(col_lines)
        return (
            "# Minimal, data-free reproduction generated by skdr-eval doctor "
            "(#246).\n"
            "# Column names/dtypes/shape only — no real values are included.\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import skdr_eval\n"
            "\n"
            f"n = {p.n_rows}\n"
            "df = pd.DataFrame(\n"
            "    {\n"
            f"{cols_block}\n"
            "    }\n"
            ")\n"
            "report = skdr_eval.doctor(\n"
            f"    df, kind={p.kind!r}, metric_col={p.metric_col!r}\n"
            ")\n"
            "print(report.to_text())\n"
        )

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


def _check_time_ordering(df: pd.DataFrame, *, time_col: str) -> Check:
    """Verify rows are chronologically ordered (time-aware CV assumes this)."""
    if time_col not in df.columns:
        return Check(
            name="time_ordering",
            status="warn",
            message=(
                f"No '{time_col}' column found; cannot verify chronological order."
            ),
            category="schema",
        )
    series = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(
        series
    ):
        # Already a sortable dtype — no parsing (and no inference warning) needed.
        parsed: pd.Series = series
    else:
        # Object/string column: try datetime, then numeric. Suppress pandas'
        # "could not infer format" UserWarning — falling back to per-element
        # parsing is exactly what we want for a best-effort ordering check.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="raise")
        except (ValueError, TypeError, OverflowError):
            try:
                parsed = pd.to_numeric(series, errors="raise")
            except (ValueError, TypeError):
                return Check(
                    name="time_ordering",
                    status="warn",
                    message=f"Could not parse '{time_col}' as datetime or numeric.",
                    fix_hint="Store arrival times as datetimes or sortable numerics.",
                    category="schema",
                )
    if bool(parsed.is_monotonic_increasing):
        return Check(
            name="time_ordering",
            status="pass",
            message=f"'{time_col}' is non-decreasing (rows are time-ordered).",
            category="schema",
        )
    arr = parsed.to_numpy()
    n_inversions = int(np.sum(arr[1:] < arr[:-1]))
    return Check(
        name="time_ordering",
        status="warn",
        message=(
            f"'{time_col}' is not sorted ({n_inversions} out-of-order rows); "
            f"time-aware splits assume chronological order."
        ),
        fix_hint=(
            f"Sort logs by '{time_col}' before evaluate_* "
            f"(df.sort_values('{time_col}'))."
        ),
        category="schema",
    )


def _check_missingness(df: pd.DataFrame, *, threshold: float = 0.2) -> Check:
    """Flag columns whose missing-value fraction exceeds ``threshold``."""
    if df.empty or df.shape[1] == 0:
        return Check(
            name="missingness",
            status="warn",
            message="No columns to check for missingness.",
            category="schema",
        )
    null_frac = df.isna().mean()
    offenders = {str(c): float(v) for c, v in null_frac.items() if float(v) > threshold}
    if not offenders:
        worst = float(null_frac.max()) if len(null_frac) else 0.0
        return Check(
            name="missingness",
            status="pass",
            message=(f"No column exceeds {threshold:.0%} missing (worst {worst:.1%})."),
            category="schema",
        )
    top = ", ".join(
        f"{c}={v:.1%}" for c, v in sorted(offenders.items(), key=lambda kv: -kv[1])[:5]
    )
    return Check(
        name="missingness",
        status="warn",
        message=f"{len(offenders)} column(s) above {threshold:.0%} missing: {top}.",
        fix_hint=(
            "Impute or drop high-missingness columns before evaluate_*; NaNs "
            "propagate through the DR estimators."
        ),
        category="schema",
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
    capabilities = get_capability_matrix()
    checks: list[Check] = [_check_environment(), _check_optional_extras()]
    input_err = _check_input_is_dataframe("logs", logs)
    if input_err is not None:
        checks.append(input_err)
        return _finalize_report(checks, capabilities=capabilities)

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
        return _finalize_report(checks, capabilities=capabilities)

    profile = _profile_dataframe(logs, kind=kind, metric_col=metric_col)

    if kind == "pairwise":
        checks.append(
            _check_pairwise_schema(logs, op_daily_df, metric_col, strict=strict)
        )
        checks.append(
            _check_no_duplicates(
                logs, key_cols=["arrival_day", "client_id", "operator_id"]
            )
        )
        time_col = "arrival_day"
        outcome_cols: tuple[str, ...] = (metric_col,)
    else:
        checks.append(_check_logs_schema(logs, metric_col=metric_col, strict=strict))
        checks.append(_check_no_duplicates(logs, key_cols=["client_id", "arrival_ts"]))
        time_col = "arrival_ts"
        outcome_cols = tuple(dict.fromkeys((metric_col, "service_time", "reward", "y")))

    checks.append(_check_time_ordering(logs, time_col=time_col))
    checks.append(_check_missingness(logs))
    checks.append(_check_finite_outcomes(logs, outcome_cols=outcome_cols))
    checks.append(_check_positivity(logs))
    checks.append(_check_sample_size(logs, n_splits=n_splits))

    return _finalize_report(checks, capabilities=capabilities, profile=profile)


def _finalize_report(
    checks: list[Check],
    *,
    capabilities: list[Capability] | None = None,
    profile: DataProfile | None = None,
) -> DoctorReport:
    summary: CheckStatus = "pass"
    for c in checks:
        if c.status == "fail":
            summary = "fail"
            break
        if c.status == "warn":
            summary = "warn"
    ok = summary != "fail"
    return DoctorReport(
        ok=ok,
        checks=checks,
        summary=summary,
        capabilities=list(capabilities) if capabilities else [],
        profile=profile,
    )


__all__ = ["Check", "DataProfile", "DoctorReport", "doctor"]
