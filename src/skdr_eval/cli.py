"""``skdr-eval`` CLI (Typer) — implements #89.

Subcommands
-----------
- ``evaluate``        — run :func:`skdr_eval.evaluate_sklearn_models`
- ``pairwise``        — run :func:`skdr_eval.evaluate_pairwise_models`
- ``card``            — re-render the YAML card from a saved artifact JSON
- ``compare``         — diff two saved artifacts; gate on verdict regressions
- ``schema``          — print the JSON Schema for the artifact or card
- ``badge``           — emit a shareable SVG/Markdown evaluation badge
- ``validate-schema`` — call ``validate_logs`` / ``validate_pairwise_inputs``
- ``doctor``          — run :func:`skdr_eval.doctor`
- ``version``         — print the package version

Exit codes
----------
* ``0`` — success: no ``do_not_deploy`` or ``insufficient_evidence`` verdict
  was produced (a recommendation that could not be computed is logged at
  WARNING and does not, by itself, change the exit code).
* ``1`` — data / schema error (the doctor or a validator flagged ``fail``).
* ``2`` — environment / import error (a required optional dep is missing).
* ``3`` — at least one evaluated estimator's recommendation verdict was
  ``do_not_deploy``; use this exit code as a CI gate. Takes precedence over
  exit code 4.
* ``4`` — at least one evaluated estimator's verdict was
  ``insufficient_evidence`` (and none was ``do_not_deploy``): the logs cannot
  yet support a deploy/don't-deploy decision. Treated as non-deployable so an
  honest "we can't tell" cannot pass a CI gate as green. See
  ``docs/report-interpretation.md``.

The verdict gate (#196, #197) inspects **every** estimator present in the
artifact (``DR``, ``SNDR``, ``MRDR``, ``SWITCH-DR``, ``DRos``, ``MIPS``, …),
not just ``DR``/``SNDR``.

The CLI is gated behind the ``[cli]`` extra (``pip install
'skdr-eval[cli]'``). Importing this module without ``typer`` installed
raises a structured :class:`ImportError`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

try:
    import typer
except ImportError as exc:  # pragma: no cover - exercised via tests for the message
    msg = (
        "The 'skdr-eval' CLI requires the [cli] extra. "
        "Install with: pip install 'skdr-eval[cli]'"
    )
    raise ImportError(msg) from exc

import pandas as pd

import skdr_eval
from skdr_eval.capabilities import get_capability_matrix
from skdr_eval.doctor import doctor as doctor_fn
from skdr_eval.reporting import (
    ArtifactSchema,
    EvaluationCard,
    _thin_artifact_from_schema,
    compare_artifacts,
    explain_artifact_schema,
)
from skdr_eval.trackers import FileTracker

logger = logging.getLogger("skdr_eval.cli")

EXIT_OK = 0
EXIT_DATA = 1
EXIT_ENV = 2
EXIT_DO_NOT_DEPLOY = 3
EXIT_INSUFFICIENT_EVIDENCE = 4

app = typer.Typer(
    name="skdr-eval",
    help="CLI for the skdr-eval offline policy-evaluation library.",
    no_args_is_help=True,
    add_completion=True,
)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Auto-detect ``.parquet`` / ``.csv`` / ``.feather`` based on suffix.

    Raises :class:`typer.Exit` with code ``EXIT_DATA`` on unknown suffix or
    read error.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif suffix in {".feather", ".arrow"}:
            df = pd.read_feather(path)
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            typer.secho(
                f"Unknown input format for {path} (suffix={suffix!r}); "
                f"expected .parquet/.csv/.tsv/.feather.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=EXIT_DATA)
    except (OSError, ValueError) as exc:
        typer.secho(f"Failed to read {path}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    return _normalize_loaded_dataframe(df)


def _normalize_loaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known quirks introduced by parquet round-trips.

    - Object columns whose first cell is a numpy ndarray are unboxed to
      Python lists (matches the in-memory shape produced by
      :func:`skdr_eval.make_pairwise_synth`).
    """
    import numpy as np  # noqa: PLC0415

    for col in df.columns:
        series = df[col]
        if series.dtype != object or series.empty:
            continue
        first_non_null = next(
            (v for v in series.to_numpy() if v is not None),
            None,
        )
        if first_non_null is None:
            continue
        if isinstance(first_non_null, np.ndarray):
            df[col] = series.map(
                lambda v: v.tolist() if isinstance(v, np.ndarray) else v
            )
    return df


def _load_model(path: Path) -> Any:
    """Load a pickled / joblib model. Refuse non-local schemes.

    .. warning::
        ``joblib.load`` uses pickle under the hood and can execute arbitrary
        code. Only load model files that you or your team created.
    """
    if "://" in str(path):
        typer.secho(
            f"Refusing to load model from {path}: only local paths are allowed.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    try:
        import joblib  # noqa: PLC0415
    except ImportError as exc:
        typer.secho(
            "joblib is required to load models. "
            "Install with: pip install 'skdr-eval[cli]'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_ENV) from exc
    if not path.exists():
        typer.secho(f"Model file not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA)
    return joblib.load(path)


def _parse_model_specs(specs: list[str]) -> dict[str, Any]:
    """Parse ``--model NAME=PATH`` tokens into a dict of fitted estimators."""
    out: dict[str, Any] = {}
    for token in specs:
        if "=" not in token:
            typer.secho(
                f"Expected '--model NAME=PATH', got {token!r}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=EXIT_DATA)
        name, _, path_str = token.partition("=")
        if not name or not path_str:
            typer.secho(
                f"Empty model name or path in {token!r}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=EXIT_DATA)
        out[name] = _load_model(Path(path_str))
    return out


def _ensure_out_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_artifact_outputs(
    artifact: skdr_eval.EvaluationArtifact, out: Path
) -> dict[str, Path]:
    """Write JSON + HTML report + one card YAML per ``(model, estimator)``."""
    out = _ensure_out_dir(out)
    paths: dict[str, Path] = {}

    schema = artifact.to_schema()
    json_path = out / "artifact.json"
    json_path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    paths["artifact.json"] = json_path

    report_path = out / "report.html"
    artifact.to_html(report_path)
    paths["report.html"] = report_path

    cards_dir = out / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for model_name in artifact.detailed:
        for estimator in ("DR", "SNDR"):
            if estimator not in artifact.detailed[model_name]:
                continue
            card = artifact.card_schema(model_name, estimator=estimator)
            safe_name = (
                model_name.replace("/", "_").replace("\\", "_").replace("..", "_")
            )
            yaml_path = cards_dir / f"{safe_name}_{estimator}.card.yaml"
            json_path_card = cards_dir / f"{safe_name}_{estimator}.card.json"
            card.to_yaml(yaml_path)
            card.to_json(json_path_card)
            paths[yaml_path.name] = yaml_path
            paths[json_path_card.name] = json_path_card

    return paths


def _verdict_exit_code(artifact: skdr_eval.EvaluationArtifact) -> int:
    """Map the artifact's recommendation verdicts to a CI-gate exit code.

    Inspects **every** estimator present for each model (not just ``DR`` /
    ``SNDR``), so a ``do_not_deploy`` from any first-class estimator
    (MRDR / SWITCH-DR / DRos / MIPS, …) trips the gate (#196).

    Returns
    -------
    int
        * ``EXIT_DO_NOT_DEPLOY`` (3) if any verdict is ``do_not_deploy`` —
          this takes precedence over everything else.
        * ``EXIT_INSUFFICIENT_EVIDENCE`` (4) if any verdict is
          ``insufficient_evidence`` and none is ``do_not_deploy`` (#197).
        * ``EXIT_OK`` (0) otherwise.

    Recommendation errors are logged at WARNING level rather than silently
    swallowed, so a broken row cannot quietly turn a gate green (#196).
    """
    saw_insufficient = False
    for model_name in artifact.detailed:
        for estimator in artifact.detailed[model_name]:
            try:
                rec = artifact.recommendation(model_name, estimator=estimator)
            except Exception as exc:
                logger.warning(
                    "Could not compute recommendation for model=%r estimator=%r: %s",
                    model_name,
                    estimator,
                    exc,
                    exc_info=True,
                )
                continue
            if rec.verdict == "do_not_deploy":
                return EXIT_DO_NOT_DEPLOY
            if rec.verdict == "insufficient_evidence":
                saw_insufficient = True
    return EXIT_INSUFFICIENT_EVIDENCE if saw_insufficient else EXIT_OK


# Selectable stdout formats for the headline report (#231).
_REPORT_FORMATS = ("table", "csv", "json", "markdown")


def _render_report(artifact: skdr_eval.EvaluationArtifact, fmt: str) -> str:
    """Render the headline report in the requested stdout format (#231)."""
    if fmt == "table":
        return str(artifact.report.to_string(index=False))
    if fmt == "csv":
        return str(artifact.report.to_csv(index=False)).rstrip("\n")
    if fmt == "json":
        return artifact.to_json_str()
    if fmt == "markdown":
        return artifact.to_markdown()
    raise typer.BadParameter(
        f"--format must be one of {_REPORT_FORMATS} (got {fmt!r}).",
    )


def _emit_report_and_confirm(
    artifact: skdr_eval.EvaluationArtifact,
    fmt: str | None,
    paths: dict[str, Path],
    out: Path,
) -> None:
    """Emit the report to stdout (if ``--format`` given) with stream separation.

    When ``fmt`` is set the machine-readable report goes to **stdout** and the
    human-readable "wrote N files" confirmation is routed to **stderr**, so the
    command composes in a pipe (``| jq``, ``| column``). With ``fmt is None``
    the legacy behaviour is preserved: the confirmation prints to stdout.
    """
    if fmt is None:
        typer.echo(f"Wrote {len(paths)} files to {out}.")
        return
    if fmt not in _REPORT_FORMATS:
        raise typer.BadParameter(
            f"--format must be one of {_REPORT_FORMATS} (got {fmt!r}).",
        )
    typer.echo(_render_report(artifact, fmt))
    typer.echo(f"Wrote {len(paths)} files to {out}.", err=True)


@app.command("version")
def version_cmd() -> None:
    """Print the installed skdr-eval version."""
    typer.echo(skdr_eval.__version__)


@app.command("doctor")
def doctor_cmd(
    logs: Path = typer.Argument(
        ..., exists=True, readable=True, help="Logs file (parquet/csv/feather)."
    ),
    kind: str = typer.Option("standard", "--kind", help="standard | pairwise"),
    op_daily: Path | None = typer.Option(
        None, "--op-daily", help="op_daily_df (pairwise only)."
    ),
    metric_col: str = typer.Option(
        "service_time", "--metric-col", help="Outcome column."
    ),
    n_splits: int = typer.Option(3, "--n-splits", min=1, help="CV split count."),
    strict: bool = typer.Option(
        False, "--strict/--no-strict", help="Strict schema validation."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a text table."
    ),
    repro: bool = typer.Option(
        False,
        "--repro",
        help="Also emit a copy-paste, data-free minimal reproduction snippet.",
    ),
) -> None:
    """Run preflight diagnostics on input data + environment."""
    if kind not in {"standard", "pairwise"}:
        typer.secho(
            f"--kind must be 'standard' or 'pairwise' (got {kind!r}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    logs_df = _load_dataframe(logs)
    op_daily_df = _load_dataframe(op_daily) if op_daily is not None else None
    report = doctor_fn(
        logs_df,
        kind=kind,  # type: ignore[arg-type]
        op_daily_df=op_daily_df,
        metric_col=metric_col,
        n_splits=n_splits,
        strict=strict,
    )
    if json_out:
        payload = report.to_dict()
        if repro:
            payload["repro"] = report.to_repro()
        typer.echo(json.dumps(payload, indent=2))
    else:
        report.print(color=sys.stdout.isatty())
        if repro:
            typer.echo("")
            typer.echo(report.to_repro())
    if not report.ok:
        raise typer.Exit(code=EXIT_DATA)


@app.command("validate-schema")
def validate_schema_cmd(
    logs: Path = typer.Argument(
        ..., exists=True, readable=True, help="Logs file (parquet/csv/feather)."
    ),
    kind: str = typer.Option("standard", "--kind", help="'standard' or 'pairwise'."),
    op_daily: Path | None = typer.Option(
        None, "--op-daily", help="op_daily_df (pairwise only)."
    ),
    metric_col: str = typer.Option(
        "service_time", "--metric-col", help="Outcome column (pairwise)."
    ),
    strict: bool = typer.Option(
        False, "--strict/--no-strict", help="Strict schema validation."
    ),
) -> None:
    """Validate that ``logs`` matches the standard or pairwise schema."""
    if kind not in {"standard", "pairwise"}:
        typer.secho(
            f"--kind must be 'standard' or 'pairwise' (got {kind!r}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    logs_df = _load_dataframe(logs)
    try:
        if kind == "standard":
            skdr_eval.validate_logs(logs_df, strict=strict)
        else:
            if op_daily is None:
                typer.secho(
                    "--op-daily is required for kind='pairwise'.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=EXIT_DATA)
            op_df = _load_dataframe(op_daily)
            skdr_eval.validate_pairwise_inputs(
                logs_df, op_df, metric_col=metric_col, strict=strict
            )
    except (
        skdr_eval.DataValidationError,
        skdr_eval.InsufficientDataError,
    ) as exc:
        typer.secho(f"schema FAIL: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    typer.secho("schema OK", fg=typer.colors.GREEN)


@app.command("evaluate")
def evaluate_cmd(
    logs: Path = typer.Argument(
        ..., exists=True, readable=True, help="Logs file (parquet/csv/feather)."
    ),
    model: list[str] = typer.Option(
        ..., "--model", help="Repeatable. Format: NAME=PATH_TO_PICKLE."
    ),
    out: Path = typer.Option(
        Path("./skdr_eval_out"), "--out", help="Output directory."
    ),
    policy_train: str = typer.Option(
        "pre_split", "--policy-train", help="'pre_split' or 'all'."
    ),
    n_splits: int = typer.Option(3, "--n-splits", min=1),
    ci_bootstrap: bool = typer.Option(
        False, "--ci-bootstrap/--no-ci-bootstrap", help="Compute bootstrap CIs."
    ),
    random_state: int = typer.Option(0, "--random-state"),
    fmt: str | None = typer.Option(
        None,
        "--format",
        help=(
            "Print the headline report to stdout in this format "
            "(table|csv|json|markdown); the 'wrote N files' line moves to "
            "stderr so stdout is pipe-clean. Omit to keep the legacy output."
        ),
    ),
    tracker_dir: Path | None = typer.Option(
        None, "--tracker-dir", help="Optional FileTracker output directory."
    ),
) -> None:
    """Run :func:`evaluate_sklearn_models` on local logs + pickled models."""
    logs_df = _load_dataframe(logs)
    models = _parse_model_specs(model)
    tracker = FileTracker(tracker_dir) if tracker_dir is not None else None
    try:
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs_df,
            models=models,
            fit_models=False,
            n_splits=n_splits,
            random_state=random_state,
            ci_bootstrap=ci_bootstrap,
            policy_train=policy_train,
            tracker=tracker,
        )
    except skdr_eval.SkdrEvalError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    except ValueError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    paths = _write_artifact_outputs(artifact, out)
    _emit_report_and_confirm(artifact, fmt, paths, out)
    raise typer.Exit(code=_verdict_exit_code(artifact))


@app.command("pairwise")
def pairwise_cmd(
    logs: Path = typer.Argument(
        ..., exists=True, readable=True, help="Logs file (parquet/csv/feather)."
    ),
    op_daily: Path = typer.Argument(
        ..., exists=True, readable=True, help="op_daily_df file."
    ),
    model: list[str] = typer.Option(
        ..., "--model", help="Repeatable. Format: NAME=PATH_TO_PICKLE."
    ),
    metric_col: str = typer.Option("service_time", "--metric-col"),
    task_type: str = typer.Option(
        "regression", "--task-type", help="regression|binary"
    ),
    direction: str = typer.Option("min", "--direction", help="min|max"),
    out: Path = typer.Option(Path("./skdr_eval_out"), "--out"),
    strategy: str = typer.Option(
        "auto", "--strategy", help="auto|direct|stream|stream_topk"
    ),
    n_splits: int = typer.Option(3, "--n-splits", min=1),
    random_state: int = typer.Option(0, "--random-state"),
    ci_bootstrap: bool = typer.Option(False, "--ci-bootstrap/--no-ci-bootstrap"),
    fmt: str | None = typer.Option(
        None,
        "--format",
        help=(
            "Print the headline report to stdout in this format "
            "(table|csv|json|markdown); the 'wrote N files' line moves to "
            "stderr so stdout is pipe-clean. Omit to keep the legacy output."
        ),
    ),
    tracker_dir: Path | None = typer.Option(
        None, "--tracker-dir", help="Optional FileTracker output directory."
    ),
) -> None:
    """Run :func:`evaluate_pairwise_models` end-to-end."""
    if task_type not in {"regression", "binary"}:
        typer.secho(
            f"--task-type must be 'regression' or 'binary' (got {task_type!r}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    if direction not in {"min", "max"}:
        typer.secho(
            f"--direction must be 'min' or 'max' (got {direction!r}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    logs_df = _load_dataframe(logs)
    op_df = _load_dataframe(op_daily)
    models = _parse_model_specs(model)
    tracker = FileTracker(tracker_dir) if tracker_dir is not None else None
    try:
        artifact = skdr_eval.evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_df,
            models=models,
            metric_col=metric_col,
            task_type=task_type,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            n_splits=n_splits,
            strategy=strategy,  # type: ignore[arg-type]
            random_state=random_state,
            ci_bootstrap=ci_bootstrap,
            tracker=tracker,
        )
    except skdr_eval.SkdrEvalError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    except ValueError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    paths = _write_artifact_outputs(artifact, out)
    _emit_report_and_confirm(artifact, fmt, paths, out)
    raise typer.Exit(code=_verdict_exit_code(artifact))


@app.command("card")
def card_cmd(
    artifact_json: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to artifact.json."
    ),
    model_name: str = typer.Option(..., "--model", help="Model name."),
    estimator: str = typer.Option("SNDR", "--estimator", help="DR or SNDR."),
    out: Path = typer.Option(Path("./card.yaml"), "--out", help="Output card path."),
    fmt: str = typer.Option("yaml", "--format", help="'yaml' or 'json'."),
) -> None:
    """Render an :class:`EvaluationCard` from a saved ``artifact.json``."""
    if fmt not in {"yaml", "json"}:
        typer.secho(
            f"--format must be 'yaml' or 'json' (got {fmt!r}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    schema = skdr_eval.load_artifact_json(artifact_json)
    card = _card_from_artifact_schema(
        schema, model_name=model_name, estimator=estimator
    )
    if fmt == "yaml":
        card.to_yaml(out)
    else:
        card.to_json(out)
    typer.echo(f"Wrote {out}")


def _card_from_artifact_schema(
    schema: skdr_eval.ArtifactSchema, *, model_name: str, estimator: str
) -> skdr_eval.EvaluationCard:
    """Reconstruct an :class:`EvaluationCard` directly from a saved schema.

    Unlike :meth:`EvaluationArtifact.card_schema`, this path does not invoke
    :func:`gate_diagnostics` or :meth:`recommendation` — the saved artifact
    does not retain the per-decision contributions those helpers can inspect.
    The resulting card has the same headline / diagnostics / sensitivity /
    provenance fields as a live ``card_schema()`` call.
    """
    from .reporting import (  # noqa: PLC0415
        DiagnosticsBlock,
        EvaluationCard,
        HeadlineBlock,
        ProvenanceBlock,
        SensitivityBlock,
        TrustBlock,
    )

    def _coerce_float(value: Any) -> float | None:
        """Coerce to float | None; non-finite → None."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        import math  # noqa: PLC0415

        return v if math.isfinite(v) else None

    row = next(
        (
            r
            for r in schema.report
            if r.model == model_name and r.estimator == estimator
        ),
        None,
    )
    if row is None:
        typer.secho(
            f"No row in artifact for model={model_name!r}, estimator={estimator!r}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_DATA)
    sens = next(
        (
            s
            for s in schema.sensitivity
            if s.model == model_name and s.estimator == estimator
        ),
        None,
    )
    warn = next(
        (
            w
            for w in schema.warnings
            if w.model == model_name and w.estimator == estimator
        ),
        None,
    )

    diag_payload = schema.diagnostics.get(model_name)

    headline = HeadlineBlock(
        estimator=estimator,
        V_hat=_coerce_float(row.V_hat),
        ci_lower=_coerce_float(row.ci_lower),
        ci_upper=_coerce_float(row.ci_upper),
        ci_alpha=_coerce_float(schema.metadata.get("alpha")),
        baseline=None,
        delta_vs_baseline=None,
        clip=_coerce_float(row.clip),
    )
    trust = TrustBlock(
        support_health=row.support_health,
        warning_codes=list(warn.warning_codes) if warn is not None else [],
        recommendation=None,
        primary_blocker=None,
    )
    n = int(schema.metadata.get("n_samples", 0)) or None
    ess_val = _coerce_float(row.ESS)
    diagnostics_block = DiagnosticsBlock(
        ESS=ess_val,
        ess_frac=(ess_val / n) if (ess_val is not None and n) else None,
        match_rate=_coerce_float(row.match_rate),
        pareto_k=_coerce_float(row.pareto_k),
        min_pscore=_coerce_float(row.min_pscore),
        tail_mass=_coerce_float(row.tail_mass),
        ece=_coerce_float(diag_payload.ece) if diag_payload else None,
        brier_score=(_coerce_float(diag_payload.brier_score) if diag_payload else None),
        gate=None,
    )
    sensitivity_block = (
        SensitivityBlock()
        if sens is None
        else SensitivityBlock(
            V_min=_coerce_float(sens.V_min),
            V_max=_coerce_float(sens.V_max),
            V_range=_coerce_float(sens.V_range),
            chosen_clip=_coerce_float(sens.chosen_clip),
            chosen_V=_coerce_float(sens.chosen_V),
            argmin_MSE_clip=_coerce_float(sens.argmin_MSE_clip),
            dr_sndr_agree=bool(sens.dr_sndr_agree),
            stable=bool(sens.stable),
        )
    )
    provenance = ProvenanceBlock(
        skdr_eval_version=schema.skdr_eval_version,
        schema_version=schema.schema_version,
        timestamp=schema.timestamp,
        n_samples=n,
        n_splits=schema.metadata.get("n_splits"),
        random_state=schema.metadata.get("random_state"),
        evaluator=schema.metadata.get("evaluator"),
    )
    return EvaluationCard(
        model_name=model_name,
        headline=headline,
        trust=trust,
        diagnostics=diagnostics_block,
        sensitivity=sensitivity_block,
        provenance=provenance,
    )


@app.command("explain")
def explain_cmd(
    artifact_json: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to artifact.json."
    ),
    model_name: str = typer.Option(..., "--model", help="Model name."),
    estimator: str = typer.Option(
        "SNDR", "--estimator", help="Estimator row to explain (e.g. DR, SNDR)."
    ),
    baseline: float | None = typer.Option(
        None, "--baseline", help="Baseline the CI is compared against (default 0)."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a text narrative."
    ),
) -> None:
    """Narrate *why* a saved artifact row got its verdict, without re-running (#201).

    Reads a saved ``artifact.json`` and explains which diagnostics gated the
    chosen ``(model, estimator)`` — each reason with its measured value and the
    threshold it was compared against.
    """
    schema = skdr_eval.load_artifact_json(artifact_json)
    try:
        explanation = explain_artifact_schema(
            schema, model_name, estimator=estimator, baseline=baseline
        )
    except skdr_eval.DataValidationError as exc:
        typer.secho(f"explain: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    if json_out:
        typer.echo(json.dumps(explanation.to_dict(), indent=2))
    else:
        typer.echo(explanation.to_text())


@app.command("capabilities")
def capabilities_cmd(
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a text table."
    ),
) -> None:
    """Show which optional extras are installed and what each one unlocks (#215)."""
    matrix = get_capability_matrix()
    if json_out:
        typer.echo(json.dumps([c.to_dict() for c in matrix], indent=2))
        return
    use_color = sys.stdout.isatty()
    typer.echo("skdr-eval capabilities")
    typer.echo("─" * 48)
    for cap in matrix:
        if cap.installed:
            mark = typer.style("✓", fg=typer.colors.GREEN) if use_color else "yes"
        else:
            mark = typer.style("✗", fg=typer.colors.YELLOW) if use_color else "no "
        typer.echo(f"[{mark}] {cap.extra}: {cap.feature}")
        if not cap.installed:
            typer.echo(f"      {cap.install_hint}")


@app.command("quickstart")
def quickstart_cmd(
    out: Path = typer.Option(
        Path("./skdr_eval_quickstart"),
        "--out",
        help="Output directory for the report + cards.",
    ),
    n: int = typer.Option(
        2000, "--n", min=1, help="Number of synthetic log rows to generate."
    ),
    seed: int = typer.Option(
        0, "--seed", help="Random seed for the synthetic logs + model."
    ),
    ci_bootstrap: bool = typer.Option(
        False,
        "--ci-bootstrap/--no-ci-bootstrap",
        help="Compute bootstrap CIs (slower; enables a deploy/don't-deploy verdict).",
    ),
) -> None:
    """Guided first run: synth logs → doctor → evaluate → card → explain (#207).

    Generates synthetic logs, runs the doctor, evaluates a default model, writes
    the HTML report + cards, and narrates the verdict — the full value loop in a
    single command, no Python required. Always exits ``0`` on success; it is an
    onboarding demo, not a CI gate (use ``evaluate`` for gating).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    typer.secho("[1/4] Generating synthetic logs…", fg=typer.colors.CYAN)
    logs_df, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=seed)

    typer.secho("[2/4] Running doctor…", fg=typer.colors.CYAN)
    report = doctor_fn(logs_df, kind="standard")
    report.print(color=sys.stdout.isatty())

    typer.secho("[3/4] Evaluating a default model…", fg=typer.colors.CYAN)
    models = {"hgb": HistGradientBoostingRegressor(max_iter=30, random_state=seed)}
    try:
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs_df,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=seed,
            ci_bootstrap=ci_bootstrap,
            policy_train="pre_split",
        )
    except skdr_eval.SkdrEvalError as exc:
        typer.secho(f"Evaluation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    paths = _write_artifact_outputs(artifact, out)

    typer.secho("[4/4] Explaining the verdict…", fg=typer.colors.CYAN)
    typer.echo(artifact.explain("hgb", estimator="SNDR").to_text())
    typer.echo("")
    typer.echo(
        f"Wrote {len(paths)} files to {out} "
        f"(open {out / 'report.html'} for the full card)."
    )


@app.command("compare")
def compare_cmd(
    baseline_json: Path = typer.Argument(
        ..., exists=True, readable=True, help="Baseline (previous) artifact.json."
    ),
    candidate_json: Path = typer.Argument(
        ..., exists=True, readable=True, help="Candidate (new) artifact.json."
    ),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text | markdown | json."
    ),
    epsilon: float = typer.Option(
        1e-9, "--epsilon", help="Numeric deltas below this are treated as unchanged."
    ),
) -> None:
    """Diff two saved artifacts and gate on verdict regressions (#184).

    Loads two ``artifact.json`` files and reports per-(model, estimator) value
    deltas, support-health transitions, warning-code changes, and verdict flips.
    Exits ``EXIT_DO_NOT_DEPLOY`` (3) when any verdict regressed vs the baseline,
    so CI can fail not only on ``do_not_deploy`` but on "the verdict got worse
    than the last accepted run".
    """
    if fmt not in {"text", "markdown", "json"}:
        raise typer.BadParameter(
            f"--format must be 'text', 'markdown', or 'json' (got {fmt!r}).",
        )
    baseline = _thin_artifact_from_schema(skdr_eval.load_artifact_json(baseline_json))
    candidate = _thin_artifact_from_schema(skdr_eval.load_artifact_json(candidate_json))
    diff = compare_artifacts(candidate, baseline, epsilon=epsilon)

    if fmt == "json":
        typer.echo(diff.model_dump_json(indent=2))
    elif fmt == "markdown":
        typer.echo(diff.to_markdown())
    else:
        for r in diff.rows:
            flag = " [REGRESSED]" if r.verdict_regressed else ""
            typer.echo(
                f"{r.model}/{r.estimator}: {r.status} "
                f"{r.verdict_before or '—'} -> {r.verdict_after or '—'}{flag}"
            )
        typer.echo(
            "Verdict regression detected."
            if diff.verdict_regressed
            else "No verdict regression."
        )
    raise typer.Exit(code=EXIT_DO_NOT_DEPLOY if diff.verdict_regressed else EXIT_OK)


@app.command("schema")
def schema_cmd(
    kind: str = typer.Option(
        "artifact", "--kind", help="Which schema to emit: 'artifact' or 'card'."
    ),
) -> None:
    """Print the JSON Schema for the artifact or card contract (#205).

    Emits the versioned JSON Schema downstream tooling can use to validate
    ``skdr-eval`` outputs without importing the library. Pipe to a file to pin
    it in Git or publish it.
    """
    if kind == "artifact":
        schema = ArtifactSchema.json_schema()
    elif kind == "card":
        schema = EvaluationCard.json_schema()
    else:
        raise typer.BadParameter(
            f"--kind must be 'artifact' or 'card' (got {kind!r}).",
        )
    typer.echo(json.dumps(schema, indent=2))


@app.command("badge")
def badge_cmd(
    artifact_json: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to artifact.json."
    ),
    model_name: str = typer.Option(..., "--model", help="Model name."),
    estimator: str = typer.Option("SNDR", "--estimator", help="DR or SNDR."),
    out: Path | None = typer.Option(
        None, "--out", help="Write the SVG here; otherwise print the Markdown snippet."
    ),
) -> None:
    """Generate a shareable evaluation badge from a saved artifact (#251).

    Colour is keyed to ``support_health`` so a thin-support result reads as
    cautionary, never oversold. With ``--out`` writes the SVG; otherwise prints
    the Markdown embed snippet to stdout.
    """
    artifact = _thin_artifact_from_schema(skdr_eval.load_artifact_json(artifact_json))
    try:
        badge = artifact.badge(model_name, estimator=estimator)
    except skdr_eval.DataValidationError as exc:
        typer.secho(f"badge: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_DATA) from exc
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(badge["svg"], encoding="utf-8")
        typer.echo(f"Wrote {out}", err=True)
    else:
        typer.echo(badge["markdown"])


if __name__ == "__main__":  # pragma: no cover
    app()
