"""``skdr-eval`` CLI (Typer) — implements #89.

Subcommands
-----------
- ``evaluate``        — run :func:`skdr_eval.evaluate_sklearn_models`
- ``pairwise``        — run :func:`skdr_eval.evaluate_pairwise_models`
- ``card``            — re-render the YAML card from a saved artifact JSON
- ``validate-schema`` — call ``validate_logs`` / ``validate_pairwise_inputs``
- ``doctor``          — run :func:`skdr_eval.doctor`
- ``version``         — print the package version

Exit codes
----------
* ``0`` — success.
* ``1`` — data / schema error (the doctor or a validator flagged ``fail``).
* ``2`` — environment / import error (a required optional dep is missing).
* ``3`` — at least one card's recommendation verdict was ``do_not_deploy``;
  use this exit code as a CI gate.

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
from skdr_eval.doctor import doctor as doctor_fn
from skdr_eval.trackers import FileTracker

logger = logging.getLogger("skdr_eval.cli")

EXIT_OK = 0
EXIT_DATA = 1
EXIT_ENV = 2
EXIT_DO_NOT_DEPLOY = 3

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
    - Object columns whose first cell parses as a datetime are coerced via
      :func:`pandas.to_datetime` so the design builder accepts them.
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
    """Load a pickled / joblib model. Refuse non-local schemes."""
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
            yaml_path = cards_dir / f"{model_name}_{estimator}.card.yaml"
            json_path_card = cards_dir / f"{model_name}_{estimator}.card.json"
            card.to_yaml(yaml_path)
            card.to_json(json_path_card)
            paths[yaml_path.name] = yaml_path
            paths[json_path_card.name] = json_path_card

    return paths


def _verdict_exit_code(artifact: skdr_eval.EvaluationArtifact) -> int:
    """Return ``EXIT_DO_NOT_DEPLOY`` if any model's recommendation is ``do_not_deploy``."""
    for model_name in artifact.detailed:
        for estimator in ("SNDR", "DR"):
            if estimator not in artifact.detailed[model_name]:
                continue
            try:
                rec = artifact.recommendation(model_name, estimator=estimator)
            except Exception:
                continue
            if rec.verdict == "do_not_deploy":
                return EXIT_DO_NOT_DEPLOY
    return EXIT_OK


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
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        report.print(color=sys.stdout.isatty())
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
    tracker_dir: Path | None = typer.Option(
        None, "--tracker-dir", help="Optional FileTracker output directory."
    ),
) -> None:
    """Run :func:`evaluate_sklearn_models` on local logs + pickled models."""
    logs_df = _load_dataframe(logs)
    models = _parse_model_specs(model)
    tracker = FileTracker(tracker_dir) if tracker_dir is not None else None
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
    paths = _write_artifact_outputs(artifact, out)
    typer.echo(f"Wrote {len(paths)} files to {out}.")
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
    paths = _write_artifact_outputs(artifact, out)
    typer.echo(f"Wrote {len(paths)} files to {out}.")
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
        _coerce_optional_float,
    )

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
        V_hat=_coerce_optional_float(row.V_hat),
        ci_lower=_coerce_optional_float(row.ci_lower),
        ci_upper=_coerce_optional_float(row.ci_upper),
        ci_alpha=_coerce_optional_float(schema.metadata.get("alpha")),
        baseline=None,
        delta_vs_baseline=None,
        clip=_coerce_optional_float(row.clip),
    )
    trust = TrustBlock(
        support_health=row.support_health,
        warning_codes=list(warn.warning_codes) if warn is not None else [],
        recommendation=None,
        primary_blocker=None,
    )
    n = int(schema.metadata.get("n_samples", 0)) or None
    ess_val = _coerce_optional_float(row.ESS)
    diagnostics_block = DiagnosticsBlock(
        ESS=ess_val,
        ess_frac=(ess_val / n) if (ess_val is not None and n) else None,
        match_rate=_coerce_optional_float(row.match_rate),
        pareto_k=_coerce_optional_float(row.pareto_k),
        min_pscore=_coerce_optional_float(row.min_pscore),
        tail_mass=_coerce_optional_float(row.tail_mass),
        ece=_coerce_optional_float(diag_payload.ece) if diag_payload else None,
        brier_score=(
            _coerce_optional_float(diag_payload.brier_score) if diag_payload else None
        ),
        gate=None,
    )
    sensitivity_block = (
        SensitivityBlock()
        if sens is None
        else SensitivityBlock(
            V_min=_coerce_optional_float(sens.V_min),
            V_max=_coerce_optional_float(sens.V_max),
            V_range=_coerce_optional_float(sens.V_range),
            chosen_clip=_coerce_optional_float(sens.chosen_clip),
            chosen_V=_coerce_optional_float(sens.chosen_V),
            argmin_MSE_clip=_coerce_optional_float(sens.argmin_MSE_clip),
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


if __name__ == "__main__":  # pragma: no cover
    app()
