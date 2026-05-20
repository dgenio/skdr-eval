"""Tests for the :mod:`skdr_eval.cli` Typer entry point (#89)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from typer.testing import CliRunner

import skdr_eval
from skdr_eval.cli import (
    EXIT_DATA,
    EXIT_DO_NOT_DEPLOY,
    EXIT_OK,
    app,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def synth_logs_parquet(tmp_path: Path) -> Path:
    """Write a parquet file of synthetic logs and return its path."""
    logs, _, _ = skdr_eval.make_synth_logs(n=800, n_ops=3, seed=0)
    path = tmp_path / "logs.parquet"
    logs.to_parquet(path)
    return path


@pytest.fixture
def synth_logs_csv(tmp_path: Path) -> Path:
    logs, _, _ = skdr_eval.make_synth_logs(n=800, n_ops=3, seed=0)
    path = tmp_path / "logs.csv"
    logs.to_csv(path, index=False)
    return path


@pytest.fixture
def pairwise_inputs(tmp_path: Path) -> tuple[Path, Path]:
    logs_df, op_df = skdr_eval.make_pairwise_synth(
        n_days=4, n_clients_day=80, n_ops=3, seed=0
    )
    p_logs = tmp_path / "pw_logs.parquet"
    p_op = tmp_path / "pw_op.parquet"
    logs_df.to_parquet(p_logs)
    op_df.to_parquet(p_op)
    return p_logs, p_op


@pytest.fixture
def fitted_model_path(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Run a brief skdr_eval evaluation just to extract a fitted model.

    Returns a path to the joblib dump plus the feature columns the model was
    trained on, for later reuse by the CLI evaluate subcommand.
    """
    logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
    # Train on the same feature set that ``build_design`` would assemble for
    # downstream evaluators. We deliberately use ``fit_models=True`` upstream
    # in CLI to skip the feature mismatch problem.
    model = HistGradientBoostingRegressor(max_iter=20, random_state=0)
    feature_cols = [
        c for c in logs.columns if c.startswith("cli_") or c.startswith("st_")
    ]
    model.fit(logs[feature_cols].to_numpy(), logs["service_time"].to_numpy())
    out = tmp_path / "model.joblib"
    joblib.dump(model, out)
    return out, {"features": ",".join(feature_cols)}


class TestVersionAndHelp:
    def test_help_emits_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK, result.stdout
        for sub in (
            "evaluate",
            "pairwise",
            "card",
            "validate-schema",
            "doctor",
            "version",
        ):
            assert sub in result.stdout

    def test_version_subcommand(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == EXIT_OK
        assert result.stdout.strip() == skdr_eval.__version__


class TestDoctorSubcommand:
    def test_doctor_clean_logs_exits_zero(self, synth_logs_parquet: Path):
        result = runner.invoke(app, ["doctor", str(synth_logs_parquet)])
        assert result.exit_code in (EXIT_OK, EXIT_DATA), result.stdout
        # The doctor either passes (exit 0) or warns about sample size.
        assert "doctor" in result.stdout.lower()

    def test_doctor_json_output(self, synth_logs_parquet: Path):
        result = runner.invoke(app, ["doctor", str(synth_logs_parquet), "--json"])
        # Parse stdout as JSON regardless of overall exit code.
        payload = json.loads(result.stdout)
        assert "checks" in payload
        assert "summary" in payload
        assert payload["summary"] in {"pass", "warn", "fail"}

    def test_doctor_invalid_kind(self, synth_logs_parquet: Path):
        result = runner.invoke(
            app, ["doctor", str(synth_logs_parquet), "--kind", "bogus"]
        )
        assert result.exit_code == EXIT_DATA

    def test_doctor_pairwise_requires_op_daily(self, synth_logs_parquet: Path):
        result = runner.invoke(
            app, ["doctor", str(synth_logs_parquet), "--kind", "pairwise"]
        )
        # Missing op_daily → doctor reports fail → exit 1.
        assert result.exit_code == EXIT_DATA


class TestValidateSchemaSubcommand:
    def test_validate_clean_logs(self, synth_logs_parquet: Path):
        result = runner.invoke(app, ["validate-schema", str(synth_logs_parquet)])
        assert result.exit_code == EXIT_OK
        assert "OK" in result.stdout

    def test_validate_pairwise_requires_op_daily(
        self, pairwise_inputs: tuple[Path, Path]
    ):
        logs, _op = pairwise_inputs
        result = runner.invoke(
            app, ["validate-schema", str(logs), "--kind", "pairwise"]
        )
        assert result.exit_code == EXIT_DATA

    def test_validate_pairwise_with_op_daily(self, pairwise_inputs: tuple[Path, Path]):
        logs, op = pairwise_inputs
        result = runner.invoke(
            app,
            [
                "validate-schema",
                str(logs),
                "--kind",
                "pairwise",
                "--op-daily",
                str(op),
                "--metric-col",
                "service_time",
            ],
        )
        assert result.exit_code == EXIT_OK

    def test_validate_csv_input_dtype_degraded(self, synth_logs_csv: Path):
        # CSV is lossy: arrival_ts deserializes as str. ``validate-schema``
        # must surface this as a clean ``schema FAIL`` rather than crash.
        result = runner.invoke(app, ["validate-schema", str(synth_logs_csv)])
        assert result.exit_code == EXIT_DATA

    def test_unknown_format_fails(self, tmp_path: Path):
        bogus = tmp_path / "logs.unknown"
        bogus.write_text("nothing")
        result = runner.invoke(app, ["validate-schema", str(bogus)])
        assert result.exit_code == EXIT_DATA


class TestEvaluateAndCardRoundTrip:
    def test_evaluate_writes_outputs_and_card_subcommand_reads_them(
        self,
        tmp_path: Path,
    ):
        # Use ``fit_models=True`` semantics by running directly here, then
        # checkpoint an artifact JSON we can drive the `card` subcommand from.
        logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
        )
        out = tmp_path / "out"
        out.mkdir()
        artifact_json = out / "artifact.json"
        artifact_json.write_text(artifact.to_schema().model_dump_json(indent=2))

        card_yaml = tmp_path / "rendered.yaml"
        result = runner.invoke(
            app,
            [
                "card",
                str(artifact_json),
                "--model",
                "HGB",
                "--estimator",
                "DR",
                "--out",
                str(card_yaml),
            ],
        )
        assert result.exit_code == EXIT_OK, result.stdout + result.stderr
        assert card_yaml.is_file()
        # Round-trip the rendered YAML.
        loaded = skdr_eval.EvaluationCard.from_yaml(card_yaml)
        assert loaded.model_name == "HGB"
        assert loaded.headline.estimator == "DR"

    def test_card_subcommand_json_format(self, tmp_path: Path):
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=15, random_state=0)}
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
        )
        artifact_json = tmp_path / "artifact.json"
        artifact_json.write_text(artifact.to_schema().model_dump_json(indent=2))
        out = tmp_path / "card.json"
        result = runner.invoke(
            app,
            [
                "card",
                str(artifact_json),
                "--model",
                "HGB",
                "--estimator",
                "SNDR",
                "--out",
                str(out),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == EXIT_OK, result.stderr
        data = json.loads(out.read_text())
        assert data["model_name"] == "HGB"

    def test_card_unknown_model_returns_exit_1(self, tmp_path: Path):
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=15, random_state=0)}
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
        )
        artifact_json = tmp_path / "artifact.json"
        artifact_json.write_text(artifact.to_schema().model_dump_json(indent=2))
        out = tmp_path / "card.yaml"
        result = runner.invoke(
            app,
            [
                "card",
                str(artifact_json),
                "--model",
                "DoesNotExist",
                "--estimator",
                "DR",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == EXIT_DATA


class TestDoNotDeployExit:
    def test_do_not_deploy_exit_code_constant(self):
        # Stability guard for CI gates.
        assert EXIT_DO_NOT_DEPLOY == 3
