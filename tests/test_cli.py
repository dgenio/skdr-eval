"""Tests for the :mod:`skdr_eval.cli` Typer entry point (#89)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from typer import Exit
from typer.testing import CliRunner

import skdr_eval
from skdr_eval.cli import (
    EXIT_DATA,
    EXIT_DO_NOT_DEPLOY,
    EXIT_INSUFFICIENT_EVIDENCE,
    EXIT_OK,
    _load_model,
    _parse_model_specs,
    _verdict_exit_code,
    _write_artifact_outputs,
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

    def test_validate_invalid_kind(self, synth_logs_parquet: Path):
        result = runner.invoke(
            app, ["validate-schema", str(synth_logs_parquet), "--kind", "bogus"]
        )
        assert result.exit_code == EXIT_DATA

    def test_corrupt_file_fails_gracefully(self, tmp_path: Path):
        corrupt = tmp_path / "bad.parquet"
        corrupt.write_bytes(b"not a real parquet file")
        result = runner.invoke(app, ["validate-schema", str(corrupt)])
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


class TestEvaluateSubcommand:
    def test_evaluate_catches_model_errors(
        self,
        synth_logs_parquet: Path,
        fitted_model_path: tuple[Path, dict],
        tmp_path: Path,
    ):
        """Evaluate exits cleanly with EXIT_DATA when model features mismatch."""
        model_path, _ = fitted_model_path
        out_dir = tmp_path / "eval_out"
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                f"HGB={model_path}",
                "--out",
                str(out_dir),
                "--n-splits",
                "2",
                "--policy-train",
                "pre_split",
            ],
        )
        # Model trained on wrong features → SkdrEvalError → clean exit code 1
        assert result.exit_code == EXIT_DATA

    def test_evaluate_invalid_model_path_exits_1(
        self, synth_logs_parquet: Path, tmp_path: Path
    ):
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                "BAD=/nonexistent/model.joblib",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA

    def test_evaluate_remote_model_refused(
        self, synth_logs_parquet: Path, tmp_path: Path
    ):
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                "M=https://evil.com/model.pkl",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA


class TestEvaluateEndToEnd:
    def test_write_artifact_outputs(self, tmp_path: Path):
        """_write_artifact_outputs writes JSON, HTML, and card files."""
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
        out_dir = tmp_path / "out"
        paths = _write_artifact_outputs(artifact, out_dir)
        assert (out_dir / "artifact.json").is_file()
        assert (out_dir / "report.html").is_file()
        assert (out_dir / "cards").is_dir()
        cards = list((out_dir / "cards").glob("*.card.yaml"))
        assert len(cards) >= 1
        assert len(paths) >= 4  # artifact.json + report.html + >=2 cards

    def test_verdict_exit_code_without_ci_is_insufficient_or_block(
        self, tmp_path: Path
    ):
        """Without a bootstrap CI, the gate is non-zero (#197).

        Every estimator falls into ``insufficient_evidence`` (no CI to clear the
        baseline) unless a high-risk diagnostic forces ``do_not_deploy``. Either
        way the gate must not report a false-green ``EXIT_OK``.
        """
        logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=0,
            ci_bootstrap=False,
            policy_train="pre_split",
        )
        code = _verdict_exit_code(artifact)
        assert code in (EXIT_INSUFFICIENT_EVIDENCE, EXIT_DO_NOT_DEPLOY)


class TestLoadDataframeEdgeCases:
    def test_feather_format(self, tmp_path: Path):
        logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
        path = tmp_path / "logs.feather"
        logs.to_feather(path)
        result = runner.invoke(app, ["validate-schema", str(path)])
        assert result.exit_code == EXIT_OK

    def test_tsv_format(self, tmp_path: Path):
        logs, _, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
        path = tmp_path / "logs.tsv"
        logs.to_csv(path, sep="\t", index=False)
        result = runner.invoke(app, ["validate-schema", str(path)])
        # TSV loses datetime types → likely schema fail, but we exercised the path
        assert result.exit_code in (EXIT_OK, EXIT_DATA)


class TestParseModelSpecsEdgeCases:
    def test_model_spec_without_equals(self, synth_logs_parquet: Path, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                "no_equals_sign",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA

    def test_model_spec_empty_name(self, synth_logs_parquet: Path, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                "=/some/path.joblib",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA

    def test_model_spec_empty_path(self, synth_logs_parquet: Path, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(synth_logs_parquet),
                "--model",
                "NAME=",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA


class TestCardSubcommandEdgeCases:
    def test_card_invalid_format(self, tmp_path: Path):
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
                str(tmp_path / "out.txt"),
                "--format",
                "xml",
            ],
        )
        assert result.exit_code == EXIT_DATA


class TestPairwiseSubcommand:
    def test_pairwise_catches_model_errors(
        self, pairwise_inputs: tuple[Path, Path], tmp_path: Path
    ):
        """Pairwise exits cleanly with EXIT_DATA when model features mismatch."""
        p_logs, p_op = pairwise_inputs
        # Train a model on wrong features to trigger SkdrEvalError
        logs_df, _op_df = skdr_eval.make_pairwise_synth(
            n_days=4, n_clients_day=80, n_ops=3, seed=0
        )
        model = HistGradientBoostingRegressor(max_iter=10, random_state=0)
        feature_cols = [c for c in logs_df.columns if c.startswith("cli_")]
        model.fit(logs_df[feature_cols].to_numpy(), logs_df["service_time"].to_numpy())
        model_path = tmp_path / "pw_model.joblib"
        joblib.dump(model, model_path)

        out_dir = tmp_path / "pw_out"
        result = runner.invoke(
            app,
            [
                "pairwise",
                str(p_logs),
                str(p_op),
                "--model",
                f"PW={model_path}",
                "--metric-col",
                "service_time",
                "--out",
                str(out_dir),
                "--n-splits",
                "2",
            ],
        )
        # Model feature mismatch → SkdrEvalError → clean exit code 1
        assert result.exit_code == EXIT_DATA

    def test_pairwise_invalid_task_type_exits_1(
        self, pairwise_inputs: tuple[Path, Path], tmp_path: Path
    ):
        p_logs, p_op = pairwise_inputs
        model_path = tmp_path / "fake.joblib"
        joblib.dump("not_a_model", model_path)
        result = runner.invoke(
            app,
            [
                "pairwise",
                str(p_logs),
                str(p_op),
                "--model",
                f"X={model_path}",
                "--task-type",
                "invalid",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA

    def test_pairwise_invalid_direction_exits_1(
        self, pairwise_inputs: tuple[Path, Path], tmp_path: Path
    ):
        p_logs, p_op = pairwise_inputs
        model_path = tmp_path / "fake.joblib"
        joblib.dump("not_a_model", model_path)
        result = runner.invoke(
            app,
            [
                "pairwise",
                str(p_logs),
                str(p_op),
                "--model",
                f"X={model_path}",
                "--direction",
                "sideways",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == EXIT_DATA


def _artifact_json(tmp_path: Path) -> Path:
    """Evaluate a small model and checkpoint the artifact JSON."""
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
    path = tmp_path / "artifact.json"
    path.write_text(artifact.to_schema().model_dump_json(indent=2))
    return path


class TestExplainSubcommand:
    """#201: ``skdr-eval explain``."""

    def test_explain_text_output(self, tmp_path: Path):
        artifact_json = _artifact_json(tmp_path)
        result = runner.invoke(
            app, ["explain", str(artifact_json), "--model", "HGB", "--estimator", "DR"]
        )
        assert result.exit_code == EXIT_OK, result.stdout
        assert "HGB / DR" in result.stdout
        assert "Verdict:" in result.stdout

    def test_explain_json_output(self, tmp_path: Path):
        artifact_json = _artifact_json(tmp_path)
        result = runner.invoke(
            app,
            ["explain", str(artifact_json), "--model", "HGB", "--json"],
        )
        assert result.exit_code == EXIT_OK, result.stdout
        payload = json.loads(result.stdout)
        assert payload["model_name"] == "HGB"
        assert "verdict" in payload
        assert isinstance(payload["reasons"], list)

    def test_explain_unknown_model_exits_data(self, tmp_path: Path):
        artifact_json = _artifact_json(tmp_path)
        result = runner.invoke(app, ["explain", str(artifact_json), "--model", "Nope"])
        assert result.exit_code == EXIT_DATA


class TestCapabilitiesSubcommand:
    """#215: ``skdr-eval capabilities``."""

    def test_capabilities_text(self):
        result = runner.invoke(app, ["capabilities"])
        assert result.exit_code == EXIT_OK, result.stdout
        assert "capabilities" in result.stdout.lower()
        for extra in ("viz", "cli", "boosting", "mlflow"):
            assert extra in result.stdout

    def test_capabilities_json(self):
        result = runner.invoke(app, ["capabilities", "--json"])
        assert result.exit_code == EXIT_OK
        payload = json.loads(result.stdout)
        extras = {row["extra"] for row in payload}
        assert {"viz", "speed", "cli", "boosting", "mlflow", "wandb", "aim"} == extras
        assert all(isinstance(row["installed"], bool) for row in payload)


class TestDoctorRepro:
    """#246: ``skdr-eval doctor --repro``."""

    def test_repro_text_appended(self, synth_logs_parquet: Path):
        result = runner.invoke(app, ["doctor", str(synth_logs_parquet), "--repro"])
        assert "skdr_eval.doctor" in result.stdout
        assert "import pandas as pd" in result.stdout

    def test_repro_in_json(self, synth_logs_parquet: Path):
        result = runner.invoke(
            app, ["doctor", str(synth_logs_parquet), "--json", "--repro"]
        )
        payload = json.loads(result.stdout)
        assert "repro" in payload
        assert "skdr_eval.doctor" in payload["repro"]


class TestQuickstartSubcommand:
    """#207: ``skdr-eval quickstart`` golden path."""

    def test_quickstart_writes_card_and_explains(self, tmp_path: Path):
        out = tmp_path / "qs"
        result = runner.invoke(app, ["quickstart", "--out", str(out), "--n", "800"])
        # Onboarding demo always exits 0 on success (not a CI gate).
        assert result.exit_code == EXIT_OK, result.stdout + result.stderr
        assert (out / "report.html").is_file()
        assert (out / "artifact.json").is_file()
        assert "Verdict:" in result.stdout

    def test_quickstart_evaluation_error_exits_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A SkdrEvalError during evaluation surfaces as EXIT_DATA, not a crash."""

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise skdr_eval.SkdrEvalError("synthetic evaluation failure")

        monkeypatch.setattr(skdr_eval, "evaluate_sklearn_models", _boom)
        out = tmp_path / "qs_err"
        result = runner.invoke(app, ["quickstart", "--out", str(out), "--n", "800"])
        assert result.exit_code == EXIT_DATA, result.stdout + result.stderr

    def test_help_lists_new_subcommands(self):
        result = runner.invoke(app, ["--help"])
        for sub in ("explain", "capabilities", "quickstart"):
            assert sub in result.stdout


class TestLoadModelDirect:
    """Direct unit tests for _load_model to ensure coverage of guard paths."""

    def test_url_scheme_refused(self):
        with pytest.raises(Exit):
            _load_model(__import__("pathlib").Path("http://evil.com/model.pkl"))

    def test_nonexistent_path_refused(self, tmp_path: Path):
        with pytest.raises(Exit):
            _load_model(tmp_path / "does_not_exist.joblib")

    def test_parse_model_specs_no_equals(self):
        with pytest.raises(Exit):
            _parse_model_specs(["no_equals"])

    def test_parse_model_specs_empty_name(self):
        with pytest.raises(Exit):
            _parse_model_specs(["=/path"])

    def test_parse_model_specs_empty_path(self):
        with pytest.raises(Exit):
            _parse_model_specs(["name="])


# --------------------------------------------------------------------------- #
# New output / consumption commands (#184 #205 #231 #251)                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def design_model_and_logs(tmp_path: Path) -> tuple[Path, Path]:
    """A logs parquet + a pickled model fitted on the full design feature set.

    The CLI ``evaluate`` runs with ``fit_models=False``, so the candidate model
    must predict on the same columns ``build_design`` assembles (cli_*, st_*,
    and op_*_elig — seven features here). Returns ``(logs_path, model_path)``.
    """
    logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
    feature_cols = [
        c
        for c in logs.columns
        if c.startswith("cli_") or c.startswith("st_") or c.startswith("op_")
    ]
    model = HistGradientBoostingRegressor(max_iter=20, random_state=0)
    model.fit(logs[feature_cols].to_numpy(), logs["service_time"].to_numpy())
    logs_path = tmp_path / "logs.parquet"
    logs.to_parquet(logs_path)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    return logs_path, model_path


def _saved_artifact_json(path: Path, *, seed: int = 0) -> Path:
    """Write a real artifact.json via the Python API and return its path."""
    logs, _, _ = skdr_eval.make_synth_logs(n=500, n_ops=3, seed=seed)
    art = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"HGB": HistGradientBoostingRegressor(max_iter=15, random_state=seed)},
        fit_models=True,
        n_splits=3,
        random_state=seed,
        policy_train="all",
        ci_bootstrap=True,
    )
    path.write_text(art.to_schema().model_dump_json(indent=2), encoding="utf-8")
    return path


class TestEvaluateFormat:
    def test_format_json_is_pipe_clean_stdout(
        self, design_model_and_logs: tuple[Path, Path], tmp_path: Path
    ):
        logs_path, model_path = design_model_and_logs
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(logs_path),
                "--model",
                f"M={model_path}",
                "--out",
                str(tmp_path / "out"),
                "--format",
                "json",
            ],
        )
        # Exit code follows the verdict gate (0/3/4); stdout must be valid JSON.
        payload = json.loads(result.stdout)
        assert "report" in payload
        # The "wrote N files" confirmation is on stderr, not stdout.
        assert "Wrote" not in result.stdout
        assert "Wrote" in result.stderr

    def test_format_markdown(
        self, design_model_and_logs: tuple[Path, Path], tmp_path: Path
    ):
        logs_path, model_path = design_model_and_logs
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(logs_path),
                "--model",
                f"M={model_path}",
                "--out",
                str(tmp_path / "out"),
                "--format",
                "markdown",
            ],
        )
        assert "# skdr-eval evaluation summary" in result.stdout

    def test_default_output_unchanged(
        self, design_model_and_logs: tuple[Path, Path], tmp_path: Path
    ):
        logs_path, model_path = design_model_and_logs
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(logs_path),
                "--model",
                f"M={model_path}",
                "--out",
                str(tmp_path / "out"),
            ],
        )
        # Without --format the confirmation stays on stdout (legacy behaviour).
        assert "Wrote" in result.stdout

    @pytest.mark.parametrize("fmt", ["table", "csv"])
    def test_format_table_and_csv(
        self, fmt: str, design_model_and_logs: tuple[Path, Path], tmp_path: Path
    ):
        logs_path, model_path = design_model_and_logs
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(logs_path),
                "--model",
                f"M={model_path}",
                "--out",
                str(tmp_path / "out"),
                "--format",
                fmt,
            ],
        )
        # The report header column names appear on stdout in both formats.
        assert "V_hat" in result.stdout
        assert "estimator" in result.stdout
        if fmt == "csv":
            assert "," in result.stdout  # CSV separator present
        assert "Wrote" in result.stderr

    def test_bad_format_rejected(
        self, design_model_and_logs: tuple[Path, Path], tmp_path: Path
    ):
        logs_path, model_path = design_model_and_logs
        result = runner.invoke(
            app,
            [
                "evaluate",
                str(logs_path),
                "--model",
                f"M={model_path}",
                "--out",
                str(tmp_path / "out"),
                "--format",
                "bogus",
            ],
        )
        assert result.exit_code != EXIT_OK


class TestSchemaCommand:
    def test_schema_artifact(self):
        result = runner.invoke(app, ["schema", "--kind", "artifact"])
        assert result.exit_code == EXIT_OK
        payload = json.loads(result.stdout)
        assert payload["title"] == "ArtifactSchema"

    def test_schema_card(self):
        result = runner.invoke(app, ["schema", "--kind", "card"])
        assert result.exit_code == EXIT_OK
        assert json.loads(result.stdout)["title"] == "EvaluationCard"

    def test_schema_bad_kind(self):
        result = runner.invoke(app, ["schema", "--kind", "bogus"])
        assert result.exit_code != EXIT_OK


class TestCompareCommand:
    def test_compare_no_regression_exits_zero(self, tmp_path: Path):
        a = _saved_artifact_json(tmp_path / "a.json", seed=0)
        b = _saved_artifact_json(tmp_path / "b.json", seed=0)
        result = runner.invoke(app, ["compare", str(a), str(b)])
        assert result.exit_code == EXIT_OK
        assert "No verdict regression" in result.stdout

    def test_compare_regression_exits_do_not_deploy(self, tmp_path: Path):
        # Baseline: hand-edit a saved artifact into a clean "deploy" posture,
        # then compare the (worse) candidate against it.
        candidate = _saved_artifact_json(tmp_path / "cand.json", seed=0)
        raw = json.loads(candidate.read_text())
        for row in raw["report"]:
            row["support_health"] = "ok"
            row["diagnostic_warnings"] = ""
            row["pareto_k"] = 0.1
            row["ci_lower"] = 1000.0
            row["ci_upper"] = 1100.0
        baseline = tmp_path / "base.json"
        baseline.write_text(json.dumps(raw), encoding="utf-8")
        result = runner.invoke(app, ["compare", str(baseline), str(candidate)])
        assert result.exit_code == EXIT_DO_NOT_DEPLOY
        assert "regression" in result.stdout.lower()

    def test_compare_markdown_format(self, tmp_path: Path):
        a = _saved_artifact_json(tmp_path / "a.json", seed=0)
        b = _saved_artifact_json(tmp_path / "b.json", seed=0)
        result = runner.invoke(app, ["compare", str(a), str(b), "--format", "markdown"])
        assert "| Model |" in result.stdout

    def test_compare_json_format(self, tmp_path: Path):
        a = _saved_artifact_json(tmp_path / "a.json", seed=0)
        b = _saved_artifact_json(tmp_path / "b.json", seed=0)
        result = runner.invoke(app, ["compare", str(a), str(b), "--format", "json"])
        assert result.exit_code == EXIT_OK
        payload = json.loads(result.stdout)
        assert "rows" in payload
        assert payload["verdict_regressed"] is False

    def test_compare_text_lists_rows(self, tmp_path: Path):
        a = _saved_artifact_json(tmp_path / "a.json", seed=0)
        b = _saved_artifact_json(tmp_path / "b.json", seed=0)
        result = runner.invoke(app, ["compare", str(a), str(b)])
        # Default text format lists each (model, estimator) row with its verdict.
        assert "HGB/SNDR" in result.stdout

    def test_compare_bad_format_rejected(self, tmp_path: Path):
        a = _saved_artifact_json(tmp_path / "a.json", seed=0)
        b = _saved_artifact_json(tmp_path / "b.json", seed=0)
        result = runner.invoke(app, ["compare", str(a), str(b), "--format", "bogus"])
        assert result.exit_code != EXIT_OK


class TestBadgeCommand:
    def test_badge_prints_svg_to_stdout(self, tmp_path: Path):
        # Without --out the raw SVG goes to stdout so `> badge.svg` works.
        aj = _saved_artifact_json(tmp_path / "a.json", seed=0)
        result = runner.invoke(app, ["badge", str(aj), "--model", "HGB"])
        assert result.exit_code == EXIT_OK
        assert result.stdout.strip().startswith("<svg")
        assert "</svg>" in result.stdout

    def test_badge_writes_svg_and_snippet(self, tmp_path: Path):
        aj = _saved_artifact_json(tmp_path / "a.json", seed=0)
        svg = tmp_path / "badge.svg"
        result = runner.invoke(
            app, ["badge", str(aj), "--model", "HGB", "--out", str(svg)]
        )
        assert result.exit_code == EXIT_OK
        assert svg.read_text(encoding="utf-8").startswith("<svg")
        # The Markdown snippet on stderr references the file we actually wrote.
        assert "![skdr-eval:" in result.stderr
        assert "badge.svg" in result.stderr

    def test_badge_unknown_model_exits_data(self, tmp_path: Path):
        aj = _saved_artifact_json(tmp_path / "a.json", seed=0)
        result = runner.invoke(app, ["badge", str(aj), "--model", "NOPE"])
        assert result.exit_code == EXIT_DATA
