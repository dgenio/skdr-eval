"""Regression tests pinning the documented CLI exit codes (#245).

The README and the ``skdr_eval.cli`` docstring advertise a stable exit-code
contract that users automate against in CI:

* ``0`` — success
* ``1`` — data / schema error
* ``2`` — environment / import error
* ``3`` — ``do_not_deploy`` verdict (CI gate)
* ``4`` — ``insufficient_evidence`` verdict

Each code is asserted against the condition that should trigger it, so the
contract cannot silently break.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import typer
from typer.testing import CliRunner

import skdr_eval
from skdr_eval.cli import (
    EXIT_DATA,
    EXIT_DO_NOT_DEPLOY,
    EXIT_ENV,
    EXIT_INSUFFICIENT_EVIDENCE,
    EXIT_OK,
    _load_model,
    app,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_exit_code_constants_match_contract() -> None:
    """The numeric values are part of the public CI contract."""
    assert (
        EXIT_OK,
        EXIT_DATA,
        EXIT_ENV,
        EXIT_DO_NOT_DEPLOY,
        EXIT_INSUFFICIENT_EVIDENCE,
    ) == (
        0,
        1,
        2,
        3,
        4,
    )


def test_exit_0_on_successful_validate(tmp_path: Path) -> None:
    """A clean validate-schema run exits 0."""
    logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=0)
    path = tmp_path / "logs.parquet"
    logs.to_parquet(path)
    result = runner.invoke(app, ["validate-schema", str(path)])
    assert result.exit_code == EXIT_OK, result.stdout


def test_exit_1_on_schema_error(tmp_path: Path) -> None:
    """A malformed-schema file fails the validator with exit 1."""
    import pandas as pd  # noqa: PLC0415

    bad = pd.DataFrame({"not_a_known_column": [1, 2, 3]})
    path = tmp_path / "bad.parquet"
    bad.to_parquet(path)
    result = runner.invoke(app, ["validate-schema", str(path)])
    assert result.exit_code == EXIT_DATA, result.stdout


def test_exit_1_on_unknown_file_format(tmp_path: Path) -> None:
    path = tmp_path / "logs.weird"
    path.write_text("nonsense", encoding="utf-8")
    result = runner.invoke(app, ["validate-schema", str(path)])
    assert result.exit_code == EXIT_DATA, result.stdout


def test_exit_2_on_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_load_model`` exits 2 when the joblib import fails (#196 env path)."""
    # Setting the entry to ``None`` makes ``import joblib`` raise ImportError.
    monkeypatch.setitem(sys.modules, "joblib", None)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"unused")
    with pytest.raises(typer.Exit) as excinfo:
        _load_model(model_path)
    assert excinfo.value.exit_code == EXIT_ENV


class _Rec:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict


class _SingleVerdictArtifact:
    """Duck-typed artifact with one estimator returning a fixed verdict."""

    def __init__(self, verdict: str) -> None:
        self.detailed = {"m": {"DR": object()}}
        self._verdict = verdict

    def recommendation(self, model_name: str, *, estimator: str = "SNDR") -> _Rec:
        return _Rec(self._verdict)


def test_exit_3_on_do_not_deploy() -> None:
    """A ``do_not_deploy`` verdict trips the CI gate with exit 3."""
    from skdr_eval import cli  # noqa: PLC0415

    assert (
        cli._verdict_exit_code(_SingleVerdictArtifact("do_not_deploy"))
        == EXIT_DO_NOT_DEPLOY
    )


def test_exit_4_on_insufficient_evidence() -> None:
    """An ``insufficient_evidence`` verdict (no block) exits 4."""
    from skdr_eval import cli  # noqa: PLC0415

    assert (
        cli._verdict_exit_code(_SingleVerdictArtifact("insufficient_evidence"))
        == EXIT_INSUFFICIENT_EVIDENCE
    )
