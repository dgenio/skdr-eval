"""Tests for Polars / PyArrow input coercion and accessors (#72)."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval._frames import coerce_to_pandas
from skdr_eval.exceptions import DataValidationError, OptionalDependencyError

_HAS_POLARS = importlib.util.find_spec("polars") is not None
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None

requires_polars = pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")
requires_pyarrow = pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")


def _evaluate(logs: object) -> skdr_eval.EvaluationArtifact:
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
    )


class TestCoerceToPandas:
    def test_pandas_passthrough_is_identity(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert coerce_to_pandas(df, name="logs") is df

    def test_unsupported_type_raises_with_param_name(self) -> None:
        with pytest.raises(DataValidationError, match="logs_df must be a pandas"):
            coerce_to_pandas([1, 2, 3], name="logs_df")

    @requires_polars
    def test_polars_frame_converted(self) -> None:
        import polars as pl  # noqa: PLC0415

        out = coerce_to_pandas(pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert out["a"].tolist() == [1, 2]

    @requires_pyarrow
    def test_arrow_table_converted(self) -> None:
        import pyarrow as pa  # noqa: PLC0415

        out = coerce_to_pandas(pa.table({"a": [1, 2], "b": [3.0, 4.0]}))
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert out["a"].tolist() == [1, 2]


@requires_polars
class TestPolarsInputEquivalence:
    def test_sklearn_polars_matches_pandas(self) -> None:
        import polars as pl  # noqa: PLC0415

        logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=1)
        art_pd = _evaluate(logs)
        art_pl = _evaluate(pl.from_pandas(logs))
        # Identical V_hat to numerical precision: conversion must not perturb
        # the evaluation.
        pd.testing.assert_series_equal(
            art_pd.report["V_hat"].reset_index(drop=True),
            art_pl.report["V_hat"].reset_index(drop=True),
        )


@requires_pyarrow
class TestArrowInputEquivalence:
    def test_sklearn_arrow_matches_pandas(self) -> None:
        import pyarrow as pa  # noqa: PLC0415

        logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=2)
        art_pd = _evaluate(logs)
        art_pa = _evaluate(pa.Table.from_pandas(logs))
        pd.testing.assert_series_equal(
            art_pd.report["V_hat"].reset_index(drop=True),
            art_pa.report["V_hat"].reset_index(drop=True),
        )


@requires_polars
def test_pairwise_polars_inputs() -> None:
    import polars as pl  # noqa: PLC0415

    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=120, n_ops=4, seed=3
    )
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
    art = skdr_eval.evaluate_pairwise_models(
        logs_df=pl.from_pandas(logs_df),
        op_daily_df=pl.from_pandas(op_daily_df),
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=3,
        fit_models=True,
        policy_train="pre_split",
        random_state=0,
    )
    assert not art.report.empty
    assert "V_hat" in art.report.columns


class TestArtifactAccessors:
    @requires_polars
    def test_to_polars_roundtrips_report(self) -> None:
        import polars as pl  # noqa: PLC0415

        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=4)
        art = _evaluate(logs)
        out = art.to_polars()
        assert isinstance(out, pl.DataFrame)
        assert out.height == len(art.report)
        assert set(out.columns) == set(art.report.columns)

    @requires_pyarrow
    def test_to_arrow_roundtrips_report(self) -> None:
        import pyarrow as pa  # noqa: PLC0415

        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=5)
        art = _evaluate(logs)
        out = art.to_arrow()
        assert isinstance(out, pa.Table)
        assert out.num_rows == len(art.report)

    def test_to_polars_without_polars_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logs, _, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=6)
        art = _evaluate(logs)
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "polars":
                raise ImportError("no polars")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(OptionalDependencyError, match="polars"):
            art.to_polars()
