"""Tests for Polars / PyArrow input coercion and accessors (#72)."""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval._frames import coerce_to_pandas
from skdr_eval.exceptions import DataValidationError, OptionalDependencyError
from skdr_eval.pairwise import _normalize_elig_cell

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


def _evaluate_pairwise(
    logs_df: object,
    op_daily_df: object,
    *,
    elig_col: str | None = "elig_mask",
    execution_mode: str = "auto",
) -> skdr_eval.EvaluationArtifact:
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
    return skdr_eval.evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=3,
        fit_models=True,
        policy_train="pre_split",
        random_state=0,
        elig_col=elig_col,
        execution_mode=execution_mode,
    )


# Eligibility feeds the DR weights, so an unhonored mask perturbs the support
# diagnostics (ESS / match_rate / SE), not just the point estimate. Asserting
# the full set of eligibility-sensitive report columns makes the equivalence
# regressions harder to satisfy while still being wrong on diagnostics.
_ELIG_SENSITIVE_COLS = ["V_hat", "SE_if", "ESS", "match_rate"]


def _assert_pairwise_report_equiv(
    art_a: skdr_eval.EvaluationArtifact, art_b: skdr_eval.EvaluationArtifact
) -> None:
    pd.testing.assert_frame_equal(
        art_a.report[_ELIG_SENSITIVE_COLS].reset_index(drop=True),
        art_b.report[_ELIG_SENSITIVE_COLS].reset_index(drop=True),
    )


class TestNormalizeEligCell:
    """`_normalize_elig_cell` canonicalizes every container type to `list`.

    Unit-level coverage of each branch (the integration equivalence tests below
    exercise the ndarray/set paths end-to-end, but the tuple branch and the
    unorderable-set fallback are only reachable here).
    """

    def test_ndarray_becomes_list_of_python_scalars(self) -> None:
        out = _normalize_elig_cell(np.array(["op_2", "op_5"]))
        assert out == ["op_2", "op_5"]
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)  # tolist() → native str

    def test_set_is_sorted_deterministically(self) -> None:
        assert _normalize_elig_cell({"op_5", "op_2", "op_1"}) == [
            "op_1",
            "op_2",
            "op_5",
        ]

    def test_frozenset_is_sorted_deterministically(self) -> None:
        assert _normalize_elig_cell(frozenset({3, 1, 2})) == [1, 2, 3]

    def test_unorderable_set_uses_repr_keyed_fallback(self) -> None:
        # Mixed types are not mutually orderable (`int` vs `str` raise TypeError
        # under `sorted`); the fallback must still be deterministic.
        value = {1, "a", 2, "b"}
        out = _normalize_elig_cell(value)
        assert isinstance(out, list)
        assert sorted(out, key=repr) == out  # deterministic, repr-keyed
        assert set(out) == value  # no elements lost
        # Stable across calls regardless of set construction order.
        assert _normalize_elig_cell({"b", 2, "a", 1}) == out

    def test_tuple_becomes_list(self) -> None:
        out = _normalize_elig_cell(("op_0", "op_1"))
        assert out == ["op_0", "op_1"]
        assert isinstance(out, list)

    def test_list_passthrough_unchanged(self) -> None:
        value = ["op_0", "op_1"]
        assert _normalize_elig_cell(value) is value

    @pytest.mark.parametrize("value", [float("nan"), None, "scalar", 3])
    def test_non_container_passthrough(self, value: object) -> None:
        # Left untouched so the missing-mask "all eligible" fallback still fires.
        assert _normalize_elig_cell(value) is value


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
    """Polars pairwise input must match the pandas path V_hat exactly (#158).

    ``coerce_to_pandas`` turns the list-valued ``elig_mask`` cells into
    ``np.ndarray`` via ``.to_pandas()``; the pairwise eligibility consumers are
    sensitive to ``list`` vs ``ndarray``, so without renormalization V_hat
    diverged from a pandas-native input (contradicting #72's equivalence
    claim). ``PairwiseDesign.from_dataframes`` now canonicalizes the cells back
    to ``list`` at ingestion, restoring exact equivalence.
    """
    import polars as pl  # noqa: PLC0415

    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=120, n_ops=4, seed=3
    )
    art_pd = _evaluate_pairwise(logs_df, op_daily_df)
    art_pl = _evaluate_pairwise(pl.from_pandas(logs_df), pl.from_pandas(op_daily_df))
    _assert_pairwise_report_equiv(art_pd, art_pl)


@requires_polars
def test_pairwise_external_policies_accepts_polars() -> None:
    # #236: the external-policy tables passed to evaluate_pairwise_models now go
    # through the same coerce_to_pandas seam, so a polars policy frame produces
    # results identical to the pandas one.
    import polars as pl  # noqa: PLC0415

    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=120, n_ops=4, seed=8
    )
    rng = np.random.default_rng(0)
    clients = logs_df["client_id"].unique()
    known_ops = op_daily_df["operator_id"].unique()
    policy = pd.DataFrame(
        {
            "client_id": clients,
            "operator_id": rng.choice(known_ops, size=len(clients)),
        }
    )

    common = {
        "logs_df": logs_df,
        "op_daily_df": op_daily_df,
        "models": {},
        "metric_col": "service_time",
        "task_type": "regression",
        "direction": "min",
        "n_splits": 2,
        "fit_models": False,  # external policies are scored, not fit
        "random_state": 0,
    }
    art_pd = skdr_eval.evaluate_pairwise_models(
        external_policies={"sim": policy}, **common
    )
    art_pl = skdr_eval.evaluate_pairwise_models(
        external_policies={"sim": pl.from_pandas(policy)}, **common
    )
    _assert_pairwise_report_equiv(art_pd, art_pl)


@requires_polars
def test_autoscaling_scenario_accepts_polars() -> None:
    # #236: simulate_autoscaling_scenario routes its logs/op-daily frames through
    # the seam, so polars inputs are accepted and match the pandas run.
    import polars as pl  # noqa: PLC0415

    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=120, n_ops=4, seed=9
    )

    def _run(logs: object, op_daily: object) -> skdr_eval.EvaluationArtifact:
        return skdr_eval.simulate_autoscaling_scenario(
            logs,
            op_daily,
            {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)},
            {"capacity_multiplier": 0.8},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            random_state=0,
            fit_models=True,
            policy_train="pre_split",
        )

    art_pd = _run(logs_df, op_daily_df)
    art_pl = _run(pl.from_pandas(logs_df), pl.from_pandas(op_daily_df))
    _assert_pairwise_report_equiv(art_pd, art_pl)


def _with_set_masks(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Re-express each list-valued ``elig_mask`` cell as an (unordered) set."""
    out = logs_df.copy()
    out["elig_mask"] = out["elig_mask"].map(set)
    return out


class TestPairwiseEligMaskTypeEquivalence:
    """A set-valued elig_mask must be honored identically to a list (#155).

    ``validate_pairwise_inputs`` blesses ``set`` masks, but most eligibility
    consumers only special-case ``(list, tuple)`` and silently fell back to
    "every operator eligible" for a ``set`` — producing incorrect, less
    restrictive eligibility. Normalizing at ingestion makes a ``set`` mask
    produce the same V_hat as the identical mask expressed as a ``list``.
    """

    @pytest.mark.parametrize("execution_mode", ["standard", "large_data"])
    def test_set_mask_matches_list_mask(self, execution_mode: str) -> None:
        logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
            n_days=3, n_clients_day=120, n_ops=5, seed=7
        )
        art_list = _evaluate_pairwise(
            logs_df, op_daily_df, execution_mode=execution_mode
        )
        art_set = _evaluate_pairwise(
            _with_set_masks(logs_df), op_daily_df, execution_mode=execution_mode
        )
        _assert_pairwise_report_equiv(art_list, art_set)

    def test_restrictive_mask_is_actually_honored(self) -> None:
        # Guard against a silent "all eligible" fallback: dropping the mask
        # (elig_col=None ⇒ every operator eligible) must change V_hat, proving
        # the restriction is honored rather than ignored. ~80% of operators are
        # eligible per row in the synthetic data, so the masks are restrictive.
        logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
            n_days=3, n_clients_day=120, n_ops=5, seed=7
        )
        art_restricted = _evaluate_pairwise(logs_df, op_daily_df)
        art_all_elig = _evaluate_pairwise(logs_df, op_daily_df, elig_col=None)
        assert not np.allclose(
            art_restricted.report["V_hat"].to_numpy(),
            art_all_elig.report["V_hat"].to_numpy(),
        )


@requires_polars
def test_polars_input_on_par_with_pandas() -> None:
    """Microbenchmark for #72's "on par or faster" acceptance criterion.

    Exercises the ``evaluate_sklearn_models`` path, where the Polars input is
    proven equivalent to pandas (see ``TestPolarsInputEquivalence``): the frame
    is converted once at the boundary and then shares the pandas pipeline, so
    wall-clock should be on par. Asserts (1) the boundary conversion does not
    perturb ``V_hat`` and (2) a generous wall-clock bound, so a pathological
    per-row conversion regression is caught without CI timing flakiness. Run
    with ``-s`` to see the reported timings.

    NOTE: the *pairwise* path is not benchmarked here — its V_hat equivalence
    across pandas/Polars input is covered by ``test_pairwise_polars_inputs``
    (the list-valued ``elig_mask`` round-trip fixed in #158).
    """
    import polars as pl  # noqa: PLC0415

    logs, _, _ = skdr_eval.make_synth_logs(n=8000, n_ops=6, seed=11)

    t0 = time.perf_counter()
    art_pd = _evaluate(logs)
    t_pandas = time.perf_counter() - t0

    logs_pl = pl.from_pandas(logs)
    t0 = time.perf_counter()
    art_pl = _evaluate(logs_pl)
    t_polars = time.perf_counter() - t0

    # Correctness: the boundary conversion must not change the estimate.
    pd.testing.assert_series_equal(
        art_pd.report["V_hat"].reset_index(drop=True),
        art_pl.report["V_hat"].reset_index(drop=True),
    )

    print(
        f"\n[#72 bench] sklearn V_hat — pandas {t_pandas:.3f}s vs "
        f"polars-input {t_polars:.3f}s (overhead {t_polars - t_pandas:+.3f}s)"
    )

    # "On par": the one-time boundary conversion is small next to evaluation.
    # The bound is deliberately generous to stay non-flaky on shared CI runners
    # while still catching a pathological (e.g. per-row) conversion regression.
    assert t_polars < t_pandas * 5.0 + 2.0


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
