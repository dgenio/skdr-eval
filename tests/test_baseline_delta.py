"""Tests for the ``baseline=`` first-class output on the evaluators (#132)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from skdr_eval import (
    DataValidationError,
    evaluate_sklearn_models,
    make_synth_logs,
)


def _logs():
    logs, _, _ = make_synth_logs(n=600, n_ops=3, seed=7)
    return logs


def test_baseline_none_produces_no_delta_columns() -> None:
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
    )
    assert art.baseline_kind is None
    assert art.baseline_value is None
    assert "delta_V_hat" not in art.report.columns


def test_baseline_scalar_produces_delta_columns() -> None:
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
        baseline=25.0,
    )
    assert art.baseline_kind == "scalar"
    assert art.baseline_value == pytest.approx(25.0)
    assert "delta_V_hat" in art.report.columns
    # delta_V_hat == V_hat - baseline_value for every row.
    for _, row in art.report.iterrows():
        assert row["delta_V_hat"] == pytest.approx(row["V_hat"] - 25.0)


def test_baseline_logged_uses_eval_slice_mean() -> None:
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
        baseline="logged",
    )
    assert art.baseline_kind == "logged"
    # The exact value is the mean of the eval slice — we just sanity-check
    # it's a finite positive number and that the delta column reflects it.
    assert art.baseline_value is not None
    assert np.isfinite(art.baseline_value)
    assert art.baseline_value > 0.0
    for _, row in art.report.iterrows():
        assert row["delta_V_hat"] == pytest.approx(row["V_hat"] - art.baseline_value)


def test_baseline_unknown_string_raises() -> None:
    with pytest.raises(DataValidationError, match="Unknown baseline"):
        evaluate_sklearn_models(
            logs=_logs(),
            models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
            policy_train="pre_split",
            baseline="not-a-baseline",
        )


def test_baseline_with_ci_produces_delta_ci_columns() -> None:
    """``baseline=`` + ``ci_bootstrap=True`` populates the delta CI columns.

    Covers the conditional branches in ``build_evaluation_artifact`` that
    derive ``delta_ci_lower`` / ``delta_ci_upper`` from ``ci_lower`` /
    ``ci_upper`` minus the baseline value, and the matching
    ``BaselineBlock`` lift in ``_build_card_from_row``.
    """
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
        baseline=25.0,
        ci_bootstrap=True,
    )
    assert "delta_V_hat" in art.report.columns
    assert "ci_lower" in art.report.columns
    assert "ci_upper" in art.report.columns
    assert "delta_ci_lower" in art.report.columns
    assert "delta_ci_upper" in art.report.columns
    # Delta-CI columns are CI columns minus the baseline, element-wise.
    for _, row in art.report.iterrows():
        assert row["delta_ci_lower"] == pytest.approx(row["ci_lower"] - 25.0)
        assert row["delta_ci_upper"] == pytest.approx(row["ci_upper"] - 25.0)
    # The card baseline block also surfaces the CI delta.
    card = art.card_schema("HGB", estimator="DR")
    assert card.baseline is not None
    assert card.baseline.kind == "scalar"
    assert card.baseline.value == pytest.approx(25.0)
    assert card.baseline.delta_V_hat is not None
    assert card.baseline.delta_ci_lower is not None
    assert card.baseline.delta_ci_upper is not None


def test_baseline_card_block_populated() -> None:
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
        baseline=20.0,
    )
    card = art.card_schema("HGB", estimator="DR")
    assert card.baseline is not None
    assert card.baseline.kind == "scalar"
    assert card.baseline.value == pytest.approx(20.0)
    assert card.baseline.delta_V_hat == pytest.approx(
        card.headline.V_hat - 20.0 if card.headline.V_hat is not None else 0.0,
    )


def test_estimand_block_default_assumption_tags() -> None:
    """Every card carries the seven canonical assumption tags (#128)."""
    art = evaluate_sklearn_models(
        logs=_logs(),
        models={"HGB": HistGradientBoostingRegressor(max_iter=20)},
        policy_train="pre_split",
    )
    card = art.card_schema("HGB", estimator="DR")
    expected = {
        "unconfoundedness",
        "overlap",
        "sutva",
        "double_robustness",
        "stochastic_logging",
        "bounded_weight_variance",
        "time_structure_respected",
    }
    assert set(card.estimand.assumptions) == expected
    assert card.estimand.estimand_tex.startswith("V(")
