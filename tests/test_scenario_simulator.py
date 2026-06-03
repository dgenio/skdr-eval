"""Tests for the what-if autoscaling scenario simulator (issue #34)."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from skdr_eval import (
    evaluate_pairwise_models,
    make_pairwise_synth,
    simulate_autoscaling_scenario,
)
from skdr_eval.exceptions import DataValidationError


def _setup(seed: int = 0):
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=150, n_ops=8, seed=seed, binary=False
    )
    feat = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    x, y = logs_df[feat].to_numpy(), logs_df["service_time"].to_numpy()
    models = {"ridge": Ridge(), "hgb": HistGradientBoostingRegressor(random_state=0)}
    for m in models.values():
        m.fit(x, y)
    return logs_df, op_daily_df, models


def _eval_common():
    return {
        "metric_col": "service_time",
        "task_type": "regression",
        "direction": "min",
        "n_splits": 2,
        "strategy": "direct",
        "random_state": 0,
        "policy_train": "all",  # pre-fitted models; avoids the pre_split nudge
    }


def test_noop_scenario_matches_baseline():
    """capacity_multiplier=1.0 + as_logged is a no-op vs evaluate_pairwise_models."""
    logs_df, op_daily_df, models = _setup()
    baseline = evaluate_pairwise_models(
        logs_df=logs_df, op_daily_df=op_daily_df, models=models, **_eval_common()
    )
    scenario = simulate_autoscaling_scenario(
        logs_df,
        op_daily_df,
        models,
        {"capacity_multiplier": 1.0, "eligibility_mode": "as_logged"},
        **_eval_common(),
    )
    cols = ["V_hat", "SE_if", "ESS", "match_rate"]
    a = baseline.report.set_index(["model", "estimator"])[cols].sort_index()
    b = scenario.report.set_index(["model", "estimator"])[cols].sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-10)


def test_scenario_metadata_recorded():
    """The applied scenario and its assumptions are attached to metadata."""
    logs_df, op_daily_df, models = _setup()
    artifact = simulate_autoscaling_scenario(
        logs_df,
        op_daily_df,
        models,
        {"capacity_multiplier": 0.5, "eligibility_mode": "restricted"},
        **_eval_common(),
    )
    meta = artifact.metadata["scenario"]
    assert meta["capacity_multiplier"] == 0.5
    assert meta["eligibility_mode"] == "restricted"
    assert isinstance(meta["assumptions"], list) and meta["assumptions"]


def test_reduced_capacity_runs_and_changes_support():
    """A reduced-capacity scenario produces a valid, finite report."""
    logs_df, op_daily_df, models = _setup()
    artifact = simulate_autoscaling_scenario(
        logs_df,
        op_daily_df,
        models,
        {"capacity_multiplier": 0.4},
        **_eval_common(),
    )
    report = artifact.report
    assert np.isfinite(report["V_hat"]).all()
    assert (report["ESS"] > 0).all()
    assert ((report["match_rate"] >= 0) & (report["match_rate"] <= 1)).all()


def test_scenario_is_reproducible():
    """Same seed + scenario -> identical V_hat."""
    logs_df, op_daily_df, models = _setup()
    scen = {"capacity_multiplier": 0.5, "eligibility_mode": "restricted"}
    a = simulate_autoscaling_scenario(
        logs_df, op_daily_df, models, scen, **_eval_common()
    )
    b = simulate_autoscaling_scenario(
        logs_df, op_daily_df, models, scen, **_eval_common()
    )
    np.testing.assert_array_equal(
        a.report["V_hat"].to_numpy(), b.report["V_hat"].to_numpy()
    )


def test_unknown_scenario_knob_raises():
    logs_df, op_daily_df, models = _setup()
    with pytest.raises(DataValidationError, match="Unknown scenario knob"):
        simulate_autoscaling_scenario(
            logs_df, op_daily_df, models, {"staffing": 0.5}, **_eval_common()
        )


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.5])
def test_bad_capacity_multiplier_raises(bad):
    logs_df, op_daily_df, models = _setup()
    with pytest.raises(DataValidationError, match="capacity_multiplier"):
        simulate_autoscaling_scenario(
            logs_df,
            op_daily_df,
            models,
            {"capacity_multiplier": bad},
            **_eval_common(),
        )


def test_bad_eligibility_mode_raises():
    logs_df, op_daily_df, models = _setup()
    with pytest.raises(DataValidationError, match="eligibility_mode"):
        simulate_autoscaling_scenario(
            logs_df,
            op_daily_df,
            models,
            {"eligibility_mode": "wide_open"},
            **_eval_common(),
        )


def test_missing_eligibility_column_raises():
    logs_df, op_daily_df, models = _setup()
    logs_df = logs_df.drop(columns=["elig_mask"])
    with pytest.raises(DataValidationError, match="eligibility column"):
        simulate_autoscaling_scenario(
            logs_df,
            op_daily_df,
            models,
            {"capacity_multiplier": 0.5},
            **_eval_common(),
        )
