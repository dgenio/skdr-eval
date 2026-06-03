"""Tests for external-policy evaluation (issue #56).

``evaluate_external_policies`` (and the ``external_policies`` parameter on
``evaluate_pairwise_models``) lets callers score policies produced by an
external decision process — e.g. a call-centre simulator that accounts for
queues and shifts — instead of inducing them greedily from candidate models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from skdr_eval import (
    evaluate_external_policies,
    evaluate_pairwise_models,
    make_pairwise_synth,
)
from skdr_eval.exceptions import DataValidationError
from skdr_eval.pairwise import PairwiseDesign, induce_policy


def _synth(seed: int = 0, n_days: int = 1):
    return make_pairwise_synth(
        n_days=n_days, n_clients_day=150, n_ops=5, seed=seed, binary=False
    )


def _random_policy(logs_df: pd.DataFrame, op_daily_df: pd.DataFrame, seed: int):
    """A valid client->operator assignment over known operators."""
    rng = np.random.default_rng(seed)
    clients = logs_df["client_id"].unique()
    known_ops = op_daily_df["operator_id"].unique()
    return pd.DataFrame(
        {"client_id": clients, "operator_id": rng.choice(known_ops, size=len(clients))}
    )


def test_evaluate_external_policies_basic():
    """A simulator-style assignment is scored end-to-end with valid metrics."""
    logs_df, op_daily_df = _synth()
    pol = _random_policy(logs_df, op_daily_df, seed=1)

    artifact = evaluate_external_policies(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        policies={"sim_a": pol},
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
    )
    report = artifact.report
    assert set(report["model"]) == {"sim_a"}
    assert set(report["estimator"]) >= {"DR", "SNDR"}
    assert np.isfinite(report["V_hat"]).all()
    assert (report["ESS"] > 0).all()
    assert ((report["match_rate"] >= 0) & (report["match_rate"] <= 1)).all()
    assert artifact.metadata["external_policies"] is True


def test_multiple_external_policies_are_compared():
    """Two simulators are scored together for a head-to-head comparison."""
    logs_df, op_daily_df = _synth()
    artifact = evaluate_external_policies(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        policies={
            "sim_a": _random_policy(logs_df, op_daily_df, seed=1),
            "sim_b": _random_policy(logs_df, op_daily_df, seed=2),
        },
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
    )
    assert set(artifact.report["model"]) == {"sim_a", "sim_b"}


def test_external_policy_matches_induced_policy():
    """Feeding an induced policy as external assignments reproduces it exactly.

    This pins the external-policy plumbing: the only thing that changes is the
    *source* of the policy, so when the externally-supplied assignment equals
    the one ``induce_policy`` produces, the DR/SNDR numbers must be identical.
    A single day guarantees a well-defined per-client assignment.
    """
    logs_df, op_daily_df = _synth(seed=3, n_days=1)
    model = Ridge()
    feat = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    model.fit(logs_df[feat].to_numpy(), logs_df["service_time"].to_numpy())

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    induced = induce_policy({"m": model}, design, strategy="direct", direction="min")[
        "m"
    ]
    ext_frame = (
        pd.DataFrame(
            {
                "client_id": design.logs_df["client_id"].to_numpy(),
                "operator_id": induced,
            }
        )
        .drop_duplicates("client_id")
        .reset_index(drop=True)
    )

    common = {
        "logs_df": logs_df,
        "op_daily_df": op_daily_df,
        "metric_col": "service_time",
        "task_type": "regression",
        "direction": "min",
        "n_splits": 2,
        "strategy": "direct",
        "random_state": 0,
    }
    induced_art = evaluate_pairwise_models(
        models={"m": model}, fit_models=False, policy_train="all", **common
    )
    external_art = evaluate_external_policies(policies={"m": ext_frame}, **common)

    cols = ["V_hat", "SE_if", "ESS", "match_rate"]
    a = induced_art.report.set_index("estimator")[cols]
    b = external_art.report.set_index("estimator")[cols]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-9)


def test_external_policies_via_evaluate_pairwise_param():
    """The ``external_policies`` parameter works directly on the main entry."""
    logs_df, op_daily_df = _synth()
    artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models={},  # not needed when external_policies is given
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        external_policies={"sim": _random_policy(logs_df, op_daily_df, seed=5)},
    )
    assert set(artifact.report["model"]) == {"sim"}


def test_missing_client_assignment_raises():
    logs_df, op_daily_df = _synth()
    pol = _random_policy(logs_df, op_daily_df, seed=1).iloc[:-10]  # drop coverage
    with pytest.raises(DataValidationError, match="does not cover"):
        evaluate_external_policies(
            logs_df, op_daily_df, {"sim": pol}, "service_time", "regression", "min"
        )


def test_unknown_operator_raises():
    logs_df, op_daily_df = _synth()
    pol = _random_policy(logs_df, op_daily_df, seed=1)
    pol.loc[0, "operator_id"] = "op_does_not_exist"
    with pytest.raises(DataValidationError, match="unknown operator"):
        evaluate_external_policies(
            logs_df, op_daily_df, {"sim": pol}, "service_time", "regression", "min"
        )


def test_duplicate_client_assignment_raises():
    logs_df, op_daily_df = _synth()
    pol = _random_policy(logs_df, op_daily_df, seed=1)
    pol = pd.concat([pol, pol.iloc[[0]]], ignore_index=True)  # duplicate one client
    with pytest.raises(DataValidationError, match="duplicate client"):
        evaluate_external_policies(
            logs_df, op_daily_df, {"sim": pol}, "service_time", "regression", "min"
        )


def test_missing_required_column_raises():
    logs_df, op_daily_df = _synth()
    bad = pd.DataFrame({"client_id": logs_df["client_id"].unique()})  # no operator_id
    with pytest.raises(DataValidationError, match="missing required column"):
        evaluate_external_policies(
            logs_df, op_daily_df, {"sim": bad}, "service_time", "regression", "min"
        )


def test_empty_policies_raises():
    logs_df, op_daily_df = _synth()
    with pytest.raises(DataValidationError, match="non-empty dict"):
        evaluate_external_policies(
            logs_df, op_daily_df, {}, "service_time", "regression", "min"
        )


def test_reserved_kwargs_rejected():
    logs_df, op_daily_df = _synth()
    pol = _random_policy(logs_df, op_daily_df, seed=1)
    with pytest.raises(TypeError, match="does not accept"):
        evaluate_external_policies(
            logs_df,
            op_daily_df,
            {"sim": pol},
            "service_time",
            "regression",
            "min",
            fit_models=True,
        )
