"""Tests for the public validate_logs / validate_pairwise_inputs helpers."""

import numpy as np
import pandas as pd
import pytest

import skdr_eval
from skdr_eval.exceptions import DataValidationError, InsufficientDataError


def test_validate_logs_accepts_synth_output():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
    # Should be a silent no-op.
    skdr_eval.validate_logs(logs)


def test_validate_logs_strict_accepts_synth_output():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=1)
    skdr_eval.validate_logs(logs, strict=True)


def test_validate_logs_missing_required_column():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=2)
    with pytest.raises(DataValidationError, match="missing required columns"):
        skdr_eval.validate_logs(logs.drop(columns=["action"]))


def test_validate_logs_no_elig_columns():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=3)
    bad = logs.drop(columns=[c for c in logs.columns if c.endswith("_elig")])
    with pytest.raises(DataValidationError, match="eligibility columns"):
        skdr_eval.validate_logs(bad)


def test_validate_logs_action_not_in_elig_columns():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=4)
    logs = logs.copy()
    logs.loc[logs.index[0], "action"] = "op_ghost"
    with pytest.raises(DataValidationError, match="not present as '\\*_elig'"):
        skdr_eval.validate_logs(logs)


def test_validate_logs_no_feature_columns():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=5)
    feature_cols = [
        c for c in logs.columns if c.startswith("cli_") or c.startswith("st_")
    ]
    bad = logs.drop(columns=feature_cols)
    with pytest.raises(DataValidationError, match="no feature columns"):
        skdr_eval.validate_logs(bad)


def test_validate_logs_strict_rejects_unsorted_timestamps():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=80, n_ops=2, seed=6)
    shuffled = logs.iloc[::-1].reset_index(drop=True)
    # Non-strict path is permissive.
    skdr_eval.validate_logs(shuffled)
    with pytest.raises(DataValidationError, match="monotonically"):
        skdr_eval.validate_logs(shuffled, strict=True)


def test_validate_logs_strict_rejects_pairwise_leakage():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=80, n_ops=2, seed=7)
    logs = logs.copy()
    logs["client_id"] = "c_0"
    with pytest.raises(DataValidationError, match="pairwise-schema columns"):
        skdr_eval.validate_logs(logs, strict=True)


def test_validate_logs_empty_raises():
    with pytest.raises(InsufficientDataError):
        skdr_eval.validate_logs(pd.DataFrame())


def test_validate_logs_rejects_nan_feature():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=8)
    logs = logs.copy()
    feature_col = next(c for c in logs.columns if c.startswith("cli_"))
    logs.loc[logs.index[0], feature_col] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        skdr_eval.validate_logs(logs)


def test_validate_pairwise_inputs_accepts_synth_output():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=50, n_ops=4, seed=0
    )
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")


def test_validate_pairwise_inputs_strict_accepts_synth_output():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=50, n_ops=4, seed=1
    )
    skdr_eval.validate_pairwise_inputs(
        logs_df, op_daily_df, metric_col="service_time", strict=True
    )


def test_validate_pairwise_inputs_missing_metric():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=2
    )
    with pytest.raises(DataValidationError, match="missing required columns"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="not_a_column"
        )


def test_validate_pairwise_inputs_no_op_features():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=3
    )
    op_daily_df = op_daily_df.drop(
        columns=[c for c in op_daily_df.columns if c.startswith("op_")]
    )
    with pytest.raises(DataValidationError, match="operator feature columns"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time"
        )


def test_validate_pairwise_inputs_strict_detects_chosen_outside_elig():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=4
    )
    logs_df = logs_df.copy()
    # Replace one row's elig_mask with a list that excludes the chosen op.
    chosen_idx = logs_df.index[0]
    chosen_op = logs_df.loc[chosen_idx, "operator_id"]
    other_ops = [op for op in op_daily_df["operator_id"].unique() if op != chosen_op]
    logs_df.at[chosen_idx, "elig_mask"] = other_ops[:2]
    # Non-strict is permissive.
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")
    with pytest.raises(DataValidationError, match="chosen operator_id"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time", strict=True
        )


def test_validate_pairwise_inputs_bad_elig_type():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=20, n_ops=3, seed=5
    )
    logs_df = logs_df.copy()
    logs_df["elig_mask"] = "all"
    with pytest.raises(DataValidationError, match="list/tuple/set"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time"
        )
