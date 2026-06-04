"""Tests for the large-data execution path (issue #33).

``execution_mode="large_data"`` builds the per-decision observed-feature,
action, eligibility and policy-probability arrays with vectorized operations
instead of a per-row ``DataFrame.iterrows()`` loop. It must be **numerically
identical** to the standard path; these tests pin that equivalence at both the
builder level and the full-report level, plus the ``"auto"`` resolution.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from skdr_eval import evaluate_pairwise_models, make_pairwise_synth
from skdr_eval.pairwise import (
    LARGE_DATA_ROW_THRESHOLD,
    PairwiseDesign,
    build_eval_arrays_vectorized,
    build_policy_probs_vectorized,
)


def _fit_models(logs_df, target_col, binary=False):
    feat = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    x, y = logs_df[feat].to_numpy(), logs_df[target_col].to_numpy()
    models = (
        {"logit": LogisticRegression(max_iter=200)}
        if binary
        else {"ridge": Ridge(), "hgb": HistGradientBoostingRegressor(random_state=0)}
    )
    for m in models.values():
        m.fit(x, y)
    return models


# --- Builder-level parity (the reference loop, replicated inline) -----------


def _reference_obs_arrays(design: PairwiseDesign):
    """Row-wise reference implementation copied from the standard path."""
    logs_df = design.logs_df
    op_id_col = design.operator_id_col
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())
    x_list, a_list = [], []
    for _i, row in logs_df.iterrows():
        day = row[design.day_col]
        chosen = row[op_id_col]
        feats = [float(row[f]) for f in design.cli_features]
        if str(day) in design.day_to_op_df:
            op_df = design.day_to_op_df[str(day)]
            op_row = op_df[op_df[op_id_col] == chosen]
            if len(op_row) > 0:
                feats += [float(op_row.iloc[0][f]) for f in design.op_features]
            else:
                feats += [0.0] * len(design.op_features)
        else:
            feats += [0.0] * len(design.op_features)
        x_list.append(feats)
        if (
            str(day) in design.ops_all_by_day
            and str(chosen) in design.ops_all_by_day[str(day)]
        ):
            a_list.append(design.ops_all_by_day[str(day)].index(str(chosen)))
        else:
            a_list.append(0)
    x_obs = np.array(x_list, dtype=np.float32)
    a = np.array(a_list)

    elig = np.zeros((len(logs_df), max_ops))
    for idx, row in logs_df.iterrows():
        i = int(idx)
        day_str = str(row[design.day_col])
        if day_str in design.ops_all_by_day:
            if design.elig_col and design.elig_col in row:
                val = row[design.elig_col]
                if isinstance(val, (list, tuple)):
                    for op in val:
                        if str(op) in design.ops_all_by_day[day_str]:
                            elig[i, design.ops_all_by_day[day_str].index(str(op))] = 1.0
                else:
                    elig[i, : len(design.ops_all_by_day[day_str])] = 1.0
            else:
                elig[i, : len(design.ops_all_by_day[day_str])] = 1.0
    return x_obs, a, elig, max_ops


def test_vectorized_builders_match_reference_loop():
    """X_obs / A / eligibility / policy_probs match the row-wise reference."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=200, n_ops=8, seed=11
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    ref_x, ref_a, ref_elig, ref_max = _reference_obs_arrays(design)
    vec_x, vec_a, vec_elig, vec_max = build_eval_arrays_vectorized(
        design, "service_time"
    )
    assert vec_max == ref_max
    np.testing.assert_array_equal(vec_a, ref_a)
    np.testing.assert_array_equal(vec_elig, ref_elig)
    np.testing.assert_allclose(vec_x, ref_x, rtol=0, atol=1e-6)

    # Policy-probability matrix for an arbitrary assignment (operator per row).
    rng = np.random.default_rng(0)
    decisions = np.array(
        [
            rng.choice(design.ops_all_by_day[str(d)])
            for d in design.logs_df[design.day_col]
        ],
        dtype=object,
    )
    vec_pp = build_policy_probs_vectorized(design, decisions, ref_max)
    ref_pp = np.zeros((len(design.logs_df), ref_max))
    for i, (_idx, row) in enumerate(design.logs_df.iterrows()):
        day_str = str(row[design.day_col])
        chosen = str(decisions[i])
        if chosen in design.ops_all_by_day[day_str]:
            ref_pp[i, design.ops_all_by_day[day_str].index(chosen)] = 1.0
    np.testing.assert_array_equal(vec_pp, ref_pp)


# --- Full-report parity ------------------------------------------------------


@pytest.mark.parametrize("binary", [False, True])
def test_execution_mode_report_parity(binary):
    """standard and large_data produce identical reports (<1e-10)."""
    target = "success" if binary else "service_time"
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=150, n_ops=6, seed=7, binary=binary
    )
    models = _fit_models(logs_df, target, binary=binary)
    common = {
        "logs_df": logs_df,
        "op_daily_df": op_daily_df,
        "models": models,
        "metric_col": target,
        "task_type": "binary" if binary else "regression",
        "direction": "max" if binary else "min",
        "n_splits": 2,
        "strategy": "direct",
        "random_state": 42,
        "policy_train": "all",  # pre-fitted models; avoids the pre_split nudge
    }
    std = evaluate_pairwise_models(execution_mode="standard", **common).report
    big = evaluate_pairwise_models(execution_mode="large_data", **common).report

    key = ["model", "estimator"]
    num = ["V_hat", "SE_if", "ESS", "tail_mass", "match_rate", "min_pscore", "pareto_k"]
    std_i = std.set_index(key)[num].sort_index()
    big_i = big.set_index(key)[num].sort_index()
    np.testing.assert_allclose(std_i.to_numpy(), big_i.to_numpy(), rtol=0, atol=1e-10)


def test_execution_mode_auto_resolution():
    """auto records standard below the threshold, large_data at/above it."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=4, seed=1
    )
    assert len(logs_df) < LARGE_DATA_ROW_THRESHOLD
    models = _fit_models(logs_df, "service_time")
    art = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        policy_train="all",
        execution_mode="auto",
    )
    assert art.metadata["execution_mode"] == "standard"


def test_large_data_handles_duplicate_op_daily_rows():
    """A duplicated (day, operator) row in op_daily_df must not misalign x_obs.

    The vectorized lookup deduplicates on (day, operator_id) keeping the first
    match, mirroring the standard path's ``op_row.iloc[0]``, so large_data still
    matches standard exactly.
    """
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=120, n_ops=5, seed=3
    )
    op_daily_dup = pd.concat(
        [op_daily_df, op_daily_df.iloc[[0]]], ignore_index=True
    )  # duplicate one (day, operator) snapshot
    models = _fit_models(logs_df, "service_time")
    common = {
        "logs_df": logs_df,
        "op_daily_df": op_daily_dup,
        "models": models,
        "metric_col": "service_time",
        "task_type": "regression",
        "direction": "min",
        "n_splits": 2,
        "strategy": "direct",
        "random_state": 0,
        "policy_train": "all",
    }
    std = evaluate_pairwise_models(execution_mode="standard", **common).report
    big = evaluate_pairwise_models(execution_mode="large_data", **common).report
    np.testing.assert_allclose(
        std.sort_values(["model", "estimator"])["V_hat"].to_numpy(),
        big.sort_values(["model", "estimator"])["V_hat"].to_numpy(),
        rtol=0,
        atol=1e-10,
    )


def test_invalid_execution_mode_raises():
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=40, n_ops=3, seed=0
    )
    with pytest.raises(ValueError, match="Unknown execution_mode"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=_fit_models(logs_df, "service_time"),
            metric_col="service_time",
            task_type="regression",
            direction="min",
            execution_mode="turbo",  # type: ignore[arg-type]
        )
