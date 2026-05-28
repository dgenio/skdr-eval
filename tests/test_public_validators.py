"""Tests for the public validate_logs / validate_pairwise_inputs helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import skdr_eval
from skdr_eval.exceptions import DataValidationError, InsufficientDataError
from skdr_eval.validation import validate_models_dict


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


def test_validate_logs_int_action_vs_str_elig_hints_dtype():
    """#115: int actions with str elig column names surface a dtype-cast hint."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=100, n_ops=3, seed=42)
    logs = logs.copy()
    elig_cols = [c for c in logs.columns if c.endswith("_elig")]
    ops = [c.removesuffix("_elig") for c in elig_cols]
    mapping = {op: i for i, op in enumerate(ops)}
    logs["action"] = logs["action"].map(mapping)
    logs = logs.rename(
        columns={c: f"{mapping[c.removesuffix('_elig')]}_elig" for c in elig_cols}
    )

    with pytest.raises(DataValidationError, match="dtype is") as excinfo:
        skdr_eval.validate_logs(logs)
    assert "astype(str)" in str(excinfo.value)


def test_validate_logs_unrelated_action_message_has_no_dtype_hint():
    """#115: genuinely-unrelated string actions keep the original message (no hint)."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=4)
    logs = logs.copy()
    logs.loc[logs.index[0], "action"] = "op_ghost"
    with pytest.raises(DataValidationError) as excinfo:
        skdr_eval.validate_logs(logs)
    assert "dtype is" not in str(excinfo.value)


def test_validate_models_dict_accepts_valid_mapping():
    """#109: a well-formed {name: estimator} dict passes silently."""
    validate_models_dict({"lr": LinearRegression()})


def test_validate_models_dict_non_dict_non_estimator_has_no_hint():
    """#109: a non-dict value without a fit method errors without a did-you-mean hint."""
    with pytest.raises(DataValidationError) as excinfo:
        validate_models_dict([1, 2, 3])
    message = str(excinfo.value)
    assert "must be a dict" in message
    assert "did you mean" not in message


def test_validate_models_dict_non_string_keys():
    """#109: non-string keys raise a clear error."""
    with pytest.raises(DataValidationError, match="keys must be strings"):
        validate_models_dict({0: LinearRegression()})


def test_validate_logs_custom_y_col():
    """#105: validate_logs accepts a non-default reward column via y_col."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=200, n_ops=3, seed=0)
    renamed = logs.rename(columns={"service_time": "reward"})

    # Default y_col now fails because 'service_time' is gone...
    with pytest.raises(DataValidationError, match="missing required columns"):
        skdr_eval.validate_logs(renamed)
    # ...but passing the actual column name validates cleanly.
    skdr_eval.validate_logs(renamed, y_col="reward")


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


def test_validate_logs_rejects_string_arrival_ts():
    """validate_logs must match build_design's contract (numeric/datetime64).

    Regression: build_design uses ``.values.astype(float)`` which raises on
    string timestamps; the preflight must catch this rather than falsely
    pass.
    """
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=9)
    logs = logs.copy()
    logs["arrival_ts"] = logs["arrival_ts"].astype(str)
    with pytest.raises(DataValidationError, match="numeric or datetime64"):
        skdr_eval.validate_logs(logs)


def test_validate_logs_accepts_datetime64_arrival_ts():
    """datetime64[ns] is the canonical type produced by make_synth_logs."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=10)
    assert pd.api.types.is_datetime64_any_dtype(logs["arrival_ts"])
    skdr_eval.validate_logs(logs)


def test_validate_logs_accepts_numeric_arrival_ts():
    """Integer epoch seconds also pass."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=11)
    logs = logs.copy()
    logs["arrival_ts"] = np.arange(len(logs), dtype=np.int64)
    skdr_eval.validate_logs(logs)


def test_validate_logs_detects_chosen_action_not_eligible_on_row():
    """Per-row check: action='op_A' with op_A_elig=0 must fail preflight."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=80, n_ops=3, seed=12)
    logs = logs.copy()
    # Pick a row, force its chosen action's elig column to 0 while keeping
    # the action value intact. Synth uses bool dtype; assign False directly.
    chosen = str(logs.loc[logs.index[0], "action"])
    logs.loc[logs.index[0], f"{chosen}_elig"] = False
    with pytest.raises(DataValidationError, match="not eligible on"):
        skdr_eval.validate_logs(logs)


def test_validate_logs_rejects_non_01_eligibility():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=40, n_ops=2, seed=13)
    logs = logs.copy()
    elig_col = next(c for c in logs.columns if c.endswith("_elig"))
    # Cast to int so we can inject an out-of-range value.
    logs[elig_col] = logs[elig_col].astype(int)
    logs.loc[logs.index[0], elig_col] = 2
    with pytest.raises(DataValidationError, match="only 0/1 values"):
        skdr_eval.validate_logs(logs)


def test_validate_logs_rejects_nan_service_time():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=40, n_ops=2, seed=14)
    logs = logs.copy()
    logs.loc[logs.index[0], "service_time"] = np.nan
    with pytest.raises(DataValidationError, match="service_time"):
        skdr_eval.validate_logs(logs)


def test_validate_logs_strict_rejects_eligible_never_chosen():
    """strict: if op is eligible somewhere but never chosen, flag it."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=80, n_ops=3, seed=15)
    logs = logs.copy()
    # Pick an op chosen at least once. Force its elig column to True
    # everywhere (so it remains eligible) and replace every row where it
    # was chosen with a different action. Result: op is eligible on every
    # row but never appears in `action`.
    target = str(logs["action"].iloc[0])
    replacement = next(op for op in logs["action"].unique() if op != target)
    # Force both ops eligible everywhere so the row-wise check passes and
    # the eligibility/observation gap becomes the *only* invariant violated.
    logs[f"{target}_elig"] = True
    logs[f"{replacement}_elig"] = True
    logs.loc[logs["action"] == target, "action"] = replacement
    skdr_eval.validate_logs(logs)
    with pytest.raises(DataValidationError, match="never chosen"):
        skdr_eval.validate_logs(logs, strict=True)


def test_validate_pairwise_inputs_no_cli_features():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=16
    )
    logs_df = logs_df.drop(columns=[c for c in logs_df.columns if c.startswith("cli_")])
    with pytest.raises(DataValidationError, match="client feature columns"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time"
        )


def test_validate_pairwise_inputs_strict_missing_op_day_pair():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=17
    )
    # Drop one operator's daily row to create a (operator, day) gap.
    bad_op = op_daily_df["operator_id"].iloc[0]
    bad_day = op_daily_df["arrival_day"].iloc[0]
    op_daily_df = op_daily_df[
        ~(
            (op_daily_df["operator_id"] == bad_op)
            & (op_daily_df["arrival_day"] == bad_day)
        )
    ].reset_index(drop=True)
    # Inject a logs row that picks the missing pair.
    logs_df = logs_df.copy()
    logs_df.loc[logs_df.index[0], "operator_id"] = bad_op
    logs_df.loc[logs_df.index[0], "arrival_day"] = bad_day
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")
    with pytest.raises(DataValidationError, match="\\(operator, day\\) pairs"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time", strict=True
        )


def test_validate_pairwise_inputs_mixed_elig_types_caught():
    """The elig_col type check now iterates all rows, not just the first."""
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=18
    )
    logs_df = logs_df.copy()
    # First row keeps a list (would pass the old first-row-only check); a
    # later row has a bad value.
    logs_df.at[logs_df.index[5], "elig_mask"] = "bad-string"
    with pytest.raises(DataValidationError, match="list/tuple/set"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time"
        )


def test_validate_pairwise_inputs_nan_elig_skipped():
    """NaN elig_mask rows are skipped (treated as 'unrestricted'), not rejected."""
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=20, n_ops=3, seed=19
    )
    logs_df = logs_df.copy()
    logs_df.at[logs_df.index[0], "elig_mask"] = np.nan
    # Should pass non-strict (NaN is treated as unrestricted, matching the
    # downstream contract in evaluate_pairwise_models).
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")


def test_validate_pairwise_inputs_strict_truncates_bad_row_examples():
    """When >5 rows fail the chosen-in-elig check, error message truncates."""
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=3, seed=20
    )
    logs_df = logs_df.copy()
    # Force every elig_mask to a single-op list that excludes the chosen op.
    for i in range(min(10, len(logs_df))):
        chosen = logs_df.loc[logs_df.index[i], "operator_id"]
        other = next(op for op in op_daily_df["operator_id"].unique() if op != chosen)
        logs_df.at[logs_df.index[i], "elig_mask"] = [other]
    with pytest.raises(DataValidationError, match="up to 5"):
        skdr_eval.validate_pairwise_inputs(
            logs_df, op_daily_df, metric_col="service_time", strict=True
        )


def test_validate_logs_rejects_nat_arrival_ts():
    """datetime64 column with NaT values is caught."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=50, n_ops=2, seed=21)
    logs = logs.copy()
    logs.loc[logs.index[0], "arrival_ts"] = pd.NaT
    with pytest.raises(DataValidationError, match="NaT/NaN"):
        skdr_eval.validate_logs(logs)
