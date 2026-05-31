"""Tests for the generic trace -> OPE-log adapter (#149)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import skdr_eval
from skdr_eval.adapters import TraceAdapterResult, from_jsonl_trace, from_records
from skdr_eval.exceptions import DataValidationError, InsufficientDataError


def _mapping_records(n: int = 60, seed: int = 0) -> list[dict]:
    """A small agent-style trace with mapping contexts and explicit timestamps."""
    rng = np.random.RandomState(seed)
    actions = ["route_fast", "route_smart", "route_cheap"]
    records = []
    for i in range(n):
        ctx = {
            "tokens": float(rng.randint(64, 1024)),
            "priority": float(rng.randint(0, 3)),
        }
        a = actions[rng.randint(len(actions))]
        records.append(
            {
                "context": ctx,
                "action": a,
                "reward": float(rng.normal(10.0, 1.0)),
                "timestamp": f"2024-01-01T00:{i % 60:02d}:00",
            }
        )
    return records


def test_from_records_mapping_context_is_schema_valid() -> None:
    res = from_records(_mapping_records())
    assert isinstance(res, TraceAdapterResult)
    # Round-trips through the public preflight with the adapter's reward column.
    skdr_eval.validate_logs(res.logs, y_col=res.reward_col)
    assert res.feature_names == ["cli_tokens", "cli_priority"]
    assert res.actions == ["route_cheap", "route_fast", "route_smart"]
    assert res.n_records == 60
    assert res.synthesized_timestamps is False
    assert res.had_logged_propensities is False
    # One elig column per action, all eligible (no per-row eligibility given).
    for a in res.actions:
        assert (res.logs[f"{a}_elig"] == 1).all()


def test_from_records_sequence_context_indexes_features() -> None:
    records = [
        {"context": [0.1, 0.2, 0.3], "action": "a", "reward": 1.0},
        {"context": [0.4, 0.5, 0.6], "action": "b", "reward": 2.0},
    ]
    res = from_records(records, timestamp_key=None)
    assert res.feature_names == ["cli_0", "cli_1", "cli_2"]
    assert res.synthesized_timestamps is True
    # Synthesized order is a monotonic integer range.
    assert res.logs["arrival_ts"].tolist() == [0, 1]


def test_missing_timestamp_synthesizes_order() -> None:
    records = _mapping_records(n=10)
    del records[3]["timestamp"]  # one row lacks a timestamp -> synthesize all
    res = from_records(records)
    assert res.synthesized_timestamps is True
    assert res.logs["arrival_ts"].tolist() == list(range(10))


def test_per_row_eligibility_is_honored() -> None:
    records = [
        {
            "context": {"x": 1.0},
            "action": "a",
            "reward": 1.0,
            "eligible_actions": ["a", "b"],
        },
        {
            "context": {"x": 2.0},
            "action": "c",
            "reward": 2.0,
            "eligible_actions": ["c"],
        },
    ]
    res = from_records(records, timestamp_key=None)
    assert res.actions == ["a", "b", "c"]
    # Row 0: a, b eligible; c not.
    assert res.logs.loc[0, "a_elig"] == 1
    assert res.logs.loc[0, "b_elig"] == 1
    assert res.logs.loc[0, "c_elig"] == 0
    # Row 1: only c eligible.
    assert res.logs.loc[1, "c_elig"] == 1
    assert res.logs.loc[1, "a_elig"] == 0


def test_chosen_action_not_eligible_raises() -> None:
    records = [
        {
            "context": {"x": 1.0},
            "action": "a",
            "reward": 1.0,
            "eligible_actions": ["b", "c"],
        },
    ]
    with pytest.raises(DataValidationError, match="not in its eligible set"):
        from_records(records, timestamp_key=None)


def test_logged_propensities_are_flagged_not_consumed() -> None:
    records = _mapping_records(n=12)
    for r in records:
        r["propensity"] = 0.33
    res = from_records(records)
    assert res.had_logged_propensities is True
    # The logged propensity must NOT leak into the logs frame.
    assert "propensity" not in res.logs.columns


def test_non_numeric_context_raises() -> None:
    records = [{"context": {"x": "high"}, "action": "a", "reward": 1.0}]
    with pytest.raises(DataValidationError, match=r"not .*numeric"):
        from_records(records, timestamp_key=None)


def test_inconsistent_context_keys_raise() -> None:
    records = [
        {"context": {"x": 1.0}, "action": "a", "reward": 1.0},
        {"context": {"y": 2.0}, "action": "b", "reward": 2.0},
    ]
    with pytest.raises(DataValidationError, match="differ from the first record"):
        from_records(records, timestamp_key=None)


def test_missing_reward_key_raises() -> None:
    records = [{"context": {"x": 1.0}, "action": "a"}]
    with pytest.raises(DataValidationError, match="missing reward key"):
        from_records(records, timestamp_key=None)


def test_empty_trace_raises() -> None:
    with pytest.raises(InsufficientDataError, match="no records"):
        from_records([])


def test_adapter_output_feeds_evaluate_sklearn_models() -> None:
    res = from_records(_mapping_records(n=400, seed=3))
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=res.logs,
        models={"linear": LinearRegression()},
        n_splits=3,
        y_col=res.reward_col,
        policy_train="pre_split",
    )
    assert "V_hat" in artifact.report.columns
    assert (artifact.report["model"] == "linear").any()


def test_from_jsonl_trace_roundtrip(tmp_path) -> None:
    records = _mapping_records(n=20)
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    res = from_jsonl_trace(path)
    assert res.n_records == 20
    skdr_eval.validate_logs(res.logs, y_col=res.reward_col)


def test_from_jsonl_trace_rejects_bad_json(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"context": {"x": 1.0}, "action": "a", "reward": 1.0}\nnot json\n')
    with pytest.raises(DataValidationError, match="not valid JSON"):
        from_jsonl_trace(path)


def test_from_jsonl_trace_rejects_non_object_line(tmp_path) -> None:
    path = tmp_path / "arr.jsonl"
    path.write_text("[1, 2, 3]\n")
    with pytest.raises(DataValidationError, match="must be a JSON object"):
        from_jsonl_trace(path)


def test_summary_is_descriptive() -> None:
    res = from_records(_mapping_records(n=5))
    summary = res.summary()
    assert "5 records" in summary
    assert "timestamps from trace" in summary
    assert "logged propensities absent" in summary


def test_numeric_timestamps_are_used_not_synthesized() -> None:
    records = [
        {"context": {"x": 1.0}, "action": "a", "reward": 1.0, "timestamp": 10},
        {"context": {"x": 2.0}, "action": "a", "reward": 2.0, "timestamp": 20},
    ]
    res = from_records(records)
    assert res.synthesized_timestamps is False
    assert res.logs["arrival_ts"].tolist() == [10.0, 20.0]


def test_unparseable_timestamps_raise() -> None:
    records = [
        {"context": {"x": 1.0}, "action": "a", "reward": 1.0, "timestamp": "nope"},
        {"context": {"x": 2.0}, "action": "a", "reward": 2.0, "timestamp": "nah"},
    ]
    with pytest.raises(DataValidationError, match="could not all be parsed"):
        from_records(records)


def test_non_finite_context_raises() -> None:
    records = [{"context": {"x": float("inf")}, "action": "a", "reward": 1.0}]
    with pytest.raises(DataValidationError, match="non-finite"):
        from_records(records, timestamp_key=None)


def test_first_record_missing_context_raises() -> None:
    with pytest.raises(DataValidationError, match="record 0: missing context key"):
        from_records([{"action": "a", "reward": 1.0}], timestamp_key=None)


def test_scalar_context_raises() -> None:
    records = [{"context": 5, "action": "a", "reward": 1.0}]
    with pytest.raises(DataValidationError, match="must be a mapping or a sequence"):
        from_records(records, timestamp_key=None)


def test_empty_context_has_no_features_raises() -> None:
    records = [{"context": {}, "action": "a", "reward": 1.0}]
    with pytest.raises(DataValidationError, match="context has no features"):
        from_records(records, timestamp_key=None)


def test_later_record_missing_context_raises() -> None:
    records = [
        {"context": {"x": 1.0}, "action": "a", "reward": 1.0},
        {"action": "b", "reward": 2.0},
    ]
    with pytest.raises(DataValidationError, match="record 1: missing context key"):
        from_records(records, timestamp_key=None)


def test_missing_action_key_raises() -> None:
    records = [{"context": {"x": 1.0}, "reward": 1.0}]
    with pytest.raises(DataValidationError, match="missing action key"):
        from_records(records, timestamp_key=None)


def test_non_numeric_reward_raises() -> None:
    records = [{"context": {"x": 1.0}, "action": "a", "reward": "high"}]
    with pytest.raises(DataValidationError, match=r"reward .* is not numeric"):
        from_records(records, timestamp_key=None)


def test_sequence_length_mismatch_raises() -> None:
    records = [
        {"context": [1.0, 2.0], "action": "a", "reward": 1.0},
        {"context": [1.0], "action": "b", "reward": 2.0},
    ]
    with pytest.raises(DataValidationError, match="context length 1 differs"):
        from_records(records, timestamp_key=None)


def test_sequence_non_numeric_element_raises() -> None:
    records = [{"context": [1.0, "x"], "action": "a", "reward": 1.0}]
    with pytest.raises(DataValidationError, match=r"index 1.* is not numeric"):
        from_records(records, timestamp_key=None)
