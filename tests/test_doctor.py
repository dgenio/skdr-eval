"""Tests for the :mod:`skdr_eval.doctor` preflight surface (#91)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import skdr_eval
from skdr_eval.doctor import Check, DoctorReport, _check_environment, doctor


def _good_logs(n: int = 800):
    logs, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=0)
    return logs


def _good_pairwise():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=10, n_clients_day=60, n_ops=3, seed=0
    )
    return logs_df, op_daily_df


class TestDoctorBasics:
    def test_returns_report_on_clean_logs(self):
        report = doctor(_good_logs())
        assert isinstance(report, DoctorReport)
        assert isinstance(report.checks, list)
        # No fails on clean synth logs.
        assert report.ok
        assert report.summary in {"pass", "warn"}

    def test_pairwise_clean_inputs(self):
        logs_df, op_df = _good_pairwise()
        report = doctor(
            logs_df, kind="pairwise", op_daily_df=op_df, metric_col="service_time"
        )
        assert report.ok
        # Pairwise schema check is in the checklist.
        assert any(c.name == "schema" and c.status == "pass" for c in report.checks)

    def test_never_raises_on_garbage_input(self):
        # The doctor's contract: returns a report, doesn't raise.
        report = doctor("not a dataframe")  # type: ignore[arg-type]
        assert isinstance(report, DoctorReport)
        assert report.summary == "fail"
        assert not report.ok
        assert any(c.name == "input_type" and c.status == "fail" for c in report.checks)

    def test_idempotent_does_not_mutate(self):
        logs = _good_logs()
        original_cols = list(logs.columns)
        snap = logs.copy(deep=True)
        doctor(logs)
        assert list(logs.columns) == original_cols
        pd.testing.assert_frame_equal(logs, snap)


class TestDoctorChecks:
    def test_missing_action_column_surfaced_as_fail(self):
        logs = _good_logs().drop(columns=["action"])
        report = doctor(logs)
        assert not report.ok
        assert any(c.name == "schema" and c.status == "fail" for c in report.checks)

    def test_nonfinite_outcome_is_flagged(self):
        logs = _good_logs()
        logs.loc[logs.index[0], "service_time"] = np.nan
        report = doctor(logs)
        assert any(
            c.name == "finite_outcomes" and c.status == "fail" for c in report.checks
        )

    def test_pairwise_missing_op_daily_returns_fail(self):
        logs_df, _ = _good_pairwise()
        report = doctor(
            logs_df, kind="pairwise", op_daily_df=None, metric_col="service_time"
        )
        assert not report.ok
        assert any(c.name == "schema" and c.status == "fail" for c in report.checks)

    def test_standard_schema_honors_custom_metric_col(self):
        # General-purpose OPE logs whose reward column is not "service_time"
        # must pass the standard schema check when metric_col names it (#149).
        logs = _good_logs().rename(columns={"service_time": "reward"})
        report = doctor(logs, metric_col="reward")
        assert report.ok
        assert any(c.name == "schema" and c.status == "pass" for c in report.checks)

    def test_small_sample_size_fails(self):
        logs = _good_logs(n=100)  # well below 500 floor
        report = doctor(logs)
        assert not report.ok
        assert any(
            c.name == "sample_size" and c.status == "fail" for c in report.checks
        )

    def test_below_fold_floor_warns(self):
        # 600 rows is above absolute min (500) but below 3 * 500 fold floor (1500).
        logs = _good_logs(n=600)
        report = doctor(logs, n_splits=3)
        # No fail from sample_size — but a warn must appear.
        assert any(
            c.name == "sample_size" and c.status == "warn" for c in report.checks
        )

    def test_positivity_warns_with_bad_propensities(self):
        logs = _good_logs()
        logs["propensity"] = 0.0001  # all rows below ε
        report = doctor(logs)
        assert any(c.name == "positivity" and c.status == "warn" for c in report.checks)

    def test_positivity_skipped_without_propensity_column(self):
        logs = _good_logs()
        # No 'propensity' column → warn with message about skip
        report = doctor(logs)
        assert any(
            c.name == "positivity" and c.status == "warn" and "skipped" in c.message
            for c in report.checks
        )

    def test_duplicates_detected_in_pairwise(self):
        logs_df, op_df = _good_pairwise()
        logs_df = pd.concat([logs_df, logs_df.head(5)], ignore_index=True)
        report = doctor(
            logs_df, kind="pairwise", op_daily_df=op_df, metric_col="service_time"
        )
        assert any(c.name == "duplicates" and c.status == "warn" for c in report.checks)


class TestDoctorReportRendering:
    def test_to_dict_round_trip_keys(self):
        report = doctor(_good_logs())
        d = report.to_dict()
        assert set(d.keys()) >= {"ok", "summary", "checks"}
        assert d["ok"] is True or d["ok"] is False
        assert all("name" in c and "status" in c for c in d["checks"])

    def test_to_markdown_returns_string_with_table(self):
        report = doctor(_good_logs())
        md = report.to_markdown()
        assert "| Status | Check | Message | Fix hint |" in md
        assert "**Overall:**" in md

    def test_to_text_handles_color_argument(self):
        report = doctor(_good_logs())
        text_plain = report.to_text(color=False)
        text_color = report.to_text(color=True)
        assert "skdr-eval doctor" in text_plain
        assert "skdr-eval doctor" in text_color


class TestCheckDataclass:
    def test_check_to_dict_keys(self):
        c = Check(name="foo", status="pass", message="ok", fix_hint="", category="env")
        d = c.to_dict()
        assert d["name"] == "foo"
        assert d["status"] == "pass"

    def test_summary_is_worst_status(self):
        # A report containing one fail must summarize as fail regardless of
        # later passes.
        report = doctor("not a dataframe")  # type: ignore[arg-type]
        assert report.summary == "fail"


class TestDoctorBadKwargs:
    def test_invalid_kind_via_pairwise_path(self):
        # Doctor itself uses Literal kind, but a wrong kind string still must
        # not raise from inside; pairwise schema branch handles missing op_df.
        logs = _good_logs()
        # Use the actual Literal values; downstream coverage of the CLI's bad
        # --kind value lives in test_cli.py.
        report = doctor(logs, kind="standard")
        assert isinstance(report, DoctorReport)

    def test_unknown_kind_returns_fail(self):
        """Doctor returns fail (not raise) when kind is unrecognized."""
        logs = _good_logs()
        report = doctor(logs, kind="bogus")  # type: ignore[arg-type]
        assert report.summary == "fail"
        assert any(
            "kind" in c.name.lower() or "bogus" in c.message for c in report.checks
        )


def test_environment_check_fails_below_minimum(monkeypatch) -> None:
    """The environment check fails when the running Python is below the
    declared minimum. Forced here by raising the floor above the runtime so
    the fail branch is exercised on the supported (3.10+) matrix."""
    # ``skdr_eval.doctor`` the package attribute is shadowed by the re-exported
    # ``doctor`` function, so reach the real module via ``sys.modules``.
    monkeypatch.setattr(sys.modules["skdr_eval.doctor"], "_PYTHON_MIN", (3, 99))
    check = _check_environment()
    assert check.status == "fail"
    assert check.category == "environment"
    assert "below" in check.message
