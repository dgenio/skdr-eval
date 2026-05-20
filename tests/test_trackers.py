"""Tests for the :mod:`skdr_eval.trackers` package (#93)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.trackers import FileTracker, NullTracker, Tracker
from skdr_eval.trackers.aim import AimTracker
from skdr_eval.trackers.mlflow import MLflowTracker
from skdr_eval.trackers.wandb import WandbTracker

if TYPE_CHECKING:
    from pathlib import Path


def _build_artifact(seed: int = 0):
    logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=seed)
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=seed)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=seed,
        policy_train="pre_split",
    )


class TestNullTracker:
    def test_satisfies_protocol(self):
        tracker = NullTracker()
        assert isinstance(tracker, Tracker)

    def test_methods_no_op(self, tmp_path: Path):
        tracker = NullTracker()
        tracker.log_metric("foo", 1.0)
        tracker.log_metric("foo", 1.0, step=5)
        tracker.set_tag("seed", "0")
        # ``log_artifact`` should accept paths but produce no side effects.
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        tracker.log_artifact(sample, artifact_path="sub/path.txt")
        # No new entries beside the original ``sample.txt``.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["sample.txt"]

    def test_context_manager(self):
        with NullTracker() as t:
            t.log_metric("x", 0.0)
        # No raise = pass.

    def test_log_card_no_op(self):
        artifact = _build_artifact()
        card = artifact.card_schema("HGB", estimator="DR")
        NullTracker().log_card(card)


class TestFileTracker:
    def test_creates_root_and_subdirs(self, tmp_path: Path):
        root = tmp_path / "run-1"
        FileTracker(root)
        assert root.is_dir()
        assert (root / "artifacts").is_dir()
        assert (root / "cards").is_dir()

    def test_log_metric_appends_jsonl(self, tmp_path: Path):
        root = tmp_path / "run-2"
        with FileTracker(root) as tracker:
            tracker.log_metric("V_hat", 1.5)
            tracker.log_metric("V_hat", 1.7, step=1)
        metrics_path = root / "metrics.jsonl"
        assert metrics_path.is_file()
        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == 2
        record0 = json.loads(lines[0])
        record1 = json.loads(lines[1])
        assert record0["name"] == "V_hat"
        assert record0["value"] == 1.5
        assert "step" not in record0
        assert record1["step"] == 1

    def test_set_tag_writes_json(self, tmp_path: Path):
        root = tmp_path / "run-3"
        tracker = FileTracker(root)
        tracker.set_tag("seed", "0")
        tracker.set_tag("model", "HGB")
        tags = json.loads((root / "tags.json").read_text())
        assert tags == {"model": "HGB", "seed": "0"}

    def test_log_artifact_copies_file(self, tmp_path: Path):
        root = tmp_path / "run-4"
        src = tmp_path / "source.txt"
        src.write_text("payload")
        tracker = FileTracker(root)
        tracker.log_artifact(src, artifact_path="sub/data.txt")
        assert (root / "artifacts" / "sub" / "data.txt").read_text() == "payload"

    def test_log_artifact_default_name(self, tmp_path: Path):
        root = tmp_path / "run-5"
        src = tmp_path / "source.bin"
        src.write_bytes(b"\x00\x01\x02")
        FileTracker(root).log_artifact(src)
        assert (root / "artifacts" / "source.bin").read_bytes() == b"\x00\x01\x02"

    def test_log_artifact_missing_file_raises(self, tmp_path: Path):
        tracker = FileTracker(tmp_path / "run-6")
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(tmp_path / "does_not_exist.txt")

    def test_log_card_writes_yaml(self, tmp_path: Path):
        root = tmp_path / "run-7"
        tracker = FileTracker(root)
        artifact = _build_artifact()
        card = artifact.card_schema("HGB", estimator="DR")
        tracker.log_card(card)
        out = root / "cards" / "HGB_DR.card.yaml"
        assert out.is_file()
        # Round-trip the YAML to ensure the card was written correctly.
        loaded = skdr_eval.EvaluationCard.from_yaml(out)
        assert loaded == card

    def test_reusing_root_preserves_tags(self, tmp_path: Path):
        root = tmp_path / "run-8"
        FileTracker(root).set_tag("a", "1")
        second = FileTracker(root)
        # Tag round-trip across two trackers on the same root.
        second.set_tag("b", "2")
        tags = json.loads((root / "tags.json").read_text())
        assert tags == {"a": "1", "b": "2"}


class TestEvaluatorTrackerWiring:
    def test_none_default_is_no_op(self, tmp_path: Path):
        """With ``tracker=None`` (default) the evaluator must not write."""
        # Nothing to assert beyond "no raise" — there is no tracker dir to
        # inspect for absence of state.
        artifact = _build_artifact()
        assert isinstance(artifact, skdr_eval.EvaluationArtifact)

    def test_null_tracker_artifact_identical_to_none(self):
        art_a = _build_artifact()
        # Re-run with NullTracker — by construction the per-row metrics are
        # identical because the tracker only observes, never mutates.
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
        art_b = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
            tracker=NullTracker(),
        )
        # Compare V_hat rows — deterministic per seed.
        cols = ["model", "estimator", "V_hat"]
        pd_a = art_a.report[cols].reset_index(drop=True)
        pd_b = art_b.report[cols].reset_index(drop=True)
        # Float equality up to determinism (same seed, same pipeline).
        assert (pd_a["V_hat"] - pd_b["V_hat"]).abs().max() < 1e-12

    def test_file_tracker_writes_metrics_and_cards(self, tmp_path: Path):
        root = tmp_path / "run-9"
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=0)
        models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
        with FileTracker(root) as tracker:
            skdr_eval.evaluate_sklearn_models(
                logs=logs,
                models=models,
                fit_models=True,
                n_splits=3,
                random_state=0,
                policy_train="pre_split",
                tracker=tracker,
            )
        # At least V_hat for DR and SNDR was logged.
        metrics_path = root / "metrics.jsonl"
        assert metrics_path.is_file()
        names = {
            json.loads(line)["name"]
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        }
        assert "HGB/DR/V_hat" in names
        assert "HGB/SNDR/V_hat" in names
        # One card per estimator.
        cards = sorted((root / "cards").iterdir())
        assert len(cards) == 2, [p.name for p in cards]

    def test_pairwise_evaluator_accepts_tracker(self, tmp_path: Path):
        logs_df, op_df = skdr_eval.make_pairwise_synth(
            n_days=4, n_clients_day=80, n_ops=3, seed=0
        )
        models = {"HGB": HistGradientBoostingRegressor(max_iter=15, random_state=0)}
        root = tmp_path / "run-pw"
        tracker = FileTracker(root)
        artifact = skdr_eval.evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=2,
            random_state=0,
            tracker=tracker,
            fit_models=True,
            policy_train="pre_split",
        )
        assert isinstance(artifact, skdr_eval.EvaluationArtifact)
        assert (root / "metrics.jsonl").is_file()


class TestTrackerStubs:
    def test_mlflow_stub_raises_on_construct(self):
        with pytest.raises(NotImplementedError, match=r"mlflow"):
            MLflowTracker()

    def test_wandb_stub_raises_on_construct(self):
        with pytest.raises(NotImplementedError, match=r"wandb"):
            WandbTracker()

    def test_aim_stub_raises_on_construct(self):
        with pytest.raises(NotImplementedError, match=r"aim"):
            AimTracker()
