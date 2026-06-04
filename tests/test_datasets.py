"""Tests for the public-dataset loaders (#70).

Offline by default: a local OBD-format fixture is fed through ``base_url`` so
the loader, cache, manifest, and schema mapping are all exercised without
network. The real-download path is opt-in via ``SKDR_EVAL_DOWNLOAD_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.datasets import (
    DatasetBundle,
    default_cache_dir,
    load_criteo_counterfactual,
    load_movielens_ope,
    load_obd,
)
from skdr_eval.datasets._cache import fetch_file, sha256_of
from skdr_eval.exceptions import DatasetError


def _write_obd_fixture(
    root: Path, behavior: str = "random", campaign: str = "all", n: int = 80
) -> Path:
    """Write a minimal OBD-format sample under ``root/<behavior>/`` and return root."""
    rng = np.random.RandomState(0)
    src = root / behavior
    src.mkdir(parents=True, exist_ok=True)
    n_items = 5
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="min").astype(str),
            "item_id": rng.randint(0, n_items, size=n),
            "position": rng.randint(1, 4, size=n),
            "click": rng.randint(0, 2, size=n),
            "propensity_score": rng.uniform(0.1, 0.9, size=n),
            "user_feature_0": rng.randint(0, 3, size=n),
            "user_feature_1": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df.to_csv(src / f"{campaign}.csv", index=False)
    pd.DataFrame(
        {
            "item_id": list(range(n_items)),
            "item_feature_0": rng.randint(0, 4, size=n_items),
        }
    ).to_csv(src / "item_context.csv", index=False)
    return root


def test_default_cache_dir_honors_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SKDR_EVAL_CACHE_DIR", str(tmp_path / "cache"))
    assert default_cache_dir() == tmp_path / "cache" / "datasets"
    monkeypatch.delenv("SKDR_EVAL_CACHE_DIR", raising=False)
    assert default_cache_dir() == Path.home() / ".skdr_eval" / "datasets"


def test_fetch_file_copies_local_and_is_cached(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    dest = tmp_path / "cache" / "src.csv"
    out = fetch_file(str(src), dest)
    assert out == dest
    assert dest.read_text(encoding="utf-8") == "a,b\n1,2\n"
    # Second call returns cache without touching the (now-removed) source.
    src.unlink()
    assert fetch_file(str(src), dest).exists()


def test_fetch_file_missing_local_source_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        fetch_file(str(tmp_path / "nope.csv"), tmp_path / "out.csv")


def test_fetch_file_network_error_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="check your network"):
        fetch_file(
            "http://127.0.0.1:0/never.csv",
            tmp_path / "out.csv",
        )


class TestLoadObdOffline:
    def test_returns_schema_valid_bundle(self, tmp_path: Path) -> None:
        base = _write_obd_fixture(tmp_path / "obd_src")
        bundle = load_obd(
            "random", "all", cache_dir=tmp_path / "cache", base_url=str(base)
        )
        assert isinstance(bundle, DatasetBundle)
        logs, ops_all, gt = bundle  # unpacks like make_synth_logs
        assert gt is None
        assert isinstance(ops_all, pd.Index)
        # Schema-valid for the standard evaluator with reward column "click".
        skdr_eval.validate_logs(logs, y_col="click")
        assert "action" in logs.columns
        assert any(c.startswith("cli_user_feature") for c in logs.columns)
        assert "cli_position" in logs.columns

    def test_manifest_written_with_sha256(self, tmp_path: Path) -> None:
        base = _write_obd_fixture(tmp_path / "obd_src")
        cache = tmp_path / "cache"
        load_obd("random", "all", cache_dir=cache, base_url=str(base))
        manifest_path = cache / "obd" / "random" / "all" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataset"] == "open_bandit_dataset"
        assert "CC BY 4.0" in manifest["license"]
        files = manifest["files"]
        log_csv = cache / "obd" / "random" / "all" / "all.csv"
        assert files["all.csv"]["sha256"] == sha256_of(log_csv)

    def test_cache_is_reproducible(self, tmp_path: Path) -> None:
        base = _write_obd_fixture(tmp_path / "obd_src")
        cache = tmp_path / "cache"
        b1 = load_obd("random", "all", cache_dir=cache, base_url=str(base))
        b2 = load_obd("random", "all", cache_dir=cache, base_url=str(base))
        pd.testing.assert_frame_equal(b1.logs, b2.logs)

    def test_max_rows_truncates(self, tmp_path: Path) -> None:
        base = _write_obd_fixture(tmp_path / "obd_src", n=80)
        bundle = load_obd(
            "random", "all", cache_dir=tmp_path / "c", base_url=str(base), max_rows=10
        )
        assert len(bundle.logs) <= 10

    def test_runs_through_evaluator(self, tmp_path: Path) -> None:
        base = _write_obd_fixture(tmp_path / "obd_src", n=80)
        bundle = load_obd("random", "all", cache_dir=tmp_path / "c", base_url=str(base))
        art = skdr_eval.evaluate_sklearn_models(
            logs=bundle.logs,
            models={"hgb": HistGradientBoostingRegressor(max_iter=15, random_state=0)},
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
            y_col="click",
        )
        assert "V_hat" in art.report.columns

    def test_invalid_behavior_policy_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Unknown behavior_policy"):
            load_obd("nope", "all", cache_dir=tmp_path)

    def test_invalid_campaign_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Unknown campaign"):
            load_obd("random", "nope", cache_dir=tmp_path)


class TestEncodingAndCleaning:
    def test_categorical_encoding_is_row_order_independent(self) -> None:
        from skdr_eval.datasets.obd import _encode_features  # noqa: PLC0415

        frame = pd.DataFrame({"f": ["b", "a", "c", "a"]})
        codes = _encode_features(frame)["f"].tolist()
        # Sorted categories: a->0, b->1, c->2 regardless of appearance order.
        assert codes == [1, 0, 2, 0]
        # Shuffling rows must not change a value's code.
        shuffled = frame.iloc[[3, 2, 1, 0]].reset_index(drop=True)
        shuffled_codes = _encode_features(shuffled)["f"].tolist()
        assert shuffled_codes == [0, 2, 0, 1]

    def test_rows_with_nan_features_are_dropped(self, tmp_path: Path) -> None:
        # A non-numeric position value coerces to NaN and the row must drop,
        # keeping the frame schema-valid for the design builder.
        src = tmp_path / "obd_src" / "random"
        src.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "item_id": [0, 1, 2],
                "position": [1, "bad", 3],
                "click": [1, 0, 1],
                "propensity_score": [0.5, 0.5, 0.5],
                "user_feature_0": [1, 2, 3],
            }
        ).to_csv(src / "all.csv", index=False)
        pd.DataFrame({"item_id": [0, 1, 2]}).to_csv(
            src / "item_context.csv", index=False
        )
        logs, _, _ = load_obd(
            "random",
            "all",
            cache_dir=tmp_path / "c",
            base_url=str(tmp_path / "obd_src"),
        )
        assert len(logs) == 2  # the "bad" position row dropped
        assert not logs["cli_position"].isna().any()
        skdr_eval.validate_logs(logs, y_col="click")


class TestStubs:
    def test_criteo_stub_raises(self) -> None:
        with pytest.raises(DatasetError, match="not implemented yet"):
            load_criteo_counterfactual()

    def test_movielens_stub_raises(self) -> None:
        with pytest.raises(DatasetError, match="not implemented yet"):
            load_movielens_ope()


@pytest.mark.skipif(
    os.environ.get("SKDR_EVAL_DOWNLOAD_TESTS") != "1",
    reason="network download test; set SKDR_EVAL_DOWNLOAD_TESTS=1 to run",
)
def test_load_obd_real_download(tmp_path: Path) -> None:
    logs, _, gt = load_obd("random", "all", cache_dir=tmp_path / "cache", max_rows=200)
    skdr_eval.validate_logs(logs, y_col="click")
    assert len(logs) > 0
    assert gt is None
