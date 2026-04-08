"""Tests for config.py — EvaluationConfig, ModelConfig, VisualizationConfig,
ConfigManager, and module-level helpers."""

from pathlib import Path

import pytest

from skdr_eval.config import (
    ConfigManager,
    EvaluationConfig,
    ModelConfig,
    VisualizationConfig,
    get_default_config,
    load_config_from_file,
    merge_configs,
    save_config_to_file,
    validate_config,
)
from skdr_eval.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.n_splits == 3
        assert cfg.n_boot == 400
        assert cfg.alpha == 0.05
        assert cfg.clip_grid == [2, 5, 10, 20, 50, float("inf")]

    def test_mutable_default_independent(self):
        """Two instances must not share the same clip_grid list."""
        a = EvaluationConfig()
        b = EvaluationConfig()
        assert a.clip_grid is not b.clip_grid

    def test_custom_clip_grid(self):
        cfg = EvaluationConfig(clip_grid=[5, 10, float("inf")])
        assert cfg.clip_grid == [5, 10, float("inf")]

    def test_invalid_n_splits(self):
        with pytest.raises(ConfigurationError, match="n_splits"):
            EvaluationConfig(n_splits=1)

    def test_invalid_min_ess_frac(self):
        with pytest.raises(ConfigurationError, match="min_ess_frac"):
            EvaluationConfig(min_ess_frac=1.5)

    def test_invalid_n_boot(self):
        with pytest.raises(ConfigurationError, match="n_boot"):
            EvaluationConfig(n_boot=10)

    def test_invalid_alpha(self):
        with pytest.raises(ConfigurationError, match="alpha"):
            EvaluationConfig(alpha=0.0)

    def test_invalid_policy_train_frac(self):
        with pytest.raises(ConfigurationError, match="policy_train_frac"):
            EvaluationConfig(policy_train_frac=1.1)

    def test_invalid_topk(self):
        with pytest.raises(ConfigurationError, match="topk"):
            EvaluationConfig(topk=0)

    def test_invalid_neg_per_pos(self):
        with pytest.raises(ConfigurationError, match="neg_per_pos"):
            EvaluationConfig(neg_per_pos=0)

    def test_invalid_chunk_pairs(self):
        with pytest.raises(ConfigurationError, match="chunk_pairs"):
            EvaluationConfig(chunk_pairs=100)

    def test_invalid_log_level(self):
        with pytest.raises(ConfigurationError, match="log_level"):
            EvaluationConfig(log_level="VERBOSE")

    def test_no_visualization_fields(self):
        """figsize/dpi/style fields were removed from EvaluationConfig."""
        cfg = EvaluationConfig()
        assert not hasattr(cfg, "figsize")
        assert not hasattr(cfg, "style")


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_type == "logistic"
        assert cfg.hyperparameters == {}
        assert cfg.cv_folds == 5

    def test_mutable_default_independent(self):
        """Two instances must not share the same hyperparameters dict."""
        a = ModelConfig()
        b = ModelConfig()
        assert a.hyperparameters is not b.hyperparameters

    def test_invalid_task_type(self):
        with pytest.raises(ConfigurationError, match="task_type"):
            ModelConfig(task_type="clustering")

    def test_invalid_cv_folds(self):
        with pytest.raises(ConfigurationError, match="cv_folds"):
            ModelConfig(cv_folds=1)

    def test_invalid_test_size(self):
        with pytest.raises(ConfigurationError, match="test_size"):
            ModelConfig(test_size=1.0)

    def test_invalid_n_jobs(self):
        with pytest.raises(ConfigurationError, match="n_jobs"):
            ModelConfig(n_jobs=-2)


# ---------------------------------------------------------------------------
# VisualizationConfig
# ---------------------------------------------------------------------------


class TestVisualizationConfig:
    def test_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.dpi == 300
        assert cfg.font_size == 12

    def test_invalid_dpi(self):
        with pytest.raises(ConfigurationError, match="dpi"):
            VisualizationConfig(dpi=10)

    def test_invalid_font_size(self):
        with pytest.raises(ConfigurationError, match="font_size"):
            VisualizationConfig(font_size=4)

    def test_invalid_save_format(self):
        with pytest.raises(ConfigurationError, match="save_format"):
            VisualizationConfig(save_format="bmp")


# ---------------------------------------------------------------------------
# ConfigManager — save/load roundtrips
# ---------------------------------------------------------------------------


class TestConfigManagerRoundtrip:
    def test_evaluation_config_roundtrip(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        original = EvaluationConfig(n_splits=5, n_boot=200)
        mgr.save_evaluation_config(original)
        loaded = mgr.load_evaluation_config()
        assert loaded.n_splits == 5
        assert loaded.n_boot == 200

    def test_evaluation_config_roundtrip_custom_file(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        original = EvaluationConfig(n_splits=4)
        custom = tmp_path / "custom_eval.yaml"
        mgr.save_evaluation_config(original, filename=str(custom))
        loaded = mgr.load_evaluation_config(filename=str(custom))
        assert loaded.n_splits == 4

    def test_model_config_roundtrip(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        original = ModelConfig(model_type="rf", cv_folds=3)
        mgr.save_model_config(original)
        loaded = mgr.load_model_config()
        assert loaded.model_type == "rf"
        assert loaded.cv_folds == 3

    def test_visualization_config_roundtrip(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        original = VisualizationConfig(dpi=150, font_size=10)
        mgr.save_visualization_config(original)
        loaded = mgr.load_visualization_config()
        assert loaded.dpi == 150
        assert loaded.font_size == 10
        assert loaded.figsize == original.figsize

    def test_global_config_roundtrip(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        data = {"version": "2.0.0", "debug": True}
        mgr.save_global_config(data)
        loaded = mgr.load_global_config()
        assert loaded["version"] == "2.0.0"
        assert loaded["debug"] is True

    def test_missing_file_returns_default_eval(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load_evaluation_config()
        assert isinstance(cfg, EvaluationConfig)

    def test_missing_file_returns_default_model(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load_model_config()
        assert isinstance(cfg, ModelConfig)

    def test_missing_file_returns_default_viz(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load_visualization_config()
        assert isinstance(cfg, VisualizationConfig)

    def test_missing_global_returns_empty_dict(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load_global_config()
        assert cfg == {}

    def test_create_default_configs(self, tmp_path):
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.create_default_configs()
        assert mgr.eval_config_file.exists()
        assert mgr.model_config_file.exists()
        assert mgr.viz_config_file.exists()
        assert mgr.global_config_file.exists()

    def test_clip_grid_yaml_roundtrip(self, tmp_path):
        """list[float] with float('inf') must survive YAML serialisation."""
        mgr = ConfigManager(config_dir=tmp_path)
        original = EvaluationConfig()
        mgr.save_evaluation_config(original)
        loaded = mgr.load_evaluation_config()
        # inf serialises as .inf in YAML and round-trips correctly
        assert loaded.clip_grid == original.clip_grid

    def test_figsize_yaml_edge_case(self, tmp_path):
        """VisualizationConfig.figsize is list[int]; YAML roundtrip must preserve values."""
        mgr = ConfigManager(config_dir=tmp_path)
        original = VisualizationConfig()
        mgr.save_visualization_config(original)
        loaded = mgr.load_visualization_config()
        assert loaded.figsize == original.figsize
        assert isinstance(loaded.figsize, list)


# ---------------------------------------------------------------------------
# load_config_from_file / save_config_to_file
# ---------------------------------------------------------------------------


class TestFileHelpers:
    def test_save_load_yaml(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        data = {"key": "value", "num": 42}
        save_config_to_file(data, f)
        loaded = load_config_from_file(f)
        assert loaded == data

    def test_save_load_json(self, tmp_path):
        f = tmp_path / "cfg.json"
        data = {"key": "value", "num": 42}
        save_config_to_file(data, f)
        loaded = load_config_from_file(f)
        assert loaded == data

    def test_load_yml_extension(self, tmp_path):
        f = tmp_path / "cfg.yml"
        f.write_text("key: value\n")
        loaded = load_config_from_file(f)
        assert loaded["key"] == "value"

    def test_unsupported_format_save(self, tmp_path):
        f = tmp_path / "cfg.toml"
        with pytest.raises(ConfigurationError, match="Unsupported file format"):
            save_config_to_file({}, f)

    def test_unsupported_format_load(self, tmp_path):
        f = tmp_path / "cfg.toml"
        f.write_text("[section]\nkey = 'value'\n")
        with pytest.raises(ConfigurationError, match="Unsupported file format"):
            load_config_from_file(f)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            load_config_from_file(tmp_path / "nonexistent.yaml")

    def test_accepts_path_object(self, tmp_path):
        f = tmp_path / "cfg.json"
        save_config_to_file({"a": 1}, f)
        loaded = load_config_from_file(Path(f))
        assert loaded == {"a": 1}

    def test_accepts_str_path(self, tmp_path):
        f = str(tmp_path / "cfg.json")
        save_config_to_file({"a": 1}, f)
        assert load_config_from_file(f) == {"a": 1}


# ---------------------------------------------------------------------------
# merge_configs
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_simple_merge(self):
        a = {"x": 1, "y": 2}
        b = {"y": 99, "z": 3}
        assert merge_configs(a, b) == {"x": 1, "y": 99, "z": 3}

    def test_nested_merge(self):
        a = {"eval": {"n_splits": 3, "alpha": 0.05}}
        b = {"eval": {"n_splits": 5}}
        result = merge_configs(a, b)
        assert result["eval"]["n_splits"] == 5
        assert result["eval"]["alpha"] == 0.05

    def test_three_way_merge(self):
        a = {"x": 1}
        b = {"y": 2}
        c = {"z": 3}
        assert merge_configs(a, b, c) == {"x": 1, "y": 2, "z": 3}

    def test_empty_merge(self):
        assert merge_configs({}, {}) == {}

    def test_single_config(self):
        a = {"x": 1}
        assert merge_configs(a) == {"x": 1}
        assert merge_configs(a) is not a  # returns a new dict

    def test_non_dict_value_overrides(self):
        a = {"x": [1, 2, 3]}
        b = {"x": [4, 5]}
        result = merge_configs(a, b)
        assert result["x"] == [4, 5]


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_empty(self):
        assert validate_config({}) is True

    def test_valid_evaluation(self):
        cfg = {"evaluation": {"n_splits": 4, "n_boot": 200}}
        assert validate_config(cfg) is True

    def test_valid_model(self):
        assert validate_config({"model": {"model_type": "rf"}}) is True

    def test_valid_visualization(self):
        assert validate_config({"visualization": {"dpi": 150}}) is True

    def test_invalid_evaluation(self):
        assert validate_config({"evaluation": {"n_splits": 1}}) is False

    def test_invalid_model(self):
        assert validate_config({"model": {"task_type": "bad"}}) is False

    def test_invalid_visualization(self):
        assert validate_config({"visualization": {"dpi": 10}}) is False

    def test_wrong_key_type_returns_false(self):
        """Bad key names cause TypeError — must return False, not propagate."""
        assert validate_config({"evaluation": {"nonexistent_field": 999}}) is False

    def test_programming_error_propagates(self):
        """Exceptions beyond ConfigurationError/TypeError/ValueError propagate."""

        original = EvaluationConfig.__post_init__

        def broken_post_init(self):
            raise RuntimeError("internal bug")

        EvaluationConfig.__post_init__ = broken_post_init
        try:
            with pytest.raises(RuntimeError, match="internal bug"):
                validate_config({"evaluation": {}})
        finally:
            EvaluationConfig.__post_init__ = original


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    def test_structure(self):
        d = get_default_config()
        assert set(d.keys()) >= {"evaluation", "model", "visualization", "global"}

    def test_evaluation_roundtrip(self):
        d = get_default_config()
        cfg = EvaluationConfig(**d["evaluation"])
        assert cfg == EvaluationConfig()

    def test_model_roundtrip(self):
        d = get_default_config()
        cfg = ModelConfig(**d["model"])
        assert cfg == ModelConfig()
