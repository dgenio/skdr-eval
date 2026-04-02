"""Configuration management for skdr-eval library."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .exceptions import ConfigurationError

logger = logging.getLogger("skdr_eval")


@dataclass
class EvaluationConfig:
    """Configuration for offline policy evaluation."""

    # Data parameters
    n_splits: int = 3
    random_state: int = 42
    min_ess_frac: float = 0.02

    # Clipping parameters
    clip_grid: list[float] = field(
        default_factory=lambda: [2, 5, 10, 20, 50, float("inf")]
    )

    # Bootstrap parameters
    n_boot: int = 400
    block_len: Optional[int] = None
    alpha: float = 0.05

    # Model parameters
    outcome_estimator: str = "hgb"
    propensity_estimator: str = "logistic"

    # Policy training parameters
    policy_train: str = "all"
    policy_train_frac: float = 0.85

    # Pairwise evaluation parameters
    strategy: str = "auto"
    propensity: str = "condlogit"
    topk: int = 20
    neg_per_pos: int = 5
    chunk_pairs: int = 2_000_000

    # Logging parameters
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:  # noqa: PLR2004
            raise ConfigurationError("n_splits must be at least 2")

        if not 0 < self.min_ess_frac < 1:
            raise ConfigurationError("min_ess_frac must be between 0 and 1")

        if self.n_boot < 100:  # noqa: PLR2004
            raise ConfigurationError("n_boot must be at least 100")

        if not 0 < self.alpha < 1:
            raise ConfigurationError("alpha must be between 0 and 1")

        if not 0 < self.policy_train_frac < 1:
            raise ConfigurationError("policy_train_frac must be between 0 and 1")

        if self.topk < 1:
            raise ConfigurationError("topk must be at least 1")

        if self.neg_per_pos < 1:
            raise ConfigurationError("neg_per_pos must be at least 1")

        if self.chunk_pairs < 1000:  # noqa: PLR2004
            raise ConfigurationError("chunk_pairs must be at least 1000")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Model type
    model_type: str = "logistic"
    task_type: str = "classification"

    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: Optional[str] = None

    # Training parameters
    random_state: int = 42
    n_jobs: int = -1

    # Evaluation parameters
    test_size: float = 0.2
    stratify: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        valid_task_types = ["classification", "regression"]
        if self.task_type not in valid_task_types:
            raise ConfigurationError(f"task_type must be one of {valid_task_types}")

        if self.cv_folds < 2:  # noqa: PLR2004
            raise ConfigurationError("cv_folds must be at least 2")

        if not 0 < self.test_size < 1:
            raise ConfigurationError("test_size must be between 0 and 1")

        if self.n_jobs < -1:
            raise ConfigurationError("n_jobs must be -1 or positive integer")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    # Figure settings
    figsize: list[int] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    style: str = "default"

    # Color settings
    color_palette: str = "husl"
    colormap: str = "viridis"

    # Font settings
    font_size: int = 12
    title_size: int = 16
    label_size: int = 14

    # Layout settings
    tight_layout: bool = True
    bbox_inches: str = "tight"

    # Save settings
    save_format: str = "png"
    save_dpi: int = 300

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.dpi < 72:  # noqa: PLR2004
            raise ConfigurationError("dpi must be at least 72")

        if self.font_size < 8:  # noqa: PLR2004
            raise ConfigurationError("font_size must be at least 8")

        if self.title_size < 8:  # noqa: PLR2004
            raise ConfigurationError("title_size must be at least 8")

        if self.label_size < 8:  # noqa: PLR2004
            raise ConfigurationError("label_size must be at least 8")

        valid_formats = ["png", "pdf", "svg", "jpg", "jpeg"]
        if self.save_format.lower() not in valid_formats:
            raise ConfigurationError(f"save_format must be one of {valid_formats}")


class ConfigManager:
    """Manager class for handling configuration files and settings."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Parameters
        ----------
        config_dir : str or Path, optional
            Directory to store configuration files. If None, uses default.
        """
        if config_dir is None:
            config_dir = Path.home() / ".skdr_eval"
        else:
            config_dir = Path(config_dir)

        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration files
        self.eval_config_file = self.config_dir / "evaluation.yaml"
        self.model_config_file = self.config_dir / "model.yaml"
        self.viz_config_file = self.config_dir / "visualization.yaml"
        self.global_config_file = self.config_dir / "global.yaml"

    def save_evaluation_config(
        self, config: EvaluationConfig, filename: Union[str, Path, None] = None
    ) -> None:
        """Save evaluation configuration to file.

        Parameters
        ----------
        config : EvaluationConfig
            Configuration to save.
        filename : str, optional
            Custom filename. If None, uses default.
        """
        path = self.eval_config_file if filename is None else Path(filename)

        config_dict = asdict(config)
        self._save_yaml(config_dict, path)
        logger.info(f"Evaluation configuration saved to {path}")

    def load_evaluation_config(
        self, filename: Union[str, Path, None] = None
    ) -> EvaluationConfig:
        """Load evaluation configuration from file.

        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses default.

        Returns
        -------
        EvaluationConfig
            Loaded configuration.
        """
        path = self.eval_config_file if filename is None else Path(filename)

        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using defaults")
            return EvaluationConfig()

        config_dict = self._load_yaml(path)
        return EvaluationConfig(**config_dict)

    def save_model_config(
        self, config: ModelConfig, filename: Union[str, Path, None] = None
    ) -> None:
        """Save model configuration to file.

        Parameters
        ----------
        config : ModelConfig
            Configuration to save.
        filename : str, optional
            Custom filename. If None, uses default.
        """
        path = self.model_config_file if filename is None else Path(filename)

        config_dict = asdict(config)
        self._save_yaml(config_dict, path)
        logger.info(f"Model configuration saved to {path}")

    def load_model_config(self, filename: Union[str, Path, None] = None) -> ModelConfig:
        """Load model configuration from file.

        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses default.

        Returns
        -------
        ModelConfig
            Loaded configuration.
        """
        path = self.model_config_file if filename is None else Path(filename)

        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using defaults")
            return ModelConfig()

        config_dict = self._load_yaml(path)
        return ModelConfig(**config_dict)

    def save_visualization_config(
        self, config: VisualizationConfig, filename: Union[str, Path, None] = None
    ) -> None:
        """Save visualization configuration to file.

        Parameters
        ----------
        config : VisualizationConfig
            Configuration to save.
        filename : str, optional
            Custom filename. If None, uses default.
        """
        path = self.viz_config_file if filename is None else Path(filename)

        config_dict = asdict(config)
        self._save_yaml(config_dict, path)
        logger.info(f"Visualization configuration saved to {path}")

    def load_visualization_config(
        self, filename: Union[str, Path, None] = None
    ) -> VisualizationConfig:
        """Load visualization configuration from file.

        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses default.

        Returns
        -------
        VisualizationConfig
            Loaded configuration.
        """
        path = self.viz_config_file if filename is None else Path(filename)

        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using defaults")
            return VisualizationConfig()

        config_dict = self._load_yaml(path)
        return VisualizationConfig(**config_dict)

    def save_global_config(
        self, config: dict[str, Any], filename: Union[str, Path, None] = None
    ) -> None:
        """Save global configuration to file.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to save.
        filename : str, optional
            Custom filename. If None, uses default.
        """
        path = self.global_config_file if filename is None else Path(filename)

        self._save_yaml(config, path)
        logger.info(f"Global configuration saved to {path}")

    def load_global_config(
        self, filename: Union[str, Path, None] = None
    ) -> dict[str, Any]:
        """Load global configuration from file.

        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, uses default.

        Returns
        -------
        Dict[str, Any]
            Loaded configuration dictionary.
        """
        path = self.global_config_file if filename is None else Path(filename)

        if not path.exists():
            logger.warning(f"Configuration file {path} not found, using empty dict")
            return {}

        return self._load_yaml(path)

    def create_default_configs(self) -> None:
        """Create default configuration files."""
        # Create default evaluation config
        eval_config = EvaluationConfig()
        self.save_evaluation_config(eval_config)

        # Create default model config
        model_config = ModelConfig()
        self.save_model_config(model_config)

        # Create default visualization config
        viz_config = VisualizationConfig()
        self.save_visualization_config(viz_config)

        # Create default global config
        global_config = {
            "version": "1.0.0",
            "created_by": "skdr_eval",
            "description": "Default configuration for skdr_eval library",
        }
        self.save_global_config(global_config)

        logger.info("Default configuration files created")

    def _save_yaml(self, data: dict[str, Any], filename: Path) -> None:
        """Save data to YAML file."""
        try:
            with filename.open("w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save YAML file {filename}: {e}") from e

    def _load_yaml(self, filename: Path) -> dict[str, Any]:
        """Load data from YAML file."""
        try:
            with filename.open() as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML file {filename}: {e}") from e


def get_default_config() -> dict[str, Any]:
    """Get default configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Default configuration.
    """
    return {
        "evaluation": asdict(EvaluationConfig()),
        "model": asdict(ModelConfig()),
        "visualization": asdict(VisualizationConfig()),
        "global": {
            "version": "1.0.0",
            "created_by": "skdr_eval",
            "description": "Default configuration for skdr_eval library",
        },
    }


def load_config_from_file(filename: Union[str, Path]) -> dict[str, Any]:
    """Load configuration from file.

    Parameters
    ----------
    filename : str or Path
        Path to configuration file.

    Returns
    -------
    Dict[str, Any]
        Loaded configuration.
    """
    filename = Path(filename)

    if not filename.exists():
        raise ConfigurationError(f"Configuration file {filename} not found")

    if filename.suffix.lower() == ".yaml" or filename.suffix.lower() == ".yml":
        try:
            with filename.open() as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML file {filename}: {e}") from e
    elif filename.suffix.lower() == ".json":
        try:
            with filename.open() as f:
                return json.load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON file {filename}: {e}") from e
    else:
        raise ConfigurationError(f"Unsupported file format: {filename.suffix}")


def save_config_to_file(config: dict[str, Any], filename: Union[str, Path]) -> None:
    """Save configuration to file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration to save.
    filename : str or Path
        Path to save configuration file.
    """
    filename = Path(filename)

    if filename.suffix.lower() == ".yaml" or filename.suffix.lower() == ".yml":
        try:
            with filename.open("w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save YAML file {filename}: {e}") from e
    elif filename.suffix.lower() == ".json":
        try:
            with filename.open("w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save JSON file {filename}: {e}") from e
    else:
        raise ConfigurationError(f"Unsupported file format: {filename.suffix}")


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Parameters
    ----------
    *configs : Dict[str, Any]
        Configuration dictionaries to merge.

    Returns
    -------
    Dict[str, Any]
        Merged configuration.
    """
    merged: dict[str, Any] = {}

    for config in configs:
        for key, value in config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value

    return merged


def validate_config(config: dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration to validate.

    Returns
    -------
    bool
        True if configuration is valid.
    """
    try:
        # Validate evaluation config if present
        if "evaluation" in config:
            EvaluationConfig(**config["evaluation"])

        # Validate model config if present
        if "model" in config:
            ModelConfig(**config["model"])

        # Validate visualization config if present
        if "visualization" in config:
            VisualizationConfig(**config["visualization"])

        return True
    except (ConfigurationError, TypeError, ValueError) as e:
        logger.error("Configuration validation failed: %s", e)
        return False
