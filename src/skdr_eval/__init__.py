"""skdr-eval: Offline policy evaluation using DR and Stabilized DR."""

import importlib

from .core import (
    Design,
    DRResult,
    block_bootstrap_ci,
    build_design,
    dr_value_with_clip,
    evaluate_pairwise_models,
    evaluate_sklearn_models,
    fit_outcome_crossfit,
    fit_propensity_timecal,
    induce_policy_from_sklearn,
)
from .config import (
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
from .pairwise import PairwiseDesign
from .synth import make_pairwise_synth, make_synth_logs

# Version is set by setuptools-scm
__version__: str = "unknown"

try:
    _version_module = importlib.import_module("skdr_eval._version")
    __version__ = _version_module.version
except ImportError:
    pass

__all__ = [
    "DRResult",
    "Design",
    "PairwiseDesign",
    "__version__",
    "block_bootstrap_ci",
    "build_design",
    "dr_value_with_clip",
    "evaluate_pairwise_models",
    "evaluate_sklearn_models",
    "fit_outcome_crossfit",
    "fit_propensity_timecal",
    "induce_policy_from_sklearn",
    "make_pairwise_synth",
    "make_synth_logs",
]
