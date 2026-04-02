"""skdr-eval: Offline policy evaluation using DR and Stabilized DR."""

import importlib

from .core import (
    Design,
    DRResult,
    block_bootstrap_ci,
    build_design,
    dr_value_with_clip,
    evaluate_pairwise_models,
    evaluate_propensity_diagnostics,
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
from .diagnostics import PropensityDiagnostics
from .exceptions import (
    BootstrapError,
    ConfigurationError,
    ConvergenceError,
    DataValidationError,
    EstimationError,
    InsufficientDataError,
    MemoryError,
    ModelValidationError,
    OutcomeModelError,
    PairwiseEvaluationError,
    PolicyInductionError,
    PropensityScoreError,
    SkdrEvalError,
    VersionError,
)
from .models import (
    ModelEvaluator,
    ModelFactory,
    ModelSelector,
    create_model_ensemble,
    get_model_recommendations,
)
from .pairwise import PairwiseDesign
from .statistical import (
    StatisticalTest,
    bootstrap_confidence_interval,
    chi_square_test,
    kolmogorov_smirnov_test,
    mann_whitney_u_test,
    multiple_comparison_correction,
    permutation_test,
    power_analysis,
    sample_size_calculation,
    t_test,
)
from .synth import make_pairwise_synth, make_synth_logs

try:
    from .visualization import (
        create_dashboard,
        plot_calibration_curve,
        plot_diagnostics_summary,
        plot_dr_results,
        plot_propensity_distribution,
        plot_roc_curve,
    )

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Version is set by setuptools-scm
__version__: str = "unknown"

try:
    _version_module = importlib.import_module("skdr_eval._version")
    __version__ = _version_module.version
except ImportError:
    pass

__all__ = [
    "HAS_VISUALIZATION",
    "BootstrapError",
    "ConfigManager",
    "ConfigurationError",
    "ConvergenceError",
    "DRResult",
    "DataValidationError",
    "Design",
    "EstimationError",
    "EvaluationConfig",
    "InsufficientDataError",
    "MemoryError",
    "ModelConfig",
    "ModelEvaluator",
    "ModelFactory",
    "ModelSelector",
    "ModelValidationError",
    "OutcomeModelError",
    "PairwiseDesign",
    "PairwiseEvaluationError",
    "PolicyInductionError",
    "PropensityDiagnostics",
    "PropensityScoreError",
    "SkdrEvalError",
    "StatisticalTest",
    "VersionError",
    "VisualizationConfig",
    "__version__",
    "block_bootstrap_ci",
    "bootstrap_confidence_interval",
    "build_design",
    "chi_square_test",
    "create_dashboard",
    "create_model_ensemble",
    "dr_value_with_clip",
    "evaluate_pairwise_models",
    "evaluate_propensity_diagnostics",
    "evaluate_sklearn_models",
    "fit_outcome_crossfit",
    "fit_propensity_timecal",
    "get_default_config",
    "get_model_recommendations",
    "induce_policy_from_sklearn",
    "kolmogorov_smirnov_test",
    "load_config_from_file",
    "make_pairwise_synth",
    "make_synth_logs",
    "mann_whitney_u_test",
    "merge_configs",
    "multiple_comparison_correction",
    "permutation_test",
    "plot_calibration_curve",
    "plot_diagnostics_summary",
    "plot_dr_results",
    "plot_propensity_distribution",
    "plot_roc_curve",
    "power_analysis",
    "sample_size_calculation",
    "save_config_to_file",
    "t_test",
    "validate_config",
]
