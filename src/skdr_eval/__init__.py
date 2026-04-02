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
from .pairwise import PairwiseDesign
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
    "ConfigurationError",
    "ConvergenceError",
    "DRResult",
    "DataValidationError",
    "Design",
    "EstimationError",
    "InsufficientDataError",
    "MemoryError",
    "ModelValidationError",
    "OutcomeModelError",
    "PairwiseDesign",
    "PairwiseEvaluationError",
    "PolicyInductionError",
    "PropensityDiagnostics",
    "PropensityScoreError",
    "SkdrEvalError",
    "VersionError",
    "__version__",
    "block_bootstrap_ci",
    "build_design",
    "create_dashboard",
    "dr_value_with_clip",
    "evaluate_pairwise_models",
    "evaluate_propensity_diagnostics",
    "evaluate_sklearn_models",
    "fit_outcome_crossfit",
    "fit_propensity_timecal",
    "induce_policy_from_sklearn",
    "make_pairwise_synth",
    "make_synth_logs",
    "plot_calibration_curve",
    "plot_diagnostics_summary",
    "plot_dr_results",
    "plot_propensity_distribution",
    "plot_roc_curve",
]
