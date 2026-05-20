"""skdr-eval: Offline policy evaluation using DR and Stabilized DR."""

import importlib

from . import estimators as estimators
from . import slate as slate
from ._simulation import CoverageResult, simulate_coverage
from .capabilities import get_capabilities
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
from .estimators import (
    ClipTransform,
    DRosShrinkTransform,
    EmbeddingSufficiencyReport,
    EstimatorStrategy,
    IdentityTransform,
    MIPSTransform,
    MRDRWeightedLoss,
    MSEOutcomeLoss,
    OutcomeLoss,
    SwitchTauTransform,
    TransformContext,
    WeightTransform,
    build_strategy,
    dr_value_with_strategy,
    embedding_sufficiency_diagnostic,
    mips_value,
)
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
from .reporting import (
    SCHEMA_VERSION,
    ArtifactSchema,
    DiagnosticGate,
    EvaluationArtifact,
    GateResult,
    Reason,
    Recommendation,
    RecommendationPolicy,
    SupportHealthThresholds,
    attach_warnings,
    build_evaluation_artifact,
    export_results,
    gate_diagnostics,
    load_artifact_json,
    render_evaluation_card,
    summarize_sensitivity,
)
from .slate import (
    SlateGroundTruth,
    SlateResult,
    make_slate_synth,
    pseudo_inverse_ips,
    reward_interaction_ips,
    slate_cascade_dr,
    slate_standard_ips,
)
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
from .validation import validate_logs, validate_pairwise_inputs

try:
    from .visualization import (  # noqa: F401
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
    "SCHEMA_VERSION",
    "ArtifactSchema",
    "BootstrapError",
    "ClipTransform",
    "ConfigManager",
    "ConfigurationError",
    "ConvergenceError",
    "CoverageResult",
    "DRResult",
    "DRosShrinkTransform",
    "DataValidationError",
    "Design",
    "DiagnosticGate",
    "EmbeddingSufficiencyReport",
    "EstimationError",
    "EstimatorStrategy",
    "EvaluationArtifact",
    "EvaluationConfig",
    "GateResult",
    "IdentityTransform",
    "InsufficientDataError",
    "MIPSTransform",
    "MRDRWeightedLoss",
    "MSEOutcomeLoss",
    "MemoryError",
    "ModelConfig",
    "ModelEvaluator",
    "ModelFactory",
    "ModelSelector",
    "ModelValidationError",
    "OutcomeLoss",
    "OutcomeModelError",
    "PairwiseDesign",
    "PairwiseEvaluationError",
    "PolicyInductionError",
    "PropensityDiagnostics",
    "PropensityScoreError",
    "Reason",
    "Recommendation",
    "RecommendationPolicy",
    "SkdrEvalError",
    "SlateGroundTruth",
    "SlateResult",
    "StatisticalTest",
    "SupportHealthThresholds",
    "SwitchTauTransform",
    "TransformContext",
    "VersionError",
    "VisualizationConfig",
    "WeightTransform",
    "__version__",
    "attach_warnings",
    "block_bootstrap_ci",
    "bootstrap_confidence_interval",
    "build_design",
    "build_evaluation_artifact",
    "build_strategy",
    "chi_square_test",
    "create_model_ensemble",
    "dr_value_with_clip",
    "dr_value_with_strategy",
    "embedding_sufficiency_diagnostic",
    "estimators",
    "evaluate_pairwise_models",
    "evaluate_propensity_diagnostics",
    "evaluate_sklearn_models",
    "export_results",
    "fit_outcome_crossfit",
    "fit_propensity_timecal",
    "gate_diagnostics",
    "get_capabilities",
    "get_default_config",
    "get_model_recommendations",
    "induce_policy_from_sklearn",
    "kolmogorov_smirnov_test",
    "load_artifact_json",
    "load_config_from_file",
    "make_pairwise_synth",
    "make_slate_synth",
    "make_synth_logs",
    "mann_whitney_u_test",
    "merge_configs",
    "mips_value",
    "multiple_comparison_correction",
    "permutation_test",
    "power_analysis",
    "pseudo_inverse_ips",
    "render_evaluation_card",
    "reward_interaction_ips",
    "sample_size_calculation",
    "save_config_to_file",
    "simulate_coverage",
    "slate",
    "slate_cascade_dr",
    "slate_standard_ips",
    "summarize_sensitivity",
    "t_test",
    "validate_config",
    "validate_logs",
    "validate_pairwise_inputs",
]

if HAS_VISUALIZATION:
    __all__.extend(
        [
            "create_dashboard",
            "plot_calibration_curve",
            "plot_diagnostics_summary",
            "plot_dr_results",
            "plot_propensity_distribution",
            "plot_roc_curve",
        ]
    )
