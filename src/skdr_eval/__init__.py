"""skdr-eval: Offline policy evaluation using DR and Stabilized DR."""

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
from .pairwise import PairwiseDesign
from .synth import make_pairwise_synth, make_synth_logs

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"

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
