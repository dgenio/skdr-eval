"""skdr-eval: Offline policy evaluation using DR and Stabilized DR."""

import contextlib

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

# Version is set by setuptools-scm
__version__: str = "unknown"

with contextlib.suppress(ImportError):
    from ._version import version  # type: ignore[import-untyped]

    __version__ = version

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
