"""Slate / top-K off-policy estimators and synthetic data (#75).

Public API:

* :func:`make_slate_synth` — synthetic generator with analytic ground truth.
* :class:`SlateGroundTruth` — return tuple from :func:`make_slate_synth`.
* :func:`slate_standard_ips`, :func:`reward_interaction_ips`,
  :func:`pseudo_inverse_ips`, :func:`slate_cascade_dr` — estimators.
* :class:`SlateResult` — per-estimator headline + diagnostics.
"""

from __future__ import annotations

from .estimators import (
    SlateResult,
    pseudo_inverse_ips,
    reward_interaction_ips,
    slate_cascade_dr,
    slate_standard_ips,
)
from .evaluate import evaluate_slate_models
from .synth import SlateGroundTruth, make_slate_synth

__all__ = [
    "SlateGroundTruth",
    "SlateResult",
    "evaluate_slate_models",
    "make_slate_synth",
    "pseudo_inverse_ips",
    "reward_interaction_ips",
    "slate_cascade_dr",
    "slate_standard_ips",
]
