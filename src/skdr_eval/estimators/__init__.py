"""Composable estimator strategies for DR/SNDR/MRDR/SWITCH-DR/DRos/MIPS.

The protocol seam (:mod:`skdr_eval.estimators.protocols`) decouples the
weight transform from the outcome loss, so new doubly-robust variants land
as small dataclasses instead of forks of ``dr_value_with_clip``.

Public API:

* :class:`WeightTransform`, :class:`OutcomeLoss`, :class:`EstimatorStrategy`,
  :class:`TransformContext` — the protocol layer.
* :class:`ClipTransform`, :class:`IdentityTransform`, :class:`SwitchTauTransform`,
  :class:`DRosShrinkTransform`, :class:`MIPSTransform` — built-in weight
  strategies.
* :class:`MSEOutcomeLoss`, :class:`MRDRWeightedLoss` — built-in outcome-loss
  strategies.
* :func:`dr_value_with_strategy` — strategy-aware estimator core.
* :func:`mips_value`, :func:`embedding_sufficiency_diagnostic` — MIPS
  convenience wrappers (#85).
* :func:`build_strategy` — named-strategy factory used by the high-level
  evaluators.
"""

from __future__ import annotations

from .core import StrategyResult, dr_value_with_strategy
from .mips import (
    EmbeddingSufficiencyReport,
    embedding_sufficiency_diagnostic,
    mips_value,
)
from .outcome_losses import MRDRWeightedLoss, MSEOutcomeLoss
from .protocols import (
    EstimatorStrategy,
    OutcomeLoss,
    TransformContext,
    WeightTransform,
)
from .weight_transforms import (
    ClipTransform,
    DRosShrinkTransform,
    IdentityTransform,
    MIPSTransform,
    SwitchTauTransform,
)

__all__ = [
    "ClipTransform",
    "DRosShrinkTransform",
    "EmbeddingSufficiencyReport",
    "EstimatorStrategy",
    "IdentityTransform",
    "MIPSTransform",
    "MRDRWeightedLoss",
    "MSEOutcomeLoss",
    "OutcomeLoss",
    "StrategyResult",
    "SwitchTauTransform",
    "TransformContext",
    "WeightTransform",
    "build_strategy",
    "dr_value_with_strategy",
    "embedding_sufficiency_diagnostic",
    "mips_value",
]


def build_strategy(
    name: str,
    *,
    clip: float = 10.0,
    tau: float = 5.0,
    lam: float = 1.0,
    action_embedding: object | None = None,
    bandwidth: float = 1.0,
    mips_clip: float = float("inf"),
) -> EstimatorStrategy:
    """Build a named built-in :class:`EstimatorStrategy`.

    Recognised names (case-insensitive after canonicalisation): ``"DR"``,
    ``"SNDR"``, ``"MRDR"``, ``"SWITCH-DR"``, ``"DRos"``, ``"MIPS"``.
    """
    canonical = name.strip().upper().replace("_", "-")
    if canonical == "DR":
        return EstimatorStrategy(
            name="DR",
            weight_transform=ClipTransform(clip=clip),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
    if canonical == "SNDR":
        return EstimatorStrategy(
            name="SNDR",
            weight_transform=ClipTransform(clip=clip),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=True,
        )
    if canonical == "MRDR":
        return EstimatorStrategy(
            name="MRDR",
            weight_transform=ClipTransform(clip=clip),
            outcome_loss=MRDRWeightedLoss(),
            self_normalised=False,
        )
    if canonical == "SWITCH-DR":
        return EstimatorStrategy(
            name="SWITCH-DR",
            weight_transform=SwitchTauTransform(tau=tau),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
    if canonical == "DROS":
        return EstimatorStrategy(
            name="DRos",
            weight_transform=DRosShrinkTransform(lam=lam),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
    if canonical == "MIPS":
        if action_embedding is None:
            raise ValueError(
                "MIPS strategy requires action_embedding=... (an "
                "(n_actions, embed_dim) array)."
            )
        import numpy as np  # noqa: PLC0415

        emb = np.asarray(action_embedding)
        return EstimatorStrategy(
            name="MIPS",
            weight_transform=MIPSTransform(
                action_embedding=emb, bandwidth=bandwidth, clip=mips_clip
            ),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
    raise ValueError(
        f"Unknown estimator name {name!r}. Known: DR, SNDR, MRDR, SWITCH-DR, "
        "DRos, MIPS."
    )
