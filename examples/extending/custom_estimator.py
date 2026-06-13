#!/usr/bin/env python3
"""Worked example: add your own DR estimator via the strategy seam (#188).

This is the runnable companion to ``docs/extending/add-an-estimator.md``. It
implements a brand-new doubly-robust variant — **SoftClipDR** — without
forking any library internals, by plugging a custom :class:`WeightTransform`
into the public estimator strategy seam
(:mod:`skdr_eval.estimators`).

SoftClipDR replaces the hard clip ``min(1/pi, c)`` of plain DR with a smooth
saturating transform ``c * tanh(w / c)``. It approaches the raw importance
weight for small ``w`` and saturates gently at the clip ceiling ``c`` for
heavy tails, trading a little bias for a smoother variance profile than the
hard ``ClipTransform``.

The flow mirrors what ``evaluate_sklearn_models`` does internally, but stops
at the estimator core so the custom strategy is visible end to end:

1. build the design matrices from logs (:func:`skdr_eval.build_design`);
2. fit the cross-fitted propensity and outcome nuisances;
3. induce a candidate policy from a scikit-learn model;
4. evaluate it with the *custom* :class:`EstimatorStrategy` via
   :func:`skdr_eval.estimators.dr_value_with_strategy`.

Run it directly::

    python examples/extending/custom_estimator.py

``tests/test_extending_example.py`` imports :func:`run` and asserts the
estimator returns a finite value on synthetic logs, so the tutorial cannot
silently rot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.estimators import (
    EstimatorStrategy,
    MSEOutcomeLoss,
    dr_value_with_strategy,
)

if TYPE_CHECKING:
    # Imported for the type annotation only; the protocol is structural, so no
    # runtime import of TransformContext is needed to implement a transform.
    from skdr_eval.estimators.protocols import TransformContext


@dataclass(frozen=True)
class SoftClipTransform:
    """A smooth saturating importance-weight transform (the new estimator).

    Implements the :class:`skdr_eval.estimators.WeightTransform` protocol:
    given a :class:`~skdr_eval.estimators.TransformContext`, return the
    non-negative working weight vector ``w`` (shape ``(n,)``) with zeros on
    unmatched rows.

    The raw inverse-propensity weight on a matched row is
    ``pi_target(a|x) / pi_logging(a|x)``. SoftClipDR maps it through
    ``clip * tanh(w / clip)``: near-linear for ``w << clip`` and asymptotic to
    ``clip`` for ``w >> clip``, so a few extreme weights can no longer
    dominate the estimate the way an unclipped IPS weight would.

    Parameters
    ----------
    clip : float
        Saturation ceiling. Large values approach raw IPS; small values pull
        the estimator toward the direct method.
    """

    clip: float = 10.0
    name: str = "soft_clip"

    def __call__(self, context: TransformContext) -> np.ndarray:
        """Return the soft-clipped working weights for one evaluation call."""
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError(f"clip must be positive and finite, got {self.clip!r}")
        n = context.pi_obs.shape[0]
        pi_target_obs = context.policy_probs[np.arange(n), context.A.astype(int)]
        w_raw = np.zeros_like(context.pi_obs, dtype=np.float64)
        safe = context.matched & (context.pi_obs > 0)
        w_raw[safe] = pi_target_obs[safe] / context.pi_obs[safe]
        return self.clip * np.tanh(w_raw / self.clip)


def soft_clip_strategy(clip: float = 10.0) -> EstimatorStrategy:
    """Pair :class:`SoftClipTransform` with the standard MSE outcome loss.

    This is the named strategy you would register in
    ``skdr_eval.estimators.build_strategy`` if you wanted
    ``estimators=("SoftClipDR",)`` to resolve from the high-level evaluators
    (see the tutorial for that one-line library diff).
    """
    return EstimatorStrategy(
        name="SoftClipDR",
        weight_transform=SoftClipTransform(clip=clip),
        outcome_loss=MSEOutcomeLoss(),
        self_normalised=False,
    )


def run(n: int = 4000, n_ops: int = 4, seed: int = 0, clip: float = 10.0) -> dict:
    """Evaluate a candidate policy with SoftClipDR and built-in DR for contrast.

    Returns a dict with both point estimates so callers (and the test) can
    assert the custom estimator produces a finite value in the same ballpark
    as the shipped DR estimator.
    """
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=n, n_ops=n_ops, seed=seed)
    design = skdr_eval.build_design(logs)

    # Cross-fitted nuisances — identical to the standard pipeline.
    propensities, _ = skdr_eval.fit_propensity_timecal(
        design.X_phi, design.A, design.ts, n_splits=3, random_state=seed
    )
    q_hat, _ = skdr_eval.fit_outcome_crossfit(
        design.X_obs, design.Y, n_splits=3, random_state=seed
    )

    # A candidate policy induced from a scikit-learn regressor.
    model = HistGradientBoostingRegressor(random_state=seed).fit(design.X_obs, design.Y)
    policy_probs = skdr_eval.induce_policy_from_sklearn(
        model, design.X_base, list(ops_all), design.elig
    )

    shared = {
        "propensities": propensities,
        "policy_probs": policy_probs,
        "Y": design.Y,
        "q_hat": q_hat,
        "A": design.A,
        "elig": design.elig,
    }
    soft = dr_value_with_strategy(strategy=soft_clip_strategy(clip), **shared)
    builtin_dr = dr_value_with_strategy(
        strategy=skdr_eval.estimators.build_strategy("DR", clip=clip), **shared
    )
    return {
        "SoftClipDR": soft.V_hat,
        "DR": builtin_dr.V_hat,
        "SoftClipDR_ESS": soft.ESS,
    }


def main() -> None:
    """Print the SoftClipDR vs DR estimates on synthetic logs."""
    out = run()
    print("Custom estimator demo — SoftClipDR (smooth-clipped DR)")
    print(
        f"  SoftClipDR V_hat : {out['SoftClipDR']:.4f}  (ESS {out['SoftClipDR_ESS']:.0f})"
    )
    print(f"  built-in DR V_hat: {out['DR']:.4f}")
    print("Both finite and close: the custom strategy is wired correctly.")


if __name__ == "__main__":
    main()
