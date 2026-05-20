"""Synthetic slate / ranking data generator for off-policy evaluation (#75).

The generator emits per-impression logs of length ``slate_size`` together
with per-position click outcomes under a closed-form click model. The
analytic ground-truth value for a uniform target policy is returned so
downstream tests can verify recovery.

Supported click models
----------------------
* ``"cascade"``: a position-dependent attention sequence where the user
  scans top-to-bottom and stops on first click (Craswell et al. 2008).
  Cascade-DR (#75) is unbiased under this model.
* ``"position_bias"``: an inverse-rank attention model — click probability
  at rank ``k`` is ``attractiveness * c / (k + 1)``. Reward-Interaction IPS
  is consistent.
* ``"linear"``: ``Σ_k attractiveness_k * gamma_k`` — straight-line additive
  reward used by Pseudo-Inverse IPS.

All click models share the same ``attractiveness ~ Beta(alpha, beta)``
random field over (impression, item) pairs so the generator's behaviour
is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "SlateGroundTruth",
    "make_slate_synth",
]


@dataclass(frozen=True)
class SlateGroundTruth:
    """Analytic per-policy expected reward for a synthetic slate run.

    Attributes
    ----------
    V_logging : float
        Expected reward under the logging policy (uniform-random slate).
    V_uniform_target : float
        Expected reward under a fixed uniform target policy.
    V_oracle_target : float
        Expected reward under the oracle "rank by attractiveness" target.
    click_model : str
        Click model used to generate the rewards.
    """

    V_logging: float
    V_uniform_target: float
    V_oracle_target: float
    click_model: str


def _position_gamma(slate_size: int, click_model: str) -> np.ndarray:
    """Per-position weight used by the click model."""
    if click_model == "cascade":
        # Geometric decay — user gives up faster down the slate.
        return 0.7 ** np.arange(slate_size, dtype=np.float64)
    if click_model == "position_bias":
        return 1.0 / (np.arange(slate_size, dtype=np.float64) + 1.0)
    if click_model == "linear":
        return np.linspace(1.0, 0.5, slate_size, dtype=np.float64)
    raise ValueError(
        f"unknown click model {click_model!r}; expected 'cascade', "
        "'position_bias', or 'linear'."
    )


def make_slate_synth(
    n_impressions: int = 500,
    n_items: int = 20,
    slate_size: int = 5,
    click_model: Literal["cascade", "position_bias", "linear"] = "cascade",
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, SlateGroundTruth]:
    """Generate a synthetic slate-OPE dataset.

    Parameters
    ----------
    n_impressions : int, default=500
        Number of logged impressions.
    n_items : int, default=20
        Cardinality of the item catalogue. Slates are sampled without
        replacement from ``range(n_items)``.
    slate_size : int, default=5
        Slate length.
    click_model : {"cascade", "position_bias", "linear"}, default="cascade"
        Click / reward model. ``"cascade"`` is the textbook cascade click
        model; ``"position_bias"`` is inverse-rank attention; ``"linear"``
        is a simple per-position additive reward.
    seed : int, default=0
        RNG seed.

    Returns
    -------
    logs : pd.DataFrame
        One row per impression with columns ``impression_id``, ``slate``
        (list of item ids), ``clicks`` (per-position 0/1 array),
        ``reward`` (per-impression scalar), and ``logging_prob`` (uniform
        slate sampling probability).
    attractiveness : np.ndarray, shape (n_impressions, n_items)
        Per-(impression, item) attractiveness used as ground truth.
    truth : SlateGroundTruth
        Analytic policy values for the logging, uniform-target, and oracle
        target policies under the chosen click model.
    """
    if slate_size < 1 or slate_size > n_items:
        raise ValueError(
            f"slate_size must be in [1, n_items={n_items}], got {slate_size}"
        )
    rng = np.random.default_rng(seed)
    gamma = _position_gamma(slate_size, click_model)

    # Per-(impression, item) attractiveness in [0, 1].
    attractiveness = rng.beta(2.0, 5.0, size=(n_impressions, n_items)).astype(
        np.float64
    )

    rows: list[dict[str, object]] = []
    # Logging policy: uniform random permutation of items, take top slate_size.
    logging_prob = 1.0 / _num_slate_permutations(n_items, slate_size)
    rewards = np.empty(n_impressions, dtype=np.float64)

    for i in range(n_impressions):
        slate = rng.permutation(n_items)[:slate_size]
        attr = attractiveness[i, slate]
        # Per-position Bernoulli click given attractiveness.
        if click_model == "cascade":
            # User examines each rank with prob gamma_k (geometric) and
            # clicks with probability attractiveness. Reward is sum of
            # clicks weighted by examination prob.
            click_probs = gamma * attr
            clicks = rng.binomial(1, np.clip(click_probs, 0.0, 1.0))
            reward = float(clicks.sum())
        elif click_model == "position_bias":
            click_probs = gamma * attr
            clicks = rng.binomial(1, np.clip(click_probs, 0.0, 1.0))
            reward = float(clicks.sum())
        else:  # linear
            click_probs = gamma * attr
            clicks = rng.binomial(1, np.clip(click_probs, 0.0, 1.0))
            reward = float((gamma * clicks).sum())
        rewards[i] = reward
        rows.append(
            {
                "impression_id": i,
                "slate": [int(s) for s in slate],
                "clicks": clicks.astype(int).tolist(),
                "reward": reward,
                "logging_prob": logging_prob,
            }
        )

    logs = pd.DataFrame(rows)

    # Analytic policy values under each click model.
    # Logging value is the empirical mean of observed rewards.
    V_logging = float(rewards.mean())
    # Uniform target picks any slate with the same probability — same
    # expectation as the logging policy in expectation; we report the
    # empirical mean to anchor to the dataset.
    V_uniform_target = V_logging

    # Oracle target: rank items by attractiveness within each impression and
    # take the top slate_size, then compute expected reward analytically
    # using gamma.
    oracle_rewards = []
    for i in range(n_impressions):
        top = np.argsort(-attractiveness[i])[:slate_size]
        attr = attractiveness[i, top]
        if click_model == "linear":
            v = float((gamma * gamma * attr).sum())
        else:
            v = float((gamma * attr).sum())  # E[click] per position
        oracle_rewards.append(v)
    V_oracle_target = float(np.mean(oracle_rewards))

    return (
        logs,
        attractiveness,
        SlateGroundTruth(
            V_logging=V_logging,
            V_uniform_target=V_uniform_target,
            V_oracle_target=V_oracle_target,
            click_model=click_model,
        ),
    )


def _num_slate_permutations(n_items: int, slate_size: int) -> float:
    """Number of ordered slates of size ``slate_size`` from ``n_items``."""
    out = 1.0
    for k in range(slate_size):
        out *= n_items - k
    return out
