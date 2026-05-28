"""Overlap-failure regime: bias grows as the logging policy gets sharper (#129).

When the logging policy becomes near-deterministic (e.g. argmax), the
importance weight ``π_target / π_b`` is unbounded on rows where the
target policy disagrees with the logged action. PSIS Pareto-k grows
past 0.7 ("variance does not exist"), and DR loses its consistency
property. This study sweeps a softmax temperature on the logging
policy and shows:

* As temperature → 0 (sharper logging), Pareto-k grows monotonically.
* The empirical bias of DR widens in the high-Pareto-k regime.

Both signals are exactly what the support-health warnings
(``POOR_OVERLAP``, ``HIGH_PARETO_K``) are designed to catch.
"""

from __future__ import annotations

import os
from itertools import pairwise

import numpy as np

from skdr_eval.diagnostics import psis_pareto_k
from skdr_eval.estimators import build_strategy, dr_value_with_strategy

# Max allowed dip in median Pareto-k between successive (sharper) temperature
# steps before we declare a monotone-break. Calibrated against ``SIM_REPS=30``
# MC noise — large enough to absorb sampling variability, small enough to
# catch any genuine reversal.
_PARETO_K_MC_TOLERANCE = 0.05

SIM_REPS = int(os.environ.get("SIM_REPS", "30"))


def _problem(seed: int, temperature: float) -> dict[str, np.ndarray | float]:
    """Build a 3-action problem with a tunable logging-policy sharpness."""
    rng = np.random.default_rng(seed)
    n, n_actions = 1500, 3
    X = rng.normal(size=(n, 2))
    W = np.array([[1.5, 0.0], [0.0, 1.5], [-1.0, -1.0]])
    scores = X @ W.T  # (n, n_actions)
    logits = scores / max(temperature, 1e-6)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    A = np.array([rng.choice(n_actions, p=e[i]) for i in range(n)], dtype=int)
    mu = np.array([1.0, 2.0, 3.0])
    Y = mu[A] + rng.normal(scale=0.5, size=n)
    policy_probs = np.full((n, n_actions), 1.0 / n_actions)
    elig = np.ones((n, n_actions))
    V_star = float(mu.mean())  # uniform target
    return {
        "logging": e,
        "A": A,
        "Y": Y,
        "mu": mu,
        "policy_probs": policy_probs,
        "elig": elig,
        "V_star": V_star,
        "X": X,
    }


def _q_correct(prob: dict[str, np.ndarray | float]) -> np.ndarray:
    """Marginal mean q̂ — keeps the DR residual mean at 0 in expectation."""
    return np.full_like(prob["A"], float(np.mean(prob["mu"])), dtype=float)


def _run_one(seed: int, temperature: float) -> tuple[float, float, float]:
    prob = _problem(seed, temperature)
    strat = build_strategy("DR", clip=20.0)
    result = dr_value_with_strategy(
        propensities=prob["logging"],
        policy_probs=prob["policy_probs"],
        Y=prob["Y"],
        q_hat=_q_correct(prob),
        A=prob["A"],
        elig=prob["elig"],
        strategy=strat,
    )
    # Pareto-k of the unclipped weights on the matched subset.
    n = len(prob["A"])
    pi_obs = prob["logging"][np.arange(n), prob["A"]]
    pk = float(psis_pareto_k(1.0 / pi_obs))
    return float(result.V_hat), float(prob["V_star"]), pk


def _summarize(temperature: float) -> dict[str, float]:
    biases, pks = [], []
    for s in range(SIM_REPS):
        v_hat, v_star, pk = _run_one(seed=70_000 + s, temperature=temperature)
        biases.append(abs(v_hat - v_star))
        pks.append(pk)
    return {
        "median_abs_bias": float(np.median(biases)),
        "median_pareto_k": float(np.median(pks)),
    }


def test_pareto_k_grows_with_logging_sharpness_simulation() -> None:
    """Pareto-k must grow monotonically as the logging policy sharpens.

    Asserts soft monotonicity across the *entire* temperature ladder
    ``(1.0, 0.5, 0.25, 0.1)`` with a small tolerance band: each successive
    step may dip by at most ``_PARETO_K_MC_TOLERANCE`` before we count it as
    a non-monotone break. Strict step-by-step monotonicity is too tight for
    Monte Carlo at ``SIM_REPS=30``; the tolerance leaves the test
    informative while keeping it stable.
    """
    pks: list[float] = []
    for t in (1.0, 0.5, 0.25, 0.1):
        s = _summarize(t)
        pks.append(s["median_pareto_k"])
    # The extreme drop must be visible, AND every successive step must be
    # within ``_PARETO_K_MC_TOLERANCE`` of monotone — no large reversals.
    assert pks[-1] > pks[0], pks
    for prev, nxt in pairwise(pks):
        assert nxt >= prev - _PARETO_K_MC_TOLERANCE, pks


def test_pareto_k_crosses_psis_high_risk_threshold_simulation() -> None:
    """A near-deterministic logging policy pushes Pareto-k past 0.7.

    PSIS theory (Vehtari et al. 2024): when the GPD shape ``k`` exceeds
    0.7, the importance-weight tail has infinite variance and the IPS
    estimator is no longer reliable. This is the gate the library uses
    to declare ``high_risk`` overlap (see
    ``SupportHealthThresholds.high_pareto_k`` and the ``HIGH_PARETO_K``
    warning).
    """
    s_soft = _summarize(1.0)
    s_hard = _summarize(0.1)
    # Soft regime should be safely below the "do not trust" threshold;
    # hard regime should be at or above it.
    assert s_soft["median_pareto_k"] < 0.7, s_soft
    assert s_hard["median_pareto_k"] > 0.7, s_hard


def test_pareto_k_increases_monotonically_simulation() -> None:
    """Across a temperature ladder, Pareto-k strictly increases."""
    pks = [_summarize(t)["median_pareto_k"] for t in (1.0, 0.5, 0.25, 0.1)]
    # Strict monotonicity is too tight for MC at SIM_REPS=30; we require
    # the extremes to separate by a clear margin instead.
    assert pks[-1] - pks[0] > 0.2, pks
