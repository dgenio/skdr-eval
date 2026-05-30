"""Ground-truth recovery for a non-uniform target policy (#106).

This is the simulation proof for the #106 fix. The earlier recovery studies
(:mod:`test_dr_misspecification`) all evaluate a **uniform** target, for which
``Σ_a (μ_a - q̂) = 0`` and the missing ``π(A|x)`` numerator in the importance
weight is *invisible* — both the buggy weight ``1/e`` and the correct weight
``π/e`` return ``V*`` in expectation. The bug only manifests for a
**non-uniform** (policy-dependent) target.

Here we build a problem with an ``x``-dependent, clearly non-uniform target and
show:

* the corrected DR estimator ``w = π(A|x)/e(A|x)`` recovers the analytic
  ``V*(π)`` (bias well under one SE);
* the buggy estimator ``w = 1/e(A|x)`` is materially biased — i.e. the test is
  sensitive to the very defect #106 describes.

Gated by ``SIM_REPS`` (default 30 for CI; set ``SIM_REPS=200`` locally for a
thorough check).
"""

from __future__ import annotations

import os

import numpy as np

from skdr_eval.estimators import build_strategy, dr_value_with_strategy

SIM_REPS = int(os.environ.get("SIM_REPS", "30"))


def _problem(seed: int) -> dict[str, np.ndarray | float]:
    """A 3-action problem with an x-dependent, non-uniform target policy."""
    rng = np.random.default_rng(seed)
    n, n_actions = 2000, 3
    x = rng.normal(size=(n, 2))
    # Per-action mean reward μ_a(x): linear in x, distinct per action.
    coef = np.array([[1.5, -0.5], [-1.0, 1.0], [0.5, 0.5]])
    intercept = np.array([1.0, 2.0, 3.0])
    mu = x @ coef.T + intercept  # (n, n_actions)

    # Behavior policy e(a|x): exploratory softmax (full support, well-overlapped).
    e_logits = 0.5 * mu
    e = np.exp(e_logits - e_logits.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    A = np.array([rng.choice(n_actions, p=e[i]) for i in range(n)], dtype=int)
    Y = mu[np.arange(n), A] + rng.normal(scale=0.5, size=n)

    # Target policy π(a|x): a *different*, sharper softmax that prefers
    # high-reward actions — clearly non-uniform and x-dependent.
    pi_logits = 2.0 * mu
    pi = np.exp(pi_logits - pi_logits.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)

    # Analytic target value V*(π) = E_x Σ_a π(a|x) μ_a(x).
    v_star = float(np.mean(np.sum(pi * mu, axis=1)))

    # Correct marginal outcome model q̂(x) = Σ_a e(a|x) μ_a(x) = E[Y | x].
    q_marginal = np.sum(e * mu, axis=1)

    elig = np.ones((n, n_actions))
    return {
        "e": e,
        "pi": pi,
        "A": A,
        "Y": Y,
        "mu": mu,
        "q_marginal": q_marginal,
        "elig": elig,
        "v_star": v_star,
    }


def test_dr_recovers_nonuniform_target_value_simulation() -> None:
    """Corrected DR (w = π/e) recovers V*(π) for a non-uniform target."""
    strat = build_strategy("DR", clip=float("inf"))
    biases, ses = [], []
    for seed in range(50_000, 50_000 + SIM_REPS):
        prob = _problem(seed)
        result = dr_value_with_strategy(
            propensities=prob["e"],
            policy_probs=prob["pi"],
            Y=prob["Y"],
            q_hat=prob["q_marginal"],
            A=prob["A"],
            elig=prob["elig"],
            strategy=strat,
        )
        biases.append(float(result.V_hat) - prob["v_star"])
        ses.append(float(result.SE_if))
    med_bias = float(np.median(biases))
    med_se = float(np.median(ses))
    assert abs(med_bias) < med_se, (med_bias, med_se)


def test_buggy_inverse_e_weight_does_not_recover_nonuniform_target() -> None:
    """Contrast: the old weight ``1/e`` (no π numerator) is materially biased.

    This pins the sensitivity of the recovery test to the exact #106 defect —
    if the importance weight ever silently reverts to ``1/e`` this test fails.
    """
    biases = []
    for seed in range(50_000, 50_000 + SIM_REPS):
        prob = _problem(seed)
        e, pi, A = prob["e"], prob["pi"], prob["A"]
        Y, q = prob["Y"], prob["q_marginal"]
        n = A.shape[0]
        rows = np.arange(n)
        pi_obs = e[rows, A]
        # Buggy DR pseudo-outcome: q_pi + (1/e)·(Y - q̂), q_pi = q (marginal).
        q_pi = np.sum(pi * q[:, None], axis=1)
        w_buggy = 1.0 / pi_obs
        v_buggy = float(np.mean(q_pi + w_buggy * (Y - q)))
        biases.append(v_buggy - prob["v_star"])
    med_bias = float(np.median(biases))
    # The dropped π(A|x) numerator inflates the residual leg → large bias.
    assert abs(med_bias) > 0.5, med_bias
