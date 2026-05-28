"""DR is robust when the outcome model is mis-specified (#129).

The double-robustness property of DR/SNDR says: as long as *either* the
propensity model OR the outcome model is correctly specified, the
estimator is consistent. This study verifies the property by varying
the outcome model q_hat while holding the (true) propensity fixed, then
holding q_hat fixed (correct) while varying the propensity model.

The convention used here is the library's: ``policy_probs`` carries the
target-policy mass; ``q_hat`` is the cross-fitted prediction; the weight
transform returns ``1/π_b``. Under that convention DR is
``E[ q_π(X) + (1/π_b(A|X)) · (Y - q_hat(X, A)) ]`` and the residual term
contributes 0 in expectation whenever ``E[Y - q_hat | X, A] = 0``.

All studies are gated by ``SIM_REPS`` (default 30 for CI; set
``SIM_REPS=200`` locally for a thorough check).
"""

from __future__ import annotations

import os

import numpy as np

from skdr_eval.estimators import build_strategy, dr_value_with_strategy

SIM_REPS = int(os.environ.get("SIM_REPS", "30"))


def _build_problem(seed: int) -> dict[str, np.ndarray | float]:
    """Build a 3-action problem with closed-form V* under uniform target."""
    rng = np.random.default_rng(seed)
    n, n_actions = 1500, 3
    logging = rng.dirichlet(np.full(n_actions, 5.0), size=n)
    A = np.array([rng.choice(n_actions, p=logging[i]) for i in range(n)], dtype=int)
    mu = np.array([1.0, 2.0, 3.0])
    Y = mu[A] + rng.normal(scale=0.5, size=n)
    policy_probs = np.full((n, n_actions), 1.0 / n_actions)
    elig = np.ones((n, n_actions))
    V_star = float(mu.mean())  # uniform target → mean of per-action means
    return {
        "logging": logging,
        "A": A,
        "Y": Y,
        "mu": mu,
        "policy_probs": policy_probs,
        "elig": elig,
        "V_star": V_star,
    }


def _run(
    prob: dict[str, np.ndarray | float],
    *,
    q_hat: np.ndarray,
    logging_pred: np.ndarray,
) -> tuple[float, float]:
    strat = build_strategy("DR", clip=20.0)
    result = dr_value_with_strategy(
        propensities=logging_pred,
        policy_probs=prob["policy_probs"],
        Y=prob["Y"],
        q_hat=q_hat,
        A=prob["A"],
        elig=prob["elig"],
        strategy=strat,
    )
    return float(result.V_hat), float(result.SE_if)


def _q_correct_marginal(prob: dict[str, np.ndarray | float]) -> np.ndarray:
    """Marginal mean reward — collapses the DR residual term in expectation."""
    return np.full_like(prob["A"], float(np.mean(prob["mu"])), dtype=float)


def _q_perturbed(
    prob: dict[str, np.ndarray | float], *, jitter: float, seed: int
) -> np.ndarray:
    """Marginal mean + zero-mean Gaussian jitter — keeps DR residual mean 0."""
    rng = np.random.default_rng(seed)
    n = prob["A"].shape[0]
    base = np.full(n, float(np.mean(prob["mu"])))
    return base + rng.normal(scale=jitter, size=n)


def test_dr_recovers_v_star_when_q_is_correct_simulation() -> None:
    """Sanity: q_hat at the marginal mean → DR recovers V*."""
    biases, ses = [], []
    for seed in range(10_000, 10_000 + SIM_REPS):
        prob = _build_problem(seed)
        v_hat, se = _run(
            prob,
            q_hat=_q_correct_marginal(prob),
            logging_pred=prob["logging"],
        )
        biases.append(v_hat - prob["V_star"])
        ses.append(se)
    med_bias = float(np.median(biases))
    med_se = float(np.median(ses))
    # The marginal-mean q_hat is the canonical "good" baseline used by the
    # library's recovery tests; bias should be well under 1 SE.
    assert abs(med_bias) < med_se, (med_bias, med_se)


def test_dr_survives_noisy_q_when_residual_unbiased_simulation() -> None:
    """Noisy q_hat with E[ε]=0 → DR still recovers V* (variance ↑, bias ≈ 0)."""
    biases, ses = [], []
    for seed in range(20_000, 20_000 + SIM_REPS):
        prob = _build_problem(seed)
        v_hat, se = _run(
            prob,
            q_hat=_q_perturbed(prob, jitter=2.0, seed=seed + 999),
            logging_pred=prob["logging"],
        )
        biases.append(v_hat - prob["V_star"])
        ses.append(se)
    med_bias = float(np.median(biases))
    med_se = float(np.median(ses))
    # Looser bound — jitter has zero mean by construction so DR's bias is
    # still 0 in expectation, but the empirical SE is larger.
    assert abs(med_bias) < 2.0 * med_se, (med_bias, med_se)


def test_dr_survives_wrong_propensity_when_q_correct_simulation() -> None:
    """Wrong propensity, correct q_hat → DR recovers V* via the DM leg.

    With ``q_hat`` set to the marginal mean and ``E[Y - q_hat | X, A] = 0`` by
    construction, the residual term contributes 0 *regardless* of the
    propensity estimate. This is the double-robustness "q saves us"
    direction, demonstrated directly.
    """
    biases = []
    for seed in range(30_000, 30_000 + SIM_REPS):
        prob = _build_problem(seed)
        # Wrong propensity = uniform (very different from the true
        # Dirichlet-sampled logging policy).
        wrong = np.full_like(prob["logging"], 1.0 / prob["logging"].shape[1])
        v_hat, _ = _run(
            prob,
            q_hat=_q_correct_marginal(prob),
            logging_pred=wrong,
        )
        biases.append(v_hat - prob["V_star"])
    med_bias = float(np.median(biases))
    # The DM leg covers us; bias is bounded by the residual jitter.
    assert abs(med_bias) < 0.5, med_bias


def test_dr_breaks_when_q_residual_has_bias_simulation() -> None:
    """A q_hat whose residual has *non-zero* conditional mean breaks DR.

    Concretely we set ``q_hat(x, a) = mu_0`` for every action — this leaves
    a non-zero residual mean on actions 1 and 2 even though e is correct.
    DR cannot save us because the IPS leg multiplies this bias by 1/π_b.
    The point of this test is to confirm the assumption boundary: DR is
    robust to *one* misspecification, not to both.
    """
    biases = []
    for seed in range(40_000, 40_000 + SIM_REPS):
        prob = _build_problem(seed)
        # q_hat = mu[0] everywhere. The residual is (Y - mu[0]) which has
        # conditional mean ``mu_a - mu_0`` != 0 for a != 0.
        n = prob["A"].shape[0]
        q_bad = np.full(n, float(prob["mu"][0]))
        v_hat, _ = _run(prob, q_hat=q_bad, logging_pred=prob["logging"])
        biases.append(v_hat - prob["V_star"])
    med_bias = float(np.median(biases))
    # The bias must be materially non-zero — this is the failure-mode
    # signature we care about. We assert > 0.5 (~25% of V_star).
    assert abs(med_bias) > 0.5, (
        f"expected DR to be biased when q_hat residual has non-zero mean; got"
        f" median bias={med_bias:.3f}"
    )
