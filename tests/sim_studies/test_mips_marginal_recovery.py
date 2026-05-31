"""Ground-truth recovery for marginalised MIPS under a non-identity kernel (#142).

The #106 recovery study (:mod:`test_policy_value_recovery`) pins the *standard*
DR weight ``π(A|x)/e(A|x)``. It cannot exercise the MIPS bug fixed in #142,
because with an identity kernel MIPS collapses to that same exact-action ratio.
The MIPS defect only manifests for a **non-identity** embedding kernel, where the
weight must be the *marginalised* density ratio

    w[i] = (Σ_a π(a|x_i) k(E_a, E_{A_i})) / (Σ_a' e(a'|x_i) k(E_{a'}, E_{A_i}))

rather than the asymmetric ``π(A_i|x_i) / (Σ_a' e(a'|x_i) k(E_{a'}, E_{A_i}))``
that kept the exact-action target numerator.

DGP. Four actions fall into two embedding clusters ({0,1} and {2,3}); the two
actions in a cluster share an identical embedding, and the per-action reward
``μ_a(x)`` depends on the action *only through its cluster*. The cluster is then
a sufficient statistic, so the cluster-level RBF kernel makes MIPS unbiased for
``V*(π)``. The target policy is non-uniform, x-dependent, and splits mass
unequally *within* a cluster — exactly the regime where the dropped numerator
marginalisation biases the estimate.

We show:

* the corrected (marginalised) MIPS recovers the analytic ``V*(π)`` (bias well
  under one SE);
* the old exact-action numerator (numerator = ``π(A|x)`` instead of the kernel
  marginal) is materially biased — pinning the test's sensitivity to the #142
  defect.

Gated by ``SIM_REPS`` (default 30 for CI; set ``SIM_REPS=200`` locally).
"""

from __future__ import annotations

import os

import numpy as np

from skdr_eval.estimators import build_strategy, dr_value_with_strategy

SIM_REPS = int(os.environ.get("SIM_REPS", "30"))

# Two embedding clusters of two actions each. Actions in a cluster share an
# identical embedding, so a small-bandwidth RBF kernel is block-uniform within
# a cluster and ~0 across clusters.
_EMBEDDING = np.array(
    [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]], dtype=np.float64
)
_BANDWIDTH = 0.1  # << the 14.1 inter-cluster distance -> clean block kernel


def _problem(seed: int) -> dict[str, np.ndarray | float]:
    """A 4-action, 2-cluster problem with reward driven only by the cluster."""
    rng = np.random.default_rng(seed)
    n, n_actions = 2000, 4
    x = rng.normal(size=(n, 2))

    # Per-cluster mean reward m_c(x): linear in x, distinct per cluster.
    cluster_coef = np.array([[1.5, -0.5], [-1.0, 1.0]])
    cluster_intercept = np.array([1.0, 3.0])
    m_cluster = x @ cluster_coef.T + cluster_intercept  # (n, 2)
    cluster_of = np.array([0, 0, 1, 1])
    mu = m_cluster[:, cluster_of]  # (n, n_actions) — reward depends on cluster

    # Behaviour policy e(a|x): exploratory softmax with a reward-independent
    # per-action bias so it splits *unequally within a cluster*. Without this
    # asymmetry the logging policy cannot distinguish the two within-cluster
    # actions, and the exact-action numerator bug stays invisible.
    e_logits = 0.4 * mu + np.array([0.9, -0.9, 0.9, -0.9])
    e = np.exp(e_logits - e_logits.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    A = np.array([rng.choice(n_actions, p=e[i]) for i in range(n)], dtype=int)
    Y = mu[np.arange(n), A] + rng.normal(scale=0.5, size=n)

    # Target policy π(a|x): sharper, x-dependent, and deliberately *unequal*
    # within a cluster (action 0 vs 1, action 2 vs 3) so the exact-action
    # numerator differs from the cluster marginal.
    pi_logits = 1.5 * mu + np.array([0.6, -0.6, 0.6, -0.6])
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


def test_marginal_mips_recovers_clustered_target_value_simulation() -> None:
    """Marginalised MIPS recovers V*(π) under a non-identity (cluster) kernel."""
    strat = build_strategy(
        "MIPS",
        action_embedding=_EMBEDDING,
        bandwidth=_BANDWIDTH,
        kernel="rbf",
        mips_clip=float("inf"),
    )
    biases, ses = [], []
    for seed in range(70_000, 70_000 + SIM_REPS):
        prob = _problem(seed)
        result = dr_value_with_strategy(
            propensities=prob["e"],
            policy_probs=prob["pi"],
            Y=prob["Y"],
            q_hat=prob["q_marginal"],
            A=prob["A"],
            elig=prob["elig"],
            strategy=strat,
            action_embedding=_EMBEDDING,
        )
        biases.append(float(result.V_hat) - prob["v_star"])
        ses.append(float(result.SE_if))
    med_bias = float(np.median(biases))
    med_se = float(np.median(ses))
    assert abs(med_bias) < med_se, (med_bias, med_se)


def test_exact_action_numerator_mips_is_biased() -> None:
    """Contrast: the pre-#142 exact-action numerator is materially biased.

    Reconstructs the old MIPS weight (numerator = the exact observed-action
    target probability, denominator = the kernel marginal of the logging
    policy) and shows it does not recover V*(π). If the numerator ever silently
    reverts to the exact action, this test fails.
    """
    emb = _EMBEDDING
    # Block-uniform kernel: row-normalised RBF at the tiny bandwidth used above.
    sq = np.sum(emb * emb, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * emb @ emb.T, 0.0)
    k = np.exp(-d2 / (2.0 * _BANDWIDTH**2))
    k /= k.sum(axis=1, keepdims=True)

    biases = []
    for seed in range(70_000, 70_000 + SIM_REPS):
        prob = _problem(seed)
        e, pi, A = prob["e"], prob["pi"], prob["A"]
        Y, q = prob["Y"], prob["q_marginal"]
        n = A.shape[0]
        rows = np.arange(n)
        kernel_at_A = k[:, A]  # (n_actions, n)
        p_e_log = np.einsum("na,an->n", e, kernel_at_A)
        # OLD (buggy) numerator: exact observed-action target probability.
        pi_obs_target = pi[rows, A]
        w_buggy = pi_obs_target / p_e_log
        q_pi = np.sum(pi * prob["mu"], axis=1)
        v_buggy = float(np.mean(q_pi + w_buggy * (Y - q)))
        biases.append(v_buggy - prob["v_star"])
    med_bias = float(np.median(biases))
    # The missing numerator marginalisation skews the residual leg materially.
    assert abs(med_bias) > 0.2, med_bias
