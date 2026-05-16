"""Trust-diagnostic simulations: PSIS Pareto-k (#80) and ECE/Brier (#84).

Per ``docs/agent-context/review-checklist.md`` and ``docs/agent-context/invariants.md``
any new statistical primitive must come with a simulation proof that the code
recovers a known ground-truth parameter.  This file contains those proofs:

- ``test_pareto_k_recovers_known_tail_simulation``: draw from a Generalized
  Pareto distribution with known shape ``k_true`` and verify the empirical
  Pareto-k recovers ``k_true`` within Monte-Carlo error.
- ``test_pareto_k_detects_heavy_tail_simulation``: PSIS Pareto-k should
  separate well-behaved (light tail) from problematic (heavy tail) weights —
  this is the same "detection power" gate that issue #81 calls out.
- ``test_ece_zero_under_perfect_calibration_simulation``: under a DGP where
  the propensity model is *exactly* correct, the empirical ECE collapses to 0
  in the large-sample limit.
- ``test_ece_grows_with_miscalibration_simulation``: a calibrated propensity
  model has materially lower ECE than a temperature-distorted one.
"""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.diagnostics import (
    compute_propensity_brier,
    compute_propensity_ece,
    psis_pareto_k,
)

# --------------------------------------------------------------------------- #
# Pareto-k simulation proofs (#80)                                            #
# --------------------------------------------------------------------------- #


def _sample_gpd(
    n: int, k_true: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF sample from the Generalized Pareto distribution.

    For ``k != 0``:  X = sigma/k * ((1 - U)^(-k) - 1).
    For ``k == 0``:  X = -sigma * log(1 - U)  (exponential).
    """
    u = rng.uniform(size=n)
    if abs(k_true) < 1e-12:
        return -sigma * np.log(1.0 - u)
    return (sigma / k_true) * ((1.0 - u) ** (-k_true) - 1.0)


@pytest.mark.parametrize("k_true", [0.3, 0.7, 1.2])
def test_pareto_k_recovers_known_tail_simulation(k_true: float) -> None:
    """Pareto-k estimate must track a known GPD shape parameter.

    Draws ``n_reps`` independent samples of size ``n=2000`` from a known GPD,
    estimates ``k`` on each, and checks that the median recovered ``k`` is
    within ``tol`` of the true ``k``.  We check the *median* (not mean)
    because the Zhang-Stephens estimator has a small finite-sample positive
    bias for ``k > 0.7``, which is documented in Vehtari et al. 2024 Section 2.4.
    """
    rng = np.random.default_rng(seed=2025)
    n = 2000
    n_reps = 30
    tol = 0.18  # MC tolerance for Zhang-Stephens at n=2000 (Vehtari Sec 2.4 Fig 1)

    estimates = []
    for _ in range(n_reps):
        weights = _sample_gpd(n, k_true=k_true, sigma=1.0, rng=rng)
        # Shift so all weights are positive (PSIS inputs ``1/π`` are >0 by
        # construction); GPD draws can be exactly 0 at the threshold.
        weights = weights + 1.0
        k_hat = psis_pareto_k(weights)
        if np.isfinite(k_hat):
            estimates.append(k_hat)

    assert len(estimates) >= n_reps // 2, (
        f"too many NaN estimates for k_true={k_true}: only {len(estimates)}/{n_reps} valid"
    )
    median_k = float(np.median(estimates))
    assert abs(median_k - k_true) < tol, (
        f"Pareto-k recovery off: median={median_k:.3f} vs true={k_true:.3f} (tol={tol})"
    )


def test_pareto_k_detects_heavy_tail_simulation() -> None:
    """PSIS Pareto-k must clearly separate light-tail from heavy-tail weights.

    This is the "detection power" gate: if the diagnostic can't distinguish
    safe from unsafe weight distributions, it provides zero value.  Owen
    (2013) §9.4 motivates this property — IS reliability depends on the
    tail shape, not the mean.
    """
    rng = np.random.default_rng(seed=42)
    n = 2000
    n_reps = 20

    light_estimates = []
    heavy_estimates = []
    for _ in range(n_reps):
        # Light tail: weights ~ Uniform(1, 2) — bounded, finite all moments.
        light_w = rng.uniform(1.0, 2.0, size=n)
        # Heavy tail: GPD with k=1.0 (no mean) shifted to be positive.
        heavy_w = _sample_gpd(n, k_true=1.0, sigma=1.0, rng=rng) + 1.0

        k_light = psis_pareto_k(light_w)
        k_heavy = psis_pareto_k(heavy_w)
        if np.isfinite(k_light):
            light_estimates.append(k_light)
        if np.isfinite(k_heavy):
            heavy_estimates.append(k_heavy)

    median_light = float(np.median(light_estimates))
    median_heavy = float(np.median(heavy_estimates))
    # Light-tail weights should sit well below the 0.7 PSIS gate; heavy-tail
    # weights should clearly fail it.
    assert median_light < 0.5, (
        f"light-tail Pareto-k median {median_light:.3f} should be < 0.5"
    )
    assert median_heavy > 0.7, (
        f"heavy-tail Pareto-k median {median_heavy:.3f} should clear the 0.7 PSIS gate"
    )
    assert median_heavy > median_light + 0.4, (
        f"insufficient detection power: heavy {median_heavy:.3f} vs light {median_light:.3f}"
    )


def test_pareto_k_degenerate_inputs() -> None:
    """Degenerate inputs return ``nan`` rather than crash."""
    rng = np.random.default_rng(seed=0)

    # All-equal weights: tail flat, k undefined.
    assert not np.isfinite(psis_pareto_k(np.full(100, 1.5)))

    # Sample too small.
    assert not np.isfinite(psis_pareto_k(rng.uniform(size=10)))

    # Empty.
    assert not np.isfinite(psis_pareto_k(np.array([])))


# --------------------------------------------------------------------------- #
# ECE / Brier simulation proofs (#84)                                         #
# --------------------------------------------------------------------------- #


def _calibrated_dgp(
    n: int, n_actions: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A DGP where the propensity model is *exactly* the data-generating model.

    Returns ``(propensities, actions)`` such that
    ``actions[i] ~ Categorical(propensities[i])`` — so the propensities are
    the true conditional distribution.  ECE → 0 as ``n → ∞``.
    """
    alpha = rng.uniform(0.5, 2.0, size=n_actions)
    propensities = rng.dirichlet(alpha, size=n)
    # Sample actions from the propensities.  np.random.Generator has
    # vectorized multinomial via 'choice' per row — but that's a Python loop.
    # Use a cumulative-sum + uniform-search trick for vectorization.
    cdf = np.cumsum(propensities, axis=1)
    u = rng.uniform(size=(n, 1))
    actions = (u <= cdf).argmax(axis=1)
    return propensities, actions


def test_ece_zero_under_perfect_calibration_simulation() -> None:
    """ECE must converge to 0 when the model matches the DGP.

    Draws large samples (n=4000) where propensities are *exactly* the true
    conditional distribution, computes empirical ECE, and asserts it is
    small (≤ 0.025 over 8 seeds).
    """
    rng = np.random.default_rng(seed=2026)
    n = 4000
    n_actions = 4
    n_reps = 8

    eces = []
    for _ in range(n_reps):
        propensities, actions = _calibrated_dgp(n, n_actions, rng)
        ece = compute_propensity_ece(propensities, actions, n_bins=15)
        eces.append(ece)

    median_ece = float(np.median(eces))
    # 15-bin ECE on a perfectly-calibrated 4-action draw with n=4000 has a
    # well-understood finite-sample noise floor of roughly 0.01-0.02; require
    # the median below 0.025 to surface any systematic bias in the binning.
    assert median_ece < 0.025, (
        f"ECE under perfect calibration too large: median={median_ece:.4f}"
    )


def test_ece_grows_with_miscalibration_simulation() -> None:
    """A miscalibrated model has materially higher ECE than a calibrated one.

    "Miscalibrated" here means temperature-distorted propensities — multiply
    logits by a factor < 1 (over-smooth) so the model emits over-confident
    middle-of-the-pack probabilities relative to the truth.
    """
    rng = np.random.default_rng(seed=11)
    n = 4000
    n_actions = 4
    n_reps = 8

    calibrated_eces = []
    miscalibrated_eces = []
    for _ in range(n_reps):
        propensities, actions = _calibrated_dgp(n, n_actions, rng)

        # Calibrated: use the true propensities.
        calibrated_eces.append(compute_propensity_ece(propensities, actions, n_bins=15))

        # Miscalibrated: temperature-distort the log probabilities.  Higher
        # temperature flattens to uniform → over-confident relative to what
        # the data actually does.
        temperature = 4.0
        log_probs = np.log(propensities + 1e-12) / temperature
        log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
        skewed = np.exp(log_probs)
        skewed = skewed / skewed.sum(axis=1, keepdims=True)
        miscalibrated_eces.append(compute_propensity_ece(skewed, actions, n_bins=15))

    med_cal = float(np.median(calibrated_eces))
    med_mis = float(np.median(miscalibrated_eces))
    # Calibrated should sit at the noise floor; miscalibrated must clear the
    # 0.10 MISCAL_PROP gate so the warning logic is actually exercised in
    # practice.
    assert med_cal < 0.04, f"calibrated ECE too high: {med_cal:.4f}"
    assert med_mis > 0.10, (
        f"miscalibrated ECE should clear 0.10 gate: got {med_mis:.4f}"
    )
    assert med_mis > med_cal + 0.05, (
        f"miscalibration not detected: calibrated={med_cal:.4f} vs miscal={med_mis:.4f}"
    )


def test_brier_zero_for_one_hot_perfect_predictions() -> None:
    """Brier score is 0 when the propensity puts mass 1 on the observed action."""
    rng = np.random.default_rng(seed=7)
    n = 100
    n_actions = 3
    actions = rng.integers(0, n_actions, size=n)
    propensities = np.zeros((n, n_actions), dtype=float)
    propensities[np.arange(n), actions] = 1.0

    brier = compute_propensity_brier(propensities, actions)
    assert brier == pytest.approx(0.0, abs=1e-12)


def test_brier_two_for_one_hot_wrong_predictions() -> None:
    """Brier score is 2.0 when the propensity puts mass 1 on the wrong action.

    Each sample contributes ``(1 - 0)^2 + (0 - 1)^2 = 2`` to the inner sum,
    averaged over the n samples gives 2.0.
    """
    n = 100
    n_actions = 2
    # Always observe action 0 but predict action 1 with certainty.
    actions = np.zeros(n, dtype=int)
    propensities = np.zeros((n, n_actions), dtype=float)
    propensities[:, 1] = 1.0

    brier = compute_propensity_brier(propensities, actions)
    assert brier == pytest.approx(2.0, abs=1e-12)
