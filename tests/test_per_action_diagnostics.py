"""Recovery proofs for per-action propensity diagnostics (#131).

Per ``docs/agent-context/review-checklist.md``: any new statistical
primitive must include a simulation that recovers a known ground-truth
parameter. This module shows:

* When the propensity model is *perfectly* calibrated for action ``a``,
  the per-action ECE for ``a`` converges to 0.
* When the propensity model is *miscalibrated* on a single rare action,
  ``per_action[<rare>].ece`` is materially higher than the global ECE.
* The ``rare`` and ``insufficient`` flags fire on a deliberately
  rare-action DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skdr_eval.diagnostics import (
    _binary_ece,
    comprehensive_propensity_diagnostics,
    compute_propensity_ece,
    per_action_propensity_diagnostics,
)
from skdr_eval.exceptions import DataValidationError
from skdr_eval.reporting import attach_warnings


def _dirichlet_dgp(n: int, n_actions: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n_actions, 5.0), size=n)
    a = np.array([rng.choice(n_actions, p=p[i]) for i in range(n)], dtype=int)
    return p, a


def test_per_action_ece_zero_under_perfect_calibration_simulation() -> None:
    """Per-action ECE converges to 0 under a Categorical(π) DGP."""
    p, a = _dirichlet_dgp(n=4000, n_actions=4, seed=11)
    rows = per_action_propensity_diagnostics(p, a, rare_action_floor=0.0)
    eces = [r.ece for r in rows if np.isfinite(r.ece)]
    assert len(eces) == 4, eces
    # Loose bound: with n=4000, n_bins=15, per-action ECE ≤ ~0.1 under
    # perfect calibration. Global ECE is the same statistic on the top-1
    # column, used here as a sanity reference.
    global_ece = compute_propensity_ece(p, a)
    assert max(eces) < 0.10, (eces, global_ece)


def test_per_action_ece_detects_one_bad_arm_simulation() -> None:
    """A targeted distortion of one action's column shows in per-action ECE.

    We take a perfectly-calibrated propensity matrix, then deliberately
    push action 0's column toward 1.0 (over-confident) while leaving
    actions 1-3 alone. The global ECE moves modestly; the per-action
    ECE for action 0 moves a lot. This is precisely the failure mode
    #131 is designed to surface.
    """
    p, a = _dirichlet_dgp(n=3000, n_actions=4, seed=22)

    # Distort only the action-0 column toward 1.0.
    p_bad = p.copy()
    p_bad[:, 0] = np.clip(0.6 + 0.4 * p[:, 0], 0.0, 1.0)
    # Renormalize so each row still sums to 1.
    p_bad /= p_bad.sum(axis=1, keepdims=True)

    rows = per_action_propensity_diagnostics(p_bad, a, rare_action_floor=0.0)
    ece_per_action = [r.ece for r in rows]
    ece_global = compute_propensity_ece(p_bad, a)

    assert np.isfinite(ece_per_action[0])
    # Action 0 is the contaminated arm; its per-action ECE should be the
    # largest by a clear margin (we assert > 2* the next-highest).
    other_max = max(ece_per_action[1:])
    assert ece_per_action[0] > 2 * other_max, (
        f"action-0 ECE={ece_per_action[0]:.3f} not >> others={ece_per_action[1:]}"
    )
    # Global ECE is less sensitive — it should be lower than the
    # contaminated arm's per-action ECE.
    assert ece_per_action[0] > ece_global


def test_per_action_rare_and_insufficient_flags() -> None:
    """A deliberately rare arm should report rare=True + insufficient=True."""
    rng = np.random.default_rng(33)
    n = 1500
    # 4 actions; action 3 has frequency ~0.2% (rare by construction).
    p = np.tile(np.array([0.40, 0.30, 0.299, 0.001]), (n, 1))
    a = np.array([rng.choice(4, p=p[0]) for _ in range(n)], dtype=int)
    rows = per_action_propensity_diagnostics(p, a, rare_action_floor=0.01)
    # action 3 should be flagged both ways.
    rare_flags = [r.rare for r in rows]
    insuff_flags = [r.insufficient for r in rows]
    assert rare_flags[3] is True, rare_flags
    assert insuff_flags[3] is True, insuff_flags
    # The other actions should not be flagged as rare.
    assert not any(rare_flags[:3]), rare_flags


def test_per_action_respects_target_support() -> None:
    """Rare logged actions outside target support are not flagged as rare."""
    rng = np.random.default_rng(44)
    n = 1500
    p = np.tile(np.array([0.40, 0.30, 0.299, 0.001]), (n, 1))
    a = np.array([rng.choice(4, p=p[0]) for _ in range(n)], dtype=int)
    # Target policy never picks action 3 → rarity is irrelevant.
    target_actions = np.zeros(n, dtype=int)  # picks action 0 always
    rows = per_action_propensity_diagnostics(
        p, a, target_actions=target_actions, rare_action_floor=0.01
    )
    assert rows[3].rare is False, "rare-but-unused arm should not be flagged"
    assert rows[3].insufficient is True, "but still insufficient (n_3 ≈ 1)"


def test_per_action_propensity_length_mismatch_raises() -> None:
    """Propensities/actions length mismatch surfaces as DataValidationError."""
    p = np.full((10, 3), 1.0 / 3.0)
    a = np.zeros(9, dtype=int)  # one short on purpose
    with pytest.raises(DataValidationError, match="length"):
        per_action_propensity_diagnostics(p, a)


def test_per_action_propensity_wrong_ndim_raises() -> None:
    """1-D propensities (instead of (n, n_actions)) surface as DataValidationError."""
    p = np.full(10, 0.5)  # 1-D — wrong shape
    a = np.zeros(10, dtype=int)
    with pytest.raises(DataValidationError, match="shape"):
        per_action_propensity_diagnostics(p, a)


def test_comprehensive_propensity_length_mismatch_raises() -> None:
    """Length mismatch in the comprehensive diagnostic raises immediately."""
    p = np.full((10, 3), 1.0 / 3.0)
    a = np.zeros(9, dtype=int)
    with pytest.raises(DataValidationError, match="length"):
        comprehensive_propensity_diagnostics(p, a)


def test_binary_ece_returns_nan_on_length_mismatch() -> None:
    """`_binary_ece` returns nan when probs/labels lengths disagree (defensive)."""
    probs = np.linspace(0.0, 1.0, 10)
    labels = np.zeros(9, dtype=int)
    assert np.isnan(_binary_ece(probs, labels, n_bins=10))


def test_binary_ece_returns_nan_when_all_bins_empty() -> None:
    """`_binary_ece` returns nan when no bin gets any mass (empty input)."""
    probs = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    assert np.isnan(_binary_ece(probs, labels, n_bins=10))


def test_n_rare_and_insufficient_actions_field_computed() -> None:
    """``comprehensive_propensity_diagnostics`` exposes the intersection count.

    When the same action is BOTH rare (below ``rare_action_floor``) AND
    insufficient (fewer than ``_MIN_ACTION_COUNT_DISC=5`` samples), the new
    ``n_rare_and_insufficient_actions`` field counts it. With one such action
    the field equals 1.
    """
    rng = np.random.default_rng(202)
    n = 800
    actions = np.full(n, 0, dtype=int)
    # action 1: 2 samples in n=800 → 0.25% (rare under 1% floor) AND
    # 2 < 5 (insufficient). Single action carries both flags simultaneously.
    actions[:2] = 1
    rng.shuffle(actions)
    p = np.full((n, 2), 0.5)
    diag = comprehensive_propensity_diagnostics(p, actions, target_actions=None)
    assert diag.n_rare_and_insufficient_actions == 1, (
        f"per_action={[(r.action, r.n, r.rare, r.insufficient) for r in diag.per_action]}"
    )


def test_rare_action_no_support_disjoint_does_not_fire() -> None:
    """Disjoint rare/insufficient actions must NOT trigger ``RARE_ACTION_NO_SUPPORT``.

    Regression guard for the audit finding: previously the warning ANDed
    two independent counts (``n_rare > 0`` AND ``n_insufficient > 0``), which
    fires even when the rare and insufficient *actions* are disjoint. The
    rate-vs-count seam makes "rare-but-sufficient" and "insufficient-but-not-rare"
    on the same sample mathematically impossible (rare needs count/n < 1%,
    insufficient needs count < 5 — these constraints can't both hold for
    different actions at any single ``n``). So drive the warning emitter
    directly with synthetic per-model dicts that encode the disjoint case.
    """
    report = pd.DataFrame(
        [
            {
                "model": "M",
                "estimator": "DR",
                "ESS": 1000.0,
                "tail_mass": 0.0,
                "match_rate": 1.0,
                "min_pscore": 0.5,
            }
        ]
    )
    # Disjoint: one rare action and one insufficient action, but they are
    # different actions, so the intersection is 0.
    enriched, _ = attach_warnings(
        report,
        n_samples=1000,
        model_n_rare_actions={"M": 1},
        model_n_insufficient_actions={"M": 1},
        model_n_rare_and_insufficient_actions={"M": 0},
    )
    assert "RARE_ACTION_NO_SUPPORT" not in enriched["diagnostic_warnings"].iloc[0]


def test_rare_action_no_support_intersection_fires() -> None:
    """When the intersection count is non-zero, ``RARE_ACTION_NO_SUPPORT`` fires."""
    report = pd.DataFrame(
        [
            {
                "model": "M",
                "estimator": "DR",
                "ESS": 1000.0,
                "tail_mass": 0.0,
                "match_rate": 1.0,
                "min_pscore": 0.5,
            }
        ]
    )
    enriched, _ = attach_warnings(
        report,
        n_samples=1000,
        model_n_rare_actions={"M": 1},
        model_n_insufficient_actions={"M": 1},
        model_n_rare_and_insufficient_actions={"M": 1},
    )
    assert "RARE_ACTION_NO_SUPPORT" in enriched["diagnostic_warnings"].iloc[0]
