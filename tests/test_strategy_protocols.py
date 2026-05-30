"""Unit tests for the ``WeightTransform`` / ``OutcomeLoss`` strategy seam (#86).

These tests pin the math identities each transform must obey so future
refactors of :mod:`skdr_eval.estimators` cannot silently change estimator
semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.estimators import (
    ClipTransform,
    DRosShrinkTransform,
    IdentityTransform,
    MIPSTransform,
    MRDRWeightedLoss,
    MSEOutcomeLoss,
    SwitchTauTransform,
    TransformContext,
)


def _context(pi_obs: np.ndarray, A: np.ndarray, n_actions: int = 3) -> TransformContext:
    n = pi_obs.shape[0]
    a_int = A.astype(int)
    # The working weight is the DR importance ratio π(A|x)/e(A|x) (#106). To
    # isolate the *propensity transform* math under test from the policy, put
    # all target mass on the observed action so π(A|x) == 1 and the ratio
    # reduces to 1/e(A|x) — the quantity these unit tests were written around.
    policy_probs = np.zeros((n, n_actions))
    policy_probs[np.arange(n), a_int] = 1.0
    elig = np.ones((n, n_actions))
    matched = pi_obs > 0
    # Build a full logging propensity matrix: pi_obs at the observed action,
    # uniform remainder over the other actions so rows sum to 1.
    propensities = np.full((n, n_actions), 0.0)
    propensities[np.arange(n), a_int] = pi_obs
    leftover = (1.0 - pi_obs) / max(n_actions - 1, 1)
    for i in range(n):
        for a in range(n_actions):
            if a != a_int[i]:
                propensities[i, a] = leftover[i]
    return TransformContext(
        pi_obs=pi_obs,
        matched=matched,
        policy_probs=policy_probs,
        A=A,
        elig=elig,
        propensities=propensities,
    )


class TestIdentityTransform:
    def test_returns_reciprocal(self) -> None:
        pi = np.array([0.5, 0.25, 0.1])
        A = np.array([0, 1, 2])
        w = IdentityTransform()(_context(pi, A))
        np.testing.assert_allclose(w, [2.0, 4.0, 10.0])

    def test_zero_on_unmatched(self) -> None:
        pi = np.array([0.5, 0.0, 0.1])
        A = np.array([0, 1, 2])
        w = IdentityTransform()(_context(pi, A))
        assert w[1] == 0.0


class TestClipTransform:
    def test_bounds_weights(self) -> None:
        pi = np.array([0.5, 0.25, 0.01])
        A = np.array([0, 1, 2])
        w = ClipTransform(clip=5.0)(_context(pi, A))
        # 1/0.5 = 2, 1/0.25 = 4, 1/0.01 = 100 -> clipped to 5
        np.testing.assert_allclose(w, [2.0, 4.0, 5.0])

    def test_inf_clip_recovers_identity(self) -> None:
        pi = np.array([0.5, 0.25, 0.1])
        A = np.array([0, 1, 2])
        ctx = _context(pi, A)
        w_clip = ClipTransform(clip=float("inf"))(ctx)
        w_id = IdentityTransform()(ctx)
        np.testing.assert_allclose(w_clip, w_id)


class TestSwitchTauTransform:
    def test_zeroes_above_threshold(self) -> None:
        pi = np.array([0.5, 0.25, 0.01])
        A = np.array([0, 1, 2])
        # tau=5: 1/0.01 = 100 > 5, so w[2] -> 0
        w = SwitchTauTransform(tau=5.0)(_context(pi, A))
        np.testing.assert_allclose(w, [2.0, 4.0, 0.0])

    def test_invalid_tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be"):
            SwitchTauTransform(tau=0.0)
        with pytest.raises(ValueError, match="tau must be"):
            SwitchTauTransform(tau=float("inf"))


class TestDRosShrinkTransform:
    def test_lam_zero_zeros_weights(self) -> None:
        pi = np.array([0.5, 0.25])
        A = np.array([0, 1])
        w = DRosShrinkTransform(lam=0.0)(_context(pi, A))
        np.testing.assert_allclose(w, [0.0, 0.0])

    def test_lam_large_recovers_raw(self) -> None:
        # As lam -> infty, w * lam / (w^2 + lam) -> w * lam / lam = w
        pi = np.array([0.5, 0.25])
        A = np.array([0, 1])
        raw = IdentityTransform()(_context(pi, A))
        shrunk = DRosShrinkTransform(lam=1e9)(_context(pi, A))
        np.testing.assert_allclose(shrunk, raw, rtol=1e-6)


class TestMIPSTransform:
    def test_identity_kernel_collapses_to_ips(self) -> None:
        # Identity embedding + small bandwidth -> kernel ≈ identity, MIPS ≈ IPS
        emb = np.eye(3, dtype=np.float64)
        pi = np.array([0.5, 0.25, 0.1])
        A = np.array([0, 1, 2])
        ctx = _context(pi, A)
        w_mips = MIPSTransform(action_embedding=emb, bandwidth=0.01)(ctx)
        w_ips = IdentityTransform()(ctx)
        np.testing.assert_allclose(w_mips, w_ips, rtol=1e-6)

    def test_uniform_kernel_collapses_to_constant_weight(self) -> None:
        # Bandwidth=inf -> kernel rows are uniform 1/n_actions, so every
        # row's embedding-marginal logging density is 1/n_actions and the
        # MIPS weight is the constant n_actions — fully marginalising over
        # the action propensity.
        emb = np.random.default_rng(0).normal(size=(3, 2))
        pi = np.array([0.5, 0.25, 0.1])
        A = np.array([0, 1, 2])
        ctx = _context(pi, A)
        w = MIPSTransform(action_embedding=emb, bandwidth=float("inf"))(ctx)
        # n_actions = 3 -> w = 3 everywhere on the matched subset.
        np.testing.assert_allclose(w, np.full(3, 3.0))

    def test_action_mismatch_raises(self) -> None:
        emb = np.eye(4)  # 4 rows but design has 3 actions
        pi = np.array([0.5])
        A = np.array([0])
        ctx = _context(pi, A, n_actions=3)
        with pytest.raises(ValueError, match="embedding must be supplied"):
            MIPSTransform(action_embedding=emb)(ctx)

    def test_requires_full_propensity_matrix(self) -> None:
        # If a caller hand-builds a TransformContext with the wrong
        # propensities shape, MIPS should fail loudly rather than silently
        # produce a wrong weight.
        emb = np.eye(3)
        pi = np.array([0.5, 0.25, 0.1])
        A = np.array([0, 1, 2])
        bad_propensities = np.zeros((3, 2))  # wrong second dim
        ctx = TransformContext(
            pi_obs=pi,
            matched=pi > 0,
            policy_probs=np.full((3, 3), 1.0 / 3),
            A=A,
            elig=np.ones((3, 3)),
            propensities=bad_propensities,
        )
        with pytest.raises(ValueError, match="full logging-policy propensity"):
            MIPSTransform(action_embedding=emb)(ctx)


class TestOutcomeLosses:
    def test_mse_returns_ones(self) -> None:
        pi = np.array([0.5, 0.25])
        A = np.array([0, 1])
        policy = np.array([[0.5, 0.5], [0.5, 0.5]])
        w = MSEOutcomeLoss()(pi_obs=pi, policy_probs=policy, A=A)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_mrdr_weights_squared_ratio(self) -> None:
        pi = np.array([0.5, 0.25])
        A = np.array([0, 1])
        policy = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = MRDRWeightedLoss(clip_floor=1e-6)
        w = loss(pi_obs=pi, policy_probs=policy, A=A)
        # (1/0.5)^2 = 4; (1/0.25)^2 = 16
        np.testing.assert_allclose(w, [4.0, 16.0])

    def test_mrdr_clip_floor_protects_against_tiny_pi(self) -> None:
        pi = np.array([1e-10])
        A = np.array([0])
        policy = np.array([[1.0, 0.0]])
        loss = MRDRWeightedLoss(clip_floor=1e-3)
        w = loss(pi_obs=pi, policy_probs=policy, A=A)
        # With clip_floor=1e-3 the denominator is bounded; weight is at most
        # 1/clip_floor^2 = 1e6.
        assert float(w[0]) <= 1.0 / (1e-3**2)
        assert w[0] > 0
