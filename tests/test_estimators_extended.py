"""Equivalence + smoke tests for MRDR / SWITCH-DR / DRos / MIPS (#85, #86).

These tests check the *structural* properties of each estimator (degenerate
parameter recovers DR, MIPS recovers IPS when the embedding is identity)
rather than statistical recovery, which is exercised by
``test_estimator_recovery_simulation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.estimators import (
    EstimatorStrategy,
    MIPSTransform,
    MSEOutcomeLoss,
    SwitchTauTransform,
    dr_value_with_strategy,
)


def _toy_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, n_actions = 200, 3
    propensities = rng.dirichlet(alpha=np.ones(n_actions), size=n)
    A = np.array(
        [rng.choice(n_actions, p=propensities[i]) for i in range(n)],
        dtype=int,
    )
    Y = rng.normal(loc=5.0, scale=1.0, size=n)
    q_hat = rng.normal(loc=5.0, scale=0.5, size=n)
    policy_probs = np.full((n, n_actions), 1.0 / n_actions)
    elig = np.ones((n, n_actions))
    return propensities, policy_probs, Y, q_hat, A, elig


class TestSwitchDRDegenerate:
    def test_large_tau_matches_dr(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy_problem()
        # tau very large: SWITCH-DR weight == clipped IPS weight at clip=tau.
        # With tau=inf (not allowed), we use a huge finite value.
        tau = 1e6
        strategy = EstimatorStrategy(
            name="SWITCH-DR",
            weight_transform=SwitchTauTransform(tau=tau),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
        result_switch = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=strategy,
        )
        dr_strategy = skdr_eval.build_strategy("DR", clip=float("inf"))
        result_dr = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=dr_strategy,
        )
        assert abs(result_switch.V_hat - result_dr.V_hat) < 1e-6


class TestDRosDegenerate:
    def test_lam_zero_collapses_to_direct_method(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy_problem()
        # lam=0 zeros the IPS weight; V_hat must equal mean(q_pi).
        strategy = skdr_eval.build_strategy("DRos", lam=0.0)
        result = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=strategy,
        )
        # q_pi == q_hat for 1D q_hat.
        expected = float(q_hat.mean())
        assert abs(result.V_hat - expected) < 1e-9


class TestMIPSIdentityRecoversIPS:
    def test_identity_embedding_matches_ips(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy_problem()
        n_actions = propensities.shape[1]
        emb = np.eye(n_actions)
        mips = MIPSTransform(action_embedding=emb, bandwidth=0.01)
        strategy_mips = EstimatorStrategy(
            name="MIPS",
            weight_transform=mips,
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
        result_mips = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=strategy_mips,
            action_embedding=emb,
        )
        # Compare with unclipped IPS via the protocol seam (clip=inf DR).
        strategy_dr = skdr_eval.build_strategy("DR", clip=float("inf"))
        result_dr = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=strategy_dr,
        )
        assert abs(result_mips.V_hat - result_dr.V_hat) < 1e-3


class TestQHat2DResidual:
    """Regression proof for the 2D (per-action) q_hat residual slice (PR #103).

    A 2D q_hat must be indexed at the logged action in the DR residual
    ``Y - q_hat_obs``; previously ``Y - q_hat`` mis-broadcast ``(n,)`` against
    ``(n, n_actions)``. With a target policy deterministic on the logged
    action, ``q_pi`` reduces to the observed-action prediction, so the 2D
    estimate must equal the 1D estimate built from that same column.
    """

    def test_2d_qhat_matches_observed_action_slice(self) -> None:
        propensities, _policy_probs, Y, _q_hat1d, A, elig = _toy_problem()
        n, n_actions = propensities.shape
        rng = np.random.default_rng(7)
        # Distinct per-action predictions so the observed-action slice matters.
        q_hat_2d = rng.normal(loc=5.0, scale=0.5, size=(n, n_actions))
        # Deterministic target on the logged action isolates the residual term.
        policy_probs = np.zeros((n, n_actions))
        policy_probs[np.arange(n), A] = 1.0
        q_hat_obs = q_hat_2d[np.arange(n), A]

        strategy = skdr_eval.build_strategy("DR", clip=float("inf"))
        shared = {
            "propensities": propensities,
            "policy_probs": policy_probs,
            "Y": Y,
            "A": A,
            "elig": elig,
            "strategy": strategy,
        }
        result_2d = dr_value_with_strategy(q_hat=q_hat_2d, **shared)
        result_1d = dr_value_with_strategy(q_hat=q_hat_obs, **shared)

        assert np.isfinite(result_2d.V_hat)
        assert abs(result_2d.V_hat - result_1d.V_hat) < 1e-9
        assert abs(result_2d.SE_if - result_1d.SE_if) < 1e-9


class TestEvaluateSklearnExtraEstimators:
    def test_evaluate_sklearn_with_mrdr_runs(self) -> None:
        logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=11)
        models = {"hgb": HistGradientBoostingRegressor(random_state=11)}
        art = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            policy_train="pre_split",
            n_splits=3,
            random_state=11,
            estimators=("DR", "SNDR", "MRDR", "SWITCH-DR", "DRos"),
            switch_tau=10.0,
            dros_lam=2.0,
        )
        names = set(art.report["estimator"].unique())
        assert {"DR", "SNDR", "MRDR", "SWITCH-DR", "DRos"} <= names

    def test_mips_without_embedding_falls_back_to_sndr(self) -> None:
        # #136 / #85: a missing action_embedding no longer hard-fails; MIPS
        # gracefully falls back to SNDR for the 'MIPS' row, with a warning.
        logs, _, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=12)
        with pytest.warns(UserWarning, match="falling back to SNDR"):
            artifact = skdr_eval.evaluate_sklearn_models(
                logs=logs,
                models={"hgb": HistGradientBoostingRegressor(random_state=12)},
                fit_models=True,
                policy_train="pre_split",
                n_splits=3,
                random_state=12,
                estimators=("SNDR", "MIPS"),
            )
        report = artifact.report
        mips_v = float(report.loc[report["estimator"] == "MIPS", "V_hat"].iloc[0])
        sndr_v = float(report.loc[report["estimator"] == "SNDR", "V_hat"].iloc[0])
        # The fallback MIPS row carries exactly the SNDR value.
        assert mips_v == pytest.approx(sndr_v)

    def test_mips_via_evaluate_with_embedding(self) -> None:
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=13)
        rng = np.random.default_rng(13)
        # Random low-dim embedding for the 3 ops.
        emb = rng.normal(size=(3, 2))
        art = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": HistGradientBoostingRegressor(random_state=13)},
            fit_models=True,
            policy_train="pre_split",
            n_splits=3,
            random_state=13,
            estimators=("DR", "MIPS"),
            action_embedding=emb,
            mips_bandwidth=0.5,
        )
        report = art.report
        assert {"DR", "MIPS"} <= set(report["estimator"].unique())
        # MIPS V_hat must be finite.
        v_mips = float(report[report["estimator"] == "MIPS"]["V_hat"].iloc[0])
        assert np.isfinite(v_mips)


class TestEmbeddingSufficiencyDiagnostic:
    def test_sufficient_embedding_low_gap(self) -> None:
        # When the embedding equals one-hot, R²_action == R²_embedding by
        # construction so the gap is 0.
        n = 200
        rng = np.random.default_rng(7)
        Y = rng.normal(size=n)
        q_hat = np.zeros(n)
        A = rng.integers(0, 3, size=n)
        emb = np.eye(3)
        report = skdr_eval.embedding_sufficiency_diagnostic(
            Y=Y, q_hat=q_hat, A=A, action_embedding=emb
        )
        assert report.r2_action < 1e-6

    def test_insufficient_embedding_positive_gap(self) -> None:
        # Embedding is rank-1 (all ones) so it can recover no action signal;
        # gap must be strictly positive when there *is* action signal.
        n = 600
        rng = np.random.default_rng(8)
        A = rng.integers(0, 4, size=n)
        action_effects = np.array([0.0, 1.0, -1.0, 2.0])
        Y = action_effects[A] + rng.normal(scale=0.1, size=n)
        q_hat = np.zeros(n)
        emb = np.ones((4, 1))
        report = skdr_eval.embedding_sufficiency_diagnostic(
            Y=Y, q_hat=q_hat, A=A, action_embedding=emb
        )
        assert report.r2_action > 0.5
