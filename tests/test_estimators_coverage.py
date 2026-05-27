"""Edge-branch coverage for the estimator strategy seam (#85, #86).

Exercises the defensive / validation branches in :mod:`skdr_eval.estimators`
that the recovery-simulation and equivalence suites leave untouched: invalid
constructor arguments, ``build_strategy`` name handling, the strategy-core
guard rails (no matched samples, malformed weight transforms, zero
self-normalisation denominator), the ``mips_value`` convenience wrapper, and
the embedding-sufficiency diagnostic's edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.estimators import (
    DRosShrinkTransform,
    EstimatorStrategy,
    MIPSTransform,
    MSEOutcomeLoss,
    build_strategy,
    dr_value_with_strategy,
    embedding_sufficiency_diagnostic,
    mips_value,
)


def _toy(seed: int = 0, n: int = 120, n_actions: int = 3):
    rng = np.random.default_rng(seed)
    propensities = rng.dirichlet(np.ones(n_actions), size=n)
    A = np.array(
        [rng.choice(n_actions, p=propensities[i]) for i in range(n)], dtype=int
    )
    Y = rng.normal(5.0, 1.0, size=n)
    q_hat = rng.normal(5.0, 0.5, size=n)
    policy_probs = np.full((n, n_actions), 1.0 / n_actions)
    elig = np.ones((n, n_actions))
    return propensities, policy_probs, Y, q_hat, A, elig


class TestBuildStrategy:
    def test_mips_without_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="MIPS strategy requires action_embedding"):
            build_strategy("MIPS")

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown estimator name"):
            build_strategy("not-an-estimator")

    @pytest.mark.parametrize(
        ("name", "expected", "self_norm"),
        [
            ("dr", "DR", False),
            ("SNDR", "SNDR", True),
            ("mrdr", "MRDR", False),
            ("switch_dr", "SWITCH-DR", False),
            ("DRos", "DRos", False),
        ],
    )
    def test_named_branches(self, name: str, expected: str, self_norm: bool) -> None:
        strat = build_strategy(name)
        assert strat.name == expected
        assert strat.self_normalised is self_norm

    def test_mips_branch_builds_transform(self) -> None:
        strat = build_strategy("MIPS", action_embedding=np.eye(3), bandwidth=0.5)
        assert strat.name == "MIPS"
        assert isinstance(strat.weight_transform, MIPSTransform)


class TestTransformValidation:
    def test_dros_negative_lam_raises(self) -> None:
        with pytest.raises(ValueError, match="lam must be"):
            DRosShrinkTransform(lam=-1.0)

    def test_dros_inf_lam_raises(self) -> None:
        with pytest.raises(ValueError, match="lam must be"):
            DRosShrinkTransform(lam=float("inf"))

    def test_mips_non_2d_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="action_embedding must be 2D"):
            MIPSTransform(action_embedding=np.ones(3))

    def test_mips_nonpositive_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="bandwidth must be"):
            MIPSTransform(action_embedding=np.eye(3), bandwidth=0.0)


class TestStrategyCoreGuards:
    def test_no_matched_samples_raises(self) -> None:
        # Zeroing the logging propensities makes every observed-action pscore
        # zero, so no row is "matched" and the core must refuse to estimate.
        propensities, policy_probs, Y, q_hat, A, elig = _toy()
        propensities = np.zeros_like(propensities)
        strat = build_strategy("DR", clip=10.0)
        with pytest.raises(ValueError, match="No matched samples"):
            dr_value_with_strategy(
                propensities=propensities,
                policy_probs=policy_probs,
                Y=Y,
                q_hat=q_hat,
                A=A,
                elig=elig,
                strategy=strat,
            )

    def test_weight_transform_wrong_shape_raises(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy()

        class _BadShape:
            name = "bad_shape"

            def __call__(self, context: object) -> np.ndarray:
                return np.ones(Y.shape[0] + 1)

        strat = EstimatorStrategy(
            name="bad_shape",
            weight_transform=_BadShape(),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
        with pytest.raises(ValueError, match="returned shape"):
            dr_value_with_strategy(
                propensities=propensities,
                policy_probs=policy_probs,
                Y=Y,
                q_hat=q_hat,
                A=A,
                elig=elig,
                strategy=strat,
            )

    def test_negative_weights_raise(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy()

        class _NegWeights:
            name = "neg"

            def __call__(self, context: object) -> np.ndarray:
                return -np.ones(Y.shape[0])

        strat = EstimatorStrategy(
            name="neg",
            weight_transform=_NegWeights(),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=False,
        )
        with pytest.raises(ValueError, match="negative"):
            dr_value_with_strategy(
                propensities=propensities,
                policy_probs=policy_probs,
                Y=Y,
                q_hat=q_hat,
                A=A,
                elig=elig,
                strategy=strat,
            )

    def test_self_normalised_zero_denominator_falls_back_to_direct(self) -> None:
        # DRos with lam=0 zeros every weight; under self-normalisation the
        # denominator Σw is zero, so V̂ must collapse to mean(q_pi). With a
        # uniform target and 1D q_hat, q_pi == q_hat.
        propensities, policy_probs, Y, q_hat, A, elig = _toy()
        strat = EstimatorStrategy(
            name="sn_zero",
            weight_transform=DRosShrinkTransform(lam=0.0),
            outcome_loss=MSEOutcomeLoss(),
            self_normalised=True,
        )
        result = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=strat,
        )
        assert abs(result.V_hat - float(q_hat.mean())) < 1e-9


class TestMipsValueWrapper:
    def test_mips_value_returns_finite_result(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy()
        emb = np.eye(propensities.shape[1])
        result = mips_value(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            action_embedding=emb,
            bandwidth=0.01,
        )
        assert np.isfinite(result.V_hat)


class TestEmbeddingDiagnosticEdges:
    def test_non_2d_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            embedding_sufficiency_diagnostic(
                Y=np.zeros(5),
                q_hat=np.zeros(5),
                A=np.zeros(5, dtype=int),
                action_embedding=np.ones(3),
            )

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            embedding_sufficiency_diagnostic(
                Y=np.zeros(5),
                q_hat=np.zeros(4),
                A=np.zeros(5, dtype=int),
                action_embedding=np.eye(3),
            )

    def test_zero_residual_variance_reports_perfect_model(self) -> None:
        n = 50
        Y = np.full(n, 2.0)
        q_hat = np.full(n, 2.0)  # residual is identically zero
        A = np.zeros(n, dtype=int)
        report = embedding_sufficiency_diagnostic(
            Y=Y, q_hat=q_hat, A=A, action_embedding=np.eye(3)
        )
        assert report.r2_action == 0.0
        assert "perfect" in report.notes
