"""Parity and follow-up tests from the audit pass (#86, #85, #75).

Covers acceptance criteria the feature PR left untested and guards the audit
fixes:

* #86 — the strategy seam's DR path (``dr_value_with_strategy`` +
  ``build_strategy("DR")``) reproduces the legacy ``dr_value_with_clip`` DR
  value at the same clip, guarding the two coexisting DR formulas against drift.
* #75 — a one-item slate (K=1) degenerates to standard slate IPS.
* Extra-estimator report rows follow the caller's ``estimators`` order
  deterministically (no set-iteration ordering).
* MIPS surfaces the embedding-sufficiency diagnostic as a warning when the
  embedding looks insufficient, and stays silent when it is sufficient
  (invariants.md).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.core import dr_value_with_clip
from skdr_eval.estimators import build_strategy, dr_value_with_strategy
from skdr_eval.slate import make_slate_synth, slate_standard_ips


def _toy_problem(seed: int = 0, n: int = 400, n_actions: int = 3):
    rng = np.random.default_rng(seed)
    propensities = rng.dirichlet(np.full(n_actions, 5.0), size=n)
    A = np.array(
        [rng.choice(n_actions, p=propensities[i]) for i in range(n)], dtype=int
    )
    mu = np.array([1.0, 2.0, 3.0])
    Y = mu[A] + rng.normal(scale=0.5, size=n)
    q_hat = np.full(n, float(mu.mean()))
    policy_probs = np.full((n, n_actions), 1.0 / n_actions)
    elig = np.ones((n, n_actions))
    return propensities, policy_probs, Y, q_hat, A, elig


class TestSeamMatchesLegacyDR:
    def test_strategy_dr_matches_dr_value_with_clip(self) -> None:
        # #86 parity: at a fixed clip, the seam's DR must reproduce the legacy
        # clip-grid DR value to floating point. Guards the duplicated formula.
        propensities, policy_probs, Y, q_hat, A, elig = _toy_problem()
        clip = 20.0
        legacy = dr_value_with_clip(
            propensities, policy_probs, Y, q_hat, A, elig, clip_grid=(clip,)
        )
        seam = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=build_strategy("DR", clip=clip),
        )
        assert legacy["DR"].clip == clip
        assert abs(seam.V_hat - legacy["DR"].V_hat) < 1e-9
        assert abs(seam.SE_if - legacy["DR"].SE_if) < 1e-9

    def test_strategy_sndr_matches_dr_value_with_clip(self) -> None:
        propensities, policy_probs, Y, q_hat, A, elig = _toy_problem(seed=1)
        clip = 20.0
        legacy = dr_value_with_clip(
            propensities, policy_probs, Y, q_hat, A, elig, clip_grid=(clip,)
        )
        seam = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat,
            A=A,
            elig=elig,
            strategy=build_strategy("SNDR", clip=clip),
        )
        assert abs(seam.V_hat - legacy["SNDR"].V_hat) < 1e-9


class TestSlateDegenerateK1:
    def test_one_item_slate_matches_manual_standard_ips(self) -> None:
        # #75 parity: with K=1 the slate-level IPS reduces to the per-impression
        # importance-weighted reward, recovered here by a manual reference.
        logs, _attr, truth = make_slate_synth(
            n_impressions=300, n_items=8, slate_size=1, click_model="cascade", seed=0
        )
        logging_prob = float(logs["logging_prob"].iloc[0])

        def target_policy(_slate: list[int]) -> float:
            return logging_prob  # uniform target == logging

        result = slate_standard_ips(logs, target_policy=target_policy)

        # Manual standard-IPS reference over the one-item slates.
        rewards = logs["reward"].to_numpy(dtype=np.float64)
        probs = logs["logging_prob"].to_numpy(dtype=np.float64)
        w = np.where(probs > 0, logging_prob / probs, 0.0)
        manual_v = float((w * rewards).mean())

        assert abs(result.V_hat - manual_v) < 1e-9
        # Uniform target == logging, so V_hat recovers the logging value.
        assert abs(result.V_hat - truth.V_logging) < 3 * max(result.SE, 0.05)


class TestExtraEstimatorOrderDeterministic:
    def test_report_rows_follow_requested_order(self) -> None:
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=5)
        models = {"hgb": HistGradientBoostingRegressor(random_state=5)}
        order = ("SNDR", "DRos", "DR", "SWITCH-DR", "MRDR")
        art = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models=models,
            fit_models=True,
            policy_train="pre_split",
            n_splits=3,
            random_state=5,
            estimators=order,
            switch_tau=10.0,
            dros_lam=2.0,
        )
        rows = art.report[art.report["model"] == "hgb"]
        seen = [e for e in rows["estimator"].tolist() if e in order]
        # Every requested estimator present, exactly once, in the requested order.
        assert seen == list(order)


class TestMipsSufficiencyWarning:
    def _problem(self, seed: int):
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=seed)
        return logs

    def test_insufficient_embedding_warns(self) -> None:
        logs = self._problem(13)
        rng = np.random.default_rng(13)
        # A random low-dim embedding is generally not a sufficient statistic.
        emb = rng.normal(size=(3, 1))
        with pytest.warns(UserWarning, match="MIPS embedding may be insufficient"):
            skdr_eval.evaluate_sklearn_models(
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

    def test_sufficient_embedding_is_silent(self) -> None:
        logs = self._problem(14)
        # Identity embedding is a sufficient statistic → no insufficiency warning.
        emb = np.eye(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            skdr_eval.evaluate_sklearn_models(
                logs=logs,
                models={"hgb": HistGradientBoostingRegressor(random_state=14)},
                fit_models=True,
                policy_train="pre_split",
                n_splits=3,
                random_state=14,
                estimators=("DR", "MIPS"),
                action_embedding=emb,
                mips_bandwidth=0.01,
            )
        assert not [
            w for w in caught if "MIPS embedding may be insufficient" in str(w.message)
        ]
