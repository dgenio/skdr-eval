"""PR-diff branch coverage for estimator wiring in ``skdr_eval.core`` (#85, #86).

Covers the validation / fallback paths added for weighted outcome losses
(``fit_outcome_crossfit(sample_weight=...)``), the estimator-name
canonicaliser, and the normal-approximation CI branch the extra estimators
(MRDR/SWITCH-DR/DRos/MIPS) take under ``ci_bootstrap=True`` in both the
sklearn and pairwise evaluators (block bootstrap is DR/SNDR-only).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import skdr_eval
from skdr_eval.core import (
    DataValidationError,
    OutcomeModelError,
    _canonical_estimator_name,
    fit_outcome_crossfit,
)


def _xy(n: int = 60, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)), rng.normal(size=n)


class TestCanonicalEstimatorName:
    def test_dros_alias(self) -> None:
        assert _canonical_estimator_name("dros") == "DRos"

    def test_known_names_pass_through(self) -> None:
        assert _canonical_estimator_name("switch_dr") == "SWITCH-DR"
        assert _canonical_estimator_name("mrdr") == "MRDR"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            _canonical_estimator_name("nope")


class TestFitOutcomeCrossfitSampleWeight:
    def test_wrong_shape_raises(self) -> None:
        X, Y = _xy()
        with pytest.raises(DataValidationError, match="sample_weight shape"):
            fit_outcome_crossfit(X, Y, n_splits=3, sample_weight=np.ones(len(Y) + 1))

    def test_negative_weight_raises(self) -> None:
        X, Y = _xy()
        sw = np.ones(len(Y))
        sw[0] = -0.5
        with pytest.raises(DataValidationError, match="non-negative"):
            fit_outcome_crossfit(X, Y, n_splits=3, sample_weight=sw)

    def test_estimator_without_sample_weight_support(self) -> None:
        X, Y = _xy()

        class _NoSampleWeight:
            def fit(self, X: np.ndarray, y: np.ndarray) -> _NoSampleWeight:
                self._mean = float(np.mean(y))
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.full(X.shape[0], self._mean)

        with pytest.raises(OutcomeModelError, match="does not accept"):
            fit_outcome_crossfit(
                X,
                Y,
                n_splits=3,
                estimator=_NoSampleWeight,
                sample_weight=np.ones(len(Y)),
            )


class TestExtraEstimatorBootstrapCI:
    def test_sklearn_extra_estimator_uses_normal_ci(self) -> None:
        logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=21)
        art = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": HistGradientBoostingRegressor(random_state=21)},
            fit_models=True,
            policy_train="pre_split",
            n_splits=3,
            random_state=21,
            estimators=("DR", "MRDR"),
            ci_bootstrap=True,
        )
        row = art.report[art.report["estimator"] == "MRDR"].iloc[0]
        assert np.isfinite(row["ci_lower"])
        assert np.isfinite(row["ci_upper"])
        assert row["ci_lower"] <= row["V_hat"] <= row["ci_upper"]

    def test_pairwise_extra_estimator_uses_normal_ci(self) -> None:
        logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
            n_days=2, n_clients_day=80, n_ops=8, seed=42, binary=False
        )
        feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
        model = Ridge()
        model.fit(logs_df[feature_cols].values, logs_df["service_time"].values)
        art = skdr_eval.evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models={"ridge": model},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=2,
            strategy="direct",
            random_state=42,
            policy_train="pre_split",
            estimators=("DR", "MRDR"),
            ci_bootstrap=True,
        )
        row = art.report[art.report["estimator"] == "MRDR"].iloc[0]
        assert np.isfinite(row["ci_lower"])
        assert np.isfinite(row["ci_upper"])
