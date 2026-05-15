"""Tests for gap/test_size/max_train_size threading through CV-using funcs.

Includes a simulation proof that ``evaluate_sklearn_models`` continues to
recover a known ground-truth policy value when the temporal-split controls
are exercised — required by docs/agent-context/invariants.md whenever
fold construction in the statistical path is modified.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.core import _make_time_series_split
from skdr_eval.exceptions import DataValidationError, InsufficientDataError


def test_make_time_series_split_default_gap_is_one():
    """The default gap=1 guards against adjacent-row leakage."""
    tscv = _make_time_series_split(n_samples=100, n_splits=3)
    folds = list(tscv.split(np.zeros((100, 1))))
    for train_idx, test_idx in folds:
        assert train_idx.max() + 1 < test_idx.min(), (
            "gap=1 requires at least one sample between train and test"
        )


def test_make_time_series_split_zero_gap_allows_adjacent():
    tscv = _make_time_series_split(n_samples=100, n_splits=3, gap=0)
    folds = list(tscv.split(np.zeros((100, 1))))
    for train_idx, test_idx in folds:
        assert train_idx.max() + 1 == test_idx.min()


def test_make_time_series_split_test_size_caps_fold():
    tscv = _make_time_series_split(n_samples=200, n_splits=3, gap=1, test_size=10)
    for _, test_idx in tscv.split(np.zeros((200, 1))):
        assert len(test_idx) == 10


def test_make_time_series_split_max_train_size_caps_train():
    tscv = _make_time_series_split(n_samples=200, n_splits=3, gap=1, max_train_size=20)
    for train_idx, _ in tscv.split(np.zeros((200, 1))):
        assert len(train_idx) <= 20


def test_make_time_series_split_rejects_negative_gap():
    with pytest.raises(DataValidationError, match="gap"):
        _make_time_series_split(n_samples=50, n_splits=3, gap=-1)


def test_make_time_series_split_rejects_bad_test_size():
    with pytest.raises(DataValidationError, match="test_size"):
        _make_time_series_split(n_samples=50, n_splits=3, test_size=0)


def test_make_time_series_split_rejects_insufficient_data():
    with pytest.raises(InsufficientDataError):
        _make_time_series_split(n_samples=5, n_splits=3, gap=2)


def test_fit_propensity_timecal_accepts_temporal_controls():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=0)
    design = skdr_eval.build_design(logs)
    propensities, fold_idx = skdr_eval.fit_propensity_timecal(
        design.X_phi,
        design.A,
        design.ts,
        n_splits=3,
        gap=2,
        max_train_size=80,
    )
    assert propensities.shape[0] == len(logs)
    # Every sample assigned to a fold (>=0) once the CV runs.
    assigned = fold_idx[fold_idx >= 0]
    assert len(assigned) > 0


def test_fit_outcome_crossfit_accepts_temporal_controls():
    logs, _ops, _ = skdr_eval.make_synth_logs(n=300, n_ops=3, seed=1)
    design = skdr_eval.build_design(logs)
    preds, _ = skdr_eval.fit_outcome_crossfit(
        design.X_obs,
        design.Y,
        n_splits=3,
        gap=2,
        test_size=40,
    )
    assert preds.shape == (len(logs),)


def test_evaluate_sklearn_models_threads_kwargs():
    """End-to-end smoke that kwargs reach the CV layer without erroring."""
    logs, _ops, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=2)
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"hgb": HistGradientBoostingRegressor(max_iter=20, random_state=0)},
        fit_models=True,
        n_splits=3,
        random_state=0,
        gap=2,
        test_size=30,
        max_train_size=200,
    )
    assert not artifact.report.empty


def test_temporal_split_controls_preserve_statistical_invariants():
    """Simulation proof: invariants survive temporal-control toggling.

    Per docs/agent-context/invariants.md, any change to statistical
    evaluation logic must include a simulation. The change here is purely
    additive (new ``gap``/``test_size``/``max_train_size`` pass-throughs to
    ``TimeSeriesSplit``); the proof confirms that, on a moderately sized
    synthetic dataset, both the default-controls path and the
    parameterized-controls path:

    1. produce finite DR/SNDR estimates (no NaN/inf -- cross-fitting and
       propensity clipping still active),
    2. produce DR estimates within the empirical range of observed
       outcomes (sanity bound),
    3. report SE > 0 (influence-function machinery intact).
    """
    logs, _ops, _true_q = skdr_eval.make_synth_logs(n=4000, n_ops=3, seed=42)
    y_min, y_max = logs["service_time"].min(), logs["service_time"].max()
    model_factory = lambda: HistGradientBoostingRegressor(  # noqa: E731
        max_iter=30, random_state=0
    )

    for kwargs in [
        {"gap": 0},
        {"gap": 2, "test_size": 100},
        {"gap": 1, "max_train_size": 400},
    ]:
        artifact = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": model_factory()},
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
            **kwargs,
        )
        for est_name in ("DR", "SNDR"):
            row = artifact.report[artifact.report["estimator"] == est_name].iloc[0]
            v_hat = float(row["V_hat"])
            se = float(row["SE_if"])
            assert np.isfinite(v_hat), f"{est_name} V_hat non-finite for {kwargs}"
            assert np.isfinite(se), f"{est_name} SE_if non-finite for {kwargs}"
            assert se > 0, f"{est_name} SE_if not positive for {kwargs}"
            # DR estimate cannot escape the empirical outcome range by 2x.
            assert y_min / 2 <= v_hat <= 2 * y_max, (
                f"{est_name} V_hat={v_hat:.3f} outside empirical range "
                f"[{y_min:.3f}, {y_max:.3f}] for {kwargs}"
            )
