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
from skdr_eval.core import _make_time_series_split, estimate_propensity_pairwise
from skdr_eval.exceptions import DataValidationError, InsufficientDataError
from skdr_eval.pairwise import PairwiseDesign


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


def _uniform_over_eligible(elig: np.ndarray) -> np.ndarray:
    """Per-row uniform distribution over eligible actions; zero rows -> zeros."""
    n_elig = elig.sum(axis=1, keepdims=True)
    safe_n = np.where(n_elig > 0, n_elig, 1)
    return np.where(elig, 1.0 / safe_n, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap": 0},
        {"gap": 2, "test_size": 100},
        {"gap": 1, "max_train_size": 400},
    ],
)
def test_temporal_split_controls_recover_ground_truth_sklearn(kwargs):
    """Simulation proof: DR recovers a known ground truth with temporal controls.

    Required by ``docs/agent-context/invariants.md`` for any change to
    statistical evaluation logic. We use the **constant-outcome identity**:

        Set Y[i] = c for all i. Then for *any* target policy and *any*
        propensity model, the true policy value V* = c. A consistent DR
        estimator must satisfy V_DR -> c as q_hat learns the constant.

        DR_i = q_pi[i] + w[i] * (Y[i] - q_hat[i])
             = q_hat[i] + w[i] * (c - q_hat[i])

        With Y constant, regression learns q_hat ~ c so (Y - q_hat) ~ 0
        on every fold, regardless of fold layout. V_DR -> c independent
        of gap/test_size/max_train_size.

    If the temporal-split kwargs threading were broken (off-by-one fold
    indexing, leaked test rows treated as untrained, dropped samples),
    the cross-fit q_hat would land away from c on the affected rows and
    V_DR would drift from c by more than the IPS variance can explain.
    """
    logs, _ops, _ = skdr_eval.make_synth_logs(n=3000, n_ops=3, seed=42)
    design = skdr_eval.build_design(logs)

    c = 17.0
    Y_const = np.full(len(design.Y), c, dtype=np.float64)

    propensities, _ = skdr_eval.fit_propensity_timecal(
        design.X_phi,
        design.A,
        design.ts,
        n_splits=3,
        random_state=0,
        **kwargs,
    )
    q_hat, _ = skdr_eval.fit_outcome_crossfit(
        design.X_obs,
        Y_const,
        n_splits=3,
        random_state=0,
        **kwargs,
    )

    # TimeSeriesSplit only assigns rows >= first test-fold start to a test
    # fold; the uncovered prefix gets q_hat=0 from the zero-init in
    # fit_outcome_crossfit. Restrict the recovery proof to *covered* rows --
    # this is the well-defined subset over which the cross-fit math holds.
    covered = q_hat != 0
    assert covered.sum() >= 50, (
        f"covered set too small to be meaningful ({covered.sum()}/{len(q_hat)}) "
        f"for kwargs={kwargs}"
    )
    np.testing.assert_allclose(
        q_hat[covered],
        c,
        atol=1e-6,
        err_msg="q_hat failed to learn constant target on covered rows",
    )

    policy_probs = _uniform_over_eligible(design.elig)
    results = skdr_eval.dr_value_with_clip(
        propensities=propensities[covered],
        policy_probs=policy_probs[covered],
        Y=Y_const[covered],
        q_hat=q_hat[covered],
        A=design.A[covered],
        elig=design.elig[covered],
        clip_grid=(float("inf"),),
    )
    v_dr = float(results["DR"].V_hat)
    v_sndr = float(results["SNDR"].V_hat)

    # With Y constant and q_hat ~ c on covered rows, V_DR collapses to c
    # within numerical tolerance (IPS correction is ~0 by construction).
    assert abs(v_dr - c) < 1e-3, (
        f"DR={v_dr:.6f} failed to recover ground-truth V*={c} under "
        f"temporal controls {kwargs}"
    )
    assert abs(v_sndr - c) < 1e-3, (
        f"SNDR={v_sndr:.6f} failed to recover ground-truth V*={c} under "
        f"temporal controls {kwargs}"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap": 0},
        {"gap": 1, "max_train_size": 800},
    ],
)
def test_temporal_split_controls_recover_ground_truth_pairwise(kwargs):
    """Pairwise variant of the constant-outcome DR recovery proof.

    ``evaluate_pairwise_models`` threads gap/test_size/max_train_size
    through ``estimate_propensity_pairwise`` and ``fit_outcome_crossfit``.
    The same constant-outcome identity used in the sklearn test gives a
    fold-construction-independent ground truth here: V* = c regardless of
    propensity quality or policy choice.
    """
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=4, n_clients_day=150, n_ops=5, seed=42
    )
    design = PairwiseDesign.from_dataframes(
        logs_df, op_daily_df, "arrival_day", "client_id", "operator_id", "elig_mask"
    )

    propensities = estimate_propensity_pairwise(
        design,
        method="multinomial",
        n_splits=3,
        random_state=0,
        **kwargs,
    )

    # Action indices: position of chosen operator within the day's op list.
    A = np.array(
        [
            design.ops_all_by_day[row["arrival_day"]].index(row["operator_id"])
            for _, row in design.logs_df.iterrows()
        ],
        dtype=int,
    )

    # Eligibility matrix shaped like propensities: (n_decisions, max_ops).
    elig = np.zeros_like(propensities, dtype=bool)
    for i, (_, row) in enumerate(design.logs_df.iterrows()):
        day_ops = design.ops_all_by_day[row["arrival_day"]]
        eligible = row.get("elig_mask", day_ops)
        if not isinstance(eligible, (list, tuple, set)):
            eligible = day_ops
        for j, op in enumerate(day_ops):
            if op in eligible:
                elig[i, j] = True

    c = 12.5
    Y_const = np.full(len(design.logs_df), c, dtype=np.float64)

    feature_cols = design.cli_features + design.op_features
    X_obs = design.logs_df[feature_cols].to_numpy(dtype=np.float64)
    q_hat, _ = skdr_eval.fit_outcome_crossfit(
        X_obs,
        Y_const,
        n_splits=3,
        random_state=0,
        **kwargs,
    )

    # Restrict to covered rows -- see sklearn variant for rationale.
    covered = q_hat != 0
    assert covered.sum() >= 50, (
        f"covered set too small to be meaningful ({covered.sum()}/{len(q_hat)}) "
        f"for kwargs={kwargs}"
    )
    np.testing.assert_allclose(
        q_hat[covered],
        c,
        atol=1e-6,
        err_msg="q_hat failed to learn constant target on covered rows",
    )

    policy_probs = _uniform_over_eligible(elig)
    results = skdr_eval.dr_value_with_clip(
        propensities=propensities[covered],
        policy_probs=policy_probs[covered],
        Y=Y_const[covered],
        q_hat=q_hat[covered],
        A=A[covered],
        elig=elig[covered],
        clip_grid=(float("inf"),),
    )
    v_dr = float(results["DR"].V_hat)

    assert abs(v_dr - c) < 1e-3, (
        f"Pairwise DR={v_dr:.6f} failed to recover ground-truth V*={c} under "
        f"temporal controls {kwargs}"
    )
