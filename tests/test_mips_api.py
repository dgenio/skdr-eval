"""Tests for the MIPS API-completeness work (#136).

Covers the median-bandwidth heuristic, the configurable kernel (rbf / linear /
callable), and column-name embedding resolution through
``evaluate_sklearn_models``. The no-embedding SNDR fallback is covered in
``test_estimators_extended.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval as se
from skdr_eval.estimators import MIPSTransform, build_strategy, median_bandwidth
from skdr_eval.estimators.protocols import TransformContext


def _context(propensities, policy_probs, A, elig):
    pi_obs = propensities[np.arange(len(A)), A.astype(int)]
    matched = (pi_obs > 0) & elig.astype(bool)[np.arange(len(A)), A.astype(int)]
    return TransformContext(
        pi_obs=pi_obs,
        matched=matched,
        policy_probs=policy_probs,
        A=A,
        elig=elig,
        propensities=propensities,
        action_embedding=None,
    )


def test_median_bandwidth_matches_manual() -> None:
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])  # dists: 5, 10, 5
    assert median_bandwidth(emb) == pytest.approx(5.0)


def test_median_bandwidth_degenerate_falls_back_to_one() -> None:
    assert median_bandwidth(np.zeros((1, 4))) == 1.0
    assert median_bandwidth(np.zeros((5, 4))) == 1.0  # all coincident


def test_build_strategy_median_resolves_bandwidth() -> None:
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    strat = build_strategy("MIPS", action_embedding=emb, bandwidth="median")
    assert isinstance(strat.weight_transform, MIPSTransform)
    assert strat.weight_transform.bandwidth == pytest.approx(5.0)


def test_build_strategy_rejects_bad_bandwidth_string() -> None:
    with pytest.raises(ValueError, match="median"):
        build_strategy("MIPS", action_embedding=np.eye(3), bandwidth="mean")


def test_linear_kernel_is_row_stochastic() -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t = MIPSTransform(action_embedding=emb, kernel="linear")
    k = t._embedding_kernel()
    assert k.shape == (3, 3)
    assert np.allclose(k.sum(axis=1), 1.0)
    assert (k >= 0).all()


def test_callable_kernel_validated_and_normalised() -> None:
    emb = np.eye(3)

    def my_kernel(e: np.ndarray) -> np.ndarray:
        return np.ones((e.shape[0], e.shape[0]))

    t = MIPSTransform(action_embedding=emb, kernel=my_kernel)
    k = t._embedding_kernel()
    assert np.allclose(k, 1.0 / 3.0)  # uniform after row-normalisation


def test_callable_kernel_rejects_negative() -> None:
    t = MIPSTransform(action_embedding=np.eye(2), kernel=lambda e: -np.ones((2, 2)))
    with pytest.raises(ValueError, match="negative"):
        t._embedding_kernel()


def test_callable_kernel_rejects_wrong_shape() -> None:
    t = MIPSTransform(action_embedding=np.eye(3), kernel=lambda e: np.ones((2, 2)))
    with pytest.raises(ValueError, match="expected"):
        t._embedding_kernel()


def test_invalid_kernel_name_raises() -> None:
    with pytest.raises(ValueError, match="kernel must be"):
        MIPSTransform(action_embedding=np.eye(3), kernel="cosine")


def test_identity_embedding_recovers_ips_under_small_bandwidth() -> None:
    # With distinct one-hot embeddings and a tiny bandwidth the RBF kernel is
    # the identity, so the MIPS weight collapses to 1 / pi_obs (per-action IPS).
    rng = np.random.default_rng(0)
    n, n_actions = 50, 3
    emb = np.eye(n_actions)
    propensities = rng.dirichlet(np.ones(n_actions), size=n)
    A = rng.integers(0, n_actions, size=n)
    elig = np.ones((n, n_actions))
    policy = rng.dirichlet(np.ones(n_actions), size=n)
    t = MIPSTransform(action_embedding=emb, bandwidth=1e-6, kernel="rbf")
    w = t(_context(propensities, policy, A, elig))
    pi_obs = propensities[np.arange(n), A]
    assert np.allclose(w, 1.0 / pi_obs)


def test_column_name_embedding_resolution() -> None:
    logs, _, _ = se.make_synth_logs(n=1500, n_ops=3, seed=21)
    logs = logs.copy()
    cats = sorted(logs["action"].unique())
    rng = np.random.default_rng(0)
    emb_by_cat = {c: rng.normal(size=4) for c in cats}
    logs["op_emb"] = [emb_by_cat[a].tolist() for a in logs["action"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art_col = se.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": HistGradientBoostingRegressor(random_state=21)},
            fit_models=True,
            policy_train="all",
            n_splits=3,
            random_state=21,
            estimators=("SNDR", "MIPS"),
            action_embedding="op_emb",
            mips_bandwidth="median",
        )
    # An equivalent explicit array (actions are label-encoded in sorted order).
    arr = np.asarray([emb_by_cat[c] for c in cats])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art_arr = se.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": HistGradientBoostingRegressor(random_state=21)},
            fit_models=True,
            policy_train="all",
            n_splits=3,
            random_state=21,
            estimators=("SNDR", "MIPS"),
            action_embedding=arr,
            mips_bandwidth="median",
        )
    col_v = float(art_col.report.query("estimator == 'MIPS'")["V_hat"].iloc[0])
    arr_v = float(art_arr.report.query("estimator == 'MIPS'")["V_hat"].iloc[0])
    assert col_v == pytest.approx(arr_v, rel=1e-9)


def test_missing_column_name_raises() -> None:
    logs, _, _ = se.make_synth_logs(n=300, n_ops=3, seed=22)
    with pytest.raises(se.DataValidationError, match="not found in logs"):
        se.evaluate_sklearn_models(
            logs=logs,
            models={"hgb": HistGradientBoostingRegressor(random_state=22)},
            fit_models=True,
            policy_train="all",
            n_splits=3,
            random_state=22,
            estimators=("SNDR", "MIPS"),
            action_embedding="does_not_exist",
        )
