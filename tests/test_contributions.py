"""Tests for per-decision contributions (issue #92).

Covers ``DRResult.contributions``, ``EvaluationArtifact.contributions``, the
``keep_contributions`` flag and its implicit activation via ``ci_bootstrap``,
the ``max_kept_contributions`` memory guard, and the card top-K block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import skdr_eval
from skdr_eval.exceptions import DataValidationError

if TYPE_CHECKING:
    from skdr_eval.reporting import EvaluationArtifact

REQUIRED_COLUMNS = (
    "decision_id",
    "q_pi",
    "q_hat",
    "weight",
    "reward",
    "contribution_to_V",
)

IDENTITY_TOL = 1e-10


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _run_sklearn(
    *,
    n: int = 400,
    seed: int = 7,
    keep_contributions: bool = False,
    ci_bootstrap: bool = False,
    policy_train: str = "all",
    max_kept_contributions: int = 100_000_000,
) -> EvaluationArtifact:
    logs, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=seed)
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=seed)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=seed,
        ci_bootstrap=ci_bootstrap,
        policy_train=policy_train,
        keep_contributions=keep_contributions,
        max_kept_contributions=max_kept_contributions,
    )


def _run_pairwise(
    *,
    keep_contributions: bool = False,
    ci_bootstrap: bool = False,
) -> EvaluationArtifact:
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=8, seed=42, binary=False
    )
    feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    model = Ridge(random_state=42)
    model.fit(X, y)
    return skdr_eval.evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models={"ridge": model},
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        random_state=42,
        ci_bootstrap=ci_bootstrap,
        keep_contributions=keep_contributions,
    )


# --------------------------------------------------------------------------- #
# Default-off behavior                                                        #
# --------------------------------------------------------------------------- #


def test_default_does_not_store_contributions():
    art = _run_sklearn()
    for est_name in ("DR", "SNDR"):
        assert art.detailed["HGB"][est_name].contributions is None


def test_default_contributions_method_raises():
    art = _run_sklearn()
    with pytest.raises(DataValidationError, match="not available"):
        art.contributions("HGB", estimator="DR")


# --------------------------------------------------------------------------- #
# Identity test (the simulation-proof for the feature)                        #
# --------------------------------------------------------------------------- #


def test_dr_contributions_mean_recovers_V_hat():
    art = _run_sklearn(keep_contributions=True)
    v_hat_dr = float(
        art.report[(art.report["model"] == "HGB") & (art.report["estimator"] == "DR")][
            "V_hat"
        ].iloc[0]
    )
    frame = art.contributions("HGB", estimator="DR")
    assert np.isclose(
        float(frame["contribution_to_V"].mean()), v_hat_dr, atol=IDENTITY_TOL
    )


def test_sndr_contributions_mean_recovers_V_hat():
    art = _run_sklearn(keep_contributions=True)
    v_hat_sndr = float(
        art.report[
            (art.report["model"] == "HGB") & (art.report["estimator"] == "SNDR")
        ]["V_hat"].iloc[0]
    )
    frame = art.contributions("HGB", estimator="SNDR")
    assert np.isclose(
        float(frame["contribution_to_V"].mean()), v_hat_sndr, atol=IDENTITY_TOL
    )


# --------------------------------------------------------------------------- #
# DataFrame shape and ordering                                                #
# --------------------------------------------------------------------------- #


def test_contributions_required_columns_and_dtypes():
    art = _run_sklearn(keep_contributions=True)
    frame = art.contributions("HGB", estimator="DR")
    assert tuple(frame.columns) == REQUIRED_COLUMNS
    assert frame["decision_id"].dtype.kind in {"i", "u"}
    assert len(frame) == int(art.metadata["n_samples"])


def test_top_k_returns_largest_magnitude():
    art = _run_sklearn(keep_contributions=True)
    full = art.contributions("HGB", estimator="DR")
    k = 5
    top = art.contributions("HGB", estimator="DR", top_k=k)
    assert len(top) == k
    # The returned subset must be exactly the k largest-by-magnitude rows.
    expected_ids = set(
        full.iloc[full["contribution_to_V"].abs().argsort().to_numpy()[::-1][:k]][
            "decision_id"
        ]
    )
    assert set(top["decision_id"]) == expected_ids


def test_top_k_zero_raises():
    art = _run_sklearn(keep_contributions=True)
    with pytest.raises(DataValidationError, match="positive integer"):
        art.contributions("HGB", estimator="DR", top_k=0)


# --------------------------------------------------------------------------- #
# Implicit activation by ci_bootstrap                                         #
# --------------------------------------------------------------------------- #


def test_ci_bootstrap_implies_keep_contributions():
    art = _run_sklearn(ci_bootstrap=True)
    frame = art.contributions("HGB", estimator="DR")
    assert set(frame.columns) == set(REQUIRED_COLUMNS)
    # CI columns from ci_bootstrap are unaffected by the contributions plumbing.
    assert "ci_lower" in art.report.columns
    assert "ci_upper" in art.report.columns


# --------------------------------------------------------------------------- #
# Decision-id offset under pre_split                                          #
# --------------------------------------------------------------------------- #


def test_pre_split_decision_id_offset():
    n = 400
    art = _run_sklearn(n=n, keep_contributions=True, policy_train="pre_split")
    frame = art.contributions("HGB", estimator="DR")
    # pre_split slices logs positionally; eval rows are the chronological tail.
    n_eval = int(art.metadata["n_samples"])
    expected_offset = n - n_eval
    assert frame["decision_id"].min() == expected_offset
    assert frame["decision_id"].max() == n - 1
    # Strictly increasing positional ids over the eval slice.
    diffs = frame["decision_id"].to_numpy()[1:] - frame["decision_id"].to_numpy()[:-1]
    assert (diffs == 1).all()


# --------------------------------------------------------------------------- #
# Memory guard                                                                #
# --------------------------------------------------------------------------- #


def test_memory_guard_raises_when_exceeded():
    with pytest.raises(DataValidationError, match="max_kept_contributions"):
        _run_sklearn(keep_contributions=True, max_kept_contributions=10)


def test_memory_guard_inactive_when_keep_off():
    art = _run_sklearn(max_kept_contributions=10)
    assert art.detailed["HGB"]["DR"].contributions is None


# --------------------------------------------------------------------------- #
# Unknown-key error paths                                                     #
# --------------------------------------------------------------------------- #


def test_contributions_unknown_model_raises():
    art = _run_sklearn(keep_contributions=True)
    with pytest.raises(DataValidationError, match="not in artifact"):
        art.contributions("nope", estimator="DR")


def test_contributions_unknown_estimator_raises():
    art = _run_sklearn(keep_contributions=True)
    with pytest.raises(DataValidationError, match="not in detailed"):
        art.contributions("HGB", estimator="MIPS")


# --------------------------------------------------------------------------- #
# Pairwise smoke                                                              #
# --------------------------------------------------------------------------- #


def test_pairwise_contributions_dr_identity():
    art = _run_pairwise(keep_contributions=True)
    [model_name] = list(art.detailed)
    v_hat_dr = float(
        art.report[
            (art.report["model"] == model_name) & (art.report["estimator"] == "DR")
        ]["V_hat"].iloc[0]
    )
    frame = art.contributions(model_name, estimator="DR")
    assert np.isclose(
        float(frame["contribution_to_V"].mean()), v_hat_dr, atol=IDENTITY_TOL
    )
    assert (frame["decision_id"] == np.arange(len(frame), dtype=np.int64)).all()


def test_pairwise_contributions_sndr_identity():
    art = _run_pairwise(keep_contributions=True)
    [model_name] = list(art.detailed)
    v_hat_sndr = float(
        art.report[
            (art.report["model"] == model_name) & (art.report["estimator"] == "SNDR")
        ]["V_hat"].iloc[0]
    )
    frame = art.contributions(model_name, estimator="SNDR")
    assert np.isclose(
        float(frame["contribution_to_V"].mean()), v_hat_sndr, atol=IDENTITY_TOL
    )


# --------------------------------------------------------------------------- #
# Card top-K block                                                            #
# --------------------------------------------------------------------------- #


def test_card_includes_top_contributors_when_present():
    art = _run_sklearn(keep_contributions=True)
    html = art.card("HGB")
    assert "Top contributors to V" in html
    assert "Bottom detractors of V" in html


def test_card_omits_top_contributors_when_absent():
    art = _run_sklearn()
    html = art.card("HGB")
    assert "Top contributors to V" not in html
    assert "Bottom detractors of V" not in html


def test_join_back_to_logs_via_decision_id():
    """Demonstrate the documented joinback pattern from the issue."""
    n = 400
    logs, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=7)
    art = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=7)},
        fit_models=True,
        n_splits=3,
        random_state=7,
        keep_contributions=True,
    )
    frame = art.contributions("HGB", estimator="DR")
    logs_with_id = logs.reset_index(names="decision_id")
    joined = pd.merge(frame, logs_with_id, on="decision_id", how="left")
    assert len(joined) == len(frame)
    # Every contribution row gets the matching log row's reward.
    assert (joined["reward"] == joined["service_time"].astype(float)).all()
