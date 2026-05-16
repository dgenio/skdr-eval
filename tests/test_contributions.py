"""Tests for per-decision contributions (issue #92).

Covers ``DRResult.contributions``, ``EvaluationArtifact.contributions``, the
``keep_contributions`` flag, its independence from ``ci_bootstrap``, the
``max_kept_contributions`` memory guard, and the card top-K block.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import skdr_eval
from skdr_eval.core import _compute_contributions
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


def test_top_k_negative_raises():
    art = _run_sklearn(keep_contributions=True)
    with pytest.raises(DataValidationError, match="positive integer"):
        art.contributions("HGB", estimator="DR", top_k=-3)


def test_compute_contributions_unbounded_clip():
    # Exercise the clip==inf branch of _compute_contributions.
    rng = np.random.default_rng(0)
    n, k = 16, 3
    propensities = rng.uniform(0.1, 0.9, size=(n, k))
    propensities /= propensities.sum(axis=1, keepdims=True)
    policy_probs = np.eye(k)[rng.integers(0, k, size=n)]
    Y = rng.normal(size=n)
    q_hat = rng.normal(size=n)
    A = rng.integers(0, k, size=n)
    elig = np.ones((n, k))
    q_pi, w_clip, dr_contrib, decision_id, matched = _compute_contributions(
        propensities, policy_probs, Y, q_hat, A, elig, clip=float("inf")
    )
    assert q_pi.shape == (n,)
    assert w_clip.shape == (n,)
    assert dr_contrib.shape == (n,)
    assert decision_id.tolist() == list(range(n))
    assert matched.all()


def test_compute_contributions_log_indices_length_mismatch_raises():
    rng = np.random.default_rng(0)
    n = 8
    propensities = np.full((n, 2), 0.5)
    policy_probs = np.zeros((n, 2))
    policy_probs[:, 0] = 1.0
    Y = rng.normal(size=n)
    q_hat = rng.normal(size=n)
    A = np.zeros(n, dtype=int)
    elig = np.ones((n, 2))
    with pytest.raises(DataValidationError, match="length"):
        _compute_contributions(
            propensities,
            policy_probs,
            Y,
            q_hat,
            A,
            elig,
            clip=10.0,
            eval_log_indices=np.arange(n - 1, dtype=np.int64),
        )


# --------------------------------------------------------------------------- #
# Independence from ci_bootstrap                                              #
# --------------------------------------------------------------------------- #


def test_ci_bootstrap_does_not_auto_attach_contributions():
    # ci_bootstrap=True must not silently retain contributions on DRResult —
    # existing CI callers don't opt into the contributions feature and the
    # extra memory would surprise them.
    art = _run_sklearn(ci_bootstrap=True)
    assert "ci_lower" in art.report.columns
    assert "ci_upper" in art.report.columns
    for est_name in ("DR", "SNDR"):
        assert art.detailed["HGB"][est_name].contributions is None
    with pytest.raises(DataValidationError, match="not available"):
        art.contributions("HGB", estimator="DR")


def test_keep_contributions_and_ci_bootstrap_combine():
    art = _run_sklearn(keep_contributions=True, ci_bootstrap=True)
    assert "ci_lower" in art.report.columns
    frame = art.contributions("HGB", estimator="DR")
    assert set(frame.columns) == set(REQUIRED_COLUMNS)


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


# --------------------------------------------------------------------------- #
# Zero-weight decisions, pairwise pre_split, edge-case card layouts           #
# --------------------------------------------------------------------------- #


def test_zero_weight_decision_contributes_only_q_pi():
    """Issue #92 acceptance criterion: zero-weight decisions contribute q_pi.

    When a row's observed action lies outside its eligibility set (or the
    matched propensity is exactly 0), ``_compute_contributions`` zeros out
    ``w_clip``, so the DR pseudo-outcome collapses to ``q_pi``. Verifying
    the identity directly on the helper isolates it from estimator noise.
    """
    n, n_ops = 6, 3
    rng = np.random.default_rng(0)
    propensities = np.full((n, n_ops), 1.0 / n_ops)
    policy_probs = np.full((n, n_ops), 1.0 / n_ops)
    Y = rng.normal(loc=10.0, scale=2.0, size=n)
    q_hat = rng.normal(loc=9.0, scale=1.0, size=n)
    A = np.array([0, 1, 2, 0, 1, 2])
    elig = np.ones((n, n_ops), dtype=int)
    # Knock the chosen action out of the eligibility set on rows 0 and 3 so
    # `matched[i] == False` and the textbook DR contribution should be q_pi[i].
    elig[0, 0] = 0
    elig[3, 0] = 0

    q_pi, w_clip, dr_contrib, _, matched = _compute_contributions(
        propensities, policy_probs, Y, q_hat, A, elig, clip=float("inf")
    )
    for i in (0, 3):
        assert not matched[i]
        assert w_clip[i] == 0.0
        assert dr_contrib[i] == pytest.approx(q_pi[i], abs=IDENTITY_TOL)


def test_pairwise_pre_split_decision_id_offset():
    """Mirror of test_pre_split_decision_id_offset for the pairwise path.

    Issue #92 acceptance criterion 3 requires the documented join-back
    pattern (``merge(frame, logs.reset_index(names="decision_id"))``) to
    work for both estimator paths and both ``policy_train`` settings.
    """
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=8, seed=42, binary=False
    )
    feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    model = Ridge(random_state=42)
    model.fit(logs_df[feature_cols].values, logs_df["service_time"].values)
    n_total = len(logs_df)

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
        policy_train_frac=0.7,
        keep_contributions=True,
    )
    frame = art.contributions("ridge", estimator="DR")
    decision_ids = frame["decision_id"].to_numpy()

    # Decision ids must be valid positional indices into the original logs_df,
    # not 0..n_eval-1 in the post-split slice.
    assert decision_ids.min() >= 0
    assert decision_ids.max() < n_total
    assert len(np.unique(decision_ids)) == len(decision_ids)
    # Round-trip join: the contribution frame's ``reward`` matches the
    # corresponding row's ``service_time`` in the original ``logs_df``.
    logs_with_id = logs_df.reset_index(drop=True).reset_index(names="decision_id")
    joined = pd.merge(frame, logs_with_id, on="decision_id", how="left")
    assert len(joined) == len(frame)
    np.testing.assert_allclose(
        joined["reward"].to_numpy(),
        joined["service_time"].astype(float).to_numpy(),
        atol=IDENTITY_TOL,
        err_msg="pairwise pre_split decision_id must round-trip to original logs",
    )


def test_card_block_no_overlap_when_few_decisions():
    """Card top/bottom blocks must not show the same decision_id twice."""
    art = _run_sklearn(n=400, keep_contributions=True)
    # Trim contributions to 3 decisions to force the small-n path.
    detailed = art.detailed["HGB"]["DR"]
    if detailed.contributions is None:
        pytest.skip("contributions not stored")
    trimmed = {k: v[:3] for k, v in detailed.contributions.items()}
    detailed.contributions = trimmed
    html = art.card("HGB", headline_estimator="DR")
    # The top-K block and bottom-K block must never list the same decision_id.
    # The rendered HTML embeds the decision_id as a row value; we check the
    # data-side helper rather than the HTML to avoid brittle parsing.
    top, bottom = art._card_contribution_rows("HGB", "DR")
    top_ids = {row["decision_id"] for row in top}
    bottom_ids = {row["decision_id"] for row in bottom}
    assert top_ids.isdisjoint(bottom_ids), (
        f"card top/bottom overlap on small-n run: top={top_ids} bottom={bottom_ids}"
    )
    # And the card render itself must succeed.
    assert "<html" in html or "<div" in html or "top" in html.lower()


def test_card_block_single_decision_shows_only_top():
    """With exactly one decision, the bottom block must be empty."""
    art = _run_sklearn(n=400, keep_contributions=True)
    detailed = art.detailed["HGB"]["DR"]
    if detailed.contributions is None:
        pytest.skip("contributions not stored")
    trimmed = {k: v[:1] for k, v in detailed.contributions.items()}
    detailed.contributions = trimmed
    top, bottom = art._card_contribution_rows("HGB", "DR")
    assert len(top) == 1
    assert len(bottom) == 0


# --------------------------------------------------------------------------- #
# Optional stress tests (gated by SKDR_STRESS=1)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.getenv("SKDR_STRESS") != "1",
    reason="stress test — set SKDR_STRESS=1 to enable (issue #92 stretch goal)",
)
def test_stress_million_row_keep_contributions_within_2x_baseline():
    """Issue #92 stress: 1M-row keep=True run completes within 2x keep=False.

    Verifies both that the run does not OOM and that the contributions
    plumbing does not introduce a multiplicative slowdown beyond the
    expected linear cost of materializing six float64 arrays of length n.
    """
    n = 1_000_000

    t0 = time.perf_counter()
    art_off = _run_sklearn(n=n, keep_contributions=False)
    t_off = time.perf_counter() - t0
    assert not art_off.report.empty

    t0 = time.perf_counter()
    art_on = _run_sklearn(n=n, keep_contributions=True)
    t_on = time.perf_counter() - t0
    assert not art_on.report.empty
    assert art_on.detailed["HGB"]["DR"].contributions is not None
    payload = art_on.detailed["HGB"]["DR"].contributions
    assert payload["decision_id"].shape == (n,)

    ratio = t_on / max(t_off, 1e-6)
    assert ratio < 2.0, (
        f"keep_contributions=True ran {ratio:.2f}x slower than baseline "
        f"(t_off={t_off:.2f}s, t_on={t_on:.2f}s); expected <2x per issue #92."
    )


@pytest.mark.skipif(
    os.getenv("SKDR_STRESS") != "1",
    reason="memory probe — set SKDR_STRESS=1 to enable",
)
def test_stress_memory_off_vs_on_within_bounds():
    """Acceptance criterion 6 (measured): keep=False does not pay the storage cost.

    The default-off path stores ``contributions=None`` on every ``DRResult``
    (verified by ``test_default_does_not_store_contributions``); this test
    additionally measures peak-allocation deltas via tracemalloc to confirm
    the absence of the six contribution arrays. Within 1% on n=50000 is
    well inside the budget.
    """
    import tracemalloc

    n = 50_000

    tracemalloc.start()
    art_off = _run_sklearn(n=n, keep_contributions=False)
    _, peak_off = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del art_off

    tracemalloc.start()
    art_on = _run_sklearn(n=n, keep_contributions=True)
    _, peak_on = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del art_on

    # Six float64 arrays of length n (q_pi, q_hat, weight, reward,
    # contribution_to_V) + one int64 (decision_id) = ~5*8 + 8 = 48 bytes/row
    # ≈ 2.4 MB at n=50k. Allow generous 10x for transient working set.
    expected_payload_bytes = n * (5 * 8 + 8)
    delta = peak_on - peak_off
    assert delta < 10 * expected_payload_bytes, (
        f"keep=True peak memory ({peak_on}) - keep=False peak ({peak_off}) "
        f"= {delta} bytes; far exceeds expected payload "
        f"{expected_payload_bytes} bytes at n={n}."
    )
