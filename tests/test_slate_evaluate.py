"""Tests for the top-level slate evaluation entry point (#135).

Covers the ``EvaluationArtifact`` integration (report / warnings / sensitivity
/ card render), the support-health behaviour (a healthy overlap reports ``ok``;
a target on rarely-logged slates trips a warning), and a recovery check that
Cascade-DR's value tracks the synthetic oracle target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skdr_eval.exceptions import DataValidationError
from skdr_eval.reporting import EvaluationArtifact
from skdr_eval.slate import evaluate_slate_models, make_slate_synth
from skdr_eval.slate.evaluate import _empirical_q_hat_per_rank


def _uniform_policy(n_items: int):
    def policy(_rank: int, _item: int) -> float:
        return 1.0 / n_items

    return policy


def _oracle_like_policy(attractiveness: np.ndarray, n_items: int, sharpness: float):
    """A per-rank policy that prefers globally-attractive items."""
    mean_attr = attractiveness.mean(axis=0)
    weights = np.exp(sharpness * mean_attr)
    weights = weights / weights.sum()

    def policy(_rank: int, item: int) -> float:
        return float(weights[item])

    return policy


def test_returns_artifact_with_full_surface() -> None:
    logs, attractiveness, _ = make_slate_synth(
        n_impressions=300, n_items=8, slate_size=3, seed=11
    )
    art = evaluate_slate_models(
        logs,
        {
            "uniform": _uniform_policy(8),
            "sharp": _oracle_like_policy(attractiveness, 8, 2.0),
        },
        estimators=("SlateStandardIPS", "RIPS", "PI-IPS", "SlateCascadeDR"),
        baseline="logged",
    )
    assert isinstance(art, EvaluationArtifact)
    # One row per (model, estimator).
    assert len(art.report) == 2 * 4
    assert set(art.report["estimator"]) == {
        "SlateStandardIPS",
        "RIPS",
        "PI-IPS",
        "SlateCascadeDR",
    }
    # Warning + sensitivity surfaces are populated and the card renders.
    assert {"support_health", "diagnostic_warnings"} <= set(art.report.columns)
    assert len(art.sensitivity) == 8
    assert len(art.to_html_str()) > 0
    assert len(art.to_json_str()) > 0
    assert art.metadata["evaluator"] == "evaluate_slate_models"
    assert art.metadata["click_model"] == "cascade"
    # baseline="logged" attaches the delta column.
    assert "delta_V_hat" in art.report.columns


def test_healthy_overlap_reports_ok() -> None:
    logs, _, _ = make_slate_synth(n_impressions=400, n_items=6, slate_size=2, seed=12)
    art = evaluate_slate_models(
        logs, {"uniform": _uniform_policy(6)}, estimators=("RIPS",)
    )
    # A uniform target over uniformly-logged slates has flat weights -> ok.
    assert (art.report["support_health"] == "ok").all()


def test_concentrated_target_trips_warning() -> None:
    # A target that concentrates its mass on a few items (but rarely matches the
    # uniformly-logged slates) yields a heavy importance-weight tail: ESS
    # collapses and the support warnings fire.
    logs, _, _ = make_slate_synth(n_impressions=200, n_items=10, slate_size=3, seed=13)

    def concentrated(_rank: int, item: int) -> float:
        base = np.full(10, 0.001)
        base[[0, 1, 2]] = 0.331
        base = base / base.sum()
        return float(base[item])

    art_conc = evaluate_slate_models(logs, {"c": concentrated}, estimators=("RIPS",))
    art_unif = evaluate_slate_models(
        logs, {"u": _uniform_policy(10)}, estimators=("RIPS",)
    )
    health = art_conc.report["support_health"].iloc[0]
    codes = art_conc.report["diagnostic_warnings"].iloc[0]
    assert health in {"caution", "high_risk"}
    assert codes != ""
    # The heavy tail must show up as a much smaller effective sample size.
    assert art_conc.report["ESS"].iloc[0] < art_unif.report["ESS"].iloc[0] / 5


def test_cascade_dr_tracks_oracle_target() -> None:
    # Recovery-style check: Cascade-DR's estimate for the attractiveness-ranked
    # oracle target should land near the analytic oracle value and clearly above
    # the uniform target. Uses the synthetic ground truth.
    logs, attractiveness, truth = make_slate_synth(
        n_impressions=1500, n_items=8, slate_size=3, click_model="cascade", seed=14
    )
    sharp = _oracle_like_policy(attractiveness, 8, 6.0)
    art = evaluate_slate_models(logs, {"sharp": sharp, "uniform": _uniform_policy(8)})
    sharp_v = float(
        art.report[
            (art.report["model"] == "sharp")
            & (art.report["estimator"] == "SlateCascadeDR")
        ]["V_hat"].iloc[0]
    )
    uniform_v = float(
        art.report[
            (art.report["model"] == "uniform")
            & (art.report["estimator"] == "SlateCascadeDR")
        ]["V_hat"].iloc[0]
    )
    # The attractiveness-preferring policy must beat the uniform one and stay in
    # the analytic [uniform, oracle] envelope (with slack for estimation noise).
    assert sharp_v > uniform_v
    assert truth.V_uniform_target - 0.2 <= sharp_v <= truth.V_oracle_target + 0.2


def test_rejects_empty_models() -> None:
    logs, _, _ = make_slate_synth(n_impressions=30, n_items=5, slate_size=2, seed=1)
    with pytest.raises(DataValidationError, match="non-empty"):
        evaluate_slate_models(logs, {})


def test_rejects_unknown_estimator() -> None:
    logs, _, _ = make_slate_synth(n_impressions=30, n_items=5, slate_size=2, seed=1)
    with pytest.raises(DataValidationError, match="unknown slate estimators"):
        evaluate_slate_models(
            logs, {"u": _uniform_policy(5)}, estimators=("NotAnEstimator",)
        )


def test_baseline_sentinel_is_logged_and_rejects_unknown() -> None:
    """The slate evaluator accepts the canonical ``"logged"`` sentinel (like the
    other evaluators) and raises on unknown strings instead of ignoring them."""
    logs, _, _ = make_slate_synth(n_impressions=120, n_items=6, slate_size=2, seed=3)
    art = evaluate_slate_models(
        logs, {"uniform": _uniform_policy(6)}, estimators=("RIPS",), baseline="logged"
    )
    assert "delta_V_hat" in art.report.columns
    # The old "logging" sentinel is now an unknown string and must raise.
    with pytest.raises(DataValidationError, match="Unknown baseline"):
        evaluate_slate_models(
            logs,
            {"uniform": _uniform_policy(6)},
            estimators=("RIPS",),
            baseline="logging",
        )


def test_cascade_dr_qhat_is_cross_fitted_out_of_sample() -> None:
    """The empirical Cascade-DR outcome model is cross-fitted: an impression's
    q̂ is estimated from the *other* folds only. When every item is observed by
    exactly one impression, each item's out-of-fold support is empty, so the
    cross-fitted q̂ for that cell is 0 — whereas an in-sample table would leak
    the impression's own click (1.0) back into its own prediction."""
    n = 12
    logs = pd.DataFrame(
        {
            "slate": [[i] for i in range(n)],  # each impression shows a unique item
            "clicks": [[1.0] for _ in range(n)],
        }
    )
    q_hat = _empirical_q_hat_per_rank(logs, slate_size=1, n_items=n, random_state=0)
    # Out-of-sample => no leakage of the row's own click into its own q̂.
    for i in range(n):
        assert q_hat[i, 0, i] == 0.0


def test_cascade_dr_qhat_single_impression_uses_full_table() -> None:
    """With fewer than two impressions there is nothing to hold out, so the
    full in-sample click-rate table is returned (broadcast over the single
    row)."""
    logs = pd.DataFrame({"slate": [[0, 1]], "clicks": [[1.0, 0.0]]})
    q_hat = _empirical_q_hat_per_rank(logs, slate_size=2, n_items=3, random_state=0)
    assert q_hat.shape == (1, 2, 3)
    # In-sample rate: rank 0 / item 0 saw a click; rank 1 / item 1 did not.
    assert q_hat[0, 0, 0] == 1.0
    assert q_hat[0, 1, 1] == 0.0
    # Unobserved (rank, item) cells stay at 0.
    assert q_hat[0, 0, 2] == 0.0
