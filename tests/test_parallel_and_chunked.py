"""Tests for the core hot-path performance group.

Covers the numerical-equivalence and behaviour guarantees of:

* ``n_jobs`` parallelism over the model loop, cross-fitting folds, and the
  bootstrap (#178) — must be bit-identical / deterministic vs serial.
* ``execution_mode="large_data"`` chunked induction (#210) — must be
  numerically identical to ``"standard"``.
* the ``requires_overlap`` fast precheck (#206).
* the single ``coerce_to_pandas`` frame seam reaching every evaluator (#236).
* guard tests locking in the already-vectorised induction (#209) and the
  fit-once-and-share nuisance design (#248).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from skdr_eval import core, make_pairwise_synth, simulate_autoscaling_scenario
from skdr_eval.core import (
    Design,
    _requires_overlap_precheck,
    build_design,
    evaluate_sklearn_models,
    induce_policy_from_sklearn,
)
from skdr_eval.exceptions import DataValidationError, InsufficientOverlapError
from skdr_eval.synth import make_synth_logs

_EQUAL_COLS = [
    "V_hat",
    "SE_if",
    "clip",
    "ESS",
    "tail_mass",
    "MSE_est",
    "match_rate",
    "min_pscore",
    "pareto_k",
]


def _models() -> dict[str, Ridge]:
    # Fresh, independent estimators each call so parallel in-place fitting can
    # never alias a shared object.
    return {"ridge_a": Ridge(alpha=1.0), "ridge_b": Ridge(alpha=2.0)}


def _evaluate(**kwargs):
    logs, _ops, _q = make_synth_logs(n=1500, n_ops=4, seed=7)
    return evaluate_sklearn_models(
        logs,
        _models(),
        policy_train="pre_split",
        random_state=0,
        estimators=("DR", "SNDR", "SWITCH-DR", "DRos"),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# #178 — n_jobs is bit-identical / deterministic                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_model_loop_parallel_matches_serial_exactly(n_jobs: int) -> None:
    serial = _evaluate(n_jobs=1).report.sort_values(["model", "estimator"])
    parallel = _evaluate(n_jobs=n_jobs).report.sort_values(["model", "estimator"])
    np.testing.assert_array_equal(
        serial[["model", "estimator"]].to_numpy(),
        parallel[["model", "estimator"]].to_numpy(),
    )
    # rtol=atol=0 → exact bit-for-bit equality, not "close".
    np.testing.assert_allclose(
        serial[_EQUAL_COLS].to_numpy(),
        parallel[_EQUAL_COLS].to_numpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_bootstrap_ci_deterministic_across_n_jobs() -> None:
    # Single model so n_jobs drives the bootstrap replicates themselves.
    logs, _ops, _q = make_synth_logs(n=1200, n_ops=4, seed=11)

    def run(n_jobs: int):
        return evaluate_sklearn_models(
            logs,
            {"ridge": Ridge()},
            policy_train="pre_split",
            ci_bootstrap=True,
            random_state=0,
            n_jobs=n_jobs,
        ).report

    a = run(1)
    b = run(3)
    np.testing.assert_allclose(
        a[["ci_lower", "ci_upper"]].to_numpy(),
        b[["ci_lower", "ci_upper"]].to_numpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_invalid_n_jobs_zero_raises() -> None:
    logs, _ops, _q = make_synth_logs(n=300, n_ops=3, seed=1)
    with pytest.raises(ValueError, match="n_jobs"):
        evaluate_sklearn_models(
            logs, {"ridge": Ridge()}, policy_train="pre_split", n_jobs=0
        )


# --------------------------------------------------------------------------- #
# #210 — large_data chunked induction is numerically identical               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk_size", [64, 256, 4096])
def test_large_data_matches_standard_exactly(chunk_size: int) -> None:
    standard = _evaluate(execution_mode="standard").report
    large = _evaluate(execution_mode="large_data", chunk_size=chunk_size).report
    np.testing.assert_allclose(
        standard[_EQUAL_COLS].to_numpy(),
        large[_EQUAL_COLS].to_numpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_execution_mode_recorded_in_metadata() -> None:
    art = _evaluate(execution_mode="large_data", chunk_size=128)
    assert art.metadata["execution_mode"] == "large_data"
    # "auto" resolves to "standard" on a small log.
    assert _evaluate(execution_mode="auto").metadata["execution_mode"] == "standard"


def test_invalid_execution_mode_and_chunk_size_raise() -> None:
    logs, _ops, _q = make_synth_logs(n=300, n_ops=3, seed=1)
    with pytest.raises(ValueError, match="execution_mode"):
        evaluate_sklearn_models(
            logs, {"ridge": Ridge()}, policy_train="pre_split", execution_mode="nope"
        )
    with pytest.raises(ValueError, match="chunk_size"):
        evaluate_sklearn_models(
            logs, {"ridge": Ridge()}, policy_train="pre_split", chunk_size=0
        )


def test_induce_chunked_matches_single_shot() -> None:
    logs, _ops, _q = make_synth_logs(n=900, n_ops=5, seed=3)
    design = build_design(logs, y_col="service_time")
    model = Ridge().fit(design.X_obs, design.Y)
    full = induce_policy_from_sklearn(model, design.X_base, design.ops_all, design.elig)
    chunked = induce_policy_from_sklearn(
        model, design.X_base, design.ops_all, design.elig, chunk_size=128
    )
    np.testing.assert_allclose(full, chunked, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# #206 — requires_overlap fast precheck                                       #
# --------------------------------------------------------------------------- #
def test_precheck_passes_on_healthy_logs() -> None:
    art = _evaluate(requires_overlap=True)
    assert art.metadata["requires_overlap"] is True
    assert len(art.report) > 0


def test_precheck_fails_fast_when_no_eligibility() -> None:
    logs, _ops, _q = make_synth_logs(n=900, n_ops=4, seed=5)
    bad = logs.copy()
    elig_cols = [c for c in bad.columns if c.endswith("_elig")]
    bad[elig_cols] = 0  # observed action never eligible -> zero match rate
    with pytest.raises(InsufficientOverlapError, match="match"):
        evaluate_sklearn_models(
            bad, {"ridge": Ridge()}, policy_train="pre_split", requires_overlap=True
        )


def test_precheck_fails_fast_on_poor_propensity_overlap() -> None:
    # Healthy eligibility (match rate passes) but an aggressive overlap_floor
    # forces the propensity-floor branch: the smallest estimated propensity of
    # an observed action falls below the floor, so the gate fires.
    logs, _ops, _q = make_synth_logs(n=900, n_ops=4, seed=5)
    with pytest.raises(InsufficientOverlapError, match="propensity"):
        evaluate_sklearn_models(
            logs,
            {"ridge": Ridge()},
            policy_train="pre_split",
            requires_overlap=True,
            overlap_floor=0.99,
        )


def test_precheck_skips_coarse_fit_when_single_action() -> None:
    # <2 observed actions: the precheck returns before the coarse propensity fit
    # (the downstream estimator raises the clearer 'need >=2 actions').
    logs, _ops, _q = make_synth_logs(n=300, n_ops=3, seed=4)
    d = build_design(logs, y_col="service_time")
    n = len(d.A)
    single = Design(
        X_base=d.X_base,
        X_obs=d.X_obs,
        X_phi=d.X_phi,
        A=np.zeros(n, dtype=d.A.dtype),  # one observed action
        Y=d.Y,
        ts=d.ts,
        ops_all=d.ops_all,
        elig=np.ones_like(d.elig),  # match_rate = 1.0, so the gate is reached
        idx=d.idx,
    )
    assert (
        _requires_overlap_precheck(
            single, random_state=0, overlap_floor=1e-3, min_match_rate=0.05
        )
        is None
    )


def test_precheck_defers_when_coarse_fit_fails() -> None:
    # A degenerate coarse fit (NaN feature) is not evidence of no overlap: the
    # precheck swallows the error and defers to the full path rather than raising.
    logs, _ops, _q = make_synth_logs(n=300, n_ops=3, seed=4)
    d = build_design(logs, y_col="service_time")
    x_phi_bad = d.X_phi.copy()
    x_phi_bad[0, 0] = np.nan  # LogisticRegression.fit raises ValueError on NaN
    bad = Design(
        X_base=d.X_base,
        X_obs=d.X_obs,
        X_phi=x_phi_bad,
        A=d.A,
        Y=d.Y,
        ts=d.ts,
        ops_all=d.ops_all,
        elig=np.ones_like(d.elig),  # match_rate passes; the fit is what fails
        idx=d.idx,
    )
    assert (
        _requires_overlap_precheck(
            bad, random_state=0, overlap_floor=1e-3, min_match_rate=0.05
        )
        is None
    )


def test_precheck_is_off_by_default() -> None:
    # With the precheck off, degenerate eligibility does *not* raise the
    # precheck error (the default path handles it downstream instead).
    logs, _ops, _q = make_synth_logs(n=900, n_ops=4, seed=5)
    bad = logs.copy()
    elig_cols = [c for c in bad.columns if c.endswith("_elig")]
    bad[elig_cols] = 0
    with pytest.raises(Exception) as exc:
        evaluate_sklearn_models(bad, {"ridge": Ridge()}, policy_train="pre_split")
    assert not isinstance(exc.value, InsufficientOverlapError)


# --------------------------------------------------------------------------- #
# #236 — every evaluator routes frames through the single coerce seam         #
# --------------------------------------------------------------------------- #
def test_scenario_accepts_polars_via_seam() -> None:
    pl = pytest.importorskip("polars")

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=120, n_ops=6, seed=2, binary=False
    )
    feat = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    x, y = logs_df[feat].to_numpy(), logs_df["service_time"].to_numpy()
    model = Ridge().fit(x, y)
    common = {
        "models": {"ridge": model},
        "scenario": {"capacity_multiplier": 1.0},
        "metric_col": "service_time",
        "task_type": "regression",
        "direction": "min",
        "n_splits": 2,
        "strategy": "direct",
        "random_state": 0,
        "policy_train": "all",  # pre-fitted model
    }
    pandas_art = simulate_autoscaling_scenario(logs_df, op_daily_df, **common)
    # Polars inputs must flow through the single coerce seam (#236) and produce
    # the identical result.
    polars_art = simulate_autoscaling_scenario(
        pl.from_pandas(logs_df), pl.from_pandas(op_daily_df), **common
    )
    np.testing.assert_allclose(
        pandas_art.report["V_hat"].to_numpy(),
        polars_art.report["V_hat"].to_numpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# #209 / #248 — guard the already-implemented vectorisation & sharing         #
# --------------------------------------------------------------------------- #
class _CountingRidge(Ridge):
    """Ridge that records predict()/fit() call counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.n_predict = 0

    def predict(self, X):  # type: ignore[override]
        self.n_predict += 1
        return super().predict(X)


def test_induction_issues_a_single_predict_call_guard_209() -> None:
    # #209: induction is vectorised to ONE predict() over the stacked matrix in
    # the default (un-chunked) path; guard against a regression to a per-row loop.
    logs, _ops, _q = make_synth_logs(n=800, n_ops=5, seed=9)
    design = build_design(logs, y_col="service_time")
    model = _CountingRidge().fit(design.X_obs, design.Y)
    model.n_predict = 0
    induce_policy_from_sklearn(model, design.X_base, design.ops_all, design.elig)
    assert model.n_predict == 1


def test_nuisances_fit_once_and_shared_across_family_guard_248(monkeypatch) -> None:
    # #248: the base propensity + outcome nuisances are fit ONCE before the
    # model loop and reused across the DR family; only MRDR adds a weighted
    # outcome refit. Spy on the cross-fitters to lock the call counts in.
    logs, _ops, _q = make_synth_logs(n=900, n_ops=4, seed=13)

    calls = {"outcome": 0, "propensity": 0}
    real_outcome = core.fit_outcome_crossfit
    real_propensity = core.fit_propensity_timecal

    def spy_outcome(*a, **k):
        calls["outcome"] += 1
        return real_outcome(*a, **k)

    def spy_propensity(*a, **k):
        calls["propensity"] += 1
        return real_propensity(*a, **k)

    monkeypatch.setattr(core, "fit_outcome_crossfit", spy_outcome)
    monkeypatch.setattr(core, "fit_propensity_timecal", spy_propensity)

    # One model, family WITHOUT MRDR: nuisances fit exactly once each.
    evaluate_sklearn_models(
        logs,
        {"ridge": Ridge()},
        policy_train="pre_split",
        random_state=0,
        estimators=("DR", "SNDR", "SWITCH-DR", "DRos"),
    )
    assert calls == {"propensity": 1, "outcome": 1}

    # Adding MRDR triggers exactly one *additional* weighted outcome fit; the
    # propensity is still fit only once.
    calls.update(outcome=0, propensity=0)
    evaluate_sklearn_models(
        logs,
        {"ridge": Ridge()},
        policy_train="pre_split",
        random_state=0,
        estimators=("DR", "SNDR", "MRDR"),
    )
    assert calls == {"propensity": 1, "outcome": 2}


def test_induce_rejects_nonpositive_chunk_size() -> None:
    logs, _ops, _q = make_synth_logs(n=200, n_ops=3, seed=1)
    design = build_design(logs, y_col="service_time")
    model = Ridge().fit(design.X_obs, design.Y)
    with pytest.raises(DataValidationError, match="chunk_size"):
        induce_policy_from_sklearn(
            model, design.X_base, design.ops_all, design.elig, chunk_size=0
        )


# --------------------------------------------------------------------------- #
# Input-validation hardening (Copilot review on #265)                         #
# --------------------------------------------------------------------------- #
def test_evaluate_rejects_zero_n_jobs() -> None:
    logs, _ops, _q = make_synth_logs(n=200, n_ops=3, seed=1)
    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        evaluate_sklearn_models(
            logs, {"ridge": Ridge()}, policy_train="pre_split", n_jobs=0
        )


@pytest.mark.parametrize("fn_name", ["fit_propensity_timecal", "fit_outcome_crossfit"])
def test_fit_crossfitters_reject_zero_n_jobs(fn_name: str) -> None:
    logs, _ops, _q = make_synth_logs(n=200, n_ops=3, seed=1)
    design = build_design(logs, y_col="service_time")
    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        if fn_name == "fit_propensity_timecal":
            core.fit_propensity_timecal(design.X_phi, design.A, design.ts, n_jobs=0)
        else:
            core.fit_outcome_crossfit(design.X_obs, design.Y, n_jobs=0)


def test_block_bootstrap_ci_rejects_zero_n_jobs() -> None:
    values = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        core.block_bootstrap_ci(values, None, values.mean(), n_jobs=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"overlap_floor": 0.0}, "overlap_floor"),
        ({"overlap_floor": 1.5}, "overlap_floor"),
        ({"min_match_rate": -0.1}, "min_match_rate"),
        ({"min_match_rate": 1.1}, "min_match_rate"),
    ],
)
def test_evaluate_rejects_out_of_range_overlap_thresholds(kwargs, match) -> None:
    logs, _ops, _q = make_synth_logs(n=200, n_ops=3, seed=1)
    with pytest.raises(ValueError, match=match):
        evaluate_sklearn_models(
            logs, {"ridge": Ridge()}, policy_train="pre_split", **kwargs
        )


def test_shared_estimator_instance_fails_fast_under_parallel_fit() -> None:
    # The same object under two keys + threaded in-place fitting would race
    # (#178 review); the guard must reject it before dispatching workers.
    logs, _ops, _q = make_synth_logs(n=600, n_ops=3, seed=2)
    shared = Ridge()
    with pytest.raises(DataValidationError, match="same estimator instance"):
        evaluate_sklearn_models(
            logs,
            {"a": shared, "b": shared},
            policy_train="pre_split",
            fit_models=True,
            n_jobs=2,
        )


def test_shared_estimator_instance_allowed_when_serial() -> None:
    # Serial fitting is deterministic even with a shared object, so no guard.
    logs, _ops, _q = make_synth_logs(n=600, n_ops=3, seed=2)
    shared = Ridge()
    artifact = evaluate_sklearn_models(
        logs,
        {"a": shared, "b": shared},
        policy_train="pre_split",
        fit_models=True,
        n_jobs=1,
    )
    assert not artifact.report.empty
