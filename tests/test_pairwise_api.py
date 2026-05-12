"""Test pairwise evaluation API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge

from skdr_eval import evaluate_pairwise_models, make_pairwise_synth
from skdr_eval import pairwise as pairwise_mod
from skdr_eval.exceptions import DataValidationError
from skdr_eval.pairwise import (
    PairwiseDesign,
    _fit_surrogate,
    _surrogate_features,
    induce_policy_stream_topk,
)


def test_pairwise_regression_basic():
    """Test basic pairwise evaluation with regression task."""
    # Generate small synthetic data
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42, binary=False
    )

    # Create simple models
    models = {
        "ridge": Ridge(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
    }

    # Fit models on pairwise features
    # For testing, we'll create a simple feature matrix
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values

    for model in models.values():
        model.fit(X, y)

    # Run pairwise evaluation
    _artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,  # Small for testing
        strategy="direct",  # Force direct strategy for small data
        random_state=42,
    )
    report, detailed_results = _artifact.report, _artifact.detailed

    # Basic checks
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "model" in report.columns
    assert "estimator" in report.columns
    assert "V_hat" in report.columns
    assert "ESS" in report.columns
    assert "match_rate" in report.columns

    # Check that all values are finite
    numeric_cols = ["V_hat", "SE_if", "ESS", "tail_mass", "MSE_est", "match_rate"]
    for col in numeric_cols:
        if col in report.columns:
            assert report[col].notna().all(), f"NaN values found in {col}"
            assert np.isfinite(report[col]).all(), f"Non-finite values found in {col}"

    # Check ESS > 0
    assert (report["ESS"] > 0).all(), "ESS should be positive"

    # Check match_rate in [0, 1]
    assert (report["match_rate"] >= 0).all(), "match_rate should be >= 0"
    assert (report["match_rate"] <= 1).all(), "match_rate should be <= 1"

    # Check detailed results structure
    assert isinstance(detailed_results, dict)
    assert len(detailed_results) == len(models)

    for model_name in models:
        assert model_name in detailed_results
        model_results = detailed_results[model_name]
        assert isinstance(model_results, dict)
        assert "DR" in model_results or "SNDR" in model_results


def test_pairwise_binary_basic():
    """Test basic pairwise evaluation with binary task."""
    # Generate small synthetic data with binary outcomes
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42, binary=True
    )

    # Create simple models
    models = {
        "logistic": LogisticRegression(random_state=42, max_iter=1000),
    }

    # Fit models on pairwise features
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["success"].values

    for model in models.values():
        model.fit(X, y)

    # Run pairwise evaluation
    report = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="success",
        task_type="binary",
        direction="max",  # Maximize success probability
        n_splits=2,
        strategy="direct",
        random_state=42,
    ).report

    # Basic checks
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "model" in report.columns
    assert "V_hat" in report.columns

    # Check that all values are finite
    numeric_cols = ["V_hat", "SE_if", "ESS", "match_rate"]
    for col in numeric_cols:
        if col in report.columns:
            assert report[col].notna().all(), f"NaN values found in {col}"
            assert np.isfinite(report[col]).all(), f"Non-finite values found in {col}"


def test_pairwise_autoscale_strategy():
    """Test that autoscale strategy selection works."""
    # Generate data that should trigger different strategies
    small_logs, small_ops = make_pairwise_synth(
        n_days=1, n_clients_day=50, n_ops=5, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Fit model
    feature_cols = [
        col for col in small_logs.columns if col.startswith(("cli_", "op_"))
    ]
    X = small_logs[feature_cols].values
    y = small_logs["service_time"].values
    models["ridge"].fit(X, y)

    # Test auto strategy (should select direct for small data)
    report = evaluate_pairwise_models(
        logs_df=small_logs,
        op_daily_df=small_ops,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        strategy="auto",
        random_state=42,
    ).report

    assert len(report) > 0

    # Test explicit direct strategy
    report_direct = evaluate_pairwise_models(
        logs_df=small_logs,
        op_daily_df=small_ops,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        strategy="direct",
        random_state=42,
    ).report

    assert len(report_direct) > 0


def test_pairwise_propensity_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that propensity='auto' runs end-to-end without crashing."""
    # Force the multinomial path so the test is deterministic across
    # environments regardless of whether SciPy is installed.
    monkeypatch.setattr("skdr_eval.core.SCIPY_AVAILABLE", False)

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=8, seed=42, binary=False
    )

    models = {
        "ridge": Ridge(random_state=42),
    }

    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    _artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        propensity="auto",
        random_state=42,
    )

    report, detailed_results = _artifact.report, _artifact.detailed

    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "ridge" in detailed_results
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()


def test_pairwise_design_creation():
    """Test PairwiseDesign creation and statistics."""

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=50, n_ops=8, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Check design attributes
    assert len(design.cli_features) > 0
    assert len(design.op_features) > 0
    assert len(design.ops_all_by_day) == 2  # 2 days
    assert all(len(ops) == 8 for ops in design.ops_all_by_day.values())

    # Check statistics
    stats = design.get_stats()
    assert stats["n_rows"] == 100  # 2 days * 50 clients
    assert stats["n_days"] == 2
    assert stats["candidate_pairs"] > 0
    assert stats["memory_gb"] > 0


def test_pairwise_with_eligibility():
    """Test pairwise evaluation with eligibility constraints."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=30, n_ops=6, seed=42
    )

    # Verify eligibility masks exist
    assert "elig_mask" in logs_df.columns
    assert all(isinstance(mask, list) for mask in logs_df["elig_mask"])

    models = {"ridge": Ridge(random_state=42)}

    # Fit model
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    # Run evaluation with eligibility
    report = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        elig_col="elig_mask",
        random_state=42,
    ).report

    assert len(report) > 0
    assert (report["ESS"] > 0).all()


def test_pairwise_error_handling():
    """Test error handling in pairwise evaluation."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Test with invalid task_type
    with pytest.raises(ValueError, match="Unknown task_type"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="invalid",
            direction="min",
        )

    # Test with invalid direction
    with pytest.raises(ValueError, match=r"Unknown.*direction"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="invalid",
        )


def test_pairwise_fit_models_true():
    """Test that fit_models=True fits unfitted models internally."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=50, n_ops=8, seed=42
    )

    # Create unfitted models
    models = {
        "ridge": Ridge(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42, max_iter=10),
    }

    # Run evaluation with fit_models=True
    _artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        fit_models=True,  # Let the function fit the models
        random_state=42,
    )
    report, detailed_results = _artifact.report, _artifact.detailed

    # Verify results are valid
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "V_hat" in report.columns
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()

    # Verify all models were evaluated
    assert len(detailed_results) == len(models)
    for model_name in models:
        assert model_name in detailed_results


def test_pairwise_stream_topk_strategy():
    """Test stream_topk strategy end-to-end with top-K filtering."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=12, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}

    # Fit model on observed data
    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    models["ridge"].fit(X, y)

    # Run evaluation with stream_topk strategy
    _artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="stream_topk",
        topk=5,  # Keep top 5 operators per client
        random_state=42,
    )
    report, detailed_results = _artifact.report, _artifact.detailed

    # Verify results are valid
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "V_hat" in report.columns
    assert report["V_hat"].notna().all()
    assert np.isfinite(report["V_hat"]).all()

    # Verify that stream_topk strategy evaluated successfully
    assert "ridge" in detailed_results
    model_results = detailed_results["ridge"]
    # Should have either DR or SNDR estimator
    assert len(model_results) > 0
    for _estimator_name, dr_result in model_results.items():
        # Check that the DRResult has valid values
        assert hasattr(dr_result, "V_hat")
        assert np.isfinite(dr_result.V_hat)


def test_pairwise_unfitted_model_fails_without_fit_models():
    """Test that unfitted models raise NotFittedError when fit_models=False."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    # Create unfitted model
    models = {"ridge": Ridge(random_state=42)}

    # Should raise NotFittedError when trying to predict with unfitted model
    with pytest.raises(NotFittedError):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            strategy="direct",
            fit_models=False,  # Don't fit models
            random_state=42,
        )


@pytest.mark.parametrize(
    "strategy",
    ["direct", "stream", "stream_topk"],
    ids=["direct_unfitted", "stream_unfitted", "stream_topk_unfitted"],
)
def test_pairwise_unfitted_models_fail_across_strategies(strategy: str):
    """Test that unfitted models raise NotFittedError immediately for all strategies."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=30, n_ops=10, seed=42
    )

    # Create unfitted model
    models = {"ridge": Ridge(random_state=42)}

    # All strategies should fail fast with NotFittedError on unfitted models
    with pytest.raises(NotFittedError):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models=models,
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=2,
            strategy=strategy,
            topk=5,
            fit_models=False,  # Don't fit models
            random_state=42,
        )


@pytest.mark.parametrize(
    "strategy",
    ["direct", "stream", "stream_topk"],
    ids=["direct_empty_elig", "stream_empty_elig", "stream_topk_empty_elig"],
)
def test_empty_eligibility_raises_across_strategies(strategy: str):
    """A client with an empty eligibility list must fail loudly.

    Per docs/agent-context/invariants.md, the previous silent first-operator
    fallback hid data-quality issues. Every induction strategy should now
    raise DataValidationError when a client has no eligible operators.
    """
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=6, seed=42
    )

    # Force the first client of the day to have an empty eligibility list.
    # Use `.at` for scalar list assignment (`.loc` tries to broadcast).
    logs_df = logs_df.copy()
    logs_df.at[logs_df.index[0], "elig_mask"] = []

    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    model = Ridge(random_state=42)
    model.fit(X, y)

    with pytest.raises(DataValidationError, match="No eligible operators"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models={"ridge": model},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=2,
            strategy=strategy,
            topk=3,
            elig_col="elig_mask",
            random_state=42,
        )


def test_stream_topk_rejects_plain_ridge_surrogate():
    """Plain ridge surrogate is degenerate (day-global top-K) and must be rejected."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=8, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Fit a model so check_is_fitted passes (we want to hit the surrogate validation)
    feature_cols = design.cli_features + design.op_features
    X = design.logs_df[feature_cols].values
    y = design.logs_df["service_time"].values
    model = Ridge()
    model.fit(X, y)

    with pytest.raises(ValueError, match="degenerate"):
        induce_policy_stream_topk(
            models={"m": model},
            design=design,
            metric_col="service_time",
            surrogate_model="ridge",  # rejected
        )


@pytest.mark.parametrize("surrogate_model", ["hgb", "ridge_interaction"])
def test_stream_topk_personalization_recovery(surrogate_model: str):
    """Simulation proof: stream_topk top-K must vary across clients.

    Required by AGENTS.md / .claude/CLAUDE.md: any change to statistical
    evaluation logic must include a simulation that proves the code recovers
    a known ground-truth parameter.

    Ground truth setup
    ------------------
    We construct synthetic data where the optimal operator depends on a
    *client* feature: clients with ``cli_type=0`` should prefer operators with
    high ``op_a``; clients with ``cli_type=1`` should prefer operators with
    high ``op_b``. The true target is

        y = - (cli_type * op_b + (1 - cli_type) * op_a) + small_noise

    so ``direction="min"`` recovers each client's preferred operator.

    A degenerate surrogate (linear on concatenated [cli | op]) cannot
    distinguish the two client types and would produce the *same* top-K for
    all clients. The fixed surrogates ("hgb", "ridge_interaction") capture
    the cli x op interaction and produce *client-specific* top-K rankings —
    we verify this directly.
    """
    rng = np.random.RandomState(7)
    n_ops = 12
    n_clients_per_type = 200
    topk = 3

    # Operators: half are op_a-heavy, half are op_b-heavy (well separated)
    op_a = np.concatenate(
        [
            rng.uniform(0.8, 1.0, n_ops // 2),
            rng.uniform(0.0, 0.2, n_ops - n_ops // 2),
        ]
    )
    op_b = np.concatenate(
        [
            rng.uniform(0.0, 0.2, n_ops // 2),
            rng.uniform(0.8, 1.0, n_ops - n_ops // 2),
        ]
    )
    op_ids = [f"op_{i}" for i in range(n_ops)]

    op_daily_df = pd.DataFrame(
        {
            "operator_id": op_ids,
            "arrival_day": ["d1"] * n_ops,
            "op_a": op_a,
            "op_b": op_b,
        }
    )

    # Two client types with opposite preferences (cli_type repurposed as cli_x)
    cli_x = np.concatenate([np.zeros(n_clients_per_type), np.ones(n_clients_per_type)])
    n_clients = len(cli_x)
    chosen_idx = rng.randint(0, n_ops, size=n_clients)
    chosen_op_a = op_a[chosen_idx]
    chosen_op_b = op_b[chosen_idx]
    noise = rng.normal(0, 0.02, size=n_clients)
    y = -(cli_x * chosen_op_b + (1 - cli_x) * chosen_op_a) + noise

    logs_df = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n_clients)],
            "operator_id": [op_ids[i] for i in chosen_idx],
            "arrival_day": ["d1"] * n_clients,
            "arrival_ts": np.arange(n_clients),
            "cli_x": cli_x,
            "op_a": chosen_op_a,
            "op_b": chosen_op_b,
            "service_time": y,
        }
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test the surrogate's *ranking* per client directly. This is the
    # personalization property. We do NOT route through the full model —
    # whether the full model can also resolve the interaction is a separate
    # concern (and depends on its capacity / data).
    X_cli_train = design.logs_df[design.cli_features].values.astype(np.float32)
    X_op_train = design.logs_df[design.op_features].values.astype(np.float32)
    y_train = design.logs_df["service_time"].values.astype(np.float64)

    surrogate, feature_mode = _fit_surrogate(
        surrogate_model, X_cli_train, X_op_train, y_train, random_state=0
    )

    # Build features for one type-0 client and one type-1 client paired with
    # *all* operators, then score with the surrogate.
    op_features_full = op_daily_df[["op_a", "op_b"]].values.astype(np.float32)

    def _score_for(cli_x_value: float) -> np.ndarray:
        X_cli = np.full((n_ops, 1), cli_x_value, dtype=np.float32)
        X_in = _surrogate_features(feature_mode, X_cli, op_features_full)
        return surrogate.predict(X_in)

    score_type0 = _score_for(0.0)
    score_type1 = _score_for(1.0)

    # Top-K (smallest scores under direction="min") for each client type
    top0 = np.argpartition(score_type0, topk - 1)[:topk]
    top1 = np.argpartition(score_type1, topk - 1)[:topk]

    # Personalization signal: top-K rankings must differ between client types.
    # A degenerate (linear-on-concat) surrogate produces identical top-K, so
    # checking set inequality is the strongest possible assertion.
    assert set(top0) != set(top1), (
        f"surrogate={surrogate_model}: top-K is identical across client types "
        f"({sorted(top0)} == {sorted(top1)}); this is the day-global degeneracy "
        f"the fix must prevent."
    )

    # Stronger correctness check: the surrogate should rank op_a-heavy
    # operators highly for type-0 clients and op_b-heavy for type-1 clients.
    # op_a-heavy operators are the first half (indices 0..n_ops/2-1).
    a_heavy_set = set(range(n_ops // 2))
    b_heavy_set = set(range(n_ops // 2, n_ops))
    overlap_type0_with_a = len(set(top0) & a_heavy_set)
    overlap_type1_with_b = len(set(top1) & b_heavy_set)
    assert overlap_type0_with_a >= topk - 1, (
        f"surrogate={surrogate_model}: type-0 top-K should be op_a-heavy; "
        f"got {sorted(top0)} (a-heavy indices: {sorted(a_heavy_set)})"
    )
    assert overlap_type1_with_b >= topk - 1, (
        f"surrogate={surrogate_model}: type-1 top-K should be op_b-heavy; "
        f"got {sorted(top1)} (b-heavy indices: {sorted(b_heavy_set)})"
    )


def test_pairwise_pre_split_holds_out_evaluation_data():
    """pre_split must reduce the evaluation set, fitting on earlier days only."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=4, n_clients_day=40, n_ops=6, seed=42
    )

    models = {"ridge": Ridge(random_state=42)}
    # pre_split should fit on first 75% of days, evaluate on remaining 25%
    report = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=2,
        strategy="direct",
        fit_models=True,
        policy_train="pre_split",
        policy_train_frac=0.75,
        random_state=42,
    ).report

    # Evaluation succeeded with held-out data
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert np.isfinite(report["V_hat"]).all()


def test_pairwise_invalid_policy_train_raises():
    """Unknown policy_train value must raise ValueError before any work."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=10, n_ops=4, seed=42
    )
    with pytest.raises(ValueError, match="Unknown policy_train"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models={"ridge": Ridge(random_state=42)},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            policy_train="bogus",
        )


@pytest.mark.parametrize("bad_frac", [-0.1, 0.0, 1.0, 1.5])
def test_pairwise_invalid_policy_train_frac_raises(bad_frac: float):
    """policy_train_frac outside (0, 1) must raise ValueError."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=10, n_ops=4, seed=42
    )
    with pytest.raises(ValueError, match=r"policy_train_frac must be in \(0, 1\)"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models={"ridge": Ridge(random_state=42)},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            policy_train_frac=bad_frac,
        )


def test_pairwise_pre_split_zero_eval_rows_raises():
    """pre_split with frac so high that the held-out tail is empty must raise."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=2, n_ops=3, seed=42
    )
    # 2 total rows * 0.999 -> int(1.998) = 1 train row, 1 eval row -> safe.
    # We need fit_models=True and frac so high the tail is empty.
    # int(2 * 0.9999) == 1, leaves 1 row. Force the edge: pick a frac whose
    # int() truncates to n_total, leaving 0 eval rows.
    # For n_total=2 we need int(2 * frac) == 2 -> frac in [1.0, ...) which is
    # rejected upstream. Instead use n_total=1 to hit the raise.
    logs_df = logs_df.iloc[:1].reset_index(drop=True)
    with pytest.raises(ValueError, match="pre_split left 0 evaluation rows"):
        evaluate_pairwise_models(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            models={"ridge": Ridge(random_state=42)},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            fit_models=True,
            policy_train="pre_split",
            policy_train_frac=0.85,
        )


def test_stream_topk_surrogate_predict_call_count_per_day(monkeypatch):
    """#63: surrogate.predict must be called O(n_chunks_per_day), not O(n_clients).

    With a chunk_pairs cap large enough to hold the whole day's grid in one
    call, the surrogate must be invoked exactly once per day. This is the
    hoist-from-per-client-to-day-level contract.
    """
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=40, n_ops=8, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    feature_cols = design.cli_features + design.op_features
    X = design.logs_df[feature_cols].values
    y = design.logs_df["service_time"].values
    model = Ridge(random_state=0)
    model.fit(X, y)

    # Wrap the surrogate's predict so we can count its invocations without
    # touching the full model. This is the hoist-from-per-client-to-day
    # contract for #63.
    original_fit_surrogate = pairwise_mod._fit_surrogate
    surrogate_calls: list[int] = []

    def counting_fit_surrogate(*args, **kwargs):
        surrogate, mode = original_fit_surrogate(*args, **kwargs)
        original_predict = surrogate.predict

        def counting_predict(X_):
            surrogate_calls.append(len(X_))
            return original_predict(X_)

        surrogate.predict = counting_predict  # type: ignore[method-assign]
        return surrogate, mode

    monkeypatch.setattr(pairwise_mod, "_fit_surrogate", counting_fit_surrogate)

    induce_policy_stream_topk(
        models={"ridge": model},
        design=design,
        metric_col="service_time",
        topk=3,
        surrogate_model="ridge_interaction",
        chunk_pairs=10_000_000,  # large enough to fit each day in one chunk
    )

    # 3 days, 1 surrogate predict per day = 3 calls total.
    assert len(surrogate_calls) == 3, (
        f"Expected one surrogate.predict per day; got {len(surrogate_calls)} "
        f"calls with sizes {surrogate_calls}"
    )
    # Each call must cover the full (n_clients_day * n_day_ops) grid.
    assert all(size == 40 * 8 for size in surrogate_calls), (
        f"Each per-day surrogate call should score n_clients_day*n_day_ops="
        f"320 rows; got sizes {surrogate_calls}"
    )


def test_stream_topk_chunk_pairs_controls_batching_and_preserves_policy():
    """#61: chunk_pairs must actually chunk along clients without changing output.

    Running stream_topk with a tiny chunk_pairs (forces multiple chunks per day)
    and a huge chunk_pairs (one chunk per day) must produce *identical* policies.
    This is the parity guarantee for the chunked rewrite.
    """
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=25, n_ops=6, seed=123
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    feature_cols = design.cli_features + design.op_features
    X = design.logs_df[feature_cols].values
    y = design.logs_df["service_time"].values
    model = Ridge(random_state=0)
    model.fit(X, y)

    common = {
        "models": {"ridge": model},
        "design": design,
        "metric_col": "service_time",
        "topk": 3,
        "surrogate_model": "ridge_interaction",
        "random_state": 0,
    }
    # Tiny chunk → chunk_size == max(1, 24 // 6) == 4 clients per chunk
    tiny = induce_policy_stream_topk(**common, chunk_pairs=24)
    # Huge chunk → one chunk per day
    huge = induce_policy_stream_topk(**common, chunk_pairs=10_000_000)

    np.testing.assert_array_equal(tiny["ridge"], huge["ridge"])


def test_stream_topk_chunk_pairs_forwarded_through_induce_policy(monkeypatch):
    """induce_policy(strategy='stream_topk', chunk_pairs=...) must forward the value."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=6, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    feature_cols = design.cli_features + design.op_features
    X = design.logs_df[feature_cols].values
    y = design.logs_df["service_time"].values
    model = Ridge(random_state=0)
    model.fit(X, y)

    captured: dict[str, int] = {}
    original_stream_topk = pairwise_mod.induce_policy_stream_topk

    def spy_stream_topk(*args, **kwargs):
        captured["chunk_pairs"] = kwargs.get("chunk_pairs")
        return original_stream_topk(*args, **kwargs)

    monkeypatch.setattr(pairwise_mod, "induce_policy_stream_topk", spy_stream_topk)

    pairwise_mod.induce_policy(
        models={"ridge": model},
        design=design,
        strategy="stream_topk",
        metric_col="service_time",
        topk=2,
        chunk_pairs=12345,
        surrogate_model="ridge_interaction",
    )

    assert captured["chunk_pairs"] == 12345, (
        f"induce_policy did not forward chunk_pairs to stream_topk; "
        f"got {captured.get('chunk_pairs')!r}"
    )


def test_stream_topk_day_with_no_operators_raises():
    """A day with clients but no operator candidates must fail loud.

    ``PairwiseDesign.from_dataframes`` keeps ``ops_all_by_day`` and
    ``day_to_op_df`` in sync, so we break the invariant directly to
    exercise the defensive raise added per the audit (F3).
    """
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=5, n_ops=4, seed=42
    )

    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values
    model = Ridge(random_state=42)
    model.fit(X, y)

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    day_key = next(iter(design.ops_all_by_day))
    design.day_to_op_df.pop(day_key)  # break the invariant

    with pytest.raises(DataValidationError, match="no operator candidates"):
        induce_policy_stream_topk(
            models={"ridge": model},
            design=design,
            metric_col="service_time",
            topk=2,
        )


if __name__ == "__main__":
    pytest.main([__file__])
