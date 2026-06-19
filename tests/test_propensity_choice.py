"""Test propensity estimation and choice models."""

from unittest.mock import patch

import numpy as np
import pytest

from skdr_eval import make_pairwise_synth
from skdr_eval.core import estimate_propensity_pairwise
from skdr_eval.pairwise import PairwiseDesign

try:
    from skdr_eval.choice import (
        SCIPY_AVAILABLE,
        fit_conditional_logit,
        fit_conditional_logit_with_sampling,
        sample_negative_pairs,
    )
except ImportError:
    SCIPY_AVAILABLE = False
    fit_conditional_logit = None
    fit_conditional_logit_with_sampling = None
    sample_negative_pairs = None


def _make_condlogit_data(n_sets, k_alts, beta, rng):
    """Build a conditional-logit dataset with a known ground-truth ``beta``.

    For each of ``n_sets`` choice sets the chosen alternative is drawn from the
    softmax over ``X @ beta``, so a correctly fitted model should recover
    ``beta`` (up to regularization and sampling noise).
    """
    n_features = len(beta)
    n_pairs = n_sets * k_alts
    x = rng.normal(0, 1, (n_pairs, n_features)).astype(np.float32)
    choice_ids = np.repeat(np.arange(n_sets), k_alts)
    y = np.zeros(n_pairs)
    utilities = x @ np.asarray(beta, dtype=np.float64)
    for s in range(n_sets):
        idx = np.where(choice_ids == s)[0]
        u = utilities[idx]
        probs = np.exp(u - u.max())
        probs /= probs.sum()
        y[rng.choice(idx, p=probs)] = 1
    return x, choice_ids, y


def test_propensity_multinomial_fallback():
    """Test that multinomial propensity works when scipy unavailable."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=100, n_ops=10, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Force multinomial method
    propensities = estimate_propensity_pairwise(
        design,
        method="multinomial",
        n_splits=2,
        random_state=42,
    )

    # Check shape
    n_decisions = len(logs_df)
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())
    assert propensities.shape == (n_decisions, max_ops)

    # Check that probabilities are non-negative
    assert (propensities >= 0).all()

    # Check that each row sums to approximately 1 (within eligible operators)
    for i in range(n_decisions):
        row_sum = np.sum(propensities[i, :])
        if row_sum > 0:  # Skip rows with no eligible operators
            assert abs(row_sum - 1.0) < 1e-6, (
                f"Row {i} probabilities don't sum to 1: {row_sum}"
            )


def test_propensity_auto_selection():
    """Test automatic propensity method selection."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=50, n_ops=8, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test auto selection (should work with small data)
    propensities = estimate_propensity_pairwise(
        design, method="auto", n_splits=2, random_state=42
    )

    assert propensities.shape[0] == len(logs_df)
    assert (propensities >= 0).all()


def test_propensity_auto_selects_multinomial_when_scipy_unavailable():
    """Test that method='auto' resolves to 'multinomial' when SciPy is unavailable."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=50, n_ops=5, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    with (
        patch("skdr_eval.core.SCIPY_AVAILABLE", False),
        patch("skdr_eval.core.fit_conditional_logit_with_sampling") as mock_condlogit,
    ):
        propensities = estimate_propensity_pairwise(
            design, method="auto", n_splits=2, random_state=42
        )
        mock_condlogit.assert_not_called()

    assert propensities.shape[0] == len(logs_df)
    assert (propensities >= 0).all()


def test_propensity_auto_selects_condlogit_when_scipy_available():
    """Test that method='auto' resolves to 'condlogit' when SciPy is available."""
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=50, n_ops=5, seed=42
    )
    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    with (
        patch("skdr_eval.core.SCIPY_AVAILABLE", True),
        patch(
            "skdr_eval.core.fit_conditional_logit_with_sampling",
            wraps=fit_conditional_logit_with_sampling,
        ) as mock_condlogit,
    ):
        propensities = estimate_propensity_pairwise(
            design, method="auto", n_splits=2, random_state=42
        )
        assert mock_condlogit.called

    assert propensities.shape[0] == len(logs_df)
    assert (propensities >= 0).all()


@pytest.mark.skipif(True, reason="SciPy conditional logit test - may not be available")
def test_conditional_logit_with_scipy():
    """Test conditional logit when scipy is available."""
    try:
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available")

        # Create simple test data
        np.random.seed(42)
        n_pairs = 1000
        n_features = 5
        n_choices = 100

        X = np.random.normal(0, 1, (n_pairs, n_features)).astype(np.float32)
        choice_ids = np.random.randint(0, n_choices, n_pairs)

        # Create realistic choice outcomes (one chosen per choice set)
        y = np.zeros(n_pairs)
        for choice_id in range(n_choices):
            mask = choice_ids == choice_id
            if np.sum(mask) > 0:
                # Randomly choose one option in each choice set
                choice_indices = np.where(mask)[0]
                chosen_idx = np.random.choice(choice_indices)
                y[chosen_idx] = 1

        # Fit conditional logit
        coef, intercept, temp = fit_conditional_logit_with_sampling(
            X, choice_ids, y, neg_per_pos=3, random_state=42
        )

        # Check that we get reasonable coefficients
        assert len(coef) == n_features
        assert isinstance(intercept, float)
        assert temp > 0

    except ImportError:
        pytest.skip("SciPy not available for conditional logit test")


def test_propensity_with_eligibility():
    """Test propensity estimation respects eligibility constraints."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=30, n_ops=6, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    propensities = estimate_propensity_pairwise(
        design, method="multinomial", n_splits=2, random_state=42
    )

    # Check that ineligible operators have zero probability
    for i, row in logs_df.iterrows():
        day = row["arrival_day"]
        elig_ops = row["elig_mask"]

        if isinstance(elig_ops, list) and day in design.ops_all_by_day:
            day_ops = design.ops_all_by_day[day]
            for j, op in enumerate(day_ops):
                if op not in elig_ops:
                    assert propensities[i, j] == 0, (
                        f"Ineligible operator {op} has non-zero probability"
                    )


def test_propensity_normalization():
    """Test that propensities are properly normalized."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=20, n_ops=5, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    propensities = estimate_propensity_pairwise(
        design, method="multinomial", n_splits=2, random_state=42
    )

    # Check normalization for each decision
    for i in range(len(logs_df)):
        day = logs_df.iloc[i]["arrival_day"]
        if day in design.ops_all_by_day:
            n_ops_day = len(design.ops_all_by_day[day])
            row_probs = propensities[i, :n_ops_day]
            row_sum = np.sum(row_probs)

            if row_sum > 0:  # Skip if no eligible operators
                assert abs(row_sum - 1.0) < 1e-6, (
                    f"Row {i} probabilities sum to {row_sum}, not 1.0"
                )


def test_choice_model_sampling():
    """Test negative sampling in choice models."""
    try:
        # Create test data with multiple choice sets
        np.random.seed(42)
        n_pairs = 200
        n_features = 4

        X = np.random.normal(0, 1, (n_pairs, n_features)).astype(np.float32)
        choice_ids = np.repeat(np.arange(20), 10)  # 20 choice sets, 10 options each

        # Create outcomes (one positive per choice set)
        y = np.zeros(n_pairs)
        for i in range(20):
            start_idx = i * 10
            chosen_idx = start_idx + np.random.randint(0, 10)
            y[chosen_idx] = 1

        # Sample negatives
        X_sampled, choice_ids_sampled, y_sampled = sample_negative_pairs(
            X, choice_ids, y, neg_per_pos=3, random_state=42
        )

        # Check that we have fewer samples
        assert len(X_sampled) < len(X)
        assert len(X_sampled) == len(choice_ids_sampled) == len(y_sampled)

        # Check that all positives are kept
        assert np.sum(y_sampled) == np.sum(y)  # Same number of positives

        # Check that we have some negatives
        assert np.sum(y_sampled == 0) > 0

    except ImportError:
        pytest.skip("Choice module not available")


def test_propensity_error_handling():
    """Test error handling in propensity estimation."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=10, n_ops=3, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Test with invalid method
    with pytest.raises(ValueError):
        estimate_propensity_pairwise(design, method="invalid_method", random_state=42)


def test_large_dataset_fallback():
    """Test that large datasets fall back to multinomial."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=100, n_ops=20, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Mock large dataset stats to force fallback
    with patch.object(design, "get_stats") as mock_stats:
        mock_stats.return_value = {
            "candidate_pairs": 100_000_000,  # Large number to force fallback
            "n_rows": 100,
            "n_days": 1,
        }

        propensities = estimate_propensity_pairwise(
            design, method="auto", n_splits=2, random_state=42
        )

        # Should still work (fallback to multinomial)
        assert propensities.shape[0] == len(logs_df)
        assert (propensities >= 0).all()


def test_sample_negative_pairs_isolated_from_global_seed():
    """Negative sampling must not depend on the global ``np.random`` state."""
    if sample_negative_pairs is None:
        pytest.skip("Choice module not available")

    rng = np.random.default_rng(0)
    n_pairs = 200
    x = rng.normal(0, 1, (n_pairs, 4)).astype(np.float32)
    choice_ids = np.repeat(np.arange(20), 10)
    y = np.zeros(n_pairs)
    for i in range(20):
        y[i * 10 + int(rng.integers(0, 10))] = 1

    np.random.seed(123)
    _, ids_a, y_a = sample_negative_pairs(
        x, choice_ids, y, neg_per_pos=3, random_state=7
    )
    np.random.seed(999)  # unrelated global reseed between calls
    _, ids_b, y_b = sample_negative_pairs(
        x, choice_ids, y, neg_per_pos=3, random_state=7
    )

    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_fit_conditional_logit_deterministic():
    """The same ``random_state`` yields identical coefficients."""
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    x, choice_ids, y = _make_condlogit_data(
        n_sets=400, k_alts=4, beta=[1.0, -0.5], rng=np.random.default_rng(0)
    )
    coef_a, int_a, _ = fit_conditional_logit(x, choice_ids, y, random_state=42)
    coef_b, int_b, _ = fit_conditional_logit(x, choice_ids, y, random_state=42)

    # Both fits run the identical computation in-process, so a tight tolerance
    # still flags any determinism/RNG regression: such a regression shifts the
    # optimizer's initialization (and trajectory) by orders of magnitude more
    # than this. Avoids brittle exact-equality on floating-point outputs.
    np.testing.assert_allclose(coef_a, coef_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(int_a, int_b, rtol=0, atol=1e-12)


def test_fit_conditional_logit_isolated_from_global_seed():
    """An unrelated ``np.random.seed`` must not change the fitted coefficients."""
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    x, choice_ids, y = _make_condlogit_data(
        n_sets=400, k_alts=4, beta=[1.0, -0.5], rng=np.random.default_rng(1)
    )

    np.random.seed(0)
    coef_a, int_a, _ = fit_conditional_logit(x, choice_ids, y, random_state=42)
    np.random.seed(123456)  # would change a global-RNG initialization
    coef_b, int_b, _ = fit_conditional_logit(x, choice_ids, y, random_state=42)

    # Tight tolerance rather than exact ==: any leak of global state into the
    # fit would move the coefficients far more than this.
    np.testing.assert_allclose(coef_a, coef_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(int_a, int_b, rtol=0, atol=1e-12)


def test_fit_conditional_logit_does_not_mutate_global_rng():
    """Fitting must not reseed or consume the global ``np.random`` stream.

    This is the regression the issue targets: the old code called
    ``np.random.seed(random_state)``, silently reseeding the process-wide RNG
    and altering a caller's subsequent draws.
    """
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    x, choice_ids, y = _make_condlogit_data(
        n_sets=300, k_alts=3, beta=[1.0, -0.5], rng=np.random.default_rng(3)
    )

    np.random.seed(0)
    expected = np.random.random(5)

    np.random.seed(0)
    fit_conditional_logit(x, choice_ids, y, random_state=42)
    after = np.random.random(5)

    np.testing.assert_array_equal(expected, after)


def test_sample_negative_pairs_does_not_mutate_global_rng():
    """Negative sampling must not reseed or consume the global ``np.random`` stream."""
    if sample_negative_pairs is None:
        pytest.skip("Choice module not available")

    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (200, 4)).astype(np.float32)
    choice_ids = np.repeat(np.arange(20), 10)
    y = np.zeros(200)
    for i in range(20):
        y[i * 10 + int(rng.integers(0, 10))] = 1

    np.random.seed(0)
    expected = np.random.random(5)

    np.random.seed(0)
    sample_negative_pairs(x, choice_ids, y, neg_per_pos=3, random_state=7)
    after = np.random.random(5)

    np.testing.assert_array_equal(expected, after)


def test_fit_conditional_logit_accepts_generator_and_none():
    """``random_state`` accepts int, Generator, and None without error."""
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    x, choice_ids, y = _make_condlogit_data(
        n_sets=200, k_alts=3, beta=[0.8, -0.3], rng=np.random.default_rng(2)
    )
    for seed in (0, np.random.default_rng(5), None):
        coef, _intercept, temp = fit_conditional_logit(
            x, choice_ids, y, random_state=seed
        )
        assert coef.shape == (2,)
        assert np.isfinite(coef).all()
        assert temp == 1.0


def test_fit_conditional_logit_recovers_ground_truth():
    """Simulation proof: the fit recovers a known conditional-logit ``beta``.

    Required for statistical-logic changes (docs/agent-context/review-checklist.md):
    initialization from a local Generator must not break parameter recovery.
    """
    if not SCIPY_AVAILABLE:
        pytest.skip("SciPy not installed in this environment")

    beta = np.array([1.5, -1.0])
    x, choice_ids, y = _make_condlogit_data(
        n_sets=4000, k_alts=4, beta=beta, rng=np.random.default_rng(2024)
    )

    coef, _, _ = fit_conditional_logit(
        x, choice_ids, y, l2=1e-3, maxiter=500, random_state=0
    )

    # Coefficients recover the ground truth within sampling/regularization noise.
    np.testing.assert_allclose(coef, beta, atol=0.2)


def test_fit_conditional_logit_with_sampling_threads_generator():
    """A ``Generator`` is threaded as one stream through sampling + init.

    ``fit_conditional_logit_with_sampling`` passes ``random_state`` to both
    ``sample_negative_pairs`` and ``fit_conditional_logit``. The docstring
    promises that passing a ``Generator`` threads a single stream through both
    steps (it is advanced, not reset/reused). This verifies that contract:
    two fresh ``default_rng(k)`` instances reproduce the same fit, and the
    Generator's state advances across the call.
    """
    if fit_conditional_logit_with_sampling is None or not SCIPY_AVAILABLE:
        pytest.skip("SciPy/choice module not available")

    x, choice_ids, y = _make_condlogit_data(
        n_sets=300, k_alts=4, beta=[1.0, -0.5], rng=np.random.default_rng(11)
    )

    # Two independent Generators seeded identically thread the same single
    # stream through negative sampling and optimizer init -> identical fit.
    coef_a, int_a, _ = fit_conditional_logit_with_sampling(
        x, choice_ids, y, neg_per_pos=3, random_state=np.random.default_rng(7)
    )
    coef_b, int_b, _ = fit_conditional_logit_with_sampling(
        x, choice_ids, y, neg_per_pos=3, random_state=np.random.default_rng(7)
    )
    np.testing.assert_allclose(coef_a, coef_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(int_a, int_b, rtol=0, atol=1e-12)

    # The Generator is consumed (threaded), not reset: a draw taken afterwards
    # does not match a fresh stream's first draw.
    gen = np.random.default_rng(7)
    fit_conditional_logit_with_sampling(
        x, choice_ids, y, neg_per_pos=3, random_state=gen
    )
    assert not np.array_equal(
        gen.normal(size=3), np.random.default_rng(7).normal(size=3)
    )


if __name__ == "__main__":
    pytest.main([__file__])
