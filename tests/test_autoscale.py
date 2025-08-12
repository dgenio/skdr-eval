"""Test autoscale strategy selection."""

import pytest

from skdr_eval import make_pairwise_synth
from skdr_eval.pairwise import PairwiseDesign, choose_strategy


def test_choose_strategy_direct():
    """Test that small datasets choose direct strategy."""
    stats = {
        "candidate_pairs": 5_000_000,  # 5M pairs
        "n_features": 10,
        "memory_gb": 0.2,
    }

    strategy = choose_strategy(stats)
    assert strategy == "direct"


def test_choose_strategy_stream():
    """Test that medium datasets choose stream strategy."""
    stats = {
        "candidate_pairs": 50_000_000,  # 50M pairs
        "n_features": 15,
        "memory_gb": 3.0,
    }

    strategy = choose_strategy(stats)
    assert strategy == "stream"


def test_choose_strategy_stream_topk():
    """Test that large datasets choose stream_topk strategy."""
    stats = {
        "candidate_pairs": 500_000_000,  # 500M pairs
        "n_features": 20,
        "memory_gb": 40.0,
    }

    strategy = choose_strategy(stats)
    assert strategy == "stream_topk"


def test_pairwise_design_stats():
    """Test PairwiseDesign statistics calculation."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3, n_clients_day=1000, n_ops=50, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    stats = design.get_stats()

    # Check basic stats
    assert stats["n_rows"] == 3000  # 3 days * 1000 clients
    assert stats["n_days"] == 3
    assert stats["avg_ops_per_day"] == 50
    assert stats["candidate_pairs"] > 0
    assert stats["n_features"] > 0
    assert stats["memory_gb"] > 0

    # Check that candidate pairs is reasonable
    # Should be roughly n_rows * avg_eligible_ops
    expected_min = stats["n_rows"] * 10  # At least 10 ops per client on average
    expected_max = stats["n_rows"] * stats["avg_ops_per_day"]  # At most all ops

    assert expected_min <= stats["candidate_pairs"] <= expected_max


def test_autoscale_with_eligibility():
    """Test autoscale calculation with eligibility constraints."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=2, n_clients_day=500, n_ops=100, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    stats = design.get_stats()

    # With eligibility, candidate pairs should be less than n_rows * n_ops
    max_possible = stats["n_rows"] * 100  # All operators for all clients
    assert stats["candidate_pairs"] < max_possible

    # But should still be substantial (at least 50% eligible on average)
    min_expected = stats["n_rows"] * 50
    assert stats["candidate_pairs"] >= min_expected


def test_strategy_boundaries():
    """Test strategy selection at boundary conditions."""
    # Test exactly at 10M boundary
    stats_10m = {"candidate_pairs": 10_000_000}
    assert choose_strategy(stats_10m) == "direct"

    stats_10m_plus = {"candidate_pairs": 10_000_001}
    assert choose_strategy(stats_10m_plus) == "stream"

    # Test exactly at 200M boundary
    stats_200m = {"candidate_pairs": 200_000_000}
    assert choose_strategy(stats_200m) == "stream"

    stats_200m_plus = {"candidate_pairs": 200_000_001}
    assert choose_strategy(stats_200m_plus) == "stream_topk"


def test_design_feature_extraction():
    """Test that PairwiseDesign correctly extracts features."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=100, n_ops=20, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)

    # Check that client features are extracted
    expected_cli_features = [col for col in logs_df.columns if col.startswith("cli_")]
    assert set(design.cli_features) == set(expected_cli_features)

    # Check that operator features are extracted
    expected_op_features = [col for col in op_daily_df.columns if col.startswith("op_")]
    assert set(design.op_features) == set(expected_op_features)

    # Check day mapping
    assert len(design.ops_all_by_day) == 1  # 1 day
    assert len(design.ops_all_by_day["day_00"]) == 20  # 20 operators


def test_memory_estimation():
    """Test memory estimation in statistics."""
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=1, n_clients_day=1000, n_ops=100, seed=42
    )

    design = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    stats = design.get_stats()

    # Memory should be positive
    assert stats["memory_gb"] > 0

    # Should be roughly candidate_pairs * n_features * 4 bytes / (1024^3)
    expected_memory = (stats["candidate_pairs"] * stats["n_features"] * 4) / (1024**3)

    # Allow some tolerance for rounding
    assert abs(stats["memory_gb"] - expected_memory) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
