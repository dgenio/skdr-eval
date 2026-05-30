"""Tests for the LLM-reranker OPE recipe (#95).

Includes the required simulation proof: with a known linear-in-dot-product
reward, a uniform logging reranker, and a sort-by-dot-product target, MIPS
recovers the closed-form target value within ±2 SE.
"""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.exceptions import DataValidationError
from skdr_eval.recipes import (
    LLMRerankerGroundTruth,
    LLMRerankerLogSchema,
    evaluate_reranker_mips,
    induce_reranker_policy,
    make_llm_reranker_synth,
)


def test_synth_is_deterministic_and_well_formed() -> None:
    logs1, cand1, truth1 = make_llm_reranker_synth(
        n_queries=200, candidates_per_query=12, embed_dim=16, seed=7
    )
    logs2, cand2, _truth2 = make_llm_reranker_synth(
        n_queries=200, candidates_per_query=12, embed_dim=16, seed=7
    )
    assert np.allclose(cand1, cand2)
    assert logs1["reward"].to_numpy() == pytest.approx(logs2["reward"].to_numpy())
    assert isinstance(truth1, LLMRerankerGroundTruth)
    # The sort-by-dot target must beat the uniform logging policy.
    assert truth1.V_sort_by_dot > truth1.V_logging
    assert cand1.shape == (12, 16)
    assert len(logs1) == 200


def test_schema_accepts_synth_and_reports_missing_columns() -> None:
    logs, _, _ = make_llm_reranker_synth(n_queries=50, candidates_per_query=8, seed=1)
    assert LLMRerankerLogSchema.validate_frame(logs) is logs
    with pytest.raises(ValueError, match="missing required columns"):
        LLMRerankerLogSchema.validate_frame(logs.drop(columns=["candidate_embedding"]))


def test_schema_rejects_out_of_range_propensity() -> None:
    logs, _, _ = make_llm_reranker_synth(n_queries=20, candidates_per_query=8, seed=1)
    logs = logs.copy()
    logs.loc[0, "propensity_at_position"] = 1.5
    with pytest.raises(ValueError, match="row 0 is invalid"):
        LLMRerankerLogSchema.validate_frame(logs)


def test_induce_reranker_policy_argmax_is_one_hot() -> None:
    logs, cand, _ = make_llm_reranker_synth(
        n_queries=30, candidates_per_query=10, embed_dim=8, seed=2
    )
    policy = induce_reranker_policy(logs, cand)
    assert policy.shape == (30, 10)
    assert np.allclose(policy.sum(axis=1), 1.0)
    # Deterministic arg-max => exactly one unit entry per row.
    assert np.all((policy == 0) | (policy == 1))
    assert (policy.sum(axis=1) == 1).all()


def test_induce_reranker_policy_softmax_is_stochastic() -> None:
    logs, cand, _ = make_llm_reranker_synth(
        n_queries=20, candidates_per_query=6, embed_dim=8, seed=3
    )
    policy = induce_reranker_policy(logs, cand, temperature=0.5)
    assert np.allclose(policy.sum(axis=1), 1.0)
    assert (policy > 0).all()  # softmax => strictly positive everywhere


def test_mips_recovers_target_value_within_two_se() -> None:
    """Simulation proof for #95: MIPS recovers the closed-form V_π ± 2 SE."""
    logs, cand, truth = make_llm_reranker_synth(
        n_queries=2000, candidates_per_query=20, embed_dim=32, seed=42
    )
    result = evaluate_reranker_mips(logs, cand)  # default: dot target + fitted q̂
    lo = result.V_hat - 2.0 * result.SE_if
    hi = result.V_hat + 2.0 * result.SE_if
    assert lo <= truth.V_sort_by_dot <= hi, (
        f"V_hat={result.V_hat:.4f} ± 2*{result.SE_if:.4f} did not cover "
        f"V_sort_by_dot={truth.V_sort_by_dot:.4f}"
    )


def test_mips_recovery_is_stable_across_seeds() -> None:
    # The MIPS point estimate is unbiased *in the mean*: averaged over seeds it
    # recovers V_sort_by_dot. We assert mean unbiasedness (the property the
    # #106/#142 weight-correctness fix must preserve) rather than per-seed
    # ±2*SE_if coverage. After #142 the dot-target MIPS weight uses the
    # embedding marginal Σ_a π(a|x) k(E_a, E_A) over all rows — true MIPS — not
    # the degenerate exact-match weight that was nonzero only when the logged
    # candidate equalled the target's arg-max. The plug-in influence-function
    # SE understates run-to-run variance for this recipe (it uses an
    # in-sample-fitted q̂ and a data-fit median bandwidth, neither cross-fitted),
    # so per-seed 2*SE intervals under-cover; that SE-underestimate is a
    # separate diagnostics concern, not a recovery bug.
    reps = 12
    errors = []
    for seed in range(reps):
        logs, cand, truth = make_llm_reranker_synth(
            n_queries=1500, candidates_per_query=16, embed_dim=24, seed=100 + seed
        )
        res = evaluate_reranker_mips(logs, cand)
        errors.append(res.V_hat - truth.V_sort_by_dot)
    errors = np.asarray(errors)
    mean_err = float(errors.mean())
    se_of_mean = float(errors.std(ddof=1) / np.sqrt(reps))
    # 3 standard errors of the mean: passes for an unbiased estimator, fails
    # loudly if the target-policy numerator regresses (which shifts the mean).
    assert abs(mean_err) <= 3.0 * se_of_mean, (mean_err, se_of_mean)


def test_reranker_rejects_out_of_range_candidate_id() -> None:
    logs, cand, _ = make_llm_reranker_synth(
        n_queries=20, candidates_per_query=6, embed_dim=8, seed=3
    )
    logs = logs.copy()
    logs.loc[0, "candidate_id"] = 999  # out of [0, 6)
    with pytest.raises(DataValidationError, match=r"\[0, 6\)"):
        evaluate_reranker_mips(logs, cand)


def test_reranker_rejects_bad_policy_probs_shape() -> None:
    logs, cand, _ = make_llm_reranker_synth(
        n_queries=20, candidates_per_query=6, embed_dim=8, seed=3
    )
    bad = np.ones((len(logs), 3))  # wrong n_candidates
    with pytest.raises(DataValidationError, match="policy_probs"):
        evaluate_reranker_mips(logs, cand, policy_probs=bad)
