"""Worked OPE recipes built on the core estimators (#95).

Recipes are opinionated, end-to-end helpers for a specific use case. They add
no new estimators — they wire the existing ones (here, embedding-MIPS) into a
ready-to-run workflow with a synthetic generator, a log-schema validator, and a
policy-induction convenience.

Public API:

* :func:`make_llm_reranker_synth` — deterministic LLM-reranker dataset with a
  known ground-truth target value.
* :class:`LLMRerankerLogSchema` — Pydantic schema validating user logs.
* :func:`induce_reranker_policy` — turn a Python scoring callable into a
  per-decision target-policy probability matrix.
* :func:`evaluate_reranker_mips` — evaluate a reranker policy with MIPS.
"""

from __future__ import annotations

from .llm_reranker import (
    LLMRerankerGroundTruth,
    LLMRerankerLogSchema,
    evaluate_reranker_mips,
    induce_reranker_policy,
    make_llm_reranker_synth,
)

__all__ = [
    "LLMRerankerGroundTruth",
    "LLMRerankerLogSchema",
    "evaluate_reranker_mips",
    "induce_reranker_policy",
    "make_llm_reranker_synth",
]
