"""LLM-reranker OPE recipe via embedding-MIPS (#95).

A team runs a *logged* reranker (the current production reranker) that, for
each query, picks one candidate from a candidate pool and observes a reward
(click / dwell / purchase). They want to evaluate a *new* reranker offline,
before A/B testing, using only the logged data and pre-computed embeddings —
**no runtime LLM dependency**.

The natural estimator is MIPS (#85): the action space (candidate pool) is
large, but the candidate embeddings carry its structure, so common support
need only hold over the embedding rather than the raw candidate id.

This module ships:

* :func:`make_llm_reranker_synth` — a deterministic synthetic generator with a
  closed-form target value ``V_π`` for the "sort by query·candidate" reranker.
* :class:`LLMRerankerLogSchema` — a Pydantic schema validating user-supplied
  logs against the documented column contract.
* :func:`induce_reranker_policy` — turn a ``score(query_emb, candidate_emb)``
  callable into a per-decision target-policy probability matrix.
* :func:`evaluate_reranker_mips` — assemble the MIPS inputs and return a
  :class:`~skdr_eval.core.DRResult`.

Modelling choice: each query logs a single *chosen* candidate (a top-1
reranker), so one log row = one decision, the candidate id is the action, and
``candidate_embedding`` is the action embedding. The logging policy is uniform
over the shared candidate pool, which makes the target value closed-form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from ..core import DRResult

__all__ = [
    "LLMRerankerGroundTruth",
    "LLMRerankerLogSchema",
    "evaluate_reranker_mips",
    "induce_reranker_policy",
    "make_llm_reranker_synth",
]

# Score callable: ``(query_embedding, candidate_embeddings) -> scores`` where
# ``candidate_embeddings`` is ``(n_candidates, embed_dim)`` and the returned
# scores are ``(n_candidates,)``.
RerankerScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

_REQUIRED_COLUMNS = (
    "query_id",
    "candidate_id",
    "candidate_rank",
    "query_embedding",
    "candidate_embedding",
    "propensity_at_position",
    "reward",
)


@dataclass(frozen=True)
class LLMRerankerGroundTruth:
    """Closed-form policy values for a synthetic LLM-reranker run.

    Attributes
    ----------
    V_logging : float
        Expected reward under the uniform logging reranker.
    V_sort_by_dot : float
        Expected reward under the target reranker that picks the candidate
        maximising the query·candidate dot product.
    theta : float
        Reward slope on the dot-product feature.
    n_candidates : int
        Size of the shared candidate pool.
    embed_dim : int
        Embedding dimension.
    """

    V_logging: float
    V_sort_by_dot: float
    theta: float
    n_candidates: int
    embed_dim: int


class LLMRerankerLogSchema(BaseModel):
    """Pydantic schema for a single LLM-reranker log row.

    Use :meth:`validate_frame` to validate a whole DataFrame; it checks the
    required columns are present and that every row parses, raising a
    ``pydantic.ValidationError`` with an actionable message otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_id: str
    candidate_id: int
    candidate_rank: int
    query_embedding: list[float]
    candidate_embedding: list[float]
    propensity_at_position: float
    reward: float

    @field_validator("propensity_at_position")
    @classmethod
    def _propensity_in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"propensity_at_position must be in (0, 1], got {v!r}")
        return v

    @field_validator("query_embedding", "candidate_embedding")
    @classmethod
    def _embedding_non_empty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("embedding vectors must be non-empty")
        return v

    @classmethod
    def validate_frame(cls, logs: pd.DataFrame) -> pd.DataFrame:
        """Validate a logs DataFrame row-by-row against the schema.

        Returns the input unchanged on success; raises with a clear message
        listing missing columns or the first offending row.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in logs.columns]
        if missing:
            raise ValueError(
                f"LLM-reranker logs are missing required columns: {missing}. "
                f"Expected columns: {list(_REQUIRED_COLUMNS)}."
            )
        for i, row in enumerate(logs.to_dict(orient="records")):
            payload = dict(row)
            payload["query_embedding"] = list(
                np.asarray(payload["query_embedding"], dtype=float).ravel()
            )
            payload["candidate_embedding"] = list(
                np.asarray(payload["candidate_embedding"], dtype=float).ravel()
            )
            try:
                cls(**payload)
            except Exception as exc:
                raise ValueError(f"LLM-reranker log row {i} is invalid: {exc}") from exc
        return logs


def _unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row to the unit sphere (zero rows left untouched)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return cast("np.ndarray", x / norms)


def make_llm_reranker_synth(
    n_queries: int = 500,
    candidates_per_query: int = 20,
    embed_dim: int = 64,
    seed: int = 42,
    *,
    theta: float = 0.5,
    noise: float = 0.1,
) -> tuple[pd.DataFrame, np.ndarray, LLMRerankerGroundTruth]:
    """Generate a deterministic LLM-reranker OPE dataset.

    The reward is a known linear function of the query·candidate dot product,
    ``r = 0.5 + theta · <q, c> + ε``, so the target value of the
    "sort by dot product" reranker is closed-form.

    Parameters
    ----------
    n_queries : int, default 500
        Number of logged queries (one chosen candidate each).
    candidates_per_query : int, default 20
        Size of the shared candidate pool.
    embed_dim : int, default 64
        Embedding dimension.
    seed : int, default 42
        RNG seed.
    theta : float, default 0.5
        Reward slope on the dot-product feature.
    noise : float, default 0.1
        Std of the Gaussian reward noise.

    Returns
    -------
    logs : pd.DataFrame
        One row per query with the columns validated by
        :class:`LLMRerankerLogSchema`.
    candidate_embeddings : np.ndarray, shape (candidates_per_query, embed_dim)
        The shared candidate pool embeddings (the MIPS action embedding).
    truth : LLMRerankerGroundTruth
        Closed-form logging / target policy values.
    """
    rng = np.random.default_rng(seed)
    n_candidates = candidates_per_query
    candidates = _unit_rows(rng.normal(size=(n_candidates, embed_dim)))
    queries = _unit_rows(rng.normal(size=(n_queries, embed_dim)))

    dot = queries @ candidates.T  # (n_queries, n_candidates)
    mu = 0.5 + theta * dot  # expected reward per (query, candidate)

    # Logging policy: uniform over the candidate pool.
    chosen = rng.integers(0, n_candidates, size=n_queries)
    propensity = np.full(n_queries, 1.0 / n_candidates, dtype=np.float64)
    mu_chosen = mu[np.arange(n_queries), chosen]
    reward = mu_chosen + rng.normal(0.0, noise, size=n_queries)

    logs = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n_queries)],
            "candidate_id": chosen.astype(int),
            "candidate_rank": np.zeros(n_queries, dtype=int),
            "query_embedding": [q.tolist() for q in queries],
            "candidate_embedding": [candidates[c].tolist() for c in chosen],
            "propensity_at_position": propensity,
            "reward": reward,
        }
    )

    # Closed-form policy values (noise is mean-zero).
    v_logging = float(mu.mean())  # uniform over candidates and queries
    v_sort_by_dot = float(mu[np.arange(n_queries), np.argmax(dot, axis=1)].mean())
    truth = LLMRerankerGroundTruth(
        V_logging=v_logging,
        V_sort_by_dot=v_sort_by_dot,
        theta=theta,
        n_candidates=n_candidates,
        embed_dim=embed_dim,
    )
    return logs, candidates, truth


def induce_reranker_policy(
    logs: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    score_fn: RerankerScoreFn | None = None,
    *,
    temperature: float = 0.0,
) -> np.ndarray:
    """Turn a scoring callable into a per-decision target-policy matrix.

    Parameters
    ----------
    logs : pd.DataFrame
        Logs with a ``query_embedding`` column (one row per decision).
    candidate_embeddings : np.ndarray, shape (n_candidates, embed_dim)
        The shared candidate pool.
    score_fn : callable, optional
        ``(query_embedding, candidate_embeddings) -> scores`` of shape
        ``(n_candidates,)``. Defaults to the dot-product reranker.
    temperature : float, default 0.0
        ``0.0`` yields a deterministic arg-max (top-1) policy; ``> 0`` yields a
        softmax over scores divided by the temperature.

    Returns
    -------
    policy_probs : np.ndarray, shape (n_decisions, n_candidates)
        Row-stochastic target-policy probabilities.
    """
    if score_fn is None:

        def score_fn(q: np.ndarray, cand: np.ndarray) -> np.ndarray:
            return cast("np.ndarray", cand @ q)

    query_emb = np.asarray(
        [np.asarray(q, dtype=np.float64).ravel() for q in logs["query_embedding"]],
        dtype=np.float64,
    )
    n = query_emb.shape[0]
    n_candidates = candidate_embeddings.shape[0]
    policy = np.zeros((n, n_candidates), dtype=np.float64)
    for i in range(n):
        scores = np.asarray(score_fn(query_emb[i], candidate_embeddings), dtype=float)
        if temperature <= 0.0:
            policy[i, int(np.argmax(scores))] = 1.0
        else:
            z = scores / temperature
            z -= z.max()
            ez = np.exp(z)
            policy[i] = ez / ez.sum()
    return policy


def _query_embeddings(logs: pd.DataFrame) -> np.ndarray:
    """Stack the per-row ``query_embedding`` column into ``(n, embed_dim)``."""
    return np.asarray(
        [np.asarray(q, dtype=np.float64).ravel() for q in logs["query_embedding"]],
        dtype=np.float64,
    )


def _fit_dot_outcome(
    logs: pd.DataFrame, candidate_embeddings: np.ndarray
) -> np.ndarray:
    """Fit a per-(query, candidate) outcome model on the dot-product feature.

    Regresses the observed reward on the observed query·candidate dot product
    with closed-form OLS, then predicts over the full candidate pool. This is
    the doubly-robust outcome model the recipe uses by default: it recovers the
    linear reward law from the logs without peeking at ground truth.
    """
    query_emb = _query_embeddings(logs)
    actions = logs["candidate_id"].to_numpy(dtype=np.int64)
    dot_full = query_emb @ candidate_embeddings.T  # (n, n_candidates)
    f_obs = dot_full[np.arange(len(logs)), actions]
    reward = logs["reward"].to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones_like(f_obs), f_obs])
    beta, *_ = np.linalg.lstsq(design, reward, rcond=None)
    return cast("np.ndarray", beta[0] + beta[1] * dot_full)


def _reranker_bootstrap_se(
    *,
    logs: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    actions: np.ndarray,
    elig: np.ndarray,
    q_hat_fixed: np.ndarray | None,
    bandwidth: float | str,
    kernel: str,
    n_bootstrap: int,
    seed: int,
) -> float:
    """Full-pipeline bootstrap SE for the reranker MIPS estimate (#142).

    Resamples decisions with replacement and, on each resample, **refits** the
    dot-feature outcome model before recomputing ``V_hat``. This captures the
    ``q̂`` estimation-error variance that the plug-in influence-function SE —
    which conditions on a fixed ``q̂`` — omits; that omission makes the IF-SE
    understate run-to-run variance (the per-seed ±2*SE interval under-covers).
    The kernel bandwidth depends only on the fixed candidate pool, so it is not
    re-resolved. When the caller supplied their own ``q_hat`` it is held fixed
    (resampled by row), since the recipe cannot know how to refit it.
    """
    from ..estimators import mips_value  # noqa: PLC0415

    n = len(Y)
    dot_full = _query_embeddings(logs) @ candidate_embeddings.T
    rng = np.random.default_rng(seed)
    v_hats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if q_hat_fixed is None:
            f_obs = dot_full[idx, actions[idx]]
            design = np.column_stack([np.ones_like(f_obs), f_obs])
            beta, *_ = np.linalg.lstsq(design, Y[idx], rcond=None)
            q_b = beta[0] + beta[1] * dot_full[idx]
        else:
            q_b = q_hat_fixed[idx]
        v_hats[b] = mips_value(
            propensities=propensities[idx],
            policy_probs=policy_probs[idx],
            Y=Y[idx],
            q_hat=q_b,
            A=actions[idx],
            elig=elig[idx],
            action_embedding=candidate_embeddings,
            bandwidth=bandwidth,
            kernel=kernel,
        ).V_hat
    return float(np.std(v_hats, ddof=1))


def evaluate_reranker_mips(
    logs: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    policy_probs: np.ndarray | None = None,
    *,
    q_hat: np.ndarray | None = None,
    bandwidth: float | str = "median",
    kernel: str = "rbf",
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> DRResult:
    """Evaluate a reranker target policy with MIPS.

    Assembles the MIPS inputs from the reranker logs and calls
    :func:`skdr_eval.mips_value`. When ``policy_probs`` is omitted, the
    deterministic dot-product reranker is used. When ``q_hat`` is omitted, a
    dot-feature outcome model is fitted from the logs (the doubly-robust form);
    pass an ``(n_decisions, n_candidates)`` array to supply your own.

    Standard error
    --------------
    By default the returned ``SE_if`` is the plug-in influence-function SE,
    which conditions on the fitted ``q̂`` and therefore **understates**
    run-to-run variance (its ±2*SE interval under-covers; #142). Pass
    ``n_bootstrap > 0`` (e.g. ``200``) to instead report a full-pipeline
    bootstrap SE that refits ``q̂`` on each resample — this captures the ``q̂``
    estimation variance and restores nominal coverage, at ``n_bootstrap`` extra
    MIPS evaluations.

    Returns
    -------
    DRResult
        With ``V_hat`` the MIPS estimate of the target reranker's value. When
        ``n_bootstrap > 0``, ``SE_if`` (and ``MSE_est``) are the bootstrap
        estimates; all other fields are the point-estimate values.
    """
    from ..estimators import mips_value  # noqa: PLC0415

    if policy_probs is None:
        policy_probs = induce_reranker_policy(logs, candidate_embeddings)

    n = len(logs)
    n_candidates = candidate_embeddings.shape[0]

    # Validate inputs up front. This is a public recipe entry point and the
    # arithmetic below uses ``candidate_id`` as a direct index into
    # ``(n, n_candidates)`` arrays, so an out-of-range or non-integer id would
    # otherwise surface as an opaque ``IndexError``.
    from ..exceptions import DataValidationError  # noqa: PLC0415

    raw_ids = logs["candidate_id"].to_numpy()
    if np.issubdtype(raw_ids.dtype, np.floating):
        if not np.all(np.mod(raw_ids, 1.0) == 0.0):
            raise DataValidationError(
                "candidate_id must be integer-valued action indices; got "
                f"non-integer values (dtype {raw_ids.dtype})."
            )
    elif not np.issubdtype(raw_ids.dtype, np.integer):
        raise DataValidationError(
            f"candidate_id must be integer action indices; got dtype {raw_ids.dtype}."
        )
    actions = logs["candidate_id"].to_numpy(dtype=np.int64)
    if actions.size and (actions.min() < 0 or actions.max() >= n_candidates):
        raise DataValidationError(
            f"candidate_id values must lie in [0, {n_candidates}) to index the "
            f"candidate pool; got range [{int(actions.min())}, "
            f"{int(actions.max())}] for {n_candidates} candidate embeddings."
        )
    expected_shape = (n, n_candidates)
    if policy_probs.shape != expected_shape:
        raise DataValidationError(
            f"policy_probs must have shape {expected_shape} "
            f"(n_decisions, n_candidates); got {policy_probs.shape}."
        )
    if q_hat is not None and q_hat.shape != expected_shape:
        raise DataValidationError(
            f"q_hat must have shape {expected_shape} "
            f"(n_decisions, n_candidates); got {q_hat.shape}."
        )

    pi_obs = logs["propensity_at_position"].to_numpy(dtype=np.float64)
    # Full logging-propensity matrix. The uniform reranker spreads mass
    # equally, so the unchosen candidates carry the residual mass; this keeps
    # each row a valid logging distribution for the MIPS embedding marginal.
    propensities = np.zeros((n, n_candidates), dtype=np.float64)
    if n_candidates > 1:
        rest = (1.0 - pi_obs) / (n_candidates - 1)
        propensities += rest[:, None]
    propensities[np.arange(n), actions] = pi_obs
    Y = logs["reward"].to_numpy(dtype=np.float64)
    user_supplied_q = q_hat is not None
    if q_hat is None:
        q_hat = _fit_dot_outcome(logs, candidate_embeddings)
    elig = np.ones((n, n_candidates), dtype=np.float64)

    result = mips_value(
        propensities=propensities,
        policy_probs=policy_probs,
        Y=Y,
        q_hat=q_hat,
        A=actions,
        elig=elig,
        action_embedding=candidate_embeddings,
        bandwidth=bandwidth,
        kernel=kernel,
    )

    if n_bootstrap > 0:
        from dataclasses import replace  # noqa: PLC0415

        se_boot = _reranker_bootstrap_se(
            logs=logs,
            candidate_embeddings=candidate_embeddings,
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            actions=actions,
            elig=elig,
            q_hat_fixed=q_hat if user_supplied_q else None,
            bandwidth=bandwidth,
            kernel=kernel,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        result = replace(result, SE_if=se_boot, MSE_est=se_boot**2)

    return result
