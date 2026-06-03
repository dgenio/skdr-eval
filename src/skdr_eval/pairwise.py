"""Pairwise evaluation design and autoscaling strategies."""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from .exceptions import DataValidationError

logger = logging.getLogger("skdr_eval")


@dataclass
class PairwiseDesign:
    """Design for pairwise (client, operator) evaluation.

    Attributes
    ----------
    logs_df : pd.DataFrame
        Observed decisions with columns: arrival_ts, client_id, operator_id, cli_*, op_*, target
    op_daily_df : pd.DataFrame
        Daily operator snapshots with columns: operator_id, arrival_day, op_*
    day_col : str
        Column name for arrival day
    client_id_col : str
        Column name for client ID
    operator_id_col : str
        Column name for operator ID
    elig_col : str | None
        Column name for eligibility mask (list of eligible operators)
    cli_features : List[str]
        Client feature column names (cli_*)
    op_features : List[str]
        Operator feature column names (op_*)
    ops_all_by_day : Dict[str, List[str]]
        Mapping from day to list of available operators
    day_to_op_df : Dict[str, pd.DataFrame]
        Cached mapping from day to operator dataframe
    """

    logs_df: pd.DataFrame
    op_daily_df: pd.DataFrame
    day_col: str
    client_id_col: str
    operator_id_col: str
    elig_col: str | None
    cli_features: list[str]
    op_features: list[str]
    ops_all_by_day: dict[str, list[str]]
    day_to_op_df: dict[str, pd.DataFrame]

    @classmethod
    def from_dataframes(
        cls,
        logs_df: pd.DataFrame,
        op_daily_df: pd.DataFrame,
        day_col: str = "arrival_day",
        client_id_col: str = "client_id",
        operator_id_col: str = "operator_id",
        elig_col: str | None = "elig_mask",
    ) -> "PairwiseDesign":
        """Create PairwiseDesign from dataframes."""
        # Reset index so propensity array positions always match DataFrame positions.
        # Callers may pass a filtered/sliced DataFrame whose index is not a
        # contiguous RangeIndex; without this reset, `propensities[i, ...]`
        # writes in estimate_propensity_pairwise would use label values as
        # positional offsets, silently corrupting data or raising IndexError.
        logs_df = logs_df.reset_index(drop=True)

        # Extract feature columns
        cli_features = [col for col in logs_df.columns if col.startswith("cli_")]
        op_features = [col for col in op_daily_df.columns if col.startswith("op_")]

        # Build day-to-operators mapping
        ops_all_by_day = {}
        day_to_op_df = {}

        for day in op_daily_df[day_col].unique():
            day_ops_df = op_daily_df[op_daily_df[day_col] == day].copy()
            ops_list = day_ops_df[operator_id_col].tolist()
            ops_all_by_day[day] = ops_list
            day_to_op_df[day] = day_ops_df

        return cls(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            day_col=day_col,
            client_id_col=client_id_col,
            operator_id_col=operator_id_col,
            elig_col=elig_col,
            cli_features=cli_features,
            op_features=op_features,
            ops_all_by_day=ops_all_by_day,
            day_to_op_df=day_to_op_df,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for autoscale decision."""
        n_rows = len(self.logs_df)
        n_days = len(self.ops_all_by_day)
        avg_ops_per_day = np.mean([len(ops) for ops in self.ops_all_by_day.values()])

        # Estimate eligibility
        if self.elig_col and self.elig_col in self.logs_df.columns:
            elig_lengths = self.logs_df[self.elig_col].apply(
                lambda x: len(x) if isinstance(x, (list, tuple)) else avg_ops_per_day
            )
            avg_elig: float = float(elig_lengths.mean())
        else:
            avg_elig = float(avg_ops_per_day)

        candidate_pairs = int(n_rows * avg_elig)
        n_features = (
            len(self.cli_features) + len(self.op_features) + 2
        )  # +2 for client_id, operator_id

        # Rough memory estimate (4 bytes per float32)
        memory_gb = (candidate_pairs * n_features * 4) / (1024**3)

        return {
            "n_rows": n_rows,
            "n_days": n_days,
            "avg_ops_per_day": avg_ops_per_day,
            "avg_elig": avg_elig,
            "candidate_pairs": candidate_pairs,
            "n_features": n_features,
            "memory_gb": memory_gb,
        }


# Strategy selection thresholds
DIRECT_STRATEGY_THRESHOLD = 10_000_000  # 10M pairs
STREAM_STRATEGY_THRESHOLD = 200_000_000  # 200M pairs


def choose_strategy(
    stats: dict[str, Any],
) -> Literal["direct", "stream", "stream_topk"]:
    """Choose autoscale strategy based on statistics."""
    candidate_pairs = stats["candidate_pairs"]

    if candidate_pairs <= DIRECT_STRATEGY_THRESHOLD:
        return "direct"
    elif candidate_pairs <= STREAM_STRATEGY_THRESHOLD:
        return "stream"
    else:
        return "stream_topk"


def build_candidate_pairs(
    design: PairwiseDesign,
    day: str,
    chunk_pairs: int = 2_000_000,
) -> Generator[pd.DataFrame, None, None]:
    """Build candidate pairs for a given day in chunks.

    Parameters
    ----------
    design : PairwiseDesign
        Pairwise design object
    day : str
        Day to build candidates for
    chunk_pairs : int
        Maximum pairs per chunk

    Yields
    ------
    pd.DataFrame
        Chunk of candidate pairs with columns: client_id, operator_id, cli_*, op_*
    """
    # Get client rows for this day
    day_clients = design.logs_df[design.logs_df[design.day_col] == day].copy()

    if len(day_clients) == 0:
        return

    # Get operator data for this day
    if day not in design.day_to_op_df:
        logger.warning(f"No operators found for day {day}")
        return

    day_ops = design.day_to_op_df[day].copy()

    # Build pairs in chunks
    pairs_generated = 0
    current_chunk = []

    for _, client_row in day_clients.iterrows():
        # Get eligible operators
        if design.elig_col and design.elig_col in client_row:
            elig_ops = client_row[design.elig_col]
            # Handle pandas Series or direct list/tuple values
            if hasattr(elig_ops, "iloc"):
                # It's a pandas Series, get the actual value
                elig_value = elig_ops.iloc[0] if len(elig_ops) > 0 else []
            else:
                elig_value = elig_ops

            if isinstance(elig_value, (list, tuple)):
                eligible_ops_df = day_ops[
                    day_ops[design.operator_id_col].isin(elig_value)
                ]
            else:
                eligible_ops_df = day_ops
        else:
            eligible_ops_df = day_ops

        # Create pairs for this client
        for _, op_row in eligible_ops_df.iterrows():
            pair_data = {
                design.client_id_col: client_row[design.client_id_col],
                design.operator_id_col: op_row[design.operator_id_col],
            }

            # Add client features
            for feat in design.cli_features:
                pair_data[feat] = client_row[feat]

            # Add operator features
            for feat in design.op_features:
                pair_data[feat] = op_row[feat]

            current_chunk.append(pair_data)
            pairs_generated += 1

            # Yield chunk if it's full
            if len(current_chunk) >= chunk_pairs:
                yield pd.DataFrame(current_chunk)
                current_chunk = []

    # Yield remaining pairs
    if current_chunk:
        yield pd.DataFrame(current_chunk)


def induce_policy_direct(
    models: dict[str, Any],
    design: PairwiseDesign,
    direction: Literal["min", "max"] = "min",
    chunk_pairs: int = 2_000_000,
) -> dict[str, np.ndarray]:
    """Induce policies using direct expansion strategy.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model_name -> fitted model
    design : PairwiseDesign
        Pairwise design object
    direction : Literal["min", "max"]
        Whether to minimize or maximize the predicted metric
    chunk_pairs : int
        Maximum pairs per chunk for memory management

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of model_name -> array of chosen operator_ids per decision
    """
    logger.info("Using direct expansion strategy")

    # Fail fast if any model is unfitted (validate once before any work)
    for model in models.values():
        check_is_fitted(model)

    policies: dict[str, list[str]] = {name: [] for name in models}

    # Process each day
    for day in sorted(design.ops_all_by_day.keys()):
        day_clients = design.logs_df[design.logs_df[design.day_col] == day]

        if len(day_clients) == 0:
            continue

        # Build a client_id -> day_clients.index mapping once per day; this
        # replaces an O(n_chunk * n_clients) iterrows-and-scan inside the
        # chunk loop with an O(n_clients) dict build + O(n_chunk) lookup.
        # First occurrence wins, matching the prior .index[0] semantics.
        unique_first = day_clients.drop_duplicates(
            subset=[design.client_id_col], keep="first"
        )
        client_id_to_idx = dict(
            zip(
                unique_first[design.client_id_col].values,
                unique_first.index.values,
                strict=False,
            )
        )

        # Collect all pairs for this day
        all_pairs = []
        client_indices: list[int] = []

        for chunk in build_candidate_pairs(design, day, chunk_pairs):
            if len(chunk) == 0:
                continue

            chunk_client_indices = (
                chunk[design.client_id_col].map(client_id_to_idx).tolist()
            )
            all_pairs.append(chunk)
            client_indices.extend(chunk_client_indices)

        if not all_pairs:
            continue

        # Combine all pairs for this day
        day_pairs_df = pd.concat(all_pairs, ignore_index=True)

        # Prepare features for prediction (exclude ID columns)
        feature_cols = design.cli_features + design.op_features
        X_pairs: np.ndarray = day_pairs_df[feature_cols].values.astype(np.float32)

        # Get predictions from each model. Prediction errors propagate (per
        # invariants.md: prefer fail-loud over silent first-operator fallback).
        for model_name, model in models.items():
            predictions = model.predict(X_pairs)

            # Group by client and find best operator
            client_indices_arr = np.asarray(client_indices)
            day_decisions = []
            for client_idx in day_clients.index:
                # Find pairs for this client
                client_mask = client_indices_arr == client_idx
                if not np.any(client_mask):
                    # No eligible operators is a data-quality issue, not a
                    # normal case. Per docs/agent-context/invariants.md,
                    # prefer fail-loud over silent first-operator fallback.
                    raise DataValidationError(
                        "No eligible operators for client; cannot induce a "
                        "policy. Check eligibility masks for empty rows.",
                        details={
                            "day": str(day),
                            "client_index": int(client_idx),
                            "strategy": "direct",
                        },
                    )

                client_preds = predictions[client_mask]
                client_ops = day_pairs_df.loc[
                    client_mask, design.operator_id_col
                ].values

                # Choose best operator
                if direction == "min":
                    best_idx = np.argmin(client_preds)
                else:
                    best_idx = np.argmax(client_preds)

                day_decisions.append(client_ops[best_idx])

            policies[model_name].extend(day_decisions)

    # Convert to numpy arrays
    return {name: np.array(decisions) for name, decisions in policies.items()}


def induce_policy_stream(
    models: dict[str, Any],
    design: PairwiseDesign,
    direction: Literal["min", "max"] = "min",
    chunk_pairs: int = 2_000_000,
) -> dict[str, np.ndarray]:
    """Induce policies using streaming strategy.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model_name -> fitted model
    design : PairwiseDesign
        Pairwise design object
    direction : Literal["min", "max"]
        Whether to minimize or maximize the predicted metric
    chunk_pairs : int
        Maximum pairs per chunk

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of model_name -> array of chosen operator_ids per decision
    """
    logger.info(f"Using streaming strategy with chunk_pairs={chunk_pairs}")

    # Fail fast if any model is unfitted (validate once before any work)
    for model in models.values():
        check_is_fitted(model)

    policies: dict[str, list[str]] = {name: [] for name in models}

    # Process each day
    for day in sorted(design.ops_all_by_day.keys()):
        day_clients = design.logs_df[design.logs_df[design.day_col] == day]

        if len(day_clients) == 0:
            continue

        # Initialize decisions for this day
        day_decisions: dict[str, dict[str, Any]] = {name: {} for name in models}

        # Process chunks. Prediction errors propagate (per invariants.md:
        # prefer fail-loud over silent first-operator fallback).
        for chunk in build_candidate_pairs(design, day, chunk_pairs):
            if len(chunk) == 0:
                continue

            # Prepare features
            feature_cols = design.cli_features + design.op_features
            X_pairs: np.ndarray = chunk[feature_cols].values.astype(np.float32)

            # Get predictions from each model
            for model_name, model in models.items():
                predictions = model.predict(X_pairs)

                # Update best decisions for each client in this chunk
                for i, (_, row) in enumerate(chunk.iterrows()):
                    client_id = str(row[design.client_id_col])
                    operator_id = str(row[design.operator_id_col])
                    pred = float(predictions[i])

                    if client_id not in day_decisions[model_name]:
                        day_decisions[model_name][client_id] = {
                            "best_op": operator_id,
                            "best_pred": pred,
                        }
                    else:
                        current_best = day_decisions[model_name][client_id]["best_pred"]
                        if (direction == "min" and pred < current_best) or (
                            direction == "max" and pred > current_best
                        ):
                            day_decisions[model_name][client_id] = {
                                "best_op": operator_id,
                                "best_pred": pred,
                            }

        # Extract decisions for this day in client order
        for model_name in models:
            day_model_decisions = []
            for _, client_row in day_clients.iterrows():
                client_id = str(client_row[design.client_id_col])
                if client_id not in day_decisions[model_name]:
                    # Streaming produced no pairs for this client, which
                    # means an empty eligibility set. Per invariants.md the
                    # project prefers fail-loud over silent fallbacks; treat
                    # this as a data-quality error.
                    raise DataValidationError(
                        "No eligible operators for client; cannot induce a "
                        "policy. Check eligibility masks for empty rows.",
                        details={
                            "day": str(day),
                            "client_id": client_id,
                            "strategy": "stream",
                            "model": model_name,
                        },
                    )
                day_model_decisions.append(
                    day_decisions[model_name][client_id]["best_op"]
                )

            policies[model_name].extend(day_model_decisions)

    return {name: np.array(decisions) for name, decisions in policies.items()}


def _make_interaction_features(X_cli: np.ndarray, X_op: np.ndarray) -> np.ndarray:
    """Build cli x op interaction features alongside the originals.

    Linear models on concatenated [X_cli | X_op] cannot rank operators
    differently across clients (the client term is constant for fixed client),
    so a Ridge surrogate would produce day-global top-K. Adding outer-product
    interaction terms restores per-client personalization.
    """
    n_rows = X_cli.shape[0]
    n_cli = X_cli.shape[1]
    n_op = X_op.shape[1]
    interactions = (X_cli[:, :, None] * X_op[:, None, :]).reshape(n_rows, n_cli * n_op)
    return np.hstack([X_cli, X_op, interactions]).astype(np.float32)


def _fit_surrogate(
    surrogate_model: str,
    X_cli: np.ndarray,
    X_op: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> tuple[Any, str]:
    """Fit a surrogate that captures client x operator interactions.

    Returns the fitted surrogate and a string indicating the feature mode:
    - "interaction" for ridge_interaction (concat + outer product)
    - "concat" for hgb (tree models capture interactions internally)
    """
    if surrogate_model == "hgb":
        X = np.hstack([X_cli, X_op]).astype(np.float32)
        model = HistGradientBoostingRegressor(
            max_iter=50, max_depth=4, random_state=random_state
        )
        model.fit(X, y)
        return model, "concat"
    elif surrogate_model == "ridge_interaction":
        X = _make_interaction_features(X_cli, X_op)
        model = Ridge()
        model.fit(X, y)
        return model, "interaction"
    elif surrogate_model == "ridge":
        raise ValueError(
            "surrogate_model='ridge' is degenerate for stream_topk: a linear "
            "model on concatenated [cli | op] features cannot rank operators "
            "differently across clients (top-K becomes day-global). Use "
            "'hgb' (default) or 'ridge_interaction' instead."
        )
    else:
        raise ValueError(
            f"surrogate_model must be 'hgb' or 'ridge_interaction', "
            f"got {surrogate_model!r}"
        )


def _surrogate_features(
    feature_mode: str, X_cli: np.ndarray, X_op: np.ndarray
) -> np.ndarray:
    """Build features matching what the surrogate was fitted on."""
    if feature_mode == "interaction":
        return _make_interaction_features(X_cli, X_op)
    return np.hstack([X_cli, X_op]).astype(np.float32)


def _build_day_elig_mask(
    day_clients: pd.DataFrame,
    day_ops: pd.DataFrame,
    elig_col: str | None,
    operator_id_col: str,
) -> np.ndarray:
    """Build a (n_day_clients, n_day_ops) boolean eligibility mask.

    Mirrors the per-client semantics previously in ``induce_policy_stream_topk``:
    if ``elig_col`` is missing on the design, or a client's value is not a
    list/tuple, the client is treated as eligible for every operator on the day.

    Precondition: ``day_ops[operator_id_col]`` must be unique within the day.
    Duplicate operator IDs surface ambiguous features for the same operator
    (the surrogate / full model would score the same ``operator_id`` more than
    once with different feature rows) and are treated as a data-quality error
    per ``docs/agent-context/invariants.md`` (prefer fail-loud). The historical
    ``isin(...)`` path coincidentally marked every duplicate row as eligible;
    the day-vectorized path raises instead.
    """
    n_clients = len(day_clients)
    n_ops = len(day_ops)

    if elig_col is None or elig_col not in day_clients.columns:
        return np.ones((n_clients, n_ops), dtype=bool)

    day_op_ids = day_ops[operator_id_col].values
    # Fail-loud on duplicate operator IDs within a day. The dict-based
    # eligibility lookup below would silently drop earlier duplicate positions
    # (last-write-wins), which diverges from the prior ``isin(...)`` semantics
    # *and* hides what is almost certainly a data-quality issue upstream.
    if len(np.unique(day_op_ids)) != len(day_op_ids):
        unique_ids, counts = np.unique(day_op_ids, return_counts=True)
        duplicates = unique_ids[counts > 1]
        raise DataValidationError(
            "Duplicate operator IDs within a day are not supported for "
            "stream_topk eligibility. Deduplicate op_daily_df upstream.",
            details={
                "n_duplicate_ids": int(duplicates.size),
                "sample_duplicate_ids": [str(x) for x in duplicates[:5].tolist()],
                "strategy": "stream_topk",
            },
        )

    op_id_to_pos: dict[Any, int] = {op_id: i for i, op_id in enumerate(day_op_ids)}
    mask = np.zeros((n_clients, n_ops), dtype=bool)
    elig_values = day_clients[elig_col].to_list()
    for client_pos, raw_value in enumerate(elig_values):
        if hasattr(raw_value, "iloc"):
            elig_value = raw_value.iloc[0] if len(raw_value) > 0 else []
        else:
            elig_value = raw_value
        if isinstance(elig_value, (list, tuple)):
            for op_id in elig_value:
                pos = op_id_to_pos.get(op_id)
                if pos is not None:
                    mask[client_pos, pos] = True
        else:
            # Non-list values (NaN, scalar, etc.) fall back to "all eligible"
            # to match the prior per-client behavior.
            mask[client_pos, :] = True
    return mask


def induce_policy_stream_topk(
    models: dict[str, Any],
    design: PairwiseDesign,
    metric_col: str,
    direction: Literal["min", "max"] = "min",
    topk: int = 20,
    surrogate_model: str = "hgb",
    chunk_pairs: int = 2_000_000,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Induce policies using streaming + top-K prefiltering strategy.

    Fits a cheap surrogate on observed (cli_*, op_*) pairs, uses it to
    prefilter to the top-K most promising operators per decision, then runs the
    full models only on those K candidates. This reduces computation for large
    operator pools while maintaining good policy quality.

    The implementation is fully day-vectorized: for each day, a single
    ``(n_chunk * n_day_ops, n_features)`` feature matrix is built and the
    surrogate runs **once per chunk** (not once per client), satisfying both
    the per-decision throughput goal (#46) and the surrogate hoist (#63).
    Clients are processed in client-axis chunks of size
    ``max(1, chunk_pairs // n_day_ops)``, giving ``chunk_pairs`` real meaning
    for this strategy (#61).

    **Surrogate choice matters for personalization.** A naive linear model on
    concatenated [cli | op] features is degenerate: for a fixed client the
    score becomes ``constant + b·op``, so the top-K is identical across all
    clients on the day (day-global rather than personalized). To avoid this:

    - ``"hgb"`` (default): HistGradientBoostingRegressor — non-linear, captures
      cli x op interactions automatically. Slightly slower than Ridge but still
      a "cheap surrogate" relative to deep models.
    - ``"ridge_interaction"``: Ridge regression with explicit cli x op outer-
      product interaction features. Preserves the linear-model speed while
      enabling per-client ranking.
    - ``"ridge"`` (rejected): plain Ridge on concatenated features — raises
      ValueError because top-K would degenerate to day-global selection.

    **Selection bias caveat.** The surrogate is trained on historically-chosen
    (client, operator) pairs from the logs. Operators that were never observed
    have no training signal and may be systematically suppressed by the top-K
    filter even if the full model would rank them highly.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of fitted models to evaluate
    design : PairwiseDesign
        Pairwise design containing data and metadata
    metric_col : str
        Target metric column in design.logs_df used to train the surrogate.
    direction : Literal["min", "max"], default="min"
        Whether to minimize or maximize the target metric
    topk : int, default=20
        Number of top operators to consider per decision after prefiltering
    surrogate_model : str, default="hgb"
        Surrogate model type for prefiltering. One of "hgb" or
        "ridge_interaction". "ridge" is rejected (degenerate).
    chunk_pairs : int, default=2_000_000
        Soft cap on the size of the ``(clients * day_ops)`` feature matrix
        held in memory per surrogate / full-model predict call. The chunk
        size in clients is ``max(1, chunk_pairs // n_day_ops)``.
    random_state : int, default=0
        Random seed for the surrogate model.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping model names to induced policies
    """
    logger.info(
        f"Using streaming + top-K strategy with topk={topk}, "
        f"surrogate={surrogate_model}, chunk_pairs={chunk_pairs}"
    )

    # Fail fast if any model is unfitted (validate once before any work)
    for model in models.values():
        check_is_fitted(model)

    # Step 1: Fit a cheap surrogate that captures cli x op interactions
    X_cli_surr = design.logs_df[design.cli_features].values.astype(np.float32)
    X_op_surr = design.logs_df[design.op_features].values.astype(np.float32)
    y_surr = design.logs_df[metric_col].values.astype(np.float64)
    surrogate, feature_mode = _fit_surrogate(
        surrogate_model, X_cli_surr, X_op_surr, y_surr, random_state=random_state
    )
    logger.info(
        f"Surrogate {surrogate_model!r} fitted on {len(X_cli_surr)} observed pairs"
    )

    # Step 2: per-day, fully vectorized top-K prefilter + full-model scoring.
    # Prediction errors propagate (per invariants.md: prefer fail-loud).
    policies: dict[str, list[str]] = {name: [] for name in models}
    sentinel = float("inf") if direction == "min" else float("-inf")

    for day in sorted(design.ops_all_by_day.keys()):
        day_clients = design.logs_df[design.logs_df[design.day_col] == day]

        if len(day_clients) == 0:
            continue

        if day not in design.day_to_op_df:
            # A day with clients but no operator candidates is a data-quality
            # error, not a normal case. Per invariants.md, prefer fail-loud.
            raise DataValidationError(
                "Day has clients but no operator candidates; cannot induce a "
                "policy. Check op_daily_df coverage.",
                details={
                    "day": str(day),
                    "n_clients": len(day_clients),
                    "strategy": "stream_topk",
                },
            )

        day_ops = design.day_to_op_df[day]
        n_day_clients = len(day_clients)
        n_day_ops = len(day_ops)

        day_cli_arr = day_clients[design.cli_features].to_numpy(dtype=np.float32)
        day_op_arr = day_ops[design.op_features].to_numpy(dtype=np.float32)
        day_op_ids = day_ops[design.operator_id_col].values

        # (n_day_clients, n_day_ops) boolean eligibility mask
        elig_mask = _build_day_elig_mask(
            day_clients, day_ops, design.elig_col, design.operator_id_col
        )

        # Per-client eligibility must be non-empty (matches prior fail-loud
        # behavior). Surface the first offending client position.
        elig_counts = elig_mask.sum(axis=1)
        empty_rows = np.where(elig_counts == 0)[0]
        if empty_rows.size > 0:
            raise DataValidationError(
                "No eligible operators for client; cannot induce a "
                "policy. Check eligibility masks for empty rows.",
                details={
                    "day": str(day),
                    "client_position": int(empty_rows[0]),
                    "strategy": "stream_topk",
                },
            )

        # Effective top-K is bounded by the operator pool size on the day.
        k = min(topk, n_day_ops)

        # Client-axis chunk size. The grid for one chunk is
        # (chunk_size * n_day_ops, n_features); chunk_pairs caps that area.
        chunk_size = max(1, chunk_pairs // max(1, n_day_ops))

        day_decisions: dict[str, list[str]] = {name: [] for name in models}

        for chunk_start in range(0, n_day_clients, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_day_clients)
            n_chunk = chunk_end - chunk_start
            cli_chunk = day_cli_arr[chunk_start:chunk_end]
            elig_chunk = elig_mask[chunk_start:chunk_end]

            # Build the (n_chunk * n_day_ops, n_cli + n_op) cross-product
            # matrix once per chunk. Surrogate runs *once* on this matrix.
            X_cli_grid = np.repeat(cli_chunk, n_day_ops, axis=0)
            X_op_grid = np.tile(day_op_arr, (n_chunk, 1))
            X_surr_grid = _surrogate_features(feature_mode, X_cli_grid, X_op_grid)
            surr_preds = surrogate.predict(X_surr_grid).reshape(n_chunk, n_day_ops)

            # Mask ineligible (client, op) cells with the sentinel so they
            # are never picked by argpartition / argmin.
            surr_preds = np.where(elig_chunk, surr_preds, sentinel)

            # Per-client top-K indices into day_op_arr. argpartition is
            # O(n_day_ops) per row; correctness does not depend on the order
            # within the K block because the full model re-scores them.
            if direction == "min":
                topk_idx_per_client = np.argpartition(surr_preds, k - 1, axis=1)[:, :k]
            else:
                topk_idx_per_client = np.argpartition(surr_preds, -k, axis=1)[:, -k:]

            # Build the top-K feature matrix for the full models:
            # (n_chunk * k, n_cli + n_op). Order: client-major, then top-K.
            cli_rep_topk = np.repeat(cli_chunk, k, axis=0)
            op_topk_idx_flat = topk_idx_per_client.reshape(-1)
            op_topk_grid = day_op_arr[op_topk_idx_flat]
            X_topk = np.hstack([cli_rep_topk, op_topk_grid]).astype(np.float32)

            # Eligibility for the selected top-K cells (used to mask any
            # ineligible carry-overs from clients with fewer than K eligible
            # ops — argpartition will have picked sentinel-cells for them).
            client_row_idx = np.arange(n_chunk)[:, None]
            full_elig_topk = elig_chunk[client_row_idx, topk_idx_per_client]

            for model_name, model in models.items():
                full_preds = model.predict(X_topk).reshape(n_chunk, k)
                # Re-apply eligibility mask after the full-model predict.
                full_preds = np.where(full_elig_topk, full_preds, sentinel)
                if direction == "min":
                    best_in_topk = np.argmin(full_preds, axis=1)
                else:
                    best_in_topk = np.argmax(full_preds, axis=1)
                best_op_idx = topk_idx_per_client[np.arange(n_chunk), best_in_topk]
                day_decisions[model_name].extend(
                    str(x) for x in day_op_ids[best_op_idx]
                )

        for model_name in models:
            policies[model_name].extend(day_decisions[model_name])

    return {name: np.array(decisions) for name, decisions in policies.items()}


def induce_policy(
    models: dict[str, Any],
    design: PairwiseDesign,
    strategy: Literal["auto", "direct", "stream", "stream_topk"] = "auto",
    direction: Literal["min", "max"] = "min",
    topk: int = 20,
    chunk_pairs: int = 2_000_000,
    metric_col: str | None = None,
    surrogate_model: str = "hgb",
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Induce policies using specified or auto-selected strategy.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model_name -> fitted model
    design : PairwiseDesign
        Pairwise design object
    strategy : Literal["auto", "direct", "stream", "stream_topk"]
        Strategy to use for policy induction
    direction : Literal["min", "max"]
        Whether to minimize or maximize the predicted metric
    topk : int
        Number of top operators for stream_topk strategy
    chunk_pairs : int
        Soft cap on the size of the candidate-pair feature matrix held in
        memory per predict call. ``"direct"`` and ``"stream"`` chunk along
        the pairs axis; ``"stream_topk"`` chunks along the client axis with
        ``chunk_size = max(1, chunk_pairs // n_day_ops)``.
    metric_col : str
        Target metric column name; required when strategy is "stream_topk"
    surrogate_model : str, default="hgb"
        Surrogate model for stream_topk prefiltering. "hgb" or
        "ridge_interaction".
    random_state : int, default=0
        Random seed used by the surrogate model.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of model_name -> array of chosen operator_ids per decision
    """
    if strategy == "auto":
        stats = design.get_stats()
        strategy = choose_strategy(stats)
        logger.info(
            f"Auto-selected strategy: {strategy} (candidate_pairs={stats['candidate_pairs']:,})"
        )

    if strategy == "direct":
        return induce_policy_direct(models, design, direction, chunk_pairs)
    elif strategy == "stream":
        return induce_policy_stream(models, design, direction, chunk_pairs)
    elif strategy == "stream_topk":
        if not metric_col:
            raise ValueError("metric_col must be provided when strategy='stream_topk'")
        return induce_policy_stream_topk(
            models,
            design,
            metric_col,
            direction,
            topk,
            surrogate_model=surrogate_model,
            chunk_pairs=chunk_pairs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Auto execution-mode threshold (#33). Above this many evaluation decisions the
# ``"auto"`` execution mode switches to the vectorized ``"large_data"`` path,
# which builds the observed-feature / action / eligibility / policy arrays
# without a per-row ``DataFrame.iterrows()`` loop. The vectorized path is
# proven numerically identical to the standard path (see
# ``tests/test_execution_mode.py``); the threshold only trades a little
# constant overhead on tiny inputs for a large speedup on big ones.
LARGE_DATA_ROW_THRESHOLD = 50_000


def _first_index_map(ops: list[Any]) -> dict[str, int]:
    """Map ``str(operator) -> first column index``.

    Mirrors the standard path's ``ops_all_by_day[day].index(str(op))`` (which
    returns the *first* occurrence) so the vectorized path stays identical even
    if a day's operator list contains duplicates.
    """
    pos: dict[str, int] = {}
    for i, op in enumerate(ops):
        pos.setdefault(str(op), i)
    return pos


def build_eval_arrays_vectorized(
    design: PairwiseDesign,
    metric_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the per-decision observed features, action indices and eligibility.

    Vectorized equivalent (#33) of the per-row ``logs_df.iterrows()`` loops in
    :func:`skdr_eval.core.evaluate_pairwise_models`. It reproduces the standard
    path's semantics **exactly** so the two execution modes return identical
    numbers (asserted in ``tests/test_execution_mode.py``):

    - observed operator features are looked up from ``design.day_to_op_df`` keyed
      by ``(str(day), raw operator_id)``; a missing day or operator yields a
      zero row (the standard fallback);
    - the action index is the position of ``str(operator_id)`` within
      ``ops_all_by_day[str(day)]``, falling back to ``0``;
    - eligibility uses the per-row ``elig_col`` list when present (only operators
      listed *and* available that day are eligible), otherwise every operator
      available that day is eligible.

    Parameters
    ----------
    design : PairwiseDesign
        Evaluation design. ``design.logs_df`` must carry a contiguous
        ``RangeIndex`` (guaranteed by :meth:`PairwiseDesign.from_dataframes`).
    metric_col : str
        Outcome column (unused here; accepted so callers can pass the same
        argument set as the loop path for symmetry).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, int]
        ``(X_obs, A, elig, max_ops)`` where ``X_obs`` has shape
        ``(n, len(cli_features) + len(op_features))`` and dtype ``float32``,
        ``A`` is ``(n,)`` int action indices, ``elig`` is ``(n, max_ops)`` and
        ``max_ops`` is the largest per-day operator count.
    """
    del metric_col  # symmetry with the loop path; features come from the design
    logs_df = design.logs_df
    n = len(logs_df)
    op_id_col = design.operator_id_col
    day_keys = logs_df[design.day_col].astype(str).to_numpy()
    op_strs = logs_df[op_id_col].astype(str).to_numpy()
    max_ops = max((len(ops) for ops in design.ops_all_by_day.values()), default=0)

    # --- Observed client features -----------------------------------------
    if design.cli_features:
        cli_part = logs_df[design.cli_features].to_numpy(dtype=np.float64)
    else:
        cli_part = np.zeros((n, 0), dtype=np.float64)

    # --- Observed operator features (chosen operator, that day) -----------
    # Lookup keyed by (str(day), raw operator_id) to mirror the standard path,
    # which filters ``op_df[op_df[op_id_col] == chosen_op]`` on the *raw* id.
    n_op_feat = len(design.op_features)
    if n_op_feat:
        frames = []
        for day_key, op_df in design.day_to_op_df.items():
            sub = op_df[[op_id_col, *design.op_features]].copy()
            sub["__day_key"] = str(day_key)
            frames.append(sub)
        # Deduplicate on (day, operator) keeping the first match, mirroring the
        # standard path's ``op_row.iloc[0]`` — otherwise a duplicated
        # (day, operator_id) row in op_daily_df would make the left-merge
        # one-to-many and misalign / inflate x_obs.
        lookup = (
            pd.concat(frames, ignore_index=True).drop_duplicates(
                ["__day_key", op_id_col], keep="first"
            )
            if frames
            else None
        )
        left = pd.DataFrame(
            {"__day_key": day_keys, op_id_col: logs_df[op_id_col].to_numpy()}
        )
        if lookup is not None:
            merged = left.merge(
                lookup, on=["__day_key", op_id_col], how="left", sort=False
            )
            op_part = merged[design.op_features].to_numpy(dtype=np.float64)
            op_part = np.nan_to_num(op_part, nan=0.0)
        else:
            op_part = np.zeros((n, n_op_feat), dtype=np.float64)
    else:
        op_part = np.zeros((n, 0), dtype=np.float64)

    x_obs = np.hstack([cli_part, op_part]).astype(np.float32)

    # --- Action indices and eligibility, grouped by day -------------------
    actions = np.zeros(n, dtype=int)
    elig = np.zeros((n, max_ops))
    has_elig_col = bool(design.elig_col) and design.elig_col in logs_df.columns
    elig_values = logs_df[design.elig_col].to_numpy() if has_elig_col else None
    for day_key, ops in design.ops_all_by_day.items():
        pos = _first_index_map(ops)
        n_ops_day = len(ops)
        rows = np.flatnonzero(day_keys == str(day_key))
        if rows.size == 0:
            continue
        # Action index: position of str(chosen_op); default 0.
        actions[rows] = [pos.get(o, 0) for o in op_strs[rows]]
        # Eligibility.
        for ri in rows:
            val = elig_values[ri] if elig_values is not None else None
            if isinstance(val, (list, tuple, np.ndarray)):
                cols = [pos[str(op)] for op in val if str(op) in pos]
                if cols:
                    elig[ri, cols] = 1.0
            else:
                elig[ri, :n_ops_day] = 1.0
    return x_obs, actions, elig, max_ops


def build_policy_probs_vectorized(
    design: PairwiseDesign,
    policy_decisions: np.ndarray,
    max_ops: int,
) -> np.ndarray:
    """Vectorized one-hot policy-probability matrix for a single policy (#33).

    Reproduces the standard per-row loop: for each decision the chosen operator
    ``str(policy_decisions[i])`` is mapped to its column in
    ``ops_all_by_day[str(day)]`` and set to ``1.0``; operators not available
    that day leave the row all-zero.

    Parameters
    ----------
    design : PairwiseDesign
        Evaluation design (provides the per-day operator ordering).
    policy_decisions : np.ndarray
        Chosen operator id per decision, aligned to ``design.logs_df`` rows.
    max_ops : int
        Column count of the returned matrix (largest per-day operator count).

    Returns
    -------
    np.ndarray
        ``(n, max_ops)`` one-hot policy-probability matrix.
    """
    logs_df = design.logs_df
    n = len(logs_df)
    day_keys = logs_df[design.day_col].astype(str).to_numpy()
    chosen = np.asarray(policy_decisions).astype(str)
    policy_probs = np.zeros((n, max_ops))
    for day_key, ops in design.ops_all_by_day.items():
        pos = _first_index_map(ops)
        rows = np.flatnonzero(day_keys == str(day_key))
        if rows.size == 0:
            continue
        cols = np.array([pos.get(c, -1) for c in chosen[rows]])
        ok = cols >= 0
        policy_probs[rows[ok], cols[ok]] = 1.0
    return policy_probs


def map_external_policies(
    external_policies: dict[str, pd.DataFrame],
    design: PairwiseDesign,
) -> dict[str, np.ndarray]:
    """Map externally-supplied client→operator assignments to policy arrays (#56).

    Each policy is a DataFrame holding at least the design's ``client_id`` and
    ``operator_id`` columns (the schema a routing/queueing simulator emits). The
    assignment is keyed by ``client_id``: every evaluation decision for a given
    client is routed to that client's assigned operator. A client that appears
    on multiple days therefore receives the same (static) assignment; per-call
    assignments are a documented follow-up.

    Eligibility is **not** silently repaired: an assignment to an operator that
    is unavailable on a decision's day simply does not match (its
    policy-probability row stays all-zero and the decision contributes through
    the direct-method term only), which surfaces in ``match_rate``. Assignments
    to operators unknown to the whole system, or missing client coverage, fail
    loud.

    Parameters
    ----------
    external_policies : dict[str, pd.DataFrame]
        Mapping of policy name to assignment frame.
    design : PairwiseDesign
        Evaluation design built from the logs being evaluated.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of policy name to an operator-id array aligned to
        ``design.logs_df`` rows — the same contract :func:`induce_policy`
        returns.

    Raises
    ------
    DataValidationError
        If ``external_policies`` is not a non-empty dict, a frame is missing the
        required columns, a client in the logs has no assignment, or an assigned
        operator is unknown to the system.
    """
    if not isinstance(external_policies, dict) or not external_policies:
        raise DataValidationError(
            "external_policies must be a non-empty dict of "
            "{policy_name: DataFrame[client_id, operator_id]}."
        )

    client_col = design.client_id_col
    op_col = design.operator_id_col
    logs_clients = design.logs_df[client_col].to_numpy()
    known_ops = {str(op) for ops in design.ops_all_by_day.values() for op in ops}

    policies: dict[str, np.ndarray] = {}
    for name, frame in external_policies.items():
        if not isinstance(frame, pd.DataFrame):
            raise DataValidationError(
                f"external_policies[{name!r}] must be a pandas DataFrame, "
                f"got {type(frame).__name__}."
            )
        missing_cols = [c for c in (client_col, op_col) if c not in frame.columns]
        if missing_cols:
            raise DataValidationError(
                f"external_policies[{name!r}] is missing required column(s) "
                f"{missing_cols}; expected at least {[client_col, op_col]}."
            )
        if frame[client_col].duplicated().any():
            dup = frame.loc[frame[client_col].duplicated(), client_col].unique()[:5]
            raise DataValidationError(
                f"external_policies[{name!r}] has duplicate client assignments "
                f"(e.g. {list(dup)}); provide one operator per client."
            )

        assignment = dict(zip(frame[client_col], frame[op_col], strict=False))

        # Validate client coverage against the evaluation logs.
        missing_clients = {c for c in logs_clients if c not in assignment}
        if missing_clients:
            sample = list(missing_clients)[:5]
            raise DataValidationError(
                f"external_policies[{name!r}] does not cover "
                f"{len(missing_clients)} client(s) present in the logs "
                f"(e.g. {sample}); every logged decision needs an assignment."
            )

        # Validate assigned operators are known to the system.
        unknown_ops = {
            str(op) for op in assignment.values() if str(op) not in known_ops
        }
        if unknown_ops:
            sample = list(unknown_ops)[:5]
            raise DataValidationError(
                f"external_policies[{name!r}] assigns unknown operator(s) "
                f"{sample} not present in op_daily_df."
            )

        decisions = np.array([assignment[c] for c in logs_clients], dtype=object)
        policies[name] = decisions

    return policies
