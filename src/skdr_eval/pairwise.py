"""Pairwise evaluation design and autoscaling strategies."""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

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
    elig_col : Optional[str]
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
    elig_col: Optional[str]
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
        elig_col: Optional[str] = "elig_mask",
    ) -> "PairwiseDesign":
        """Create PairwiseDesign from dataframes."""
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

    policies: dict[str, list[str]] = {name: [] for name in models}

    # Process each day
    for day in sorted(design.ops_all_by_day.keys()):
        day_clients = design.logs_df[design.logs_df[design.day_col] == day]

        if len(day_clients) == 0:
            continue

        # Collect all pairs for this day
        all_pairs = []
        client_indices = []

        for chunk in build_candidate_pairs(design, day, chunk_pairs):
            if len(chunk) == 0:
                continue

            # Track which client each pair belongs to
            chunk_client_indices = []
            for _, row in chunk.iterrows():
                client_idx = day_clients[
                    day_clients[design.client_id_col] == row[design.client_id_col]
                ].index[0]
                chunk_client_indices.append(client_idx)

            all_pairs.append(chunk)
            client_indices.extend(chunk_client_indices)

        if not all_pairs:
            continue

        # Combine all pairs for this day
        day_pairs_df = pd.concat(all_pairs, ignore_index=True)

        # Prepare features for prediction (exclude ID columns)
        feature_cols = design.cli_features + design.op_features
        X_pairs: np.ndarray = day_pairs_df[feature_cols].values.astype(np.float32)

        # Get predictions from each model
        for model_name, model in models.items():
            try:
                predictions = model.predict(X_pairs)

                # Group by client and find best operator
                day_decisions = []
                for client_idx in day_clients.index:
                    # Find pairs for this client
                    client_mask = np.array(client_indices) == client_idx
                    if not np.any(client_mask):
                        # No eligible operators - use first available
                        day_decisions.append(design.ops_all_by_day[day][0])
                        continue

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

            except Exception as e:
                logger.error(f"Error predicting with model {model_name}: {e}")
                # Fallback to first available operator
                fallback_decisions = [design.ops_all_by_day[day][0]] * len(day_clients)
                policies[model_name].extend(fallback_decisions)

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

    policies: dict[str, list[str]] = {name: [] for name in models}

    # Process each day
    for day in sorted(design.ops_all_by_day.keys()):
        day_clients = design.logs_df[design.logs_df[design.day_col] == day]

        if len(day_clients) == 0:
            continue

        # Initialize decisions for this day
        day_decisions: dict[str, dict[str, Any]] = {name: {} for name in models}

        # Process chunks
        for chunk in build_candidate_pairs(design, day, chunk_pairs):
            if len(chunk) == 0:
                continue

            # Prepare features
            feature_cols = design.cli_features + design.op_features
            X_pairs: np.ndarray = chunk[feature_cols].values.astype(np.float32)

            # Get predictions from each model
            for model_name, model in models.items():
                try:
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
                            current_best = day_decisions[model_name][client_id][
                                "best_pred"
                            ]
                            if (direction == "min" and pred < current_best) or (
                                direction == "max" and pred > current_best
                            ):
                                day_decisions[model_name][client_id] = {
                                    "best_op": operator_id,
                                    "best_pred": pred,
                                }

                except Exception as e:
                    logger.error(f"Error predicting with model {model_name}: {e}")

        # Extract decisions for this day in client order
        for model_name in models:
            day_model_decisions = []
            for _, client_row in day_clients.iterrows():
                client_id = str(client_row[design.client_id_col])
                if client_id in day_decisions[model_name]:
                    day_model_decisions.append(
                        day_decisions[model_name][client_id]["best_op"]
                    )
                else:
                    # Fallback to first available operator
                    day_model_decisions.append(design.ops_all_by_day[day][0])

            policies[model_name].extend(day_model_decisions)

    return {name: np.array(decisions) for name, decisions in policies.items()}


def induce_policy_stream_topk(
    models: dict[str, Any],
    design: PairwiseDesign,
    direction: Literal["min", "max"] = "min",
    topk: int = 20,
    chunk_pairs: int = 2_000_000,
    surrogate_model: str = "ridge",  # noqa: ARG001
) -> dict[str, np.ndarray]:
    """Induce policies using streaming + top-K prefiltering strategy.

    This strategy first uses a surrogate model to prefilter to the top-K
    most promising operators per decision, then runs the full models only
    on those candidates. This reduces computational cost for large operator
    pools while maintaining good performance.

    **Note**: This strategy is not fully implemented and will fall back to
    the streaming strategy. The surrogate model prefiltering is planned
    for future releases.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of fitted models to evaluate
    design : PairwiseDesign
        Pairwise design containing data and metadata
    direction : Literal["min", "max"], default="min"
        Whether to minimize or maximize the target metric
    topk : int, default=20
        Number of top operators to consider per decision
    chunk_pairs : int, default=2_000_000
        Maximum number of pairs to process in memory at once
    surrogate_model : str, default="ridge"
        Model to use for prefiltering (currently unused)

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping model names to induced policies

    Notes
    -----
    This strategy is designed for scenarios with very large operator pools
    where evaluating all pairs would be computationally prohibitive.
    Currently falls back to streaming strategy.
    """
    logger.info(f"Using streaming + top-K strategy with topk={topk}")

    # For now, fallback to streaming strategy
    # In a full implementation, we would:
    # 1. Train a cheap surrogate model on a sample
    # 2. Use it to prefilter to top-K operators per decision
    # 3. Then run the full models only on those K candidates

    logger.warning("Top-K strategy not fully implemented, falling back to streaming")
    return induce_policy_stream(models, design, direction, chunk_pairs)


def induce_policy(
    models: dict[str, Any],
    design: PairwiseDesign,
    strategy: Literal["auto", "direct", "stream", "stream_topk"] = "auto",
    direction: Literal["min", "max"] = "min",
    topk: int = 20,
    chunk_pairs: int = 2_000_000,
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
        Maximum pairs per chunk

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
        return induce_policy_stream_topk(models, design, direction, topk, chunk_pairs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
