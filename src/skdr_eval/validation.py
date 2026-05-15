"""Input validation utilities for skdr-eval library."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import (
    DataValidationError,
    InsufficientDataError,
    MemoryError,
    ModelValidationError,
)

logger = logging.getLogger("skdr_eval")

# Pairwise-schema columns. In strict mode, presence in a single-action logs
# DataFrame indicates the caller likely meant validate_pairwise_inputs.
_PAIRWISE_KNOWN_COLUMNS = frozenset({"arrival_day", "client_id", "operator_id"})


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
    allow_empty: bool = False,
) -> None:
    """Validate a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    name : str
        Name of the DataFrame for error messages
    required_columns : list[str], optional
        Required column names
    min_rows : int, default=1
        Minimum number of rows required
    allow_empty : bool, default=False
        Whether to allow empty DataFrames

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}"
        )

    if df.empty and not allow_empty:
        raise InsufficientDataError(f"{name} is empty")

    if len(df) < min_rows:
        raise InsufficientDataError(
            f"{name} has {len(df)} rows, need at least {min_rows}"
        )

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"{name} missing required columns: {sorted(missing_cols)}"
            )


def validate_numpy_array(
    arr: np.ndarray,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: type | None = None,
    min_size: int = 1,
    allow_empty: bool = False,
) -> None:
    """Validate a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    name : str
        Name of the array for error messages
    expected_shape : tuple[int, ...], optional
        Expected shape (None for any dimension)
    expected_dtype : type, optional
        Expected dtype
    min_size : int, default=1
        Minimum number of elements required
    allow_empty : bool, default=False
        Whether to allow empty arrays

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not isinstance(arr, np.ndarray):
        raise DataValidationError(
            f"{name} must be a numpy array, got {type(arr).__name__}"
        )

    if arr.size == 0 and not allow_empty:
        raise InsufficientDataError(f"{name} is empty")

    if arr.size < min_size:
        raise InsufficientDataError(
            f"{name} has {arr.size} elements, need at least {min_size}"
        )

    if expected_shape is not None:
        if len(arr.shape) != len(expected_shape):
            raise DataValidationError(
                f"{name} has {len(arr.shape)} dimensions, expected {len(expected_shape)}"
            )
        for i, (actual, expected) in enumerate(
            zip(arr.shape, expected_shape, strict=False)
        ):
            if expected is not None and actual != expected:
                raise DataValidationError(
                    f"{name} dimension {i} has size {actual}, expected {expected}"
                )

    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise DataValidationError(
            f"{name} has dtype {arr.dtype}, expected {expected_dtype}"
        )


def validate_sklearn_estimator(
    estimator: Any,
    name: str,
    required_methods: list[str] | None = None,
) -> None:
    """Validate a sklearn estimator.

    Parameters
    ----------
    estimator : Any
        Estimator to validate
    name : str
        Name of the estimator for error messages
    required_methods : list[str], optional
        Required methods (e.g., ['fit', 'predict'])

    Raises
    ------
    ModelValidationError
        If validation fails
    """
    if not hasattr(estimator, "fit"):
        raise ModelValidationError(f"{name} must have a 'fit' method")

    if required_methods:
        missing_methods = [m for m in required_methods if not hasattr(estimator, m)]
        if missing_methods:
            raise ModelValidationError(
                f"{name} missing required methods: {missing_methods}"
            )


def validate_probabilities(
    probs: np.ndarray,
    name: str,
    axis: int = 1,
    tolerance: float = 1e-10,
) -> None:
    """Validate that an array contains valid probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Array to validate
    name : str
        Name of the array for error messages
    axis : int, default=1
        Axis along which probabilities should sum to 1
    tolerance : float, default=1e-10
        Tolerance for sum check

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if np.any(probs < 0):
        raise DataValidationError(f"{name} contains negative probabilities")

    if np.any(probs > 1):
        raise DataValidationError(f"{name} contains probabilities > 1")

    row_sums = np.sum(probs, axis=axis)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise DataValidationError(f"{name} probabilities don't sum to 1 (axis={axis})")


def validate_positive_values(
    values: np.ndarray,
    name: str,
    strict: bool = True,
) -> None:
    """Validate that an array contains positive values.

    Parameters
    ----------
    values : np.ndarray
        Array to validate
    name : str
        Name of the array for error messages
    strict : bool, default=True
        If True, requires all values > 0; if False, allows values >= 0

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if strict:
        if np.any(values <= 0):
            raise DataValidationError(f"{name} contains non-positive values")
    elif np.any(values < 0):
        raise DataValidationError(f"{name} contains negative values")


def validate_finite_values(
    values: np.ndarray,
    name: str,
) -> None:
    """Validate that an array contains finite values.

    Parameters
    ----------
    values : np.ndarray
        Array to validate
    name : str
        Name of the array for error messages

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not np.all(np.isfinite(values)):
        raise DataValidationError(f"{name} contains non-finite values")


def validate_memory_usage(
    estimated_memory_gb: float,
    max_memory_gb: float = 8.0,
) -> None:
    """Validate that estimated memory usage is reasonable.

    Parameters
    ----------
    estimated_memory_gb : float
        Estimated memory usage in GB
    max_memory_gb : float, default=8.0
        Maximum allowed memory usage in GB

    Raises
    ------
    MemoryError
        If memory usage exceeds limit
    """
    if estimated_memory_gb > max_memory_gb:
        raise MemoryError(
            f"Estimated memory usage {estimated_memory_gb:.2f} GB exceeds limit {max_memory_gb} GB"
        )


def validate_parameter_range(
    value: int | float,
    name: str,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
) -> None:
    """Validate that a parameter is within a specified range.

    Parameters
    ----------
    value : int | float
        Value to validate
    name : str
        Name of the parameter for error messages
    min_val : int | float, optional
        Minimum allowed value
    max_val : int | float, optional
        Maximum allowed value

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if min_val is not None and value < min_val:
        raise DataValidationError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise DataValidationError(f"{name} must be <= {max_val}, got {value}")


def validate_string_choice(
    value: str,
    name: str,
    choices: list[str],
    case_sensitive: bool = True,
) -> None:
    """Validate that a string is one of the allowed choices.

    Parameters
    ----------
    value : str
        Value to validate
    name : str
        Name of the parameter for error messages
    choices : list[str]
        Allowed choices
    case_sensitive : bool, default=True
        Whether comparison should be case sensitive

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not case_sensitive:
        value = value.lower()
        choices = [c.lower() for c in choices]

    if value not in choices:
        raise DataValidationError(f"{name} must be one of {choices}, got '{value}'")


def validate_positive_integer(
    value: int,
    name: str,
) -> None:
    """Validate that a value is a positive integer.

    Parameters
    ----------
    value : int
        Value to validate
    name : str
        Name of the parameter for error messages

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not isinstance(value, int):
        raise DataValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )

    if value <= 0:
        raise DataValidationError(f"{name} must be positive, got {value}")


def validate_random_state(
    random_state: int | np.random.RandomState | None,
    name: str = "random_state",
) -> None:
    """Validate a random state parameter.

    Parameters
    ----------
    random_state : int | np.random.RandomState | None
        Random state to validate
    name : str, default="random_state"
        Name of the parameter for error messages

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if random_state is not None:
        if not isinstance(random_state, (int, np.random.RandomState)):
            raise DataValidationError(
                f"{name} must be an integer or RandomState, got {type(random_state).__name__}"
            )
        if isinstance(random_state, int) and random_state < 0:
            raise DataValidationError(
                f"{name} must be non-negative, got {random_state}"
            )


def _columns_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    if not prefix:
        return []
    return [c for c in df.columns if c.startswith(prefix)]


def validate_logs(
    logs: pd.DataFrame,
    *,
    cli_pref: str = "cli_",
    st_pref: str = "st_",
    strict: bool = False,
) -> None:
    """Validate a single-action logs DataFrame for ``evaluate_sklearn_models``.

    Performs a fast, surface-level check that ``logs`` carries the canonical
    schema consumed by :func:`skdr_eval.build_design`:

    - required columns ``arrival_ts``, ``action``, ``service_time``;
    - at least one eligibility column matching ``*_elig``;
    - at least one feature column matching ``cli_pref`` or ``st_pref``;
    - every ``action`` value is present as an ``*_elig`` operator;
    - feature, outcome, eligibility, and timestamp columns are finite numeric.

    Parameters
    ----------
    logs : pd.DataFrame
        Decision logs to validate.
    cli_pref : str, default="cli_"
        Prefix for client feature columns.
    st_pref : str, default="st_"
        Prefix for service-time feature columns.
    strict : bool, default=False
        When True, additionally require:

        - ``arrival_ts`` is monotonically non-decreasing (time order);
        - no column from the pairwise schema leaks in
          (``client_id``/``operator_id``/``arrival_day``);
        - every operator listed as eligible appears at least once in ``action``.

        Strict mode is intended for preflight checks; downstream functions
        only require the non-strict invariants.

    Raises
    ------
    DataValidationError
        If any non-strict invariant is violated.
    InsufficientDataError
        If ``logs`` is empty.
    """
    validate_dataframe(
        logs,
        "logs",
        required_columns=["arrival_ts", "action", "service_time"],
        min_rows=1,
    )

    elig_cols = [c for c in logs.columns if c.endswith("_elig")]
    if not elig_cols:
        raise DataValidationError(
            "logs has no eligibility columns (expected at least one '*_elig')"
        )

    feature_cols = _columns_with_prefix(logs, cli_pref) + _columns_with_prefix(
        logs, st_pref
    )
    if not feature_cols:
        raise DataValidationError(
            f"logs has no feature columns with prefixes '{cli_pref}' or '{st_pref}'"
        )

    ops_all = [c.removesuffix("_elig") for c in elig_cols]
    invalid_actions = set(logs["action"]) - set(ops_all)
    if invalid_actions:
        raise DataValidationError(
            f"logs.action contains values not present as '*_elig' columns: "
            f"{sorted(invalid_actions)} (eligible operators: {sorted(ops_all)})"
        )

    elig_values = logs[elig_cols].to_numpy()
    if not np.all(np.isin(elig_values, [0, 1])):
        raise DataValidationError(
            "logs eligibility columns must contain only 0/1 values"
        )

    feature_values = logs[feature_cols].to_numpy(dtype=float, copy=False)
    validate_finite_values(feature_values, "logs[feature_cols]")

    service_time = logs["service_time"].to_numpy(dtype=float, copy=False)
    validate_finite_values(service_time, "logs.service_time")

    arrival_ts = pd.to_datetime(logs["arrival_ts"], errors="coerce")
    if arrival_ts.isna().any():
        raise DataValidationError(
            "logs.arrival_ts contains values that cannot be parsed as timestamps"
        )

    if strict:
        if not arrival_ts.is_monotonic_increasing:
            raise DataValidationError(
                "strict: logs.arrival_ts is not monotonically non-decreasing"
            )

        leaked = _PAIRWISE_KNOWN_COLUMNS & set(logs.columns)
        if leaked:
            raise DataValidationError(
                f"strict: logs has pairwise-schema columns {sorted(leaked)}; "
                "use validate_pairwise_inputs instead"
            )

        eligible_anywhere = {
            op for op, col in zip(ops_all, elig_cols, strict=False) if logs[col].any()
        }
        observed_actions = set(logs["action"])
        unobserved_eligible = eligible_anywhere - observed_actions
        if unobserved_eligible:
            raise DataValidationError(
                f"strict: operators eligible but never chosen as action: "
                f"{sorted(unobserved_eligible)}"
            )


def validate_pairwise_inputs(
    logs_df: pd.DataFrame,
    op_daily_df: pd.DataFrame,
    *,
    metric_col: str,
    day_col: str = "arrival_day",
    client_id_col: str = "client_id",
    operator_id_col: str = "operator_id",
    elig_col: str | None = "elig_mask",
    cli_pref: str = "cli_",
    op_pref: str = "op_",
    strict: bool = False,
) -> None:
    """Validate inputs for :func:`skdr_eval.evaluate_pairwise_models`.

    Parameters
    ----------
    logs_df : pd.DataFrame
        Observed decisions. Must contain ``day_col``, ``client_id_col``,
        ``operator_id_col``, ``metric_col``, at least one ``cli_pref`` feature,
        and the optional eligibility column ``elig_col``.
    op_daily_df : pd.DataFrame
        Daily operator snapshots. Must contain ``day_col``, ``operator_id_col``,
        and at least one ``op_pref`` feature.
    metric_col : str
        Target column on ``logs_df``.
    day_col, client_id_col, operator_id_col, elig_col, cli_pref, op_pref : str
        Schema knobs matching ``evaluate_pairwise_models``.
    strict : bool, default=False
        When True, additionally require:

        - every ``(operator_id, day)`` pair observed in ``logs_df`` also
          appears in ``op_daily_df``;
        - if ``elig_col`` is present, every chosen ``operator_id`` is
          contained in the corresponding eligibility list.

    Raises
    ------
    DataValidationError
        If any non-strict invariant is violated.
    InsufficientDataError
        If either input is empty.
    """
    required_logs = [day_col, client_id_col, operator_id_col, metric_col]
    validate_dataframe(logs_df, "logs_df", required_columns=required_logs, min_rows=1)

    required_daily = [day_col, operator_id_col]
    validate_dataframe(
        op_daily_df, "op_daily_df", required_columns=required_daily, min_rows=1
    )

    cli_cols = _columns_with_prefix(logs_df, cli_pref)
    if not cli_cols:
        raise DataValidationError(
            f"logs_df has no client feature columns with prefix '{cli_pref}'"
        )

    op_cols = _columns_with_prefix(op_daily_df, op_pref)
    if not op_cols:
        raise DataValidationError(
            f"op_daily_df has no operator feature columns with prefix '{op_pref}'"
        )

    if elig_col is not None and elig_col in logs_df.columns:
        sample = logs_df[elig_col].dropna()
        if not sample.empty and not isinstance(sample.iloc[0], (list, tuple, set)):
            raise DataValidationError(
                f"logs_df[{elig_col!r}] must contain list/tuple/set values, "
                f"got {type(sample.iloc[0]).__name__}"
            )

    metric_values = logs_df[metric_col].to_numpy(dtype=float, copy=False)
    validate_finite_values(metric_values, f"logs_df[{metric_col!r}]")

    cli_values = logs_df[cli_cols].to_numpy(dtype=float, copy=False)
    validate_finite_values(cli_values, "logs_df[cli_*]")

    op_values = op_daily_df[op_cols].to_numpy(dtype=float, copy=False)
    validate_finite_values(op_values, "op_daily_df[op_*]")

    if strict:
        logs_pairs = set(zip(logs_df[operator_id_col], logs_df[day_col], strict=False))
        daily_pairs = set(
            zip(op_daily_df[operator_id_col], op_daily_df[day_col], strict=False)
        )
        missing = logs_pairs - daily_pairs
        if missing:
            sample_missing = sorted(missing)[:5]
            raise DataValidationError(
                f"strict: {len(missing)} (operator, day) pairs in logs_df "
                f"missing from op_daily_df (sample: {sample_missing})"
            )

        if elig_col is not None and elig_col in logs_df.columns:
            max_examples = 5
            bad_rows: list[int] = []
            for idx, (chosen, eligible) in enumerate(
                zip(logs_df[operator_id_col], logs_df[elig_col], strict=False)
            ):
                if isinstance(eligible, (list, tuple, set)) and chosen not in eligible:
                    bad_rows.append(idx)
                    if len(bad_rows) >= max_examples:
                        break
            if bad_rows:
                raise DataValidationError(
                    f"strict: chosen operator_id missing from elig_mask in rows "
                    f"{bad_rows} (showing up to {max_examples})"
                )
