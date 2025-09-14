"""Input validation utilities for skdr-eval library."""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .exceptions import (
    DataValidationError,
    InsufficientDataError,
    ModelValidationError,
    MemoryError,
)

logger = logging.getLogger("skdr_eval")


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: Optional[list[str]] = None,
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
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_dtype: Optional[type] = None,
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
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
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
    required_methods: Optional[list[str]] = None,
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
        raise DataValidationError(
            f"{name} probabilities don't sum to 1 (axis={axis})"
        )


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
    else:
        if np.any(values < 0):
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
    value: Union[int, float],
    name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> None:
    """Validate that a parameter is within a specified range.

    Parameters
    ----------
    value : Union[int, float]
        Value to validate
    name : str
        Name of the parameter for error messages
    min_val : Union[int, float], optional
        Minimum allowed value
    max_val : Union[int, float], optional
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
        raise DataValidationError(
            f"{name} must be one of {choices}, got '{value}'"
        )


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
        raise DataValidationError(f"{name} must be an integer, got {type(value).__name__}")

    if value <= 0:
        raise DataValidationError(f"{name} must be positive, got {value}")


def validate_random_state(
    random_state: Optional[Union[int, np.random.RandomState]],
    name: str = "random_state",
) -> None:
    """Validate a random state parameter.

    Parameters
    ----------
    random_state : Optional[Union[int, np.random.RandomState]]
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
            raise DataValidationError(f"{name} must be non-negative, got {random_state}")