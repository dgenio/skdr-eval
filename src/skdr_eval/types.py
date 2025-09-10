"""Type definitions and protocols for skdr-eval."""

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing_extensions import TypeAlias

# Type variables
T = TypeVar("T")
ModelType = TypeVar("ModelType", bound=BaseEstimator)

# Array types
ArrayLike: TypeAlias = Union[np.ndarray, pd.Series, list[float]]
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.floating]]
IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.integer]]
BoolArray: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]

# DataFrame types
LogsDataFrame: TypeAlias = pd.DataFrame
OperatorDataFrame: TypeAlias = pd.DataFrame

# Estimator types
EstimatorName: TypeAlias = str
EstimatorFactory: TypeAlias = Any  # Callable[[], BaseEstimator]

# Strategy types
AutoscaleStrategy: TypeAlias = str  # "direct" | "stream" | "stream_topk"
PropensityMethod: TypeAlias = str  # "condlogit" | "multinomial"
TaskType: TypeAlias = str  # "regression" | "binary"
Direction: TypeAlias = str  # "min" | "max"

# Result types
ClipValue: TypeAlias = Union[float, int]
ConfidenceInterval: TypeAlias = tuple[float, float]


class EstimatorProtocol(Protocol):
    """Protocol for sklearn-compatible estimators."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> "EstimatorProtocol":
        """Fit the estimator to data."""
        ...

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Make predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predict class probabilities (for classifiers)."""
        ...


class RegressorProtocol(EstimatorProtocol):
    """Protocol for regression estimators."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> FloatArray:
        """Make regression predictions."""
        ...


class ClassifierProtocol(EstimatorProtocol):
    """Protocol for classification estimators."""

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Predict class probabilities."""
        ...


class ModelDict(Protocol):
    """Protocol for model dictionaries."""

    def __getitem__(self, key: str) -> EstimatorProtocol:
        """Get model by name."""
        ...

    def __setitem__(self, key: str, value: EstimatorProtocol) -> None:
        """Set model by name."""
        ...

    def __iter__(self) -> Any:
        """Iterate over model names."""
        ...

    def items(self) -> Any:
        """Iterate over (name, model) pairs."""
        ...


# Validation functions
def validate_float_array(arr: Any, name: str = "array") -> FloatArray:
    """Validate and convert to float array."""
    if not isinstance(arr, (np.ndarray, pd.Series, list)):
        raise TypeError(f"{name} must be array-like, got {type(arr)}")

    arr_np = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr_np).all():
        raise ValueError(f"{name} contains non-finite values")

    return cast("FloatArray", arr_np)


def validate_int_array(arr: Any, name: str = "array") -> IntArray:
    """Validate and convert to integer array."""
    if not isinstance(arr, (np.ndarray, pd.Series, list)):
        raise TypeError(f"{name} must be array-like, got {type(arr)}")

    arr_np = np.asarray(arr, dtype=np.int64)
    return cast("IntArray", arr_np)


def validate_bool_array(arr: Any, name: str = "array") -> BoolArray:
    """Validate and convert to boolean array."""
    if not isinstance(arr, (np.ndarray, pd.Series, list)):
        raise TypeError(f"{name} must be array-like, got {type(arr)}")

    arr_np = np.asarray(arr, dtype=bool)
    return cast("BoolArray", arr_np)


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Validate positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be integer, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

    return int(value)


def validate_probability(value: Any, name: str = "value") -> float:
    """Validate probability value in [0, 1]."""
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    value_float = float(value)
    if not 0 <= value_float <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value_float}")

    return value_float


def validate_clip_grid(grid: Any) -> tuple[float, ...]:
    """Validate clipping grid."""
    if not isinstance(grid, (tuple, list)):
        raise TypeError("clip_grid must be tuple or list")

    if len(grid) == 0:
        raise ValueError("clip_grid cannot be empty")

    validated_grid = []
    for i, val in enumerate(grid):
        if val == float("inf"):
            validated_grid.append(float("inf"))
        elif isinstance(val, (int, float, np.number)) and val > 0:
            validated_grid.append(float(val))
        else:
            raise ValueError(
                f"clip_grid[{i}] must be positive number or inf, got {val}"
            )

    return tuple(validated_grid)
