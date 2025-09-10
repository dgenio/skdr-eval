"""Custom exceptions for skdr-eval."""

from typing import Any, Optional


class SkdrEvalError(Exception):
    """Base exception for skdr-eval errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initialize exception with message and optional details.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(SkdrEvalError):
    """Raised when input data validation fails."""

    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
    ) -> None:
        """Initialize data validation error.

        Args:
            message: Error message
            column: Column name that failed validation
            expected_type: Expected data type
            actual_type: Actual data type found
        """
        details = {}
        if column:
            details["column"] = column
        if expected_type:
            details["expected_type"] = expected_type
        if actual_type:
            details["actual_type"] = actual_type

        super().__init__(message, details)
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type


class ModelFittingError(SkdrEvalError):
    """Raised when model fitting fails."""

    def __init__(
        self, message: str, model_name: Optional[str] = None, fold: Optional[int] = None
    ) -> None:
        """Initialize model fitting error.

        Args:
            message: Error message
            model_name: Name of the model that failed
            fold: Cross-validation fold where failure occurred
        """
        details = {}
        if model_name:
            details["model_name"] = model_name
        if fold is not None:
            details["fold"] = str(fold)

        super().__init__(message, details)
        self.model_name: Optional[str] = model_name
        self.fold: Optional[int] = fold


class PropensityScoreError(SkdrEvalError):
    """Raised when propensity score estimation fails."""

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        min_propensity: Optional[float] = None,
    ) -> None:
        """Initialize propensity score error.

        Args:
            message: Error message
            method: Propensity estimation method used
            min_propensity: Minimum propensity score found
        """
        details = {}
        if method:
            details["method"] = method
        if min_propensity is not None:
            details["min_propensity"] = str(min_propensity)

        super().__init__(message, details)
        self.method: Optional[str] = method
        self.min_propensity: Optional[float] = min_propensity


class EvaluationError(SkdrEvalError):
    """Raised when policy evaluation fails."""

    def __init__(
        self,
        message: str,
        estimator: Optional[str] = None,
        match_rate: Optional[float] = None,
    ) -> None:
        """Initialize evaluation error.

        Args:
            message: Error message
            estimator: Estimator that failed (DR, SNDR, etc.)
            match_rate: Match rate achieved
        """
        details = {}
        if estimator:
            details["estimator"] = estimator
        if match_rate is not None:
            details["match_rate"] = str(match_rate)

        super().__init__(message, details)
        self.estimator: Optional[str] = estimator
        self.match_rate: Optional[float] = match_rate


class ConfigurationError(SkdrEvalError):
    """Raised when configuration parameters are invalid."""

    def __init__(
        self, message: str, parameter: Optional[str] = None, value: Optional[Any] = None
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            parameter: Parameter name that is invalid
            value: Invalid parameter value
        """
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = value

        super().__init__(message, details)
        self.parameter = parameter
        self.value = value
