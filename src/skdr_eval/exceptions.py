"""Custom exceptions for skdr-eval library."""

from typing import Any, Optional


class SkdrEvalError(Exception):
    """Base exception for skdr-eval library."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DataValidationError(SkdrEvalError):
    """Raised when input data validation fails."""

    pass


class ModelValidationError(SkdrEvalError):
    """Raised when model validation fails."""

    pass


class EstimationError(SkdrEvalError):
    """Raised when estimation fails."""

    pass


class ConfigurationError(SkdrEvalError):
    """Raised when configuration is invalid."""

    pass


class ConvergenceError(EstimationError):
    """Raised when optimization fails to converge."""

    pass


class InsufficientDataError(DataValidationError):
    """Raised when there's insufficient data for estimation."""

    pass


class PropensityScoreError(EstimationError):
    """Raised when propensity score estimation fails."""

    pass


class OutcomeModelError(EstimationError):
    """Raised when outcome model fitting fails."""

    pass


class PolicyInductionError(EstimationError):
    """Raised when policy induction fails."""

    pass


class BootstrapError(EstimationError):
    """Raised when bootstrap estimation fails."""

    pass


class PairwiseEvaluationError(EstimationError):
    """Raised when pairwise evaluation fails."""

    pass


class MemoryError(SkdrEvalError):
    """Raised when memory requirements exceed available resources."""

    pass


class VersionError(SkdrEvalError):
    """Raised when version compatibility issues are detected."""

    pass
