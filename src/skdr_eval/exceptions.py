"""Custom exceptions for skdr-eval."""


class SkdrEvalError(Exception):
    """Base exception for skdr-eval errors."""


class DataValidationError(SkdrEvalError):
    """Raised when input data fails validation checks."""


class InsufficientDataError(SkdrEvalError):
    """Raised when there is not enough data for a computation."""
