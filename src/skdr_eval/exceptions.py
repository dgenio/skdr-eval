"""Custom exceptions for skdr-eval library."""

from typing import Any


class SkdrEvalError(Exception):
    """Base exception for skdr-eval library."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
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


class OptionalDependencyError(SkdrEvalError):
    """Raised when a feature needs an optional dependency that is not installed.

    Used by the optional-extra surfaces (e.g. ``EvaluationArtifact.to_polars``,
    the boosting adapters, the dataset loaders) so callers get a single,
    actionable error type carrying the missing package and the ``pip install``
    hint that would enable the feature.

    Parameters
    ----------
    feature : str
        Human-readable name of the feature that needs the dependency.
    package : str
        Importable module name that is missing (e.g. ``"polars"``).
    extra : str, optional
        The ``skdr-eval[<extra>]`` extra that installs ``package``. When
        provided, the message includes the ``pip install`` hint.
    """

    def __init__(
        self,
        feature: str,
        package: str,
        extra: str | None = None,
    ) -> None:
        if extra is not None:
            hint = f"pip install 'skdr-eval[{extra}]'"
        else:
            hint = f"pip install {package}"
        message = (
            f"{feature} requires the optional dependency '{package}', which is "
            f"not installed. Install it with: {hint}"
        )
        super().__init__(
            message,
            details={"feature": feature, "package": package, "extra": extra},
        )
        self.feature = feature
        self.package = package
        self.extra = extra


class DatasetError(SkdrEvalError):
    """Raised when a public-dataset loader cannot produce a usable dataset.

    Covers the fail-loud paths of :mod:`skdr_eval.datasets`: network
    unavailable, insufficient disk space, a license that has not been
    accepted, or a downloaded file that fails its integrity check.
    """

    pass
