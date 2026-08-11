"""Explicit evaluated-population contracts for cross-fitted OPE.

Time-forward cross-fitting can leave warm-up/training-only rows without a genuine
out-of-fold prediction.  The validated path must exclude those rows explicitly
(or fail), never populate statistical defaults merely to keep the evaluation
rectangular (#280 / #297).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .exceptions import DataValidationError, InsufficientDataError


@dataclass(frozen=True)
class EvaluationPopulation:
    """Immutable description of the rows that contribute to an evaluation."""

    input_rows: int
    evaluated_mask: np.ndarray
    exclusion_reasons: tuple[tuple[str, ...], ...]

    @property
    def evaluated_rows(self) -> int:
        return int(self.evaluated_mask.sum())

    @property
    def excluded_rows(self) -> int:
        return self.input_rows - self.evaluated_rows

    @property
    def evaluated_indices(self) -> np.ndarray:
        out = np.flatnonzero(self.evaluated_mask)
        out.setflags(write=False)
        return out

    def exclusion_counts(self) -> dict[str, int]:
        """Count each exclusion reason independently across rows."""

        counts: dict[str, int] = {}
        for reasons in self.exclusion_reasons:
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def apply(self, values: np.ndarray, *, name: str = "values") -> np.ndarray:
        """Return only evaluated rows from a row-aligned array."""

        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.input_rows:
            raise DataValidationError(
                f"{name} must have {self.input_rows} rows; got shape {array.shape}"
            )
        return array[self.evaluated_mask]


def coverage_from_fold_assignments(
    fold_indices: np.ndarray,
    *,
    n_rows: int | None = None,
) -> np.ndarray:
    """Convert fold provenance into a boolean OOF-coverage mask.

    A row is covered iff its fold assignment is a non-negative integer.  The
    function does not invent a fold for uncovered rows.
    """

    folds = np.asarray(fold_indices)
    if folds.ndim != 1:
        raise DataValidationError(
            f"fold_indices must be one-dimensional; got shape {folds.shape}"
        )
    if n_rows is not None and folds.shape != (n_rows,):
        raise DataValidationError(
            f"fold_indices must have shape ({n_rows},); got {folds.shape}"
        )
    if not np.issubdtype(folds.dtype, np.integer):
        raise DataValidationError("fold_indices must contain integer fold identifiers")
    return folds >= 0


def build_evaluation_population(
    n_rows: int,
    required_coverage: Mapping[str, np.ndarray],
    *,
    minimum_evaluated_rows: int = 1,
) -> EvaluationPopulation:
    """Intersect required prediction-coverage masks and record exclusions.

    Parameters
    ----------
    n_rows:
        Number of input rows before cross-fitting exclusions.
    required_coverage:
        Mapping from a stable reason/source name (for example ``"outcome_oof"``)
        to a boolean mask where ``True`` means that row has valid evidence from
        that source.  All required masks must be true for a row to contribute.
    minimum_evaluated_rows:
        Minimum number of rows required after intersection.  The default of one
        merely prevents an empty estimand; higher statistical support thresholds
        belong to the evidence/diagnostic layer.

    Returns
    -------
    EvaluationPopulation
        Explicit row mask plus per-row exclusion reasons.

    Raises
    ------
    DataValidationError
        For malformed/ambiguous coverage inputs.
    InsufficientDataError
        When fewer than ``minimum_evaluated_rows`` remain.
    """

    if not isinstance(n_rows, int) or n_rows < 0:
        raise DataValidationError(f"n_rows must be a non-negative integer; got {n_rows!r}")
    if not isinstance(minimum_evaluated_rows, int) or minimum_evaluated_rows < 1:
        raise DataValidationError(
            "minimum_evaluated_rows must be a positive integer; "
            f"got {minimum_evaluated_rows!r}"
        )
    if not required_coverage:
        raise DataValidationError(
            "required_coverage must name at least one coverage requirement"
        )

    normalized: dict[str, np.ndarray] = {}
    for raw_name, raw_mask in required_coverage.items():
        name = str(raw_name).strip()
        if not name:
            raise DataValidationError("coverage requirement names must be non-empty")
        if name in normalized:
            raise DataValidationError(f"duplicate coverage requirement name: {name!r}")

        mask = np.asarray(raw_mask)
        if mask.shape != (n_rows,):
            raise DataValidationError(
                f"coverage mask {name!r} must have shape ({n_rows},); got {mask.shape}"
            )
        if mask.dtype != np.bool_:
            if not np.all(np.isin(mask, (0, 1))):
                raise DataValidationError(
                    f"coverage mask {name!r} must contain only booleans or 0/1 values"
                )
            mask = mask.astype(bool)
        else:
            mask = mask.astype(bool, copy=False)
        normalized[name] = mask

    evaluated = np.ones(n_rows, dtype=bool)
    for mask in normalized.values():
        evaluated &= mask

    reasons: list[tuple[str, ...]] = []
    for row in range(n_rows):
        row_reasons = tuple(name for name, mask in normalized.items() if not mask[row])
        reasons.append(row_reasons)

    n_evaluated = int(evaluated.sum())
    if n_evaluated < minimum_evaluated_rows:
        counts: dict[str, int] = {}
        for row_reasons in reasons:
            for reason in row_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        raise InsufficientDataError(
            "Explicit OOF population left too few evaluated rows: "
            f"{n_evaluated}/{n_rows} < minimum_evaluated_rows="
            f"{minimum_evaluated_rows}; exclusion counts={counts}"
        )

    stable_mask = evaluated.copy()
    stable_mask.setflags(write=False)
    return EvaluationPopulation(
        input_rows=n_rows,
        evaluated_mask=stable_mask,
        exclusion_reasons=tuple(reasons),
    )


__all__ = [
    "EvaluationPopulation",
    "build_evaluation_population",
    "coverage_from_fold_assignments",
]
