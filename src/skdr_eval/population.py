"""Explicit evaluated-population contracts for cross-fitted OPE.

Time-forward cross-fitting can leave warm-up/training-only rows without a genuine
out-of-fold prediction. The validated path must exclude those rows explicitly
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
    """Immutable, self-validating description of rows that contribute.

    Direct construction enforces the same invariants as
    :func:`build_evaluation_population`; callers cannot manufacture an
    inconsistent evaluated mask and exclusion record after validation.
    """

    input_rows: int
    evaluated_mask: np.ndarray
    exclusion_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_rows, int) or self.input_rows < 0:
            raise DataValidationError(
                f"input_rows must be a non-negative integer; got {self.input_rows!r}"
            )

        raw_mask = np.asarray(self.evaluated_mask)
        if raw_mask.shape != (self.input_rows,):
            raise DataValidationError(
                "evaluated_mask must have shape "
                f"({self.input_rows},); got {raw_mask.shape}"
            )
        if raw_mask.dtype != np.bool_:
            if not np.all(np.isin(raw_mask, (0, 1))):
                raise DataValidationError(
                    "evaluated_mask must contain only booleans or 0/1 values"
                )
            raw_mask = raw_mask.astype(bool)
        else:
            raw_mask = raw_mask.astype(bool, copy=False)

        if len(self.exclusion_reasons) != self.input_rows:
            raise DataValidationError(
                "exclusion_reasons must contain one tuple per input row; "
                f"got {len(self.exclusion_reasons)} for {self.input_rows} rows"
            )

        normalized_reasons: list[tuple[str, ...]] = []
        for row, raw_reasons in enumerate(self.exclusion_reasons):
            reasons = tuple(str(reason).strip() for reason in raw_reasons)
            if any(not reason for reason in reasons):
                raise DataValidationError(
                    f"exclusion reasons must be non-empty strings; row {row} is invalid"
                )
            if len(set(reasons)) != len(reasons):
                raise DataValidationError(
                    f"exclusion reasons must be unique per row; row {row} has duplicates"
                )
            if raw_mask[row] and reasons:
                raise DataValidationError(
                    "evaluated rows cannot carry exclusion reasons; "
                    f"row {row} has {reasons}"
                )
            if not raw_mask[row] and not reasons:
                raise DataValidationError(
                    "excluded rows must record at least one exclusion reason; "
                    f"row {row} has none"
                )
            normalized_reasons.append(reasons)

        stable_mask = raw_mask.copy()
        stable_mask.setflags(write=False)
        object.__setattr__(self, "evaluated_mask", stable_mask)
        object.__setattr__(self, "exclusion_reasons", tuple(normalized_reasons))

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

    A row is covered iff its fold assignment is a non-negative integer. The
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
        that source. All required masks must be true for a row to contribute.
    minimum_evaluated_rows:
        Minimum number of rows required after intersection. The default of one
        merely prevents an empty estimand; higher statistical support thresholds
        belong to the evidence/diagnostic layer.
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

    return EvaluationPopulation(
        input_rows=n_rows,
        evaluated_mask=evaluated,
        exclusion_reasons=tuple(reasons),
    )


__all__ = [
    "EvaluationPopulation",
    "build_evaluation_population",
    "coverage_from_fold_assignments",
]
