"""Tests for the explicit OOF evaluation-population contract (#280)."""

from __future__ import annotations

import numpy as np
import pytest

from skdr_eval.exceptions import DataValidationError, InsufficientDataError
from skdr_eval.population import (
    EvaluationPopulation,
    build_evaluation_population,
    coverage_from_fold_assignments,
)


def test_fold_assignments_preserve_uncovered_prefix() -> None:
    folds = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64)
    covered = coverage_from_fold_assignments(folds)
    np.testing.assert_array_equal(
        covered,
        np.array([False, False, True, True, True, True]),
    )
    # The contract explicitly does not replace -1 with a synthetic fold.
    np.testing.assert_array_equal(folds[:2], np.array([-1, -1]))


def test_population_intersects_all_required_coverage() -> None:
    population = build_evaluation_population(
        5,
        {
            "outcome_oof": np.array([0, 1, 1, 1, 1]),
            "policy_holdout": np.array([1, 1, 1, 0, 1]),
        },
    )
    assert isinstance(population, EvaluationPopulation)
    np.testing.assert_array_equal(
        population.evaluated_mask,
        np.array([False, True, True, False, True]),
    )
    assert population.input_rows == 5
    assert population.evaluated_rows == 3
    assert population.excluded_rows == 2
    assert population.exclusion_reasons[0] == ("outcome_oof",)
    assert population.exclusion_reasons[3] == ("policy_holdout",)
    assert population.exclusion_counts() == {
        "outcome_oof": 1,
        "policy_holdout": 1,
    }


def test_multiple_exclusion_reasons_are_preserved_per_row() -> None:
    population = build_evaluation_population(
        3,
        {
            "outcome_oof": np.array([0, 1, 1]),
            "propensity_oof": np.array([0, 1, 1]),
        },
    )
    assert population.exclusion_reasons[0] == (
        "outcome_oof",
        "propensity_oof",
    )
    assert population.exclusion_counts() == {
        "outcome_oof": 1,
        "propensity_oof": 1,
    }


def test_apply_filters_row_aligned_arrays() -> None:
    population = build_evaluation_population(
        4,
        {"outcome_oof": np.array([0, 1, 0, 1])},
    )
    one_d = np.array([10, 20, 30, 40])
    two_d = np.arange(8).reshape(4, 2)
    np.testing.assert_array_equal(population.apply(one_d), np.array([20, 40]))
    np.testing.assert_array_equal(
        population.apply(two_d),
        np.array([[2, 3], [6, 7]]),
    )


def test_apply_rejects_non_aligned_array() -> None:
    population = build_evaluation_population(
        3,
        {"outcome_oof": np.array([1, 1, 1])},
    )
    with pytest.raises(DataValidationError, match="3 rows"):
        population.apply(np.array([1, 2]), name="q_hat")


def test_empty_or_too_small_population_fails_closed() -> None:
    with pytest.raises(InsufficientDataError, match="0/3"):
        build_evaluation_population(
            3,
            {"outcome_oof": np.array([0, 0, 0])},
        )
    with pytest.raises(InsufficientDataError, match="minimum_evaluated_rows=3"):
        build_evaluation_population(
            4,
            {"outcome_oof": np.array([0, 1, 0, 1])},
            minimum_evaluated_rows=3,
        )


def test_missing_coverage_contract_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="at least one"):
        build_evaluation_population(3, {})


def test_coverage_mask_shape_and_values_are_validated() -> None:
    with pytest.raises(DataValidationError, match="shape"):
        build_evaluation_population(
            3,
            {"outcome_oof": np.array([1, 1])},
        )
    with pytest.raises(DataValidationError, match="0/1"):
        build_evaluation_population(
            2,
            {"outcome_oof": np.array([1.0, 0.5])},
        )


def test_evaluated_mask_is_immutable_copy() -> None:
    source = np.array([1, 0, 1])
    population = build_evaluation_population(3, {"outcome_oof": source})
    source[:] = 0
    np.testing.assert_array_equal(
        population.evaluated_mask,
        np.array([True, False, True]),
    )
    with pytest.raises(ValueError):
        population.evaluated_mask[0] = False


def test_evaluated_indices_are_explicit() -> None:
    population = build_evaluation_population(
        5,
        {"outcome_oof": np.array([0, 1, 0, 1, 1])},
    )
    indices = population.evaluated_indices
    np.testing.assert_array_equal(indices, np.array([1, 3, 4]))
    with pytest.raises(ValueError):
        indices[0] = 99


def test_non_integer_fold_assignments_rejected() -> None:
    with pytest.raises(DataValidationError, match="integer"):
        coverage_from_fold_assignments(np.array([-1.0, 0.0, 1.0]))
