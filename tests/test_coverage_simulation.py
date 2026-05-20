"""Tests for the coverage-probability simulation harness (issue #81).

Validates :func:`~skdr_eval._simulation.simulate_coverage` under the three
canonical DGPs at small n_reps for fast CI, plus a negative-control test that
confirms AR(1) data with an inadequate block length of 1 under-covers.
"""

import pytest

from skdr_eval._simulation import CoverageResult, simulate_coverage

# ---------------------------------------------------------------------------
# Fast smoke tests at small n_reps (speed vs. statistical precision trade-off)
# ---------------------------------------------------------------------------


class TestSimulateCoverageSmoke:
    """Structural sanity checks with small n_reps."""

    def test_returns_coverage_result(self) -> None:
        result = simulate_coverage(dgp="iid", n=500, n_reps=20, seed=0)
        assert isinstance(result, CoverageResult)

    def test_fields_populated(self) -> None:
        result = simulate_coverage(dgp="iid", n=500, n_reps=20, seed=0)
        assert result.dgp == "iid"
        assert result.n == 500
        assert result.n_reps == 20
        assert 0.0 <= result.empirical_coverage <= 1.0
        lo, hi = result.ci_for_coverage
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert result.block_length_strategy == "auto"
        assert result.block_len is None

    def test_ar1_fields(self) -> None:
        result = simulate_coverage(dgp="ar1", n=500, n_reps=20, seed=1)
        assert result.dgp == "ar1"

    def test_seasonal_fields(self) -> None:
        result = simulate_coverage(dgp="seasonal", n=500, n_reps=20, seed=2)
        assert result.dgp == "seasonal"

    def test_fixed_block_len_stored(self) -> None:
        result = simulate_coverage(
            dgp="iid",
            n=500,
            n_reps=10,
            block_length_strategy="fixed",
            block_len=5,
            seed=3,
        )
        assert result.block_len == 5
        assert result.block_length_strategy == "fixed"


class TestSimulateCoverageValidation:
    """Error-path validation."""

    def test_unknown_dgp_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown DGP"):
            simulate_coverage(dgp="poisson", n=100, n_reps=5)  # type: ignore[arg-type]

    def test_fixed_strategy_without_block_len_raises(self) -> None:
        with pytest.raises(ValueError, match="block_len must be provided"):
            simulate_coverage(
                dgp="iid",
                n=100,
                n_reps=5,
                block_length_strategy="fixed",
                block_len=None,
            )

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown block_length_strategy"):
            simulate_coverage(
                dgp="iid",
                n=100,
                n_reps=5,
                block_length_strategy="cubic",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Statistical coverage tests at medium n_reps
# These are inherently stochastic; tighter tolerance = slower but more
# informative. n_reps=100 gives ~0.03 Wilson half-width.
# ---------------------------------------------------------------------------


class TestSimulateCoverageNominal:
    """Coverage probability tests: empirical coverage should be near 1-alpha."""

    @pytest.mark.parametrize("dgp", ["iid", "ar1", "seasonal"])
    def test_nominal_coverage(self, dgp: str) -> None:
        """Empirical coverage should be within tolerance of the nominal rate."""
        result = simulate_coverage(
            dgp=dgp,
            n=2000,
            n_reps=100,
            alpha=0.05,
            seed=42,
            tolerance=0.10,  # generous tolerance for fast CI
        )
        # The Wilson lower bound should be above 1-alpha minus tolerance.
        # passes_nominal is computed the same way internally.
        assert result.passes_nominal, (
            f"Coverage too low for DGP={dgp}: "
            f"empirical={result.empirical_coverage:.2%}, "
            f"Wilson CI={result.ci_for_coverage}"
        )


# ---------------------------------------------------------------------------
# Negative control: block_len=1 on AR(1) should under-cover
# ---------------------------------------------------------------------------


class TestNegativeControl:
    """Confirm that a mis-specified block length leads to under-coverage."""

    def test_block_len_1_undercoverage_ar1(self) -> None:
        """AR(1) with block_len=1 (i.i.d. bootstrap) should under-cover.

        This is a probabilistic test.  A block length of 1 breaks the block
        structure, ignoring serial correlation, so the CI will be too narrow.
        With n=2000, rho=0.5, and n_reps=200 the probability of falsely
        "passing" is extremely small but we document rather than hard-fail
        to avoid flakiness in adversarial random seeds.
        """
        result = simulate_coverage(
            dgp="ar1",
            n=2000,
            n_reps=200,
            alpha=0.05,
            block_length_strategy="fixed",
            block_len=1,
            seed=0,
            tolerance=0.02,  # tight: expect coverage well below 0.93
        )
        # Assert that the negative control actually detects under-coverage:
        # with block_len=1 on AR(1) and a fixed seed, passes_nominal should
        # be False (coverage < nominal - tolerance).
        assert not result.passes_nominal
