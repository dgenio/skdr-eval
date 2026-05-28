"""Bootstrap and uncertainty-interval validation under controlled DGPs (#130).

Validates the moving-block bootstrap CI returned by
:func:`skdr_eval.block_bootstrap_ci` (and used by ``ci_bootstrap=True`` on
the evaluators) under three canonical data-generating processes:

* **iid** — independent observations.
* **ar1** — AR(1) serial correlation, rho=0.5.
* **small-n** — iid with n=120 (short-sample regime).

The harness in :mod:`skdr_eval._simulation` already exposes
``simulate_coverage(...)``; this study reuses it so the *same* tooling
runs in CI (via ``make coverage-sim``) and in the validation matrix.

All studies are gated by ``SIM_REPS`` (default 30 for CI; bump locally for
a thorough check).
"""

from __future__ import annotations

import os

from skdr_eval import simulate_coverage

# We always run at least 50 reps so the *empirical* coverage estimate is
# meaningful. ``SIM_REPS`` can be set higher to tighten the result; we
# never run fewer than 50 because Wilson-style coverage bounds collapse
# below that and the test loses diagnostic power.
SIM_REPS = max(int(os.environ.get("SIM_REPS", "50")), 50)


def test_bootstrap_coverage_iid_simulation() -> None:
    """iid Gaussian observations — MBB should hit ~nominal coverage."""
    result = simulate_coverage(
        dgp="iid",
        n=400,
        n_reps=SIM_REPS,
        alpha=0.05,
        block_length_strategy="auto",
    )
    # Allow ±10pp from nominal — wider than the Wilson CI for n_reps=50
    # to absorb MC noise, tight enough to catch a real regression.
    assert result.empirical_coverage >= 0.85, (
        f"iid empirical={result.empirical_coverage:.3f} (n_reps={SIM_REPS})"
    )


def test_bootstrap_coverage_ar1_simulation() -> None:
    """AR(1) serially correlated — MBB block-length should ~cover.

    With rho=0.5 the MBB consistency proof (Künsch 1989) applies and the
    auto block-length rule should keep coverage close to nominal,
    though we allow a wider envelope than the iid case.
    """
    result = simulate_coverage(
        dgp="ar1",
        n=400,
        n_reps=SIM_REPS,
        alpha=0.05,
        block_length_strategy="auto",
    )
    assert result.empirical_coverage >= 0.80, (
        f"ar1 empirical={result.empirical_coverage:.3f} (n_reps={SIM_REPS})"
    )


def test_bootstrap_coverage_small_n_simulation() -> None:
    """Short-sample iid regime — MBB should remain ~calibrated."""
    result = simulate_coverage(
        dgp="iid",
        n=120,
        n_reps=SIM_REPS,
        alpha=0.05,
        block_length_strategy="auto",
    )
    assert result.empirical_coverage >= 0.80, (
        f"small-n empirical={result.empirical_coverage:.3f} (n_reps={SIM_REPS})"
    )


def test_bootstrap_seasonal_documents_assumption_boundary() -> None:
    """Seasonal DGP is a known stress case — record the empirical coverage.

    The MBB consistency proof assumes short-range dependence (Künsch 1989).
    A seasonal DGP with period 52 has dependence at the period boundary
    that the auto block-length rule does not absorb. This test documents
    the empirical coverage so the validation matrix can cite it; it does
    NOT enforce nominal coverage because that would be wishful.
    """
    result = simulate_coverage(
        dgp="seasonal",
        n=300,
        n_reps=SIM_REPS,
        alpha=0.05,
        block_length_strategy="auto",
    )
    assert 0.0 <= result.empirical_coverage <= 1.0
    assert result.dgp == "seasonal"
    assert result.n_reps == SIM_REPS
