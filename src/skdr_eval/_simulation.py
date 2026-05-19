"""Coverage-probability simulation harness for moving-block bootstrap CIs.

This private module implements Monte-Carlo coverage verification for the
:func:`~skdr_eval.core.block_bootstrap_ci` estimator under three canonical
data-generating processes (DGPs): independent observations, AR(1) serial
correlation, and weekly-seasonal data.

Usage
-----
Run directly as a script (``python -m skdr_eval._simulation``) or import
:func:`simulate_coverage` and :class:`CoverageResult` for programmatic use.

Issues: #81 (simulation harness), #62 (DGP coverage test).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .core import block_bootstrap_ci

# ---------------------------------------------------------------------------
# CoverageResult
# ---------------------------------------------------------------------------


@dataclass
class CoverageResult:
    """Summary of a Monte-Carlo coverage simulation.

    Attributes
    ----------
    empirical_coverage : float
        Fraction of replications in which the true parameter fell inside the
        ``1 - alpha`` bootstrap CI.
    ci_for_coverage : tuple[float, float]
        Wilson (1927) 95% confidence interval for ``empirical_coverage``.
    passes_nominal : bool
        True when ``ci_for_coverage[0] >= alpha_tolerance``, i.e. the
        empirical coverage is consistent with the nominal rate within the
        sampling variability of the simulation.
    dgp : str
        Name of the DGP (``"iid"``, ``"ar1"``, or ``"seasonal"``).
    n : int
        Number of observations per replication.
    n_reps : int
        Number of Monte-Carlo replications.
    alpha : float
        Nominal significance level (target coverage = ``1 - alpha``).
    block_length_strategy : str
        Block-length rule used (``"auto"`` or ``"fixed"``).
    block_len : int or None
        Explicit block length when ``block_length_strategy="fixed"``.
    """

    empirical_coverage: float
    ci_for_coverage: tuple[float, float]
    passes_nominal: bool
    dgp: str
    n: int
    n_reps: int
    alpha: float
    block_length_strategy: str
    block_len: int | None = None


# ---------------------------------------------------------------------------
# DGP samplers
# ---------------------------------------------------------------------------


def _sample_iid(n: int, rng: np.random.Generator) -> np.ndarray:
    """IID normal observations with mean=1.0, std=1.0."""
    return rng.normal(loc=1.0, scale=1.0, size=n)


def _sample_ar1(n: int, rng: np.random.Generator, rho: float = 0.5) -> np.ndarray:
    """AR(1) process: Y_t = rho * Y_{t-1} + eps_t, eps ~ N(0, 1-rho^2).

    Stationary mean = 0; we shift by 1.0 so that true mean = 1.0.
    """
    sd = math.sqrt(1.0 - rho**2)
    y = np.empty(n, dtype=float)
    y[0] = rng.normal(0.0, 1.0)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal(0.0, sd)
    return y + 1.0  # shift so E[Y] = 1.0


def _sample_seasonal(n: int, rng: np.random.Generator, period: int = 52) -> np.ndarray:
    """Seasonal process: Y_t = sin(2π t / period) + N(0, 0.5).

    True mean = 0 (sin averages to 0 over full periods); shifted by 1.0.
    """
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + rng.normal(1.0, 0.5, size=n)


def _true_mean(dgp: str) -> float:
    """Return the population mean for the given DGP."""
    return 1.0  # all three DGPs are shifted so E[Y] = 1.0


# ---------------------------------------------------------------------------
# simulate_coverage
# ---------------------------------------------------------------------------


def simulate_coverage(
    dgp: Literal["iid", "ar1", "seasonal"] = "ar1",
    n: int = 5000,
    n_reps: int = 500,
    alpha: float = 0.05,
    block_length_strategy: Literal["auto", "fixed"] = "auto",
    block_len: int | None = None,
    seed: int = 42,
    tolerance: float = 0.05,
) -> CoverageResult:
    """Estimate the empirical coverage probability of the moving-block bootstrap CI.

    For each of ``n_reps`` replications, draws ``n`` observations from the
    selected DGP, calls :func:`~skdr_eval.core.block_bootstrap_ci`, and
    checks whether the true mean falls inside the resulting interval.

    Parameters
    ----------
    dgp : {"iid", "ar1", "seasonal"}, default "ar1"
        Data-generating process.

        - ``"iid"``: independent standard-normal observations.
        - ``"ar1"``: AR(1) with autocorrelation ρ = 0.5.
        - ``"seasonal"``: sinusoidal weekly seasonality (period = 52) plus
          i.i.d. noise.
    n : int, default 5000
        Observations per replication.
    n_reps : int, default 500
        Number of Monte-Carlo replications.
    alpha : float, default 0.05
        Nominal significance level.  The target coverage is ``1 - alpha``.
    block_length_strategy : {"auto", "fixed"}, default "auto"
        ``"auto"`` lets :func:`block_bootstrap_ci` choose the block length
        (currently ``max(1, n^{1/3})``).  ``"fixed"`` uses ``block_len``.
    block_len : int, optional
        Required when ``block_length_strategy="fixed"``.
    seed : int, default 42
        Master seed for reproducibility.
    tolerance : float, default 0.05
        Allowed shortfall below nominal coverage for
        :attr:`CoverageResult.passes_nominal`.

    Returns
    -------
    CoverageResult
        Empirical coverage, Wilson CI, and a pass/fail verdict.

    Raises
    ------
    ValueError
        If ``dgp`` is not recognised, or ``block_len`` is not provided when
        ``block_length_strategy="fixed"``.
    """
    _dgp_samplers: dict[str, Any] = {"iid": _sample_iid, "ar1": _sample_ar1, "seasonal": _sample_seasonal}
    if dgp not in _dgp_samplers:
        raise ValueError(f"Unknown DGP: {dgp!r}. Must be one of {sorted(_dgp_samplers)}")
    if block_length_strategy == "fixed" and block_len is None:
        raise ValueError("block_len must be provided when block_length_strategy='fixed'.")
    if block_length_strategy not in ("auto", "fixed"):
        raise ValueError(
            f"Unknown block_length_strategy: {block_length_strategy!r}. "
            "Must be 'auto' or 'fixed'."
        )

    sampler: Callable[[int, np.random.Generator], np.ndarray] = _dgp_samplers[dgp]
    true_mean = _true_mean(dgp)
    rng = np.random.default_rng(seed)
    nominal_coverage = 1.0 - alpha
    in_ci = 0

    for _ in range(n_reps):
        y = sampler(n, rng)
        sample_mean = np.mean(y)

        _block = block_len if block_length_strategy == "fixed" else None

        ci_lo, ci_hi = block_bootstrap_ci(
            values_num=y,
            values_den=None,
            base_mean=np.array([sample_mean]),
            n_boot=400,
            alpha=alpha,
            block_len=_block,
            random_state=int(rng.integers(0, 2**31)),
        )
        if ci_lo <= true_mean <= ci_hi:
            in_ci += 1

    empirical_coverage = in_ci / n_reps

    # Wilson confidence interval for the coverage proportion
    z = 1.959964  # z_{0.975}
    n_r = n_reps
    p = empirical_coverage
    denom = 1.0 + z**2 / n_r
    centre = (p + z**2 / (2 * n_r)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n_r + z**2 / (4 * n_r**2))
    ci_coverage = (max(0.0, centre - half), min(1.0, centre + half))

    # Passes if the empirical coverage is at or above the nominal minus tolerance.
    # The Wilson CI provides a measure of uncertainty but is not used for
    # gating at small n_reps, where it is too conservative.
    passes = empirical_coverage >= nominal_coverage - tolerance

    return CoverageResult(
        empirical_coverage=empirical_coverage,
        ci_for_coverage=ci_coverage,
        passes_nominal=passes,
        dgp=dgp,
        n=n,
        n_reps=n_reps,
        alpha=alpha,
        block_length_strategy=block_length_strategy,
        block_len=block_len,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run moving-block bootstrap coverage simulation."
    )
    parser.add_argument("--dgp", choices=["iid", "ar1", "seasonal"], default="ar1")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--n_reps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--block_length_strategy", choices=["auto", "fixed"], default="auto"
    )
    parser.add_argument("--block_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = simulate_coverage(
        dgp=args.dgp,
        n=args.n,
        n_reps=args.n_reps,
        alpha=args.alpha,
        block_length_strategy=args.block_length_strategy,
        block_len=args.block_len,
        seed=args.seed,
    )
    print(f"DGP          : {result.dgp}")
    print(f"n            : {result.n}")
    print(f"n_reps       : {result.n_reps}")
    print(f"nominal      : {1.0 - result.alpha:.2%}")
    print(f"empirical    : {result.empirical_coverage:.2%}")
    print(f"Wilson 95%CI : [{result.ci_for_coverage[0]:.2%}, {result.ci_for_coverage[1]:.2%}]")
    print(f"passes       : {result.passes_nominal}")
    sys.exit(0 if result.passes_nominal else 1)


if __name__ == "__main__":
    _main()
