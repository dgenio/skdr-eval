"""Simulation proof: the reseeded / parallel moving-block bootstrap recovers a
known ground-truth mean with nominal coverage (#178).

#178 changed ``block_bootstrap_ci`` from a single sequential ``RandomState`` to
one independent ``SeedSequence`` child per replicate so the bootstrap can run on
joblib workers. That is a change to statistical-evaluation logic, so per
``.claude/CLAUDE.md`` §2 it ships with a simulation that (a) proves the new
estimator still recovers a known ground-truth parameter and (b) proves the
parallel path is identical to the serial one.

DGP: iid Normal(mu, sigma^2) with a *known* mu. Over many independent samples
the (1 - alpha) percentile-bootstrap interval must cover mu about (1 - alpha) of
the time. We assert the empirical coverage lands in a Monte-Carlo-tolerant band
around nominal.
"""

from __future__ import annotations

import os

import numpy as np

from skdr_eval import block_bootstrap_ci

SIM_REPS = max(int(os.environ.get("SIM_REPS", "200")), 100)
TRUE_MU = 3.0
SIGMA = 1.0
N = 500
ALPHA = 0.05


def test_parallel_bootstrap_recovers_known_mean() -> None:
    """Empirical coverage of the true mean is close to the nominal 1 - alpha."""
    covered = 0
    for rep in range(SIM_REPS):
        rng = np.random.default_rng(rep)
        values = rng.normal(TRUE_MU, SIGMA, size=N)
        lo, hi = block_bootstrap_ci(
            values_num=values,
            values_den=None,
            base_mean=np.array([values.mean()]),
            n_boot=400,
            alpha=ALPHA,
            random_state=rep,
        )
        covered += int(lo <= TRUE_MU <= hi)

    coverage = covered / SIM_REPS
    # Percentile bootstrap of a mean under-covers slightly in finite samples;
    # allow a generous band that still catches a real regression (e.g. a broken
    # reseed collapsing all replicates to one draw would tank coverage).
    assert 0.88 <= coverage <= 0.99, (
        f"empirical coverage {coverage:.3f} of mu={TRUE_MU} is off nominal "
        f"{1 - ALPHA:.2f} (n_reps={SIM_REPS})"
    )


def test_parallel_path_is_identical_to_serial() -> None:
    """n_jobs must not change the interval: same random_state -> same CI."""
    rng = np.random.default_rng(0)
    values = rng.normal(TRUE_MU, SIGMA, size=N)
    kwargs = {
        "values_num": values,
        "values_den": None,
        "base_mean": np.array([values.mean()]),
        "n_boot": 400,
        "alpha": ALPHA,
        "random_state": 123,
    }
    serial = block_bootstrap_ci(n_jobs=1, **kwargs)
    parallel = block_bootstrap_ci(n_jobs=4, **kwargs)
    assert serial == parallel, (serial, parallel)
