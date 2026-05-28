#!/usr/bin/env python3
"""Non-stationary logs — moving-block bootstrap CI assumption boundary (#134).

The moving-block bootstrap CI assumes the data-generating process has
short-range dependence: blocks of length ~``O(n^(1/3))`` capture the
serial structure (Künsch 1989). When the reward distribution *drifts*
between fold 1 and fold N, that assumption is violated: blocks within
the drift window are *not* exchangeable with blocks before/after, and
the bootstrap CI under-covers.

This script generates a deliberately non-stationary log (reward mean
shifts halfway through the sample) and runs the evaluator with
``ci_bootstrap=True``. The resulting CIs may or may not cover the true
value — and that variability is precisely the failure mode we want
the user to see.

Cross-references:

- ``docs/concepts/estimands-and-assumptions.md`` §2 (assumption #7:
  time structure respected).
- ``docs/statistical-validation-matrix.md`` (bootstrap-validity row).
- ``tests/sim_studies/test_bootstrap_validity.py``
  (``test_bootstrap_seasonal_documents_assumption_boundary``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def _build_drifting_logs(n: int = 2400, seed: int = 31) -> pd.DataFrame:
    """Generate logs whose reward mean shifts at row n//2."""
    rng = np.random.default_rng(seed)
    n_ops = 3
    op_names = [f"op_{i}" for i in range(n_ops)]
    X = rng.normal(size=(n, 2))
    # Drift: amplitude doubles in the second half. This is a textbook
    # *regime change* — exactly the kind of thing a moving-block
    # bootstrap cannot absorb.
    drift = np.where(np.arange(n) < n // 2, 1.0, 2.0)
    W = np.array([[1.0, 0.0], [0.0, 1.0], [-0.5, -0.5]])
    scores = (X @ W.T) * drift[:, None]

    # Logging policy: softmax over scores with moderate temperature.
    logits = 0.8 * scores
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    A = np.array([rng.choice(n_ops, p=e[i]) for i in range(n)], dtype=int)
    service_time = -scores[np.arange(n), A] + rng.normal(scale=0.5, size=n) + 20.0

    elig_cols = {f"{op}_elig": np.ones(n, dtype=int) for op in op_names}
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    cli_cols = {f"cli_{j}": X[:, j] for j in range(2)}
    st_cols = {f"st_{op}": rng.normal(size=n) for op in op_names}

    return pd.DataFrame(
        {
            "arrival_ts": timestamps,
            "action": [op_names[a] for a in A],
            "service_time": service_time,
            **cli_cols,
            **st_cols,
            **elig_cols,
        }
    )


def main() -> None:
    print("skdr-eval — known failure: non-stationary logs")
    print("=" * 60)
    print("Reward mean drifts at row n//2. Block-bootstrap CI may under-cover.")

    logs = _build_drifting_logs(n=2400, seed=31)
    skdr_eval.validate_logs(logs)

    art = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"HGB": HistGradientBoostingRegressor(max_iter=80, random_state=31)},
        policy_train="pre_split",
        ci_bootstrap=True,
        alpha=0.05,
    )

    print("\nReport (with 95% MBB CI):")
    cols = [
        "model",
        "estimator",
        "V_hat",
        "ci_lower",
        "ci_upper",
        "ESS",
        "support_health",
    ]
    print(art.report[cols].to_string(index=False))

    print("\nInterpretation:")
    print(
        "- The point estimate V_hat is still well-defined, but the CI"
        " assumes short-range serial dependence."
    )
    print(
        "- The drift at row n/2 violates that assumption; the empirical"
        " CI may under-cover the true V."
    )
    print(
        "- ``tests/sim_studies/test_bootstrap_validity.py`` documents the"
        " coverage shortfall on a seasonal/non-stationary DGP."
    )
    print(
        "- Remediation: re-split the logs at the regime change and"
        " evaluate each window separately, or use a longer block length."
    )


if __name__ == "__main__":
    main()
