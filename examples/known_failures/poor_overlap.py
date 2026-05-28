#!/usr/bin/env python3
"""Poor-overlap failure mode (#134).

When the logging policy is near-argmax — a very common pattern when the
"logging" policy is itself a deployed greedy ranker — the importance
weight ``π_target / π_b`` is unbounded on the rows where the target
policy disagrees. PSIS Pareto-k grows past 0.7 ("variance does not
exist") and DR loses its consistency property.

This script reproduces that regime deliberately so a newcomer can see:

1. What the `support_health` banner looks like under poor overlap.
2. Which warning codes fire (``HIGH_PARETO_K``, ``EXTREME_CLIP``,
   ``POOR_OVERLAP``, often ``LOW_ESS``).
3. Why the resulting `V_hat` should not be treated as a deployment
   signal.

Cross-references:

- ``docs/concepts/estimands-and-assumptions.md`` §2 (assumption #2:
  overlap / positivity).
- ``docs/statistical-validation-matrix.md`` (overlap-failure row).
- ``tests/sim_studies/test_overlap_failure.py`` (the simulation that
  characterises this regime quantitatively).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def _build_near_argmax_logs(n: int = 1500, seed: int = 17) -> pd.DataFrame:
    """Generate logs where the logging policy is nearly argmax over 4 ops."""
    rng = np.random.default_rng(seed)
    n_ops = 4
    op_names = [f"op_{i}" for i in range(n_ops)]
    # Per-row "true" scores for each operator; the candidate ranker will
    # try to learn these. We make them deterministic enough that the
    # logging policy's argmax is very sharp.
    X = rng.normal(size=(n, 3))
    W = np.array(
        [[1.5, -0.5, 0.0], [-1.0, 1.5, 0.0], [0.0, -0.5, 1.5], [0.5, 0.5, -1.5]]
    )
    scores = X @ W.T  # (n, 4)
    # Near-argmax logging: softmax with very low temperature.
    temperature = 0.05
    logits = scores / temperature
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    # Sample logged action from this near-argmax policy.
    A = np.array(
        [rng.choice(n_ops, p=e[i]) for i in range(n)],
        dtype=int,
    )
    service_time = -scores[np.arange(n), A] + rng.normal(scale=0.5, size=n) + 20.0

    # Eligibility: every operator is eligible for every row.
    elig_cols = {f"{op}_elig": np.ones(n, dtype=int) for op in op_names}

    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    cli_cols = {f"cli_{j}": X[:, j] for j in range(3)}
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
    print("skdr-eval — known failure: poor overlap")
    print("=" * 60)
    print(
        "Logging policy is near-argmax. The candidate policy disagrees"
        " on most rows.\nExpect support_health = high_risk."
    )

    logs = _build_near_argmax_logs(n=1500, seed=17)
    skdr_eval.validate_logs(logs)

    art = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={
            "HGB": HistGradientBoostingRegressor(max_iter=80, random_state=17),
        },
        policy_train="pre_split",
    )

    print("\nReport:")
    print(
        art.report[
            [
                "model",
                "estimator",
                "V_hat",
                "pareto_k",
                "min_pscore",
                "tail_mass",
                "support_health",
                "diagnostic_warnings",
            ]
        ].to_string(index=False)
    )
    print("\nInterpretation:")
    print(
        "- pareto_k well above 0.7 → variance does not exist; the IPS"
        " leg cannot be trusted."
    )
    print(
        "- tail_mass close to 1.0 → almost every row was clipped at the"
        " max weight; the estimator collapsed to the direct method."
    )
    print(
        "- support_health == 'high_risk' → do NOT treat V_hat as a"
        " deployment signal. See docs/statistical-validation-matrix.md."
    )


if __name__ == "__main__":
    main()
