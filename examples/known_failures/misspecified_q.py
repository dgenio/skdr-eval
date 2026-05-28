#!/usr/bin/env python3
"""Mis-specified outcome model — DR survives via the IPS leg (#134).

This script demonstrates the *positive* side of double robustness:

- The outcome model q̂ is deliberately under-fit (a constant predictor).
- The propensity model is fit normally and is reasonably calibrated.
- DR still recovers a sensible policy value because the IPS leg of the
  estimator carries the load.

Compared to ``poor_overlap.py`` (where overlap collapses and DR
genuinely breaks), this regime is the *good* failure: the diagnostics
flag the situation but the headline `V_hat` remains usable.

Cross-references:

- ``docs/concepts/estimands-and-assumptions.md`` §2 (assumption #4:
  double robustness).
- ``docs/statistical-validation-matrix.md`` (DR row, "q wrong, e correct"
  cell).
- ``tests/sim_studies/test_dr_misspecification.py``
  (``test_dr_survives_noisy_q_when_residual_unbiased_simulation``).
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def main() -> None:
    print("skdr-eval — known failure: misspecified q")
    print("=" * 60)
    print(
        "Outcome model is a constant (DummyRegressor). The propensity is"
        " fit normally.\nDR should still recover a usable V_hat via the"
        " IPS leg; DM will not."
    )

    logs, _, _ = skdr_eval.make_synth_logs(n=2000, n_ops=4, seed=23)

    # The outcome model used inside DR is selected via ``outcome_estimator``;
    # we deliberately pick ``"mean"`` (a constant predictor) for the
    # misspecified leg.
    art_bad_q = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={
            "HGB_target": HistGradientBoostingRegressor(max_iter=80, random_state=23)
        },
        outcome_estimator=lambda: DummyRegressor(strategy="mean"),
        policy_train="pre_split",
    )

    art_good_q = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={
            "HGB_target": HistGradientBoostingRegressor(max_iter=80, random_state=23)
        },
        outcome_estimator="hgb",
        policy_train="pre_split",
    )

    cols = ["model", "estimator", "V_hat", "SE_if", "ESS", "support_health"]
    print("\nWith DELIBERATELY MISSPECIFIED q (DummyRegressor):")
    print(art_bad_q.report[cols].to_string(index=False))

    print("\nWith well-specified q (HistGradientBoosting):")
    print(art_good_q.report[cols].to_string(index=False))

    print("\nInterpretation:")
    print(
        "- DR's V_hat is comparable between the two regimes — the IPS leg"
        " carries the load when q̂ is constant."
    )
    print(
        "- SE_if is larger in the misspecified regime (variance ↑) —"
        " this is the double-robustness trade-off in action."
    )
    print(
        "- Diagnostics may still flag low ESS or extreme-clip warnings;"
        " those are about overlap, not about q̂."
    )


if __name__ == "__main__":
    main()
