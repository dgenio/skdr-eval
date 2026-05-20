#!/usr/bin/env python3
"""E-commerce ranking — offline evaluation of a candidate recommender.

This use case mirrors a common e-commerce setup:

- A *logging* recommender served items to sessions and the platform
  recorded (session features, served item, observed reward).
- A data-science team has trained a *candidate* recommender and wants to
  estimate its policy value **before** A/B testing.

We treat each operator-id in the synthetic generator as a candidate item
and each "service time" as the negated reward (lower is better). The
``evaluate_sklearn_models`` entry point is the right tool for the
contextual-bandit / one-item-per-impression case; for *slate* / top-K
ranking estimators, see open roadmap issue #75.

Stakeholder reading:

* ``V_hat`` — estimated mean reward under the candidate policy.
* ``support_health`` — green/yellow/red trust banner.
* ``pareto_k`` — PSIS Pareto-k; ``< 0.5`` is good, ``≥ 0.7`` is a red
  flag for weight-tail risk.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import skdr_eval


def main() -> None:
    print("skdr-eval: E-commerce ranking use case")
    print("=" * 60)

    # 1. Build a logged-recommender dataset (small for fast CI smoke).
    print("\n1. Building synthetic ranking logs (n=3000, 5 candidate items)...")
    logs, items, _ = skdr_eval.make_synth_logs(n=3000, n_ops=5, seed=7)
    skdr_eval.validate_logs(logs)
    print(f"   Sessions: {len(logs):,}")
    print(f"   Candidate items: {list(items)}")

    # 2. Define candidate ranking models. Each model assigns a *score* to
    # every (session, item) pair; the top-scoring item is the policy's
    # pick. We use lower-is-better semantics (reward = -service_time-ish).
    print("\n2. Defining candidate ranking models...")
    candidates = {
        "RF_ranker": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=7
        ),
        "GBM_ranker": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=7
        ),
    }

    # 3. Offline-evaluate with DR + SNDR.
    print("\n3. Running DR / SNDR with time-aware splits...")
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=candidates,
        fit_models=True,
        n_splits=3,
        outcome_estimator="hgb",
        random_state=7,
        policy_train="pre_split",
        policy_train_frac=0.8,
    )

    # 4. Stakeholder summary.
    print("\n4. Stakeholder summary")
    print("-" * 60)
    cols = [
        "model",
        "estimator",
        "V_hat",
        "SE_if",
        "ESS",
        "match_rate",
        "support_health",
    ]
    print(artifact.report[cols].round(4).to_string(index=False))

    # 5. Trust diagnostics.
    print("\n5. Trust diagnostics")
    print("-" * 60)
    print(artifact.warnings.to_string(index=False))

    # 6. Best candidate by DR estimate.
    dr_rows = artifact.report[artifact.report["estimator"] == "DR"]
    best_name = dr_rows.loc[dr_rows["V_hat"].idxmin(), "model"]
    print(f"\nBest candidate by DR: {best_name}")

    print("\nDone. (For slate / top-K estimators, see roadmap issue #75.)")


if __name__ == "__main__":
    main()
