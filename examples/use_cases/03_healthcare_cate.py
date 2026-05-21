#!/usr/bin/env python3
"""Healthcare CATE — offline evaluation of a treatment-assignment policy.

This use case mirrors a retrospective EHR-style setup:

- Patients arrived; a *baseline-of-care* policy assigned one of several
  treatments; an outcome was recorded.
- A *candidate* policy (e.g., a risk-stratified rule) is being
  considered. We need an estimate of its conditional-average treatment
  effect (CATE) over the population *without* running a clinical trial.

For demonstration we reuse the synthetic generator with three
"treatments" mapped onto operator ids. `evaluate_sklearn_models` returns
the same `EvaluationArtifact` and the same support-health diagnostics
apply — they're particularly important in healthcare, where overlap
violations (e.g., almost no patient with profile X received treatment
B in the logs) are common and dangerous.

The script reports both DR and SNDR; we recommend reading both — large
disagreement is itself a trust signal.
"""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skdr_eval


def main() -> None:
    print("skdr-eval: Healthcare CATE use case")
    print("=" * 60)

    # 1. Build retrospective patient logs (small for CI smoke).
    print("\n1. Building synthetic patient logs (n=2500, 3 treatments)...")
    logs, treatments, _ = skdr_eval.make_synth_logs(n=2500, n_ops=3, seed=23)
    skdr_eval.validate_logs(logs)
    print(f"   Patients: {len(logs):,}")
    print(f"   Treatments: {list(treatments)}")

    # 2. Candidate treatment-assignment policies.
    candidates = {
        "RF_rule": RandomForestRegressor(
            n_estimators=120, max_depth=6, random_state=23
        ),
        "HGB_rule": HistGradientBoostingRegressor(max_iter=120, random_state=23),
    }

    # 3. Evaluate with DR + SNDR; bootstrap CIs on for clinical reporting.
    print("\n2. Running DR / SNDR with bootstrap CIs...")
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=candidates,
        fit_models=True,
        n_splits=3,
        outcome_estimator="hgb",
        random_state=23,
        policy_train="pre_split",
        policy_train_frac=0.8,
        ci_bootstrap=True,
        alpha=0.05,
    )

    # 4. Stakeholder summary.
    print("\n3. Stakeholder summary")
    print("-" * 60)
    cols = [
        "model",
        "estimator",
        "V_hat",
        "CI_lo",
        "CI_hi",
        "ESS",
        "match_rate",
        "support_health",
    ]
    cols = [c for c in cols if c in artifact.report.columns]
    print(artifact.report[cols].round(4).to_string(index=False))

    # 5. Surface DR-vs-SNDR disagreement explicitly — a divergence between
    # the two estimators is the most common trust failure in healthcare
    # OPE because thin overlap inflates DR variance.
    print("\n4. DR vs. SNDR divergence (large gap = trust risk)")
    print("-" * 60)
    for model_name in candidates:
        rows = artifact.report[artifact.report["model"] == model_name]
        dr = rows[rows["estimator"] == "DR"]["V_hat"].iloc[0]
        sndr = rows[rows["estimator"] == "SNDR"]["V_hat"].iloc[0]
        gap = abs(dr - sndr)
        print(f"   {model_name}: |DR - SNDR| = {gap:.3f}")

    # 6. Render a stakeholder card for the best model.
    dr_rows = artifact.report[artifact.report["estimator"] == "DR"]
    best_name = dr_rows.loc[dr_rows["V_hat"].idxmin(), "model"]
    card_path = artifact.save_card(
        f"artifacts/healthcare_cate_{best_name}_card.html", best_name
    )
    print(f"\n5. Stakeholder card: {card_path}")

    print("\nDone. CATE-specific subgroup slicing is tracked in issue #87.")


if __name__ == "__main__":
    main()
