#!/usr/bin/env python3
"""Ad targeting — offline evaluation of a candidate bidding policy.

This use case mirrors a counterfactual-ads setup (Criteo-style):

- A *logging* targeting policy decided which ad creative to show to each
  impression and recorded (impression features, served creative,
  observed reward).
- A new candidate policy is trained; we want a value estimate **before**
  any live spend is at risk.

For demonstration we reuse the synthetic generator with a few candidate
creatives and run the DR / SNDR path. On real Criteo logs the call
signature is identical — see
[#70](https://github.com/dgenio/skdr-eval/issues/70) for the public
dataset loader roadmap item.

The script ends by rendering an HTML stakeholder card under
``artifacts/ad_targeting_card.html`` so a non-technical PM can read the
verdict.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import skdr_eval


def main() -> None:
    print("skdr-eval: Ad targeting use case")
    print("=" * 60)

    # 1. Logged impressions with multiple candidate creatives.
    print("\n1. Building synthetic ad-impression logs (n=3000, 4 creatives)...")
    logs, creatives, _ = skdr_eval.make_synth_logs(n=3000, n_ops=4, seed=11)
    skdr_eval.validate_logs(logs)
    print(f"   Impressions: {len(logs):,}")
    print(f"   Candidate creatives: {list(creatives)}")

    # 2. Candidate bidding-policy backbones.
    models = {
        "HGB_bidder": HistGradientBoostingRegressor(max_iter=120, random_state=11),
        "Ridge_bidder": Ridge(alpha=1.0, random_state=11),
    }

    # 3. Evaluate. We deliberately keep clip_grid wide because real ad
    # logs tend to have heavy weight tails; the support-health banner
    # will surface that as `caution` or `high_risk`.
    print("\n2. Running DR / SNDR with wide clip grid...")
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        outcome_estimator="hgb",
        random_state=11,
        policy_train="pre_split",
        policy_train_frac=0.8,
        clip_grid=(2.0, 5.0, 10.0, 20.0, 50.0, float("inf")),
    )

    # 4. Stakeholder summary.
    print("\n3. Stakeholder summary")
    print("-" * 60)
    cols = [
        "model",
        "estimator",
        "V_hat",
        "SE_if",
        "ESS",
        "match_rate",
        "support_health",
        "pareto_k",
    ]
    cols = [c for c in cols if c in artifact.report.columns]
    print(artifact.report[cols].round(4).to_string(index=False))

    # 5. Pick the best model by DR and write a stakeholder card.
    dr_rows = artifact.report[artifact.report["estimator"] == "DR"]
    best_name = dr_rows.loc[dr_rows["V_hat"].idxmin(), "model"]
    card_path = artifact.save_card(
        f"artifacts/ad_targeting_{best_name}_card.html", best_name
    )
    print(f"\n4. Stakeholder card written to: {card_path}")

    print(
        "\nDone. (Replace make_synth_logs with a Criteo loader when issue #70 ships.)"
    )


if __name__ == "__main__":
    main()
