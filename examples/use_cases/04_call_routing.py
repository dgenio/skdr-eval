#!/usr/bin/env python3
"""Call routing — pairwise offline evaluation of a routing policy.

This is the original motivating use case for ``skdr-eval``: a call
center logs (client, operator, service_time) tuples under a baseline
routing policy and wants to evaluate a candidate routing model that
chooses an operator per client.

The pairwise API differs from the standard contextual-bandit API in
two ways:

1. The action space (operators) varies by day (eligibility / shifts).
2. Service-time is the *target* to minimize, not maximize.

Both shape differences are handled by ``evaluate_pairwise_models``; the
returned ``EvaluationArtifact`` has the same shape as in the other use
cases, so the same stakeholder card / warnings / sensitivity surface
applies.
"""

from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def main() -> None:
    print("skdr-eval: Call routing (pairwise) use case")
    print("=" * 60)

    # 1. Build pairwise synthetic data — small for CI smoke.
    print("\n1. Building synthetic pairwise logs (3 days, 300 clients/day, 6 ops)...")
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=300, n_ops=6, seed=29
    )
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")
    print(f"   Decisions: {len(logs_df):,}")
    print(f"   Operator-day snapshots: {len(op_daily_df):,}")

    # 2. Train a candidate routing model on the observed columns.
    feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    candidate = HistGradientBoostingRegressor(max_iter=120, random_state=29)
    candidate.fit(logs_df[feature_cols].values, logs_df["service_time"].values)

    # 3. Pairwise evaluation under the autoscaling / streaming strategy.
    print("\n2. Running pairwise evaluation with stream_topk...")
    artifact = skdr_eval.evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models={"HGB_router": candidate},
        metric_col="service_time",
        task_type="regression",
        direction="min",
        strategy="stream_topk",
        topk=4,
        n_splits=3,
        random_state=29,
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
    ]
    cols = [c for c in cols if c in artifact.report.columns]
    print(artifact.report[cols].round(4).to_string(index=False))

    # 5. Render the stakeholder card.
    card_path = artifact.save_card(
        "artifacts/call_routing_HGB_router_card.html", "HGB_router"
    )
    print(f"\n4. Stakeholder card: {card_path}")

    # 6. Compare to the observed baseline (mean service_time).
    baseline = logs_df["service_time"].mean()
    dr = artifact.report.query("model == 'HGB_router' and estimator == 'DR'")[
        "V_hat"
    ].iloc[0]
    improvement = (baseline - dr) / baseline * 100
    print(
        f"\n5. Baseline mean service_time: {baseline:.2f}  "
        f"Candidate DR estimate: {dr:.2f}  Improvement: {improvement:+.1f}%"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
