#!/usr/bin/env python3
"""External simulator policies + what-if scenarios (issues #56, #34, #33).

This use case extends the call-routing story (see ``04_call_routing.py``)
beyond model-induced policies:

1. **External policies (#56)** — a discrete-event call-centre *simulator*
   produces ``client_id -> operator_id`` assignments that account for queues
   and shifts. ``evaluate_external_policies`` scores those assignments against
   the logged outcomes with the same DR/SNDR trust diagnostics, so two
   simulators can be compared head-to-head.

2. **What-if scenarios (#34)** — ``simulate_autoscaling_scenario`` re-evaluates
   a candidate routing model under reduced operator capacity, surfacing how the
   support diagnostics (ESS / match_rate) react when fewer operators are
   available.

3. **Large-data execution (#33)** — every entry point accepts
   ``execution_mode="large_data"``, a vectorized path that is numerically
   identical to the default but avoids the per-row Python loop on big inputs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def _simulator_assignment(logs_df: pd.DataFrame, op_daily_df: pd.DataFrame, seed: int):
    """Stand-in for a real simulator: one operator per client.

    A real simulator would account for queues, shift schedules and sequential
    dependencies; here we just produce a valid, reproducible assignment so the
    evaluation path is exercised.
    """
    rng = np.random.default_rng(seed)
    clients = logs_df["client_id"].unique()
    operators = op_daily_df["operator_id"].unique()
    return pd.DataFrame(
        {"client_id": clients, "operator_id": rng.choice(operators, size=len(clients))}
    )


def main() -> None:
    print("skdr-eval: External simulator policies + what-if scenarios")
    print("=" * 60)

    # 1. Build pairwise synthetic logs — small for CI smoke.
    print("\n1. Building synthetic pairwise logs (3 days, 200 clients/day, 6 ops)...")
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=200, n_ops=6, seed=29
    )
    skdr_eval.validate_pairwise_inputs(logs_df, op_daily_df, metric_col="service_time")
    print(f"   Decisions: {len(logs_df):,}")

    # 2. External policies (#56): compare two "simulators".
    print("\n2. Evaluating two external simulator policies (#56)...")
    sim_artifact = skdr_eval.evaluate_external_policies(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        policies={
            "simulator_a": _simulator_assignment(logs_df, op_daily_df, seed=1),
            "simulator_b": _simulator_assignment(logs_df, op_daily_df, seed=2),
        },
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=3,
        random_state=29,
        execution_mode="large_data",  # (#33) identical numbers, vectorized path
    )
    print(
        sim_artifact.report[
            ["model", "estimator", "V_hat", "ESS", "match_rate", "support_health"]
        ].to_string(index=False)
    )

    # 3. What-if scenario (#34): a model-induced policy under reduced capacity.
    print("\n3. What-if scenario: 60% operator capacity (#34)...")
    feature_cols = [c for c in logs_df.columns if c.startswith(("cli_", "op_"))]
    router = HistGradientBoostingRegressor(max_iter=120, random_state=29)
    router.fit(logs_df[feature_cols].to_numpy(), logs_df["service_time"].to_numpy())

    scenario_artifact = skdr_eval.simulate_autoscaling_scenario(
        logs_df,
        op_daily_df,
        models={"HGB_router": router},
        scenario={"capacity_multiplier": 0.6, "eligibility_mode": "as_logged"},
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=3,
        strategy="direct",
        random_state=29,
        policy_train="all",
    )
    print(
        scenario_artifact.report[
            ["model", "estimator", "V_hat", "ESS", "match_rate", "support_health"]
        ].to_string(index=False)
    )
    print("\n   Scenario assumptions (also on artifact.metadata['scenario']):")
    for note in scenario_artifact.metadata["scenario"]["assumptions"]:
        print(f"     - {note}")

    print("\nDone. Read support_health before trusting any V_hat.")


if __name__ == "__main__":
    main()
