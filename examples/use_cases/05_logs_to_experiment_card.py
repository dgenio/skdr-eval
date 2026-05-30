#!/usr/bin/env python3
"""End-to-end practitioner recipe: raw-ish logs -> experiment-review card.

This is the canonical "default way to use the library" walkthrough referenced
by ``docs/recipes/logs-to-experiment-card.md`` (issue #124). It is deliberately
more narrative than ``examples/quickstart.py``: it starts from a realistic
problem statement, builds *well-explored* logged decisions, and then reads the
resulting ``EvaluationArtifact`` in the order a practitioner should:

    support_health  ->  warnings  ->  sensitivity  ->  calibration  ->  V_hat

The logging policy here is epsilon-greedy with a large exploration rate, so the
candidate policy is genuinely supported by the logs and the headline row reports
``support_health == "ok"``. Contrast this with
``examples/known_failures/`` (deliberately unsupported) and
``examples/notebooks/06_good_vs_bad_support.ipynb`` (#121), which show what an
*untrustworthy* evaluation looks like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skdr_eval


def make_logged_decisions(
    n: int = 6000, seed: int = 7, explore: float = 0.65
) -> tuple[pd.DataFrame, list[str]]:
    """Simulate the logs a practitioner would already have on disk.

    Three always-eligible operators; an *exploratory* (epsilon-greedy) baseline
    routing policy so every action keeps a healthy logged probability. The
    observed ``service_time`` depends on client features, so a learned model has
    real signal to exploit.
    """
    rng = np.random.RandomState(seed)
    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    cli_urgency = rng.normal(0.0, 1.0, n)
    cli_tenure = rng.uniform(0.0, 1.0, n)

    ops = ["agent_alpha", "agent_bravo", "agent_charlie"]
    n_ops = len(ops)
    # True per-operator service time as a function of client features.
    mu = np.stack(
        [
            10.0 + 2.0 * cli_urgency + 3.0 * cli_tenure,
            11.0 - 1.0 * cli_urgency + 1.0 * cli_tenure,
            9.0 + 1.0 * cli_urgency + 4.0 * cli_tenure,
        ],
        axis=1,
    )
    greedy = mu.argmin(axis=1)
    # Epsilon-greedy baseline: explore uniformly with prob `explore`.
    behavior = np.full((n, n_ops), explore / n_ops)
    behavior[np.arange(n), greedy] += 1.0 - explore
    actions = np.array([rng.choice(n_ops, p=behavior[i]) for i in range(n)])
    service_time = np.maximum(
        mu[np.arange(n), actions] + rng.normal(0.0, 1.0, n), 0.1
    )

    data: dict[str, object] = {
        "arrival_ts": ts,
        "cli_urgency": cli_urgency,
        "cli_tenure": cli_tenure,
    }
    for op in ops:
        data[f"{op}_elig"] = np.ones(n, dtype=bool)
    data["action"] = [ops[a] for a in actions]
    data["service_time"] = service_time
    return pd.DataFrame(data), ops


def main() -> None:
    print("skdr-eval recipe: logs -> experiment-review card")
    print("=" * 60)

    # 1. The problem: "We trained a new routing model. Is it worth A/B testing?"
    # 2. The data we already log: context, action taken, observed outcome.
    print("\n1. Loading logged decisions (context, action, service_time)...")
    logs, ops = make_logged_decisions()
    print(f"   Decisions: {len(logs):,}   Operators: {ops}")

    # 3. Preflight: validate the schema before spending compute.
    skdr_eval.validate_logs(logs)
    print("   validate_logs: OK")

    # 4. Fit candidate policies (any sklearn-compatible regressor).
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=8, random_state=7
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=120, max_depth=6, random_state=7
        ),
    }

    # 5. Evaluate with DR / SNDR and time-aware splits.
    print("\n2. Running DR / SNDR evaluation (time-aware splits)...")
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        n_splits=3,
        outcome_estimator="hgb",
        random_state=7,
        policy_train="pre_split",
        policy_train_frac=0.8,
        baseline="logged",
    )

    # 6. Read the artifact in the correct order.
    print("\n3. support_health FIRST (read before any V_hat)")
    print("-" * 60)
    health_cols = [
        c
        for c in ("model", "estimator", "support_health", "ESS", "match_rate")
        if c in artifact.report.columns
    ]
    print(artifact.report[health_cols].round(4).to_string(index=False))

    print("\n4. Warnings (the trust contract)")
    print("-" * 60)
    print(artifact.warnings.to_string(index=False))

    print("\n5. Clip-grid sensitivity (is V_hat stable?)")
    print("-" * 60)
    print(artifact.sensitivity.to_string(index=False))

    print("\n6. Propensity calibration (are the weights honest?)")
    print("-" * 60)
    for model_name, diag in artifact.diagnostics.items():
        print(f"   {model_name}: ECE={diag.ece:.4f}  Brier={diag.brier_score:.4f}")

    print("\n7. Only now: the value estimate (lower service_time is better)")
    print("-" * 60)
    value_cols = [
        c
        for c in ("model", "estimator", "V_hat", "SE_if", "delta_V_hat")
        if c in artifact.report.columns
    ]
    print(artifact.report[value_cols].round(4).to_string(index=False))

    # 7. Hand-off artifact: a stakeholder card.
    dr = artifact.report[artifact.report["estimator"] == "DR"]
    best = dr.loc[dr["V_hat"].idxmin(), "model"]
    card_path = artifact.save_card(
        f"artifacts/recipe_{best}_card.html", best
    )
    print(f"\n8. Wrote stakeholder card for '{best}': {card_path}")
    print("\nDecision rule:")
    print("  - support_health == 'ok'  -> proceed to A/B-test planning.")
    print("  - 'caution' / 'high_risk' -> improve logging/exploration first;")
    print("    the offline estimate is not yet deployment evidence.")


if __name__ == "__main__":
    main()
