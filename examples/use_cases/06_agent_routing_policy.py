#!/usr/bin/env python3
"""Evaluate a candidate agent routing / tool-selection policy from logged traces.

The companion story made concrete (#150): you have **logged agent decisions** —
for each request, the route/tool the agent chose and what it cost (latency +
failure penalty) — and you want to know whether a *candidate* routing policy
would be cheaper, **before** you ship it.

The trace is a list of generic ``(context, action, reward)`` records, exactly
the shape an agent framework (e.g. the Weaver stack) emits. We map it into the
skdr-eval log schema with :func:`skdr_eval.adapters.from_records` (#149), then
evaluate the candidate policy with DR / SNDR and read the trust verdict.

``reward`` here is a **cost** (lower is better), which matches the
policy-induction convention in :func:`skdr_eval.induce_policy_from_sklearn`
(it up-weights actions with lower predicted outcome).

We run two regimes on the same problem:

* **healthy** — the logging agent explored enough, so the candidate is
  supported by the logs and ``support_health == "ok"``: the offline estimate
  is usable decision evidence.
* **unhealthy** — the logging agent was near-deterministic, so overlap is poor
  and ``support_health == "high_risk"``: the honest answer is "your logs don't
  support this question yet", not a number to act on.

See ``docs/weaver-stack.md`` (#148) for where this fits, and
``examples/notebooks/06_good_vs_bad_support.ipynb`` (#121) for the same
good-vs-bad-support lesson on the standard API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skdr_eval

ROUTES = ["route_fast", "route_smart", "route_cheap"]


def make_agent_traces(
    n: int = 6000, seed: int = 11, explore: float = 0.6
) -> list[dict]:
    """Synthesize logged agent routing decisions as generic trace records.

    Each record carries a ``context`` (request features), the ``action`` the
    logging agent chose, the observed ``reward`` (cost; lower is better) and a
    ``timestamp``. ``explore`` is the epsilon of the epsilon-greedy logging
    agent: high exploration keeps every route's logged probability healthy.
    """
    rng = np.random.RandomState(seed)
    req_tokens = rng.uniform(0.0, 1.0, n)  # normalized prompt size
    priority = rng.uniform(0.0, 1.0, n)  # request priority
    retrieval = rng.normal(0.0, 1.0, n)  # retrieval-confidence signal

    n_routes = len(ROUTES)
    # True per-route cost as a function of request features (lower is better).
    cost = np.stack(
        [
            6.0 + 5.0 * req_tokens + 1.0 * priority,  # fast: cheap on small reqs
            5.0
            + 1.0 * req_tokens
            - 2.0 * retrieval,  # smart: best when retrieval helps
            7.0 - 1.0 * priority + 2.0 * req_tokens,  # cheap: steady, mediocre
        ],
        axis=1,
    )
    greedy = cost.argmin(axis=1)
    behavior = np.full((n, n_routes), explore / n_routes)
    behavior[np.arange(n), greedy] += 1.0 - explore
    actions = np.array([rng.choice(n_routes, p=behavior[i]) for i in range(n)])
    observed_cost = np.maximum(
        cost[np.arange(n), actions] + rng.normal(0.0, 0.5, n), 0.1
    )

    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="s")
    records: list[dict] = []
    for i in range(n):
        records.append(
            {
                "context": {
                    "req_tokens": float(req_tokens[i]),
                    "priority": float(priority[i]),
                    "retrieval": float(retrieval[i]),
                },
                "action": ROUTES[actions[i]],
                "reward": float(observed_cost[i]),
                "timestamp": ts[i].isoformat(),
            }
        )
    return records


def evaluate_regime(name: str, explore: float) -> str:
    """Adapt one trace regime and evaluate the candidate routing policy."""
    print(
        f"\n{'=' * 64}\n{name.upper()} regime (logging explore={explore})\n{'=' * 64}"
    )

    # 1. Logged agent decisions arrive as generic trace records.
    trace = make_agent_traces(explore=explore)

    # 2. Map the trace into the skdr-eval log schema (no hand-shaping).
    adapted = skdr_eval.adapters.from_records(trace, reward_col="cost")
    print(f"   adapter: {adapted.summary()}")

    # 3. Candidate routing policies: any sklearn-compatible cost regressor.
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=8, random_state=11
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=120, max_depth=6, random_state=11
        ),
    }

    # 4. Evaluate with DR / SNDR on the adapted logs (cost: lower is better).
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=adapted.logs,
        models=models,
        n_splits=3,
        y_col=adapted.reward_col,
        outcome_estimator="hgb",
        random_state=11,
        policy_train="pre_split",
        policy_train_frac=0.8,
        baseline="logged",
    )

    cols = [
        c
        for c in ("model", "estimator", "V_hat", "ESS", "support_health")
        if c in artifact.report.columns
    ]
    print(artifact.report[cols].round(4).to_string(index=False))

    health = set(artifact.report["support_health"])
    if "ok" in health:
        verdict = "DEPLOY-CANDIDATE: support is healthy; estimate is usable evidence."
    else:
        verdict = (
            "DO-NOT-TRUST-YET: poor support; improve agent exploration / logging "
            "before treating the offline estimate as deployment evidence."
        )
    print(f"   verdict: {verdict}")
    return "ok" if "ok" in health else "high_risk"


def main() -> None:
    print("skdr-eval: evaluate a candidate agent routing policy from logged traces")
    healthy = evaluate_regime("healthy", explore=0.6)
    unhealthy = evaluate_regime("unhealthy", explore=0.03)

    print(f"\n{'=' * 64}")
    print("Summary")
    print(f"{'=' * 64}")
    print(f"   healthy regime   -> support_health includes '{healthy}'")
    print(f"   unhealthy regime -> support_health includes '{unhealthy}'")
    print(
        "\nThe point of offline evaluation for agent policies is not just the "
        "number — it is knowing when the logs cannot answer the question."
    )


if __name__ == "__main__":
    main()
