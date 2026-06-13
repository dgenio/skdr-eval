#!/usr/bin/env python3
"""Evaluate an Open Bandit Pipeline (OBP) dataset with skdr-eval (#179).

Runnable companion to ``docs/recipes/obp-interop.md``. It takes logged bandit
feedback in the shape an OBP user already has it — an
``obp``-style ``bandit_feedback`` dict of numpy arrays (``context``,
``action``, ``reward``, ``pscore``, ``position``, ``n_actions``) — maps each
field onto the skdr-eval log schema via the generic trace adapter
(:func:`skdr_eval.adapters.from_records`), and runs DR / SNDR with the full
trust-diagnostics surface.

Two data sources are supported:

* ``--obd`` — load a slice of the real Open Bandit Dataset via
  :func:`skdr_eval.load_obd` (downloads a small public mirror).
* default — a self-contained, synthetic ``bandit_feedback`` dict so the recipe
  runs offline and in CI with **no dependency on OBP itself**. OBP stays a
  docs/notebook-only library; nothing here imports ``obp``.

Honest note carried into the recipe: skdr-eval calibrates its own
propensities, so the logged ``pscore`` is *reported* (the adapter records its
presence) but not *consumed* by the estimators today. First-class logged
propensities are tracked separately in issue #167.

Run it::

    python examples/obp_interop.py            # offline synthetic OBP-shaped data
    python examples/obp_interop.py --obd       # real Open Bandit Dataset slice
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval

# OBP term  ->  skdr-eval log-schema column. The mapping the recipe documents.
OBP_FIELD_MAP = {
    "context": "cli_* feature columns (one per context dimension)",
    "action": "action (integer index, 0..n_actions-1)",
    "reward": "reward (the y_col / reward_col)",
    "pscore": "propensity (reported via from_records, not consumed yet — #167)",
    "position": "no equivalent — skdr-eval is single-slot; see the slate API",
    "timestamp": "arrival_ts (skdr-eval is time-aware; OBP is not)",
}


def make_obp_bandit_feedback(
    n: int = 5000, n_actions: int = 4, seed: int = 0
) -> dict[str, Any]:
    """Synthesize an ``obp``-style ``bandit_feedback`` dict of numpy arrays.

    Mirrors the structure ``obp.dataset.*.obtain_batch_bandit_feedback()``
    returns: ``context`` ``(n, dim)``, ``action`` ``(n,)``, ``reward`` ``(n,)``,
    ``pscore`` ``(n,)`` (logged action propensity), ``position`` ``(n,)`` and
    ``n_actions``. This is deliberately OBP's vocabulary so the field-mapping
    step in the recipe is concrete.
    """
    rng = np.random.RandomState(seed)
    context = rng.normal(size=(n, 3))
    logits = context @ rng.normal(size=(3, n_actions))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    action = np.array([rng.choice(n_actions, p=probs[i]) for i in range(n)])
    pscore = probs[np.arange(n), action]
    action_effect = (np.arange(n_actions) - 1.5) * 0.5
    mean_reward = 1.0 + 0.6 * context[:, 0] - action_effect[action] * context[:, 1]
    reward = mean_reward + rng.normal(scale=0.3, size=n)
    return {
        "context": context,
        "action": action,
        "reward": reward,
        "pscore": pscore,
        "position": np.zeros(n, dtype=int),
        "n_actions": n_actions,
    }


def bandit_feedback_to_records(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    """Map an OBP ``bandit_feedback`` dict to generic trace records.

    Each record carries a ``context`` (sequence of numeric features), the
    logged ``action`` (int), the observed ``reward`` and the logged
    ``propensity`` (OBP's ``pscore``). This is the one field-mapping step a
    practitioner coming from OBP has to perform.
    """
    context = feedback["context"]
    action = feedback["action"]
    reward = feedback["reward"]
    pscore = feedback["pscore"]
    return [
        {
            "context": context[i].tolist(),
            "action": int(action[i]),
            "reward": float(reward[i]),
            "propensity": float(pscore[i]),
        }
        for i in range(len(action))
    ]


def print_field_map() -> None:
    """Print the OBP -> skdr-eval field-mapping table from the recipe."""
    print("OBP field  ->  skdr-eval log schema")
    for obp_field, skdr_col in OBP_FIELD_MAP.items():
        print(f"  {obp_field:<10} -> {skdr_col}")


def evaluate_records(records: list[dict[str, Any]]) -> skdr_eval.EvaluationArtifact:
    """Adapt generic records to logs and evaluate a candidate policy."""
    adapted = skdr_eval.adapters.from_records(records)
    print(
        f"\nlogged propensities present: {adapted.had_logged_propensities} "
        "(reported, not consumed — see #167)"
    )
    models = {"hgb": HistGradientBoostingRegressor(random_state=0)}
    return skdr_eval.evaluate_sklearn_models(
        logs=adapted.logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
        y_col="reward",
    )


def run_synthetic() -> skdr_eval.EvaluationArtifact:
    """Evaluate a candidate policy on offline OBP-shaped synthetic feedback."""
    feedback = make_obp_bandit_feedback()
    return evaluate_records(bandit_feedback_to_records(feedback))


def run_obd_slice(max_rows: int = 4000) -> skdr_eval.EvaluationArtifact:
    """Load a slice of the real Open Bandit Dataset and evaluate a candidate."""
    bundle = skdr_eval.load_obd(max_rows=max_rows)
    models = {"hgb": HistGradientBoostingRegressor(random_state=0)}
    return skdr_eval.evaluate_sklearn_models(
        logs=bundle.logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
    )


def main() -> None:
    """Run the interop demo and print the headline report + trust verdict."""
    parser = argparse.ArgumentParser(description="OBP -> skdr-eval interop demo")
    parser.add_argument(
        "--obd",
        action="store_true",
        help="Use the real Open Bandit Dataset via load_obd (downloads a slice).",
    )
    args = parser.parse_args()

    print_field_map()
    artifact = run_obd_slice() if args.obd else run_synthetic()

    print("\nHeadline report (skdr-eval adds trust diagnostics on top of DR/SNDR):")
    cols = [
        c
        for c in ("model", "estimator", "V_hat", "SE_if", "ESS")
        if c in artifact.report.columns
    ]
    print(artifact.report[cols].to_string(index=False))
    print("\nsupport_health per row:")
    print(
        artifact.warnings[["model", "estimator", "support_health"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
