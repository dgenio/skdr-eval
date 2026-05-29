"""Quickstart: slate / top-K off-policy estimators (#75, #135).

Demonstrates the four ranking-OPE estimators on a synthetic cascade-click
dataset and the top-level :func:`evaluate_slate_models` entry point (#135),
which bundles them into an ``EvaluationArtifact`` with support-health warnings,
a sensitivity summary, and a renderable card — the same surface you get from
:func:`evaluate_sklearn_models`.

* :func:`slate_standard_ips`
* :func:`reward_interaction_ips`
* :func:`pseudo_inverse_ips`
* :func:`slate_cascade_dr`

Run with: ``python examples/quickstart_slate.py``
"""

from __future__ import annotations

import numpy as np

import skdr_eval


def main() -> None:
    logs, attractiveness, truth = skdr_eval.make_slate_synth(
        n_impressions=500,
        n_items=10,
        slate_size=3,
        click_model="cascade",
        seed=42,
    )
    n_items = attractiveness.shape[1]
    slate_size = len(logs["slate"].iloc[0])

    def uniform_target_per_rank(rank: int, item: int) -> float:  # noqa: ARG001
        return 1.0 / n_items

    def slate_target_policy(slate: list[int]) -> float:  # noqa: ARG001
        return float(logs["logging_prob"].iloc[0])

    # Per-rank empirical click rate as a quick q̂.
    q_per_rank = np.array(
        [np.mean([row[k] for row in logs["clicks"]]) for k in range(slate_size)]
    )
    q_hat = np.tile(q_per_rank[None, :, None], (len(logs), 1, n_items))

    res_ips = skdr_eval.slate_standard_ips(logs, target_policy=slate_target_policy)
    res_rips = skdr_eval.reward_interaction_ips(
        logs, target_policy_per_rank=uniform_target_per_rank
    )
    res_pi = skdr_eval.pseudo_inverse_ips(
        logs, target_policy_per_rank=uniform_target_per_rank, n_items=n_items
    )
    res_dr = skdr_eval.slate_cascade_dr(
        logs,
        target_policy_per_rank=uniform_target_per_rank,
        q_hat_per_rank=q_hat,
    )

    print(f"Ground truth (logging) V*: {truth.V_logging:.4f}")
    for r in (res_ips, res_rips, res_pi, res_dr):
        print(f"  {r.name:18s}: V_hat={r.V_hat:.4f} ± {r.SE:.4f} (ESS={r.ESS:.1f})")

    # Top-level entry point (#135): one call, full EvaluationArtifact surface.
    artifact = skdr_eval.evaluate_slate_models(
        logs,
        models={"uniform_rerank": uniform_target_per_rank},
        estimators=("RIPS", "PI-IPS", "SlateCascadeDR"),
        baseline="logging",
    )
    print("\nevaluate_slate_models report:")
    print(
        artifact.report[
            ["model", "estimator", "V_hat", "SE_if", "ESS", "support_health"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
