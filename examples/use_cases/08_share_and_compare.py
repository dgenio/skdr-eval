#!/usr/bin/env python3
"""Share and compare evaluation results (#184 #234 #237 #238 #249 #251).

Once you have an ``EvaluationArtifact``, the last mile is *communicating* it and
*tracking it over time*. This use case shows the export / consumption surface:

* ``rows()`` — a typed view over the report (read by attribute, not by column
  string).
* ``to_markdown()`` — a paste-ready summary for a PR or ticket.
* ``decision_summary()`` — one honest, stakeholder-readable line.
* ``to_summary_facts()`` — a provider-agnostic payload for an LLM summary
  (see ``docs/recipes/llm-summary-prompt.md``); the library adds no LLM
  dependency.
* ``badge()`` — a dependency-free SVG badge whose colour tracks support-health.
* ``compare()`` — diff a new run against a previous one and detect verdict
  regressions (the CI-gate story: fail when the verdict got *worse*).

Runs in a few seconds, no network or credentials.
"""

from __future__ import annotations

import json

from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def _evaluate(seed: int) -> skdr_eval.EvaluationArtifact:
    logs, _, _ = skdr_eval.make_synth_logs(n=1500, n_ops=3, seed=seed)
    models = {
        "candidate": HistGradientBoostingRegressor(max_iter=40, random_state=seed)
    }
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=seed,
        ci_bootstrap=True,
        policy_train="pre_split",
    )


def main() -> None:
    artifact = _evaluate(seed=0)

    print("=" * 64)
    print("TYPED ROWS (#234) — read results by attribute, not column string")
    print("=" * 64)
    for row in artifact.rows():
        print(
            f"  {row.model}/{row.estimator}: "
            f"V_hat={row.V_hat:.4g} verdict={row.verdict} "
            f"support={row.support_health}"
        )

    print("\n" + "=" * 64)
    print("MARKDOWN SUMMARY (#237) — paste into a PR or ticket")
    print("=" * 64)
    print(artifact.to_markdown())

    print("\n" + "=" * 64)
    print("DECISION SUMMARY (#238) — one honest line for a stakeholder")
    print("=" * 64)
    summary = artifact.decision_summary("candidate", estimator="SNDR")
    print("  " + summary["summary"])

    print("\n" + "=" * 64)
    print("LLM-READY FACTS (#249) — feed to any model; no LLM dependency here")
    print("=" * 64)
    facts = artifact.to_summary_facts("candidate", estimator="SNDR")
    print(json.dumps(facts, indent=2, default=str))

    print("\n" + "=" * 64)
    print("BADGE (#251) — colour tracks support-health, never oversold")
    print("=" * 64)
    badge = artifact.badge("candidate", estimator="SNDR")
    print(f"  message={badge['message']!r} colour={badge['color']}")
    print(f"  markdown snippet: {badge['markdown']}")

    print("\n" + "=" * 64)
    print("COMPARE (#184) — diff a new run against a previous one")
    print("=" * 64)
    previous = _evaluate(seed=1)
    diff = artifact.compare(previous)
    print(diff.to_markdown())
    print(
        f"\n  verdict regressed vs previous run: {diff.verdict_regressed} "
        "(a CI gate would fail on True)"
    )


if __name__ == "__main__":
    main()
