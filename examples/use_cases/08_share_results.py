#!/usr/bin/env python3
"""Share evaluation results: typed rows, Markdown export, and the JSON Schema.

Once you have an ``EvaluationArtifact``, the last mile is *reading* and
*communicating* it. This use case shows the additive, low-risk consumption
surface:

* ``rows()`` / ``row()`` — a typed view over the report (read by attribute,
  not by DataFrame column string) (#234).
* ``to_markdown()`` / ``export(formats=["markdown"])`` — a paste-ready summary
  of V̂, CI, and the trust diagnostics for a PR or ticket (#237).
* ``ArtifactSchema.json_schema()`` — the published JSON Schema contract that
  downstream tooling can validate against without importing the library (#205).

Deployment-verdict surfaces (comparison gates, decision summaries, badges) are
intentionally **not** shown here: they depend on the verdict contract that is
under the July 2026 experiment-eligibility audit and will land separately.

Runs in a few seconds, no network or credentials.
"""

from __future__ import annotations

import json

from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def main() -> None:
    logs, _, _ = skdr_eval.make_synth_logs(n=1500, n_ops=3, seed=0)
    models = {"candidate": HistGradientBoostingRegressor(max_iter=40, random_state=0)}
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        ci_bootstrap=True,
        policy_train="pre_split",
    )

    print("=" * 64)
    print("TYPED ROWS (#234) — read results by attribute, not column string")
    print("=" * 64)
    for row in artifact.rows():
        print(
            f"  {row.model}/{row.estimator}: "
            f"V_hat={row.V_hat:.4g} support={row.support_health} "
            f"ESS={row.ESS:.1f}"
        )

    print("\n" + "=" * 64)
    print("MARKDOWN SUMMARY (#237) — paste into a PR or ticket")
    print("=" * 64)
    print(artifact.to_markdown())

    print("\n" + "=" * 64)
    print("PUBLISHED JSON SCHEMA (#205) — validate artifacts without importing us")
    print("=" * 64)
    schema = skdr_eval.ArtifactSchema.json_schema()
    print(f"  schema title: {schema['title']}")
    print(f"  top-level properties: {sorted(schema['properties'])}")
    print(f"  (full schema is {len(json.dumps(schema))} bytes; see docs/schemas/)")


if __name__ == "__main__":
    main()
