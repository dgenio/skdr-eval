"""Quickstart: MIPS estimator with embedding-sufficiency diagnostic (#85, #136).

Demonstrates the workflow for using ``MIPS`` in skdr-eval:

1. Build or load a per-action embedding matrix ``(n_actions, embed_dim)`` — or
   pass a **logs column name** holding a per-row embedding (#136).
2. Pass it via ``action_embedding=`` to ``evaluate_sklearn_models``, optionally
   choosing the kernel (``mips_kernel="rbf"|"linear"|callable``) and the
   bandwidth (``mips_bandwidth=0.5`` or ``"median"`` for the median heuristic).
3. Inspect :class:`EmbeddingSufficiencyReport` to confirm the embedding
   captures enough of the action-driven reward signal for MIPS to be
   approximately unbiased.

Note: if ``"MIPS"`` is requested without ``action_embedding=``, MIPS now
gracefully falls back to SNDR with a warning (#136) rather than failing.

Run with: ``python examples/quickstart_mips.py``
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def main() -> None:
    rng = np.random.default_rng(11)
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=1500, n_ops=5, seed=11)

    # Build a simple 2-D embedding per operator. In practice this comes
    # from upstream representation learning (skill / capacity vectors,
    # learned action embeddings, etc.).
    n_actions = len(ops_all)
    embedding = rng.normal(scale=1.0, size=(n_actions, 2))

    models = {"hgb": HistGradientBoostingRegressor(random_state=11)}
    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        policy_train="pre_split",
        n_splits=3,
        random_state=11,
        estimators=("DR", "SNDR", "MIPS"),
        action_embedding=embedding,
        mips_bandwidth="median",  # median-heuristic RBF bandwidth (#136)
        mips_kernel="rbf",
    )
    print("Estimator report (with MIPS):")
    print(
        artifact.report[["model", "estimator", "V_hat", "SE_if", "ESS"]].to_string(
            index=False
        )
    )

    # Embedding sufficiency diagnostic — flags whether the embedding
    # captures enough of the action-driven reward signal.
    design = skdr_eval.build_design(logs)
    Y = design.Y
    A = design.A
    # Use the marginal-mean q̂ as a quick stand-in; in real workflows pull
    # the cross-fitted q_hat from the artifact (artifact.detailed["hgb"]).
    q_hat = np.full_like(Y, Y.mean())
    report = skdr_eval.embedding_sufficiency_diagnostic(
        Y=Y, q_hat=q_hat, A=A, action_embedding=embedding
    )
    print("\nEmbedding sufficiency diagnostic:")
    print(f"  R^2 gap (action - embedding): {report.r2_action:.4f}")
    print(f"  notes: {report.notes}")


if __name__ == "__main__":
    main()
