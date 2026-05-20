"""Quickstart: composable estimator strategies (#86).

Runs DR, SNDR, MRDR, SWITCH-DR, and DRos on the same synthetic logs to
show how the new ``estimators=`` kwarg surfaces alternative DR variants
without touching the core call site.

Run with: ``python examples/quickstart_estimators.py``
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def main() -> None:
    logs, _, _ = skdr_eval.make_synth_logs(n=1500, n_ops=4, seed=7)
    models = {"hgb": HistGradientBoostingRegressor(random_state=7)}

    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        policy_train="pre_split",
        n_splits=3,
        random_state=7,
        estimators=("DR", "SNDR", "MRDR", "SWITCH-DR", "DRos"),
        switch_tau=10.0,
        dros_lam=2.0,
    )
    report = artifact.report
    print("Estimator family report:")
    print(
        report[["model", "estimator", "V_hat", "SE_if", "ESS"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
