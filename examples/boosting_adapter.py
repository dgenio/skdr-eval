#!/usr/bin/env python3
"""Non-sklearn outcome models via the boosting adapters (#71).

``evaluate_sklearn_models`` accepts any object with the sklearn ``fit`` /
``predict`` surface — the name is historical, not a restriction. This example
shows two ways to bring a non-sklearn model:

1. :class:`skdr_eval.adapters.LGBMRegressorAdapter` — a thin wrapper that
   forwards LightGBM's native fit kwargs.
2. :class:`skdr_eval.adapters.CallableModelAdapter` — wrap a bare ``predict``
   function when you "just have a model".

Run: ``python examples/boosting_adapter.py``
(install the backend first: ``pip install 'skdr-eval[boosting]'``).
"""

from __future__ import annotations

import importlib.util

from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.adapters import CallableModelAdapter, LGBMRegressorAdapter


def main() -> None:
    print("skdr-eval: non-sklearn model adapters (#71)")
    print("=" * 60)

    logs, _, _ = skdr_eval.make_synth_logs(n=1500, n_ops=4, seed=0)

    # 1. CallableModelAdapter — always available (no extra dependency). Wrap an
    #    sklearn model behind bare fit/predict callables; the evaluator drives
    #    fit on its internal policy-feature matrix.
    inner = HistGradientBoostingRegressor(max_iter=50, random_state=0)
    callable_model = CallableModelAdapter(predict_fn=inner.predict, fit_fn=inner.fit)

    # 2. LightGBM adapter — only when the [boosting] extra is installed.
    models: dict[str, object] = {"callable_hgb": callable_model}
    if importlib.util.find_spec("lightgbm") is not None:
        models["lgbm"] = LGBMRegressorAdapter(
            n_estimators=100, num_leaves=31, random_state=0, verbose=-1
        )
    else:
        print("(lightgbm not installed — skipping the LGBM adapter row)")

    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
    )

    print("\nHeadline V_hat by model / estimator:")
    print(
        artifact.report[["model", "estimator", "V_hat", "ESS"]].to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
