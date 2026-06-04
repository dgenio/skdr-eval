#!/usr/bin/env python3
"""Load the Open Bandit Dataset and run an evaluation (#70).

``skdr_eval.datasets.load_obd`` returns the same ``(logs, ops_all,
ground_truth)`` shape as ``make_synth_logs``, so real benchmark data flows
through the standard evaluator unchanged. The reward column is ``"click"``.

This script is self-contained: it writes a tiny OBD-format sample to a temp
directory and loads it via ``base_url`` so it always runs in seconds with no
network. To load the real dataset instead, drop the ``base_url`` argument:

    logs, ops_all, _ = skdr_eval.datasets.load_obd(
        "random", "all", max_rows=5000
    )

which lazily downloads + caches the public sample under ``~/.skdr_eval``.
The full 26M-row dataset lives at https://research.zozo.com/data.html.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval


def _write_obd_sample(root: Path, n: int = 600, n_items: int = 6) -> str:
    """Write a minimal OBD-format ``random/all.csv`` + ``item_context.csv``."""
    rng = np.random.RandomState(0)
    src = root / "random"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="min").astype(str),
            "item_id": rng.randint(0, n_items, size=n),
            "position": rng.randint(1, 4, size=n),
            "click": rng.randint(0, 2, size=n),
            "propensity_score": rng.uniform(0.1, 0.9, size=n),
            "user_feature_0": rng.randint(0, 4, size=n),
            "user_feature_1": rng.randint(0, 4, size=n),
        }
    ).to_csv(src / "all.csv", index=False)
    pd.DataFrame({"item_id": range(n_items)}).to_csv(
        src / "item_context.csv", index=False
    )
    return str(root)


def main() -> None:
    print("skdr-eval: Open Bandit Dataset loader (#70)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        base_url = _write_obd_sample(Path(tmp) / "obd_src")
        logs, ops_all, ground_truth = skdr_eval.datasets.load_obd(
            "random", "all", cache_dir=Path(tmp) / "cache", base_url=base_url
        )

    print(f"   Decisions: {len(logs):,}")
    print(f"   Action catalog: {len(ops_all)} items")
    print(f"   Ground truth available: {ground_truth is not None}")
    skdr_eval.validate_logs(logs, y_col="click")
    print("   Schema validated (reward column = 'click').")

    artifact = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"hgb": HistGradientBoostingRegressor(max_iter=30, random_state=0)},
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
        y_col="click",
    )
    print("\nHeadline V_hat (estimated click rate under the candidate policy):")
    print(
        artifact.report[["model", "estimator", "V_hat", "ESS"]].to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
