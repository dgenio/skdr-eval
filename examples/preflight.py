#!/usr/bin/env python3
"""Preflight smoke for skdr-eval: capability detection + schema validation.

Run this before kicking off a longer evaluation to confirm:
- the package imports cleanly,
- optional extras (``[choice]``, ``[viz]``, ``[speed]``) are available
  if your pipeline needs them,
- your logs DataFrame matches the schema consumed by
  :func:`skdr_eval.evaluate_sklearn_models`,
- your pairwise inputs match
  :func:`skdr_eval.evaluate_pairwise_models`.

Exits non-zero on any validation failure so it can guard CI.
"""

from __future__ import annotations

import sys

import skdr_eval


def main() -> int:
    print("skdr-eval Preflight")
    print("=" * 50)
    print(f"Version: {skdr_eval.__version__}")

    print("\nCapabilities")
    print("-" * 50)
    caps = skdr_eval.get_capabilities()
    for key in ("viz", "speed"):
        status = "yes" if caps[key] else "no"
        print(f"  {key:8s} {status}")
    if caps["missing_extras"]:
        print(
            f"  -> install with: pip install 'skdr-eval[{','.join(caps['missing_extras'])}]'"
        )
    else:
        print("  all optional extras installed")

    print("\nLogs schema (single-action)")
    print("-" * 50)
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=500, n_ops=3, seed=0)
    print(f"  rows={len(logs)} ops={list(ops_all)}")
    skdr_eval.validate_logs(logs, strict=True)
    print("  validate_logs(strict=True): OK")

    print("\nPairwise inputs schema")
    print("-" * 50)
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=3, n_clients_day=80, n_ops=4, seed=0
    )
    print(f"  decisions={len(logs_df)} operator-days={len(op_daily_df)}")
    skdr_eval.validate_pairwise_inputs(
        logs_df, op_daily_df, metric_col="service_time", strict=True
    )
    print("  validate_pairwise_inputs(strict=True): OK")

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
