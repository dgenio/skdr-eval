# Known-failure-mode examples

The scripts in this directory are *intentionally bad* offline evaluations.
They exist so newcomers can see what an offline evaluation looks like
when one of the standing assumptions (`docs/concepts/estimands-and-assumptions.md`)
is violated, and so they can recognize the same warnings in their own
runs.

| Script | Failure regime | What the diagnostics should say |
|---|---|---|
| [`poor_overlap.py`](poor_overlap.py) | Logging policy is near-argmax; the target policy disagrees on most rows | `support_health = high_risk`, `HIGH_PARETO_K`, `POOR_OVERLAP` / `EXTREME_CLIP`, large `pareto_k` |
| [`misspecified_q.py`](misspecified_q.py) | Outcome model is severely under-fit (constant) | DR still recovers V* (the IPS leg carries the load); DM is biased |
| [`non_stationary.py`](non_stationary.py) | Reward distribution drifts between fold 1 and fold N | Moving-block bootstrap absorbs short-range dependence; long-range drift still under-covers — confirms the assumption boundary |

These scripts are deliberately kept **outside** `examples/use_cases/`
because the use-case gallery is the *happy-path* gallery (#78). Mixing
intentional failures into the happy path would defeat the gallery's
purpose. The CI smoke job that runs `examples/use_cases/` does NOT run
these scripts; they are run by `make known-failures` (see Makefile).

Cross-references:

- `docs/statistical-validation-matrix.md` indexes each failure mode to a
  simulation under `tests/sim_studies/`.
- `docs/concepts/estimands-and-assumptions.md` is the prose source of
  the assumption list each script is designed to violate.
