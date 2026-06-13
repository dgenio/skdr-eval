# What skdr-eval claims (and what it does not)

A receipts-first page. Every claim below links to something you can run — a
command, an example script, a test, or a notebook — so you can check it
yourself rather than take our word for it. The companion non-claims section is
just as important: it states, explicitly, what offline evaluation cannot do.

## Estimator scope

`skdr-eval` evaluates **one-shot contextual-bandit policies** from logged
decisions. It ships a doubly-robust estimator family and a slate family.

| Claim | Estimators | Receipt |
|---|---|---|
| DR and Stabilized DR (SNDR) on sklearn-compatible policies | `DR`, `SNDR` | `examples/quickstart.py`; `tests/test_dr_sndr_smoke.py` |
| MRDR / SWITCH-DR / DRos / MIPS via a composable strategy seam | `MRDR`, `SWITCH-DR`, `DRos`, `MIPS` | `build_strategy(...)`; `tests/test_estimators_extended.py`, `tests/test_mips_api.py` |
| Slate / top-K off-policy evaluation | `slate_standard_ips`, `pseudo_inverse_ips`, `reward_interaction_ips`, `slate_cascade_dr` | `tests/test_slate_coverage.py` |
| Pairwise / autoscaling evaluation | `evaluate_pairwise_models` | `examples/use_cases/04_call_routing.py` |
| Estimators recover a known ground-truth value on synthetic DGPs | all | `tests/test_estimator_recovery_simulation.py`, `tests/sim_studies/` |

See [Choosing an estimator](docs/choosing-an-estimator.md) for which one to
run and [methods](docs/methods.md) for the math.

## Supported logged-data assumptions

The estimates are valid under the standard OPE assumptions; they are stated,
not hidden:

- **Unconfoundedness** — the logged action's propensity is explainable from
  the recorded context.
- **Overlap** — the candidate policy only takes actions the logging policy
  also took with positive probability. When this fails, `support_health`
  reports `high_risk`.
- **Stable data-generating process** across the log window (time-aware splits
  and moving-block bootstrap respect the temporal correlation).
- **Useful nuisance models** — the propensity and outcome models are not
  arbitrarily misspecified.

Receipt: the assumption tags travel with every artifact
(`DEFAULT_ASSUMPTION_TAGS`) and are printed on the card; see
[estimands & assumptions](docs/concepts/estimands-and-assumptions.md).

## Trust diagnostics

The differentiator is that `skdr-eval` tells you when **not** to trust the
estimate. Each diagnostic has a receipt.

| Diagnostic | What it catches | Receipt |
|---|---|---|
| `support_health` (`ok` / `caution` / `high_risk`) | poor overlap, low ESS, low match-rate | `docs/recipes/good-vs-bad-support.md`; `tests/test_diagnostics_trust.py` |
| PSIS Pareto-k | heavy-tailed importance weights (variance may not exist) | `tests/test_diagnostics_trust.py` |
| Effective sample size (ESS) and tail mass | a few rows dominating the estimate | `report` columns `ESS`, `tail_mass` |
| Propensity calibration (ECE / Brier) | miscalibrated logging-policy model | `evaluate_propensity_diagnostics`; `tests/test_propensity_diagnostics.py` |
| Clip-grid sensitivity | estimate that moves with the clip threshold | `summarize_sensitivity`; `tests/test_reporting_artifact.py` |
| Deploy / don't-deploy verdict | turning all of the above into a decision | `EvaluationArtifact.recommendation(...)`; `tests/test_card_schema.py` |

The CLI turns the verdict into a CI gate: `skdr-eval evaluate` exits with code
`3` when any model's verdict is `do_not_deploy`.

## Reproducible examples

| Want to see... | Run |
|---|---|
| The 10-minute quickstart | `python examples/quickstart.py` |
| Preflight on your own logs | `skdr-eval doctor your_logs.parquet --json` |
| A good vs. bad support contrast | `python examples/use_cases/06_agent_routing_policy.py` |
| Coming from Open Bandit Pipeline | `python examples/obp_interop.py` |
| What a bad evaluation looks like | `make known-failures` |
| Coverage of the bootstrap CI | `make coverage-sim` |

## Non-claims (read these)

- **Offline evaluation does not replace an online experiment.** A green
  `skdr-eval` verdict is evidence that a candidate is *worth* an A/B test, not
  proof it will win one. See the [rollout checklist](docs/rollout-checklist.md).
- **No method can fix no-overlap logs.** If the candidate only takes actions
  the logging policy never explored, there is no counterfactual signal in the
  data; `skdr-eval` returns `support_health = high_risk` rather than a
  confident number.
- **Not for sequential / reinforcement-learning problems.** State transitions
  and long horizons are out of scope — use SCOPE-RL or d3rlpy
  (see [comparisons](docs/comparisons.md)).
- **Logged propensities are reported, not consumed (yet).** `skdr-eval`
  estimates calibrated propensities internally; first-class use of a logged
  `pscore` is tracked in issue #167.
- **Diagnostics are signals, not proofs.** They help you decide whether an
  estimate is worth acting on; they do not certify it is correct.
