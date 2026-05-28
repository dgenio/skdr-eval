# How skdr-eval compares: OBP, SCOPE-RL, d3rlpy, banditml

If you are choosing an offline-evaluation toolkit, the honest question is *"why
this one instead of the established options?"*. This page answers that without
claiming `skdr-eval` is globally "better" — it is a practitioner-first tool for
a specific job, and there are jobs the other libraries do better.

For the deeper methodological positioning (estimators, calibrated propensities,
time-aware splits), see [`methods.md`](methods.md).

## What skdr-eval is for

`skdr-eval` is built for a **practitioner data-science** workflow: you have
logged contextual decisions, a candidate policy that is (or can be wrapped as)
a scikit-learn-compatible model, time-correlated logs, and you need a
decision artifact a non-researcher stakeholder can read — *before* you commit
to an A/B test.

## Comparison at a glance

| Dimension | skdr-eval | Open Bandit Pipeline (OBP) | SCOPE-RL | d3rlpy | banditml |
|---|---|---|---|---|---|
| Primary audience | practitioner DS teams | OPE researchers / bandit benchmarks | offline RL researchers & practitioners | offline / deep RL | bandit deployment |
| Decision setting | contextual bandit / logged decisions | contextual bandit / bandit datasets | sequential RL / OPE / OPL | offline RL | contextual bandits |
| Model interface | sklearn-compatible (`fit`/`predict`) | OBP abstractions | RL abstractions | deep-RL models | bandit models |
| Time-aware logs | first-class (time-series splits, MBB CIs) | not a focus | depends on setup | not a focus | not a focus |
| Trust diagnostics | support-health, PSIS Pareto-k, calibration, sensitivity | estimator-focused | broader RL metrics | RL evaluation | deployment-focused |
| Stakeholder artifact | HTML report + machine-readable card | not core focus | not core focus | not core focus | not core focus |
| Best used when | pre-A/B evaluation of sklearn-like policies from logs | benchmark/research OPE, broad estimator coverage | sequential / RL problems | offline-RL training & eval | running an online bandit |

## How to choose

- **Use OBP** when you need established public bandit **benchmarks** and the
  broadest catalogue of OPE estimators for research comparison.
- **Use SCOPE-RL or d3rlpy** when your problem is **sequential / offline RL**
  (state transitions, long horizons), not a one-shot contextual decision.
- **Use banditml** when you are running an **online** bandit system end to end.
- **Use skdr-eval** when you have **logged contextual decisions**, sklearn-like
  candidate models, **temporal** structure in the logs, and you need a
  **decision artifact** (support-health + calibration + sensitivity + an
  HTML/card export) that a PM or experiment reviewer can actually read.

## When *not* to use skdr-eval

- Your problem is sequential decision-making / reinforcement learning with
  state transitions — reach for SCOPE-RL or d3rlpy.
- You need a wide bank of research estimators or to reproduce published bandit
  benchmarks — OBP is the reference implementation.
- Your candidate policy cannot be expressed behind a `fit`/`predict`
  (or `predict_proba`) interface.
- Your logs have **no overlap** with what the candidate policy would do — no
  OPE library can rescue that, and skdr-eval will tell you so via
  `support_health = high_risk` rather than returning a confident number.

## What this page does not claim

`skdr-eval` does not provide stronger statistical guarantees than DR/SNDR allow
under standard OPE assumptions (unconfoundedness, overlap, a stable
data-generating process, and useful nuisance models). Its diagnostics are
**trust signals**, not proof of correctness, and offline evaluation does not
replace online validation. See
[estimand and assumptions](concepts/estimands-and-assumptions.md) for the
formal contract.
