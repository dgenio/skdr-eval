# What skdr-eval claims (and what it does not)

This page is the public claim boundary for `skdr-eval`. It is deliberately
conservative: implementation validity, evidence quality, and an organization's
decision to run an online experiment are separate concerns.

> **Current status:** the first DR/SNDR implementation path is
> `validation_pending`. The July/August 2026 audit identified correctness and
> inference work that must land before a strong validated-path claim is made.
> Existing experimental APIs remain available, but breadth is not evidence of
> validation.

## The claim we are working toward

The initial validated-v1 envelope is intentionally narrow:

> `skdr-eval` estimates the value of an explicit, finite-discrete, one-step
> target policy from logged contextual decisions with recorded behavior
> propensities, and reports whether the available evidence supports that
> estimate under a documented validation envelope.

The validated-v1 work is tracked by #297 and its blockers.

## What validated-v1 is intended to require

- **One-shot contextual decisions.** Sequential / long-horizon RL is out of
  scope.
- **Finite, explicit actions.** The action vocabulary and per-row eligibility
  must be represented explicitly when they vary.
- **An explicit target policy.** The evaluated policy must provide an action
  probability distribution; it is not inferred implicitly from arbitrary
  outcome-model scores in the validated path (#279).
- **Recorded behavior propensities.** The chosen-action probability logged at
  decision time is required for the first validated path (#167). Estimated
  propensities remain experimental until separately validated.
- **A genuine evaluation population.** Every row contributing to a validated
  estimate must have valid, non-leaky OOF nuisance predictions; statistical
  defaults must not be invented to keep an evaluation running (#280).
- **Correct multi-action DR semantics.** `q_hat(x,a)`, `q_obs`, and `q_pi` must
  have explicit, action-specific meaning (#58).
- **Only validated inference regimes.** Temporal leakage controls do not imply
  that arbitrary adaptive logging or dependence has been solved. Confidence
  intervals and dependence claims are limited to regimes validated by #62.

## Implementation maturity is not evidence quality

An estimator implementation can be validated while a particular evaluation is
unsupported.

For example, a correct DR implementation can still receive weak or invalid
inputs: poor overlap, invalid propensities, a target policy outside the logging
support, unsupported dependence, missing diagnostics, or an ambiguous action
mapping.

`skdr-eval` therefore separates:

1. **implementation maturity** — whether an estimator/configuration has passed
   the project's validation requirements; and
2. **evidence status** — whether the concrete evaluation has enough support to
   sustain the reported estimate under the declared validation envelope.

This separation is tracked by #282 and #245.

## Evidence status, not deployment authorization

The core library should describe evidence, not authorize deployment or an online
experiment.

The target semantics are evidence-oriented states such as:

- `estimate_supported`
- `inconclusive`
- `insufficient_evidence`
- `unsupported`
- `invalid_evaluation`

Exact names are being finalized under #245.

**Migration note:** released versions still expose legacy deployment/experiment
verdict names such as `deploy`, `ab_test`, and `do_not_deploy`, and some CLI/docs
still describe exit codes in those terms. Those are compatibility-era semantics,
not the target claim of the core library. Issue #245 owns their migration to the
evidence-only state machine; until that work lands, callers must not interpret a
legacy positive verdict as deployment or experiment authorization.

A positive evidence state means that the statistical/evidence contract passed
under the declared validation envelope. It does **not** mean that deployment or
an online experiment is safe, ethical, approved, reversible, sufficiently
monitored, or commercially justified. Those decisions require information the
core library does not possess.

## Validation evidence required before strong claims

No estimator/configuration becomes strongly validated from internal unit tests
alone. The validation program requires multiple independent forms of evidence:

1. formula and invariant tests;
2. analytic known-ground-truth data-generating processes;
3. randomized simulation stress tests;
4. estimator-specific external/reference agreement where semantics genuinely
   match (#281);
5. empirical bias, RMSE, interval coverage, failure/abstention and
   **false-reassurance** validation (#62);
6. appropriate public logged-policy data where useful;
7. retrospective offline-vs-online comparisons when available;
8. independent human methodological review before an
   `independently_validated` claim (#298).

The public validation-lab work is tracked by #223.

## Current non-claims

- **No deployment recommendation.** Core statistical output is not a deployment
  or experiment-approval decision. Legacy verdict labels remain temporarily for
  compatibility while #245 migrates the public state machine.
- **No guarantee from diagnostics.** ESS, overlap, Pareto-k, calibration and
  sensitivity are evidence signals. Their thresholds must be empirically
  validated; they are not proofs of correctness.
- **No rescue for no-overlap logs.** If the target policy takes actions outside
  logging support, counterfactual evidence is absent.
- **No general sequential / RL claim.** State transitions and long horizons are
  out of scope.
- **No general adaptive-logging claim.** Time-aware splits and block bootstrap
  do not by themselves validate inference under arbitrary adaptive behavior
  policies.
- **No validated estimated-propensity path yet.** The first validated envelope
  requires logged propensities.
- **No production healthcare-treatment claim.** Clinical/treatment examples, if
  retained, are educational/experimental and must not be presented as a
  validated production workflow.
- **No validated general LLM/agent-routing claim yet.** Variable action sets,
  changing model catalogues, multi-objective rewards and missing exploration can
  violate the initial envelope. Agent/model-routing examples remain experimental
  unless they satisfy the same validated contract.
- **Offline evaluation does not replace online validation.** Even supported
  offline evidence is not proof that a policy will win online.

## Known pre-validation correctness work

The audit identified material issues that affect historical result semantics,
including the multi-action DR/SNDR outcome representation (#58) and rows without
genuine OOF nuisance predictions (#280).

When corrected releases are available, #300 requires explicit historical
advisories describing affected configurations and which results should be rerun.
The project will not minimize an estimand-affecting issue as a mere precision or
bootstrap change.

## Product principle

The intended differentiator is not estimator count. It is making offline policy
evidence difficult to misuse:

> **Know when an offline policy estimate deserves to be believed — and get an
> explicit refusal when the logs cannot support the question.**

Until the validation gates and external review are complete, claims in README,
PyPI metadata, examples and release notes should not exceed this page.
