# Validated v1 evidence envelope

> **Status:** proposed validation contract. Until the gates below pass, DR/SNDR remain validation-pending and experimental surfaces must not inherit validated claims.

`skdr-eval` is an offline policy-evaluation toolkit for discrete, one-step contextual decisions. Its strongest intended promise is not that every estimate is trustworthy; it is that the library makes the implementation, assumptions, evidence quality, and reasons to abstain explicit enough for a practitioner to review.

The first validated path is deliberately narrow. Breadth can be added later only through separate validation.

## The claim

For the validated-v1 path, `skdr-eval` aims to:

> estimate the value of an explicit, finite-discrete, one-step target policy from logged contextual decisions with recorded behavior propensities, and report whether the available evidence supports that estimate under a documented validation envelope.

This is an **evidence claim**, not an authorization to deploy or run an online experiment.

## Validated-v1 setting

### Decision setting

- one-shot / contextual decisions;
- finite discrete action set;
- explicit action vocabulary and ordering;
- explicit per-row eligibility where action availability varies;
- no state transitions or long-horizon/sequential-RL interpretation.

### Target policy

The target policy must be explicit. It returns a probability distribution over the eligible actions for every evaluated context.

Deterministic policies are represented as one-hot distributions. The validated path must not infer a deployment policy implicitly from arbitrary outcome-model scores.

Tracked in #279.

### Behavior policy

The first validated path requires the probability of the logged action to have been recorded at decision time.

- finite values in `(0, 1]`;
- explicit provenance;
- aligned with the same action vocabulary and eligibility represented by the row;
- no estimated propensity may silently replace a missing logged propensity in the validated path.

Estimated-propensity workflows may remain available, but they are validation-pending/experimental until separately validated.

Tracked in #167.

### Effective evaluation population

Every observation contributing to a validated estimate must have nuisance predictions that are valid under the declared cross-fitting contract.

Warm-up/training-only rows, excluded rows, and exclusion reasons must be explicit. Statistical defaults must never be invented merely to keep an evaluation running.

Tracked in #280.

### Estimand and outcome prediction

For multi-action DR-style evaluation:

- `q_obs[i] = q_hat(x_i, a_i)` for the logged action;
- `q_pi[i] = sum_a pi(a | x_i) q_hat(x_i, a)` for the target policy;
- action-specific outcome predictions and action mapping are explicit;
- silent `(n,) -> (n, n_actions)` broadcasting is not a supported validated shortcut.

Tracked in #58.

### Inference

Only inferential regimes that have passed end-to-end validation may be described as validated.

Temporal leakage controls, dependence-aware resampling, and support diagnostics solve different problems. The project must not use broad language such as “time-aware OPE” to imply that arbitrary adaptive logging or temporal dependence has been solved.

Tracked in #62.

## Two independent status axes

A validated implementation does **not** imply supported evidence for a particular dataset.

### Implementation maturity

Examples:

- `experimental`
- `validation_pending`
- `reference_validated`
- `independently_validated`
- `deprecated`

### Evidence state

A concrete evaluation independently reports whether its evidence is usable. Candidate vocabulary:

- `estimate_supported`
- `inconclusive`
- `insufficient_evidence`
- `unsupported`
- `invalid_evaluation`

A validated estimator on weak-overlap or otherwise unsupported data must still return a non-positive evidence state.

Tracked in #282 and #245.

## No deployment authorization in the core evidence state

The statistical core does not know enough to declare deployment or an online experiment safe. Those decisions can depend on harm, reversibility, regulatory constraints, monitoring, rollout size, business risk, and other organization-specific considerations.

Core evidence output therefore should not mean `deploy`, `do_not_deploy`, or `eligible_for_guarded_online_test`.

Organizations may layer a separate explicit decision policy over an `EvaluationArtifact`, but that policy is not the statistical evidence state.

Tracked in #245.

## Validation requirements

Validation is cumulative. No single line of evidence is sufficient.

1. formula and invariant tests;
2. analytic known-ground-truth DGPs;
3. randomized parameter-sweep simulation;
4. estimator-specific independent/reference agreement where semantic equivalence exists;
5. end-to-end bias/RMSE/coverage/interval-width/failure validation;
6. evidence-health calibration, including false-reassurance rate;
7. appropriate public logged-bandit data;
8. retrospective offline-vs-online comparisons when available;
9. independent human methodological review before `independently_validated` status.

Reference implementations are evidence, not mathematical oracles. An estimator must never be forced into comparison with a superficially similar but semantically different reference method.

Tracked in #281, #62, #223, and #298.

## Evidence-health validation

Diagnostics such as overlap, ESS, Pareto-k, propensity calibration, and sensitivity should not be trusted merely because they are intuitive.

Across the validation matrix the project must test whether these signals actually predict large estimation error and calibrate thresholds accordingly.

A key metric is **false reassurance rate**: among cases where the estimate is materially wrong, how often does the evidence layer nevertheless classify it as supported/healthy?

Tracked in #62 and #223.

## Explicitly outside the initial validated claim

Until separately validated, the following are experimental or unsupported for validated-v1 claims:

- sequential/offline RL;
- slate / top-k policy evaluation;
- continuous actions;
- estimated behavior propensities;
- arbitrary adaptive contextual-bandit dependence;
- multi-objective decision semantics;
- broad post-deploy monitoring semantics;
- agent/LLM routing with changing action catalogues unless the data exactly satisfy this validated contract;
- healthcare/clinical treatment recommendations as a production claim;
- automatic deployment or experiment authorization.

Examples may demonstrate experimental surfaces, but their status must be visible and they must not borrow validated-v1 language.

## Historical correctness

Material correctness problems found before the validated release must be documented explicitly, including which versions/configurations may need to be rerun.

Known tracked items include #58 and #280. The migration/advisory work is tracked in #300.

## Product/adoption gate

Statistical validation alone does not prove product-market fit. Before broad feature expansion, the project should validate with data-ready practitioners that:

- teams actually have the required behavior propensities/action provenance;
- first-use can be completed without maintainer intervention;
- the evidence artifact enters a real experiment-review decision;
- the workflow is sufficiently differentiated from direct use of established OPE tooling.

Tracked in #299.

## Freeze rule

Until the validated-v1 path and external review pass:

- do not add new estimator families to the validated roadmap;
- do not add new validated decision domains;
- defer speculative presentation/integration/product breadth unless directly required by validation or repeated design-partner evidence;
- prioritize correctness, reproducibility, explicit failure, external review, and real-user validation.

Tracked in #297 and #301.

## Project promise

A useful mental model for the project is:

> **Know when your offline policy estimate deserves to be believed.**

A high-quality result can therefore be an explicit abstention:

> **The logs cannot answer this question under the validated evidence contract.**
