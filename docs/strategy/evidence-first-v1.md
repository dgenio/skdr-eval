# Evidence-first v1 strategy

This document records the product and validation strategy adopted after the August 2026 red-team review. It is intentionally falsifiable. If the validation or market evidence fails, the project should narrow, pivot, or stop rather than weaken the claim.

## Product thesis

`skdr-eval` should become the **evidence-quality layer for offline evaluation of discrete, one-step decision policies**.

Its job is not to be the library with the largest estimator catalogue. Its job is to make it difficult to mistake a numerically produced estimate for usable evidence.

The working promise is:

> **Know when an offline policy estimate deserves to be believed.**

A first-class outcome is an explicit refusal when the logs cannot support the question.

## Validated-v1 envelope

The first strong claim is deliberately narrow:

- one-shot/contextual decisions;
- finite discrete action set;
- explicit action mapping and eligibility;
- explicit target-policy action distribution;
- scalar reward/outcome;
- behavior propensity logged at decision time;
- genuine OOF evaluation population;
- corrected multi-action DR semantics;
- inference only for dependence regimes explicitly validated by the project.

Initial implementation candidate: **DR**. SNDR follows only after its estimator-specific validation and reference strategy are defensible. Other estimators and decision domains remain experimental until separately promoted.

## Three separate decisions

Never collapse these into one field:

### 1. Implementation maturity

Does the estimator/configuration have enough validation evidence?

Examples: `experimental`, `validation_pending`, `reference_validated`, `independently_validated`.

### 2. Evidence status

Does this concrete run have enough support under the estimator's validation envelope?

Examples: `estimate_supported`, `inconclusive`, `insufficient_evidence`, `unsupported`, `invalid_evaluation`.

### 3. Organizational action

Should a team run an online experiment or deploy?

This requires risk, harm, reversibility, monitoring, traffic, regulatory and business inputs that the statistical core does not know. Any automatic organizational gate must be a separate policy layer over the evidence artifact.

## Validation ladder

A strong validation claim requires multiple kinds of evidence:

1. formula/unit invariants;
2. analytic known-ground-truth DGPs;
3. randomized simulation parameter sweeps;
4. estimator-specific cross-implementation/reference agreement where semantics genuinely match;
5. bias/RMSE/coverage/interval-width/failure validation;
6. false-reassurance and false-abstention validation of the evidence-health layer;
7. appropriate public logged-policy datasets;
8. retrospective offline-vs-online comparisons when available;
9. independent human methodological review.

Reference agreement is never treated as mathematical ground truth.

## Diagnostic validation

The evidence layer itself must be tested.

Across broad known-ground-truth simulations, measure whether diagnostics such as overlap, ESS, Pareto-k, propensity quality and sensitivity actually discriminate regimes with large estimation error.

Required metrics include:

- bias;
- RMSE;
- empirical interval coverage;
- interval width;
- abstention/failure rate;
- evidence-status distribution;
- false reassurance: materially wrong estimate classified as supported;
- false abstention / unnecessary conservatism where meaningful;
- diagnostic-to-error association/discrimination.

Thresholds that do not perform adequately must be revised rather than defended by convention.

## Public validation lab

Publish reproducible validation evidence, not a leaderboard.

The validation site/report should show successful, failed and unsupported scenarios from the same machine-readable configuration. Releases making validation claims must identify the exact package commit/version and reference dependencies used to generate the evidence.

## External review gate

Before a 1.0 release or `independently_validated` status, obtain independent human review covering at least:

- OPE/contextual-bandit estimator and estimand semantics;
- causal/statistical inference and diagnostics;
- production logging and experiment-review workflow.

Publish material objections, maintainer responses and unresolved limitations. This is methodological review, not testimonials.

## Design-partner validation

Technical correctness cannot prove product-market fit.

Interview approximately 15–20 qualified practitioners, then attempt hands-on use with at least five data-ready users/teams where possible. Measure:

- whether behavior propensities/action provenance are actually available;
- time to valid preflight;
- time to first interpretable artifact;
- maintainer assistance required;
- whether the artifact changes or enters a real experiment-review decision;
- repeat use;
- recurring adoption blockers.

Do not implement one-off requests unless the same problem recurs independently.

## Product pivots are allowed

Three outcomes remain legitimate:

### Native product succeeds
Users value both the native estimator path and the evidence workflow.

### Evidence layer wins
Users prefer established estimator backends but value skdr-eval's evidence/validation/artifact surface. Make external backends first-class rather than defending native code for its own sake.

### OPE-readiness wins
Qualified teams commonly lack valid behavior propensities/action provenance. Pivot toward instrumentation/readiness diagnostics rather than pretending the addressable OPE workflow is larger than it is.

## Kill criteria

- If DR cannot meet predeclared correctness/inferential/evidence-health thresholds, it does not receive validated status.
- If evidence-health diagnostics frequently classify materially wrong estimates as supported, redesign the evidence layer.
- If external reviewers identify unresolved fundamental estimand/inference problems, block 1.0 and narrow/redesign.
- If fewer than roughly 3 of 15 qualified target teams have the required data contract, reassess the market and consider OPE-readiness tooling.
- If data-ready users consistently prefer direct established tools and do not value the evidence artifact, pivot or stop building product/reporting breadth.
- If first use consistently requires maintainer intervention, simplify before distribution.
- If agent/model-routing traces routinely violate the validated envelope, keep that domain experimental.

## Execution allocation until validation succeeds

Approximate effort allocation:

- **70%** correctness, inferential validation, evidence-health calibration and scope reduction;
- **20%** external methodological/design-partner validation;
- **10%** everything else.

Net-new estimator/domain breadth is frozen unless it directly unblocks validated-v1 or is pulled by repeated external demand.

## Deferred by default

Until validated-v1 and design-partner evidence succeed, treat the following as deferred unless directly required by external validation:

- plugin ecosystems;
- interactive decision workspaces;
- LLM-generated summaries;
- shareable badges;
- rolling monitoring;
- A/B power planning;
- multi-objective expansion;
- broad agent-specific trace tooling;
- broad performance work beyond demonstrated validation/adoption bottlenecks;
- new estimator families.

Preserve good ideas, but do not present them as current commitments.

## Naming decision before 1.0

The project must explicitly decide whether `skdr-eval` still fits a policy-first, implementation-agnostic direction. Renaming is not required now, but it must not become an accidental post-1.0 constraint.

## Related work

- #297 — validated-v1 envelope
- #298 — independent methodological review
- #299 — design-partner / market validation
- #300 — historical correctness advisories
- #301 — backlog/adoption reset
- #279 — explicit Policy contract
- #280 — genuine OOF population
- #167 — logged propensities
- #58 — corrected multi-action DR/SNDR estimand
- #281 — estimator-specific reference agreement
- #62 — end-to-end inference and false reassurance
- #245 — evidence-state semantics
- #259 — fail-closed diagnostic states
- #282 — implementation maturity vs evidence quality
- #223 — public validation lab
