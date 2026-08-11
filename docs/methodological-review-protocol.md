# Independent methodological review protocol

This document operationalizes issue #298. It is a review protocol, not a testimonial template.

The purpose of independent review is to challenge the **specific validation envelope** claimed by `skdr-eval`, identify unsupported assumptions or misleading semantics, and create a public record of what was reviewed and what remains unresolved.

## When to run the review

Do not ask reviewers to certify a moving target.

Prepare the review packet only after the validated-v1 contracts and corresponding evidence are coherent enough to inspect:

- `docs/validated-v1.md` / #297;
- explicit target-policy contract (#279);
- effective OOF evaluation population (#280);
- logged-propensity path (#167);
- corrected multi-action estimand (#58);
- estimator-specific reference agreement (#281);
- end-to-end validation report (#62/#223);
- evidence-state semantics (#245/#282);
- historical correctness advisory (#300).

The reviewed commit and validation-report version must be immutable and recorded.

## Reviewer independence

Aim for three distinct reviewer perspectives. One person may cover more than one perspective only when their expertise genuinely spans them, but at least two independent humans should review the package before an `independently_validated` claim is made.

### OPE / contextual-bandit methods reviewer

Challenge:

- exact estimand and formulas;
- behavior-policy and target-policy semantics;
- deterministic vs stochastic policy representation;
- action ordering and eligibility;
- logged propensity interpretation;
- DR/SNDR implementation details;
- reference-equivalence assumptions;
- overlap/positivity boundaries.

### Causal / statistical-ML reviewer

Challenge:

- nuisance fitting and OOF guarantees;
- evaluated population and exclusions;
- uncertainty target and CI semantics;
- temporal/dependence claims;
- diagnostic thresholds and false reassurance;
- distinction between empirical validation and untestable causal assumptions;
- unsupported observational/clinical claims.

### Production experimentation / recommender reviewer

Challenge:

- whether real logging systems can satisfy the data contract;
- propensity provenance and instrumentation failure modes;
- action-catalogue / eligibility drift;
- artifact usefulness in a real review process;
- ways a practitioner could misread `estimate_supported`;
- whether the core accidentally authorizes experiments or deployment.

## Review packet

Provide the reviewer with a single versioned packet containing:

1. exact package commit/tag;
2. validated-v1 scope;
3. public API examples that exercise only the claimed validated path;
4. estimand/formula documentation;
5. target-policy/action-distribution contract;
6. behavior-propensity contract;
7. effective evaluation-population semantics;
8. reference-agreement reports and reference versions;
9. known-ground-truth simulation configuration;
10. validation report with bias, RMSE, coverage, interval width, abstention/failure, and false reassurance;
11. evidence-state definitions and thresholds;
12. explicit non-claims and unsupported regimes;
13. historical correctness advisory;
14. instructions to reproduce the validation evidence.

Do not send only the README or a curated success notebook.

## Required reviewer questions

Each reviewer should answer, in their own words:

1. **What exact statistical question does the validated path answer?**
2. **Which assumptions are required but cannot be verified from the artifact?**
3. **Can the implementation accidentally evaluate a policy different from the one the user intended?**
4. **Can rows without valid nuisance information silently enter the estimate?**
5. **Are reference comparisons semantically equivalent, not merely numerically close?**
6. **What does the reported confidence interval cover, and what uncertainty is excluded?**
7. **Are any temporal/dependence claims broader than the validation evidence?**
8. **Can a validated implementation still produce unsupported evidence? Is that obvious to users?**
9. **How often can evidence diagnostics falsely reassure according to the validation study? Is that acceptable for the stated claim?**
10. **Which examples/docs are most likely to encourage misuse?**
11. **Which claim would you remove or narrow before 1.0?**
12. **What test or evidence would most increase your confidence?**

## Required public review record

With reviewer permission, publish:

- reviewer identity or role/background;
- date;
- exact commit/tag reviewed;
- validation-report identifier;
- material findings, including negative findings;
- severity / blocking status;
- maintainer response;
- code/docs changes triggered by the finding;
- unresolved limitations;
- reviewer conclusion on whether the **stated envelope** is defensible.

A reviewer conclusion is not a claim that every future evaluation is correct.

## Blocking findings

The following are release blockers until resolved or the validated envelope is narrowed to exclude them:

- wrong or ambiguous estimand;
- target-policy semantics that can differ from the evaluated policy without detection;
- evaluated observations lacking required non-leaky nuisance information;
- a claimed reference comparison that is not semantically equivalent;
- material undercoverage or false reassurance outside the predeclared validation tolerance;
- evidence states that can imply support when required diagnostics are unknown;
- unsupported temporal/dependence claims;
- documentation that presents a validation-pending/experimental path as validated.

## Review closure

A review is complete only when every material finding has one of:

- fixed and verified;
- explicitly accepted as a limitation **inside** the validated envelope;
- removed from the validated envelope;
- left unresolved and therefore blocking validation/1.0.

Do not close a finding because the implementation is inconvenient to change.
