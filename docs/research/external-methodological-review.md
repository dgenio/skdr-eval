# External methodological review packet

This document operationalizes #298. The purpose is independent challenge, not endorsement marketing.

## Review target

Every review must identify the exact immutable commit/tag and validation-report version being reviewed.

The reviewer is assessing whether the stated **validation envelope is defensible**, not whether `skdr-eval` is universally correct or safe for deployment.

## Minimum reviewer perspectives

Seek at least three independent perspectives:

1. **OPE/contextual-bandit methods** — estimand, DR/SNDR formulas, behavior/target policy semantics, overlap and reference-validation design.
2. **Causal/statistical ML** — nuisance fitting, OOF guarantees, inference, diagnostic calibration, confounding/non-claims.
3. **Production experimentation/recommenders** — logging propensities, eligibility/action provenance, artifact usability and misuse risk.

Reviewers should be independent of the implementation authorship they are evaluating.

## Packet contents

Provide:

- exact source commit/tag;
- `CLAIMS.md`;
- validated-v1 envelope / strategy;
- exact estimand and formula documentation;
- explicit `Policy`/action-distribution contract;
- behavior-propensity semantics;
- effective evaluation-population rules;
- known-ground-truth simulations;
- estimator-specific reference-agreement reports;
- bias/RMSE/coverage/false-reassurance validation report;
- evidence-state and implementation-maturity schemas;
- historical correctness advisories relevant to previous releases;
- unsupported/non-claim list;
- reproducibility instructions.

## Questions for the methods reviewer

- Is the target estimand defined unambiguously?
- Does the implementation compute that estimand for deterministic and stochastic policies?
- Are `q_obs` and `q_pi` action-specific and correctly composed?
- Are behavior and target policy probabilities semantically distinct everywhere?
- Is overlap/positivity handled honestly?
- Are reference comparisons actually comparing equivalent estimators/configurations?
- Are any reference backends being treated as an oracle rather than one line of evidence?
- Which documented regimes should be removed from the validation envelope?

## Questions for the statistical/causal reviewer

- Does every evaluated row have the required non-leaky nuisance predictions?
- Are the effective population and exclusions explicit?
- What uncertainty is each interval actually claiming to include?
- Are dependence assumptions named precisely enough?
- Are diagnostics empirically calibrated against estimation error?
- Is the false-reassurance metric defined meaningfully?
- Are causal/unconfoundedness assumptions described without implying they can be verified from diagnostics?
- Are any high-stakes/observational examples likely to be misread as validated causal guidance?

## Questions for the production reviewer

- Can a real logging system satisfy the propensity/action/eligibility contract without heroic reconstruction?
- Is policy provenance auditable?
- Does `doctor` fail early on the mistakes practitioners actually make?
- Can a non-methods reviewer understand why evidence is unsupported?
- Does the artifact fit an experiment-review workflow?
- Does any language look like deployment/experiment authorization?
- What would make this artifact useful enough to attach to a real decision record?

## Required review output

With reviewer permission, publish:

- reviewer identity or role/background;
- date;
- exact commit/tag reviewed;
- validation-report identifier;
- major findings/objections;
- severity/blocking status;
- maintainer response;
- resulting code/docs changes;
- unresolved limitations;
- reviewer conclusion on whether the **stated envelope** is defensible.

Do not reduce findings to a testimonial quote.

## Finding severity

Suggested categories:

- **Blocker** — estimand/inference/evidence claim is not defensible; blocks validation promotion/1.0.
- **Major** — substantial misuse/correctness risk; must be resolved or narrow the envelope.
- **Minor** — clarity/robustness improvement that does not invalidate the current envelope.
- **Question** — requires explicit answer or documentation.

## Promotion rule

An unresolved blocker prevents `independently_validated` status. The response is to fix or narrow the claim, not weaken the review gate.
