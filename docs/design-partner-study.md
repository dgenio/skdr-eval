# Design-partner validation study

This document operationalizes issue #299. The purpose is to falsify or validate the product thesis with practitioners who have the data required for defensible one-step OPE.

This is **not** a marketing interview script and should not be used to solicit stars, endorsements, or feature wishlists.

## Questions the study must answer

1. Do enough target teams actually log the data required by validated-v1?
2. Can a competent practitioner map those logs to the contract without maintainer intervention?
3. Does the evidence artifact change or improve a real experiment-review decision?
4. Is the evidence-quality layer meaningfully more useful than direct use of established OPE tooling?
5. Which recurring obstacles justify product work after validation?

## Recruitment

Target approximately 15–20 practitioner conversations across several settings:

- recommender / personalization teams;
- advertising / targeting / contextual-bandit experimentation;
- operational routing or assignment systems with explicit exploration;
- model/LLM routing only when the team genuinely logs action eligibility and behavior probabilities.

Prefer practitioners who have participated in real randomized online experiments and can discuss their logging/review process concretely.

Do not count generic ML practitioners with no policy-logging workflow as successful design partners; their lack of readiness is itself useful market evidence.

## Screening

Before a hands-on attempt, establish whether the team can answer **yes** to the key data questions:

- Is the action selected at each decision explicitly recorded?
- Is the eligible action set known or reconstructable for that decision?
- Is the probability of the logged action recorded at decision time?
- Is the logging-policy provenance auditable?
- Is there a scalar outcome/reward tied to the decision?
- Can the target policy produce an explicit action distribution over the same vocabulary?
- Is the problem one-step/contextual rather than long-horizon sequential RL?

Record `yes`, `no`, or `unknown`; do not reinterpret missing information as available.

## Interview guide

### Existing decision process

- What happens between an offline model improvement and permission to run an online experiment?
- Who reviews the evidence?
- Which offline metrics are currently considered persuasive?
- Which failure modes make reviewers reject an offline result?
- How much does a failed online experiment cost in time, traffic, revenue, operational risk, or opportunity cost?

### Logging contract

- Why does the behavior policy randomize, if it does?
- Where is the action probability generated and logged?
- Can probabilities be lost, rounded, recomputed, or joined incorrectly?
- Does action eligibility change per request/user/time?
- Can deployed policy versions be tied to each log row?
- Are outcomes delayed, censored, or missing?

### Current OPE behavior

- Do you already run IPS/DR/OPE?
- Which implementation do you trust and why?
- How do you assess overlap/support today?
- Do you ever refuse to report an offline estimate because support is inadequate?
- How are confidence intervals interpreted in experiment review?

### Artifact usefulness

Show the narrow evidence concept, not a polished sales demo.

Ask:

- Which fields would you use in an actual review?
- Which fields would you distrust or ignore?
- What could be misinterpreted?
- Would you attach this artifact to a PR, experiment ticket, model card, or review document?
- What evidence is missing before you would rely on it?

## Hands-on study

Aim for at least five data-ready attempts.

Use the same validated-v1 workflow for all participants unless a compatibility bug prevents it. Avoid bespoke integrations during the study.

Record timestamps for:

1. start;
2. data-contract understanding;
3. first `doctor`/validation result;
4. first successful evaluation, if reached;
5. first evidence artifact;
6. participant's interpretation of the result.

Record every maintainer intervention and classify it:

- unclear documentation;
- schema mismatch;
- missing instrumentation;
- software bug;
- unsupported problem setting;
- statistical concept confusion;
- missing integration;
- performance/resource problem;
- other.

## Outcome record per participant

Use an anonymized record such as:

```yaml
participant_id: P01
domain: recommender
qualified_for_validated_v1: true
logged_propensity_available: true
eligibility_available: true
target_policy_distribution_available: true
minutes_to_valid_schema: 18
minutes_to_first_artifact: 42
maintainer_interventions: 1
artifact_used_in_real_decision: true
repeat_evaluation_within_study: false
outcome:
  - narrowed experiment population
  - identified weak support
blocking_friction:
  - eligibility column mapping unclear
requested_features:
  - parquet CLI example
```

Never store confidential business data, raw logs, participant PII, or proprietary model information in the public study record.

## Aggregate metrics

At minimum report:

- number screened;
- percentage with logged propensities;
- percentage with explicit/reconstructable eligibility;
- percentage satisfying full validated-v1 data contract;
- median time to valid schema;
- median time to first artifact among qualified attempts;
- percentage completing without maintainer intervention;
- number of artifacts used in a real review decision;
- number of repeat evaluations;
- recurring friction themes;
- recurring feature requests, counting only independently repeated requests;
- reasons for non-adoption.

## Predeclared product decisions

### Continue native-product strategy

Evidence:

- multiple independent teams satisfy the data contract;
- users can complete the workflow with low maintainer intervention;
- evidence artifacts enter real review decisions;
- users value native estimation plus evidence quality.

### Pivot toward an evidence/reporting layer over established backends

Evidence:

- users prefer established estimator backends such as OBP/VW for computation;
- they still value skdr-eval's evidence validation, abstention, provenance, and artifact semantics.

### Pivot toward OPE-readiness / decision-logging tooling

Evidence:

- otherwise relevant teams repeatedly fail screening because behavior probabilities, action eligibility, or policy provenance are not logged;
- instrumentation/readiness is a more common pain than estimator execution.

### Narrow or stop

Evidence:

- qualified teams already solve the problem adequately with existing tooling;
- the evidence artifact does not change review decisions;
- the workflow requires persistent maintainer involvement;
- the addressable set of data-ready teams is too small for the maintenance burden.

## Kill criteria

Trigger an explicit product-strategy review when any of the following holds:

- fewer than roughly 3 of the first 15 qualified-target conversations have the validated-v1 data contract;
- fewer than 2 of 5 data-ready attempts produce an artifact used in a real review decision;
- most successful users prefer a direct established backend and see no material value in the evidence layer;
- more than half of data-ready attempts require nontrivial maintainer intervention after obvious docs bugs are fixed.

Do not move the goalposts after observing the results without documenting why.

## Feature-request rule

A design-partner request is not automatically roadmap evidence.

Promote a feature only when it:

- fixes a correctness or misuse problem; or
- is independently requested/observed across multiple design partners; or
- directly unblocks the validated-v1 workflow.

Record one-off requests, but keep them deferred.
