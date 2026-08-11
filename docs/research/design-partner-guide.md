# Design-partner validation guide

This guide operationalizes #299. The goal is to test whether the validated-v1 workflow solves a real practitioner problem for teams that actually have the data contract OPE requires.

Do not optimize this research for GitHub stars, testimonials, or feature requests.

## Who qualifies

Prioritize practitioners who have operated or reviewed one-step decision systems such as:

- recommender/personalization policies;
- advertising/targeting/bandit systems;
- routing/assignment policies with explicit exploration;
- model/LLM routing only when the behavior action probability and available action set are genuinely logged.

A useful participant has direct experience with at least one real online policy experiment or experiment-review workflow.

## Interview questions

Record concrete facts rather than opinions where possible.

### Logging contract

1. What is one row / decision in your logs?
2. Is the action set explicit at decision time?
3. Can the available/eligible actions vary per decision? If so, is that set logged?
4. Is the behavior policy stochastic/randomized?
5. Is the probability of the chosen action logged at decision time?
6. Can you identify which logging-policy version/configuration produced the row?
7. Can that propensity/action provenance be audited later?

### Target policy

8. How is the candidate represented today: action probabilities, scores, a deterministic rule, or only model predictions?
9. Can you reconstruct the exact policy that would be evaluated online?
10. Can the candidate choose actions that historical logging rarely/never selected?

### Outcome

11. What reward/outcome determines policy value?
12. When does it become available?
13. Is one scalar reward sufficient for the actual review decision?
14. Are there important harms/guardrails not represented by that reward?

### Current experiment-review workflow

15. What evidence do reviewers require before approving an A/B test?
16. What do you currently do when offline metrics look good but logging support is weak?
17. Which counterfactual/OPE tooling, if any, do you already use?
18. Which result would make you explicitly refuse to proceed?
19. Where would a portable evidence artifact live: PR, ticket, experiment platform, notebook, dashboard, other?
20. Who needs to understand it besides the model author?

## Hands-on session

For data-ready partners, avoid custom implementation first. Use the narrow validated-v1 workflow.

Observe and timestamp:

1. install/start;
2. map log schema;
3. run `doctor`/preflight;
4. define the explicit target policy;
5. reach first valid or explicitly refused evaluation;
6. inspect the evidence artifact;
7. use/share it in the real decision workflow if possible.

Record every point where the participant needs maintainer help.

## Session result template

Do not store proprietary logs or sensitive values in this repository.

```text
participant/session id: anonymous identifier
system/domain: recommender | ads | routing | other
real online experimentation experience: yes/no

DATA READINESS
action set logged: yes/no/partial
eligibility logged: yes/no/not-applicable
chosen-action propensity logged: yes/no/partial
logging policy provenance available: yes/no/partial
target policy representable explicitly: yes/no/partial
validated-v1 data contract satisfied: yes/no

USABILITY
time to valid preflight:
time to first evidence artifact:
maintainer interventions:
blocking confusion:

DECISION VALUE
artifact entered a real review: yes/no
changed/narrowed/blocked/informed decision: describe without sensitive details
repeat-use intent: yes/no/unknown

RECURRING NEEDS
request(s):
seen independently before: yes/no

NON-ADOPTION REASON
if not adopted, why:
```

## Aggregate metrics

After approximately 15–20 interviews/observations, report:

- share with valid logged propensities;
- share with explicit action/eligibility provenance;
- share satisfying the whole validated-v1 data contract;
- median time to valid preflight;
- median time to first evidence artifact;
- percentage completing without maintainer intervention;
- number of artifacts entering a real decision review;
- repeat usage;
- recurring adoption blockers.

## Decision rules

### Continue native product
Multiple independent teams satisfy the data contract and value both the estimator path and evidence workflow.

### Evidence-layer pivot
Teams prefer established estimator backends but value skdr-eval's evidence, validation and artifact contract.

### OPE-readiness pivot
Most qualified teams lack propensity/action provenance and primarily need instrumentation/readiness diagnostics.

### Narrow/stop
Data-ready teams already solve the workflow adequately elsewhere and do not value the evidence layer enough to adopt it.

## Kill criterion

If fewer than roughly 3 of 15 qualified target teams have the validated-v1 data contract, explicitly reassess the addressable OPE market before adding product breadth.
