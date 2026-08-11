# Explicit target-policy contract

The validated one-step OPE path must evaluate an **explicit decision policy**.
A prediction model, score, or outcome regressor is not automatically the policy
that would be run online.

Issue #279 introduces the representation seam before the estimator kernel is
migrated to consume it.

## Contract

A policy returns an action-probability matrix:

```python
class Policy(Protocol):
    def action_distribution(
        self,
        contexts,
        *,
        actions,
        eligible_actions=None,
    ) -> np.ndarray:
        ...
```

The matrix has shape `(n_rows, n_actions)` and its columns are in exactly the
order supplied by `actions`.

Every row must:

- contain finite probabilities;
- contain no negative probabilities;
- sum to one;
- assign zero probability to ineligible actions.

Deterministic policies are ordinary one-hot distributions. They do not use a
separate estimand.

## Already-computed probabilities

When another model or service has already produced the exact candidate action
probabilities, bind them to the action vocabulary explicitly:

```python
from skdr_eval.policy import ExplicitPolicy

candidate = ExplicitPolicy(
    candidate_probabilities,
    actions=("control", "offer_a", "offer_b"),
    name="candidate-v17",
)
```

If a later caller requests the same probability matrix under a different action
ordering, evaluation fails instead of silently reinterpreting the columns.

## One resolver for native and reference backends

```python
from skdr_eval.policy import resolve_action_distribution

pi = resolve_action_distribution(
    candidate,
    contexts,
    actions=("control", "offer_a", "offer_b"),
    eligible_actions=eligibility,
)
```

Native estimators and estimator-specific reference adapters should resolve the
policy through this seam rather than independently interpreting model scores.
This is necessary for meaningful cross-implementation agreement: both sides must
be evaluating the same target distribution over the same ordered action space.

## What this does not do yet

This contract PR deliberately does **not** change DR/SNDR mathematics and does
not make the current `evaluate_sklearn_models` path validated. Follow-up work
must:

1. migrate the standard evaluator to consume an explicit policy or action
   probability matrix;
2. deprecate implicit policy induction from arbitrary outcome-model scores in
   the validated path;
3. record policy type, action mapping and safe configuration/provenance in the
   evidence artifact;
4. use the same resolved distribution in #58 corrected multi-action DR and #281
   estimator-specific reference agreement.

Until those migrations and the remaining validation gates land, this module is
the explicit **contract seam**, not evidence that the estimator path is already
validated.
