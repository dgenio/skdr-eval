# When to use slate vs pairwise vs standard evaluation

`skdr-eval` ships three top-level evaluation entry points. They share the same
trust machinery (support-health warnings, sensitivity, the `EvaluationArtifact`
card) but model different decision shapes. Pick the one that matches how your
policy actually makes a decision.

## Quick decision guide

| Your decision looks like… | Use | Entry point |
|---|---|---|
| Pick **one action** per context from a fixed set (recommend, target, classify, treat) | **Standard** | `evaluate_sklearn_models` |
| Assign **clients to operators** under capacity / eligibility constraints (routing, autoscaling) | **Pairwise** | `evaluate_pairwise_models` |
| Show an **ordered list (slate / top-K)** and observe per-position clicks | **Slate** | `evaluate_slate_models` |

## Standard — `evaluate_sklearn_models`

The contextual-bandit workhorse. One logged decision is `(context x, action a,
reward y)`; the candidate policy is a scikit-learn-protocol model. Estimators:
DR, SNDR, MRDR, SWITCH-DR, DRos, and MIPS (for large action spaces — see
[the MIPS notes](#mips-large-action-spaces)).

Reach for it when each decision picks a single action and you can wrap your
policy behind `fit` / `predict`.

## Pairwise — `evaluate_pairwise_models`

For client→operator assignment problems where decisions are **coupled** by
shared capacity and eligibility (a call-routing or autoscaling layer). The
policy is induced over `(client, operator)` pairs per day, not independently
per row. Use it when "who handles whom" is constrained by availability rather
than chosen one row at a time.

## Slate — `evaluate_slate_models`

For ranking / top-K policies that show an **ordered slate** of items and
observe **per-position** outcomes (clicks). The logged decision is a slate plus
a per-rank click vector; the candidate policy is a **per-rank policy**
`(rank, item) -> probability`.

```python
import skdr_eval

logs, attractiveness, truth = skdr_eval.make_slate_synth(
    n_impressions=500, n_items=10, slate_size=3, click_model="cascade", seed=0
)

def my_reranker(rank: int, item: int) -> float:
    ...  # probability the target policy places `item` at `rank`

artifact = skdr_eval.evaluate_slate_models(
    logs,
    models={"my_reranker": my_reranker},
    estimators=("RIPS", "PI-IPS", "SlateCascadeDR"),
    baseline="logging",
)
print(artifact.report)        # per-(model, estimator) values + support_health
artifact.to_html_str()        # renderable stakeholder card
```

Estimator choice within the slate family:

- **`SlateCascadeDR`** — doubly-robust, unbiased under the cascade click model;
  the default first choice when the logs come from cascade-style browsing.
- **`RIPS`** (reward-interaction IPS) — consistent under position-bias click
  models; no outcome model required.
- **`PI-IPS`** (pseudo-inverse IPS) — lower variance when the per-rank
  examination matrix is well-conditioned.
- **`SlateStandardIPS`** — vanilla slate-level IPS; highest variance, useful as
  a baseline.

### Reading slate support-health

Slate support diagnostics are computed from the **slate-level importance
weight** `π_target(slate) / π_logging(slate)` — an estimator-agnostic measure
of how well the logged slates overlap the target policy. Because a slate-joint
probability lives on a different scale than a per-action propensity, the
`min_pscore` / `POOR_OVERLAP` diagnostic uses the weight-implied *effective
propensity* `1 / w`, so it fires on genuine slate-overlap failures (a single
slate weight exceeding the sample size) rather than firing by construction on
any non-tiny catalogue. As always, lean on `ESS` and `pareto_k`: a heavy
importance-weight tail (low ESS, high Pareto-k) means the target policy prefers
slates the logging policy rarely showed.

## MIPS (large action spaces)

When the action set is very large (candidate-set rerankers, big operator
pools), standard per-action IPS has no overlap. **MIPS** marginalises the
propensity over an **action embedding**, so common support only needs to hold
over the embedding. It is available in `evaluate_sklearn_models` /
`evaluate_pairwise_models` via `estimators=(..., "MIPS")` and is the engine
behind the [LLM-reranker recipe](../README.md#recipes--llm-reranker-ope). MIPS
is biased when the embedding is not a sufficient statistic for the reward —
always read `embedding_sufficiency_diagnostic`.

## All three share the same trust story

Whichever entry point you use, the returned `EvaluationArtifact` exposes the
same `report`, `warnings`, `sensitivity`, and card surface, and the same rule
applies: **offline evaluation is decision support, not a replacement for an
online A/B test.**
