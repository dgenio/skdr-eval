# Recipe: evaluate an LLM reranker offline

**How do you know a new LLM reranker is actually better — without shipping it?**

You run a production reranker that, for each query, picks one candidate from a
pool and observes a reward (click / dwell / purchase). You have a *new* reranker
and want to estimate its value **offline**, from logged decisions and
pre-computed embeddings — **CPU-only, no runtime LLM dependency** — before you
risk an A/B test.

This is the single best on-ramp for the LLM/agents audience into trustworthy
offline policy evaluation. It runs end-to-end in well under two minutes.

> **Runnable notebook:**
> [`examples/notebooks/10_llm_reranker_ope.ipynb`](https://github.com/dgenio/skdr-eval/blob/main/examples/notebooks/10_llm_reranker_ope.ipynb)
> · [Open in Colab](https://colab.research.google.com/github/dgenio/skdr-eval/blob/main/examples/notebooks/10_llm_reranker_ope.ipynb)

## Why MIPS

The action space (the candidate pool) is large, so per-candidate overlap is
hopeless. **MIPS** (Marginalized IPS) fixes this: the candidate *embeddings*
carry the structure of the action space, so common support only needs to hold
over the embedding, not the raw candidate id.

## The shape of the workflow

```python
from skdr_eval.recipes import (
    make_llm_reranker_synth,
    induce_reranker_policy,
    evaluate_reranker_mips,
)

# 1. Logs with a known ground-truth target value (for this recipe's proof).
logs, candidate_embeddings, truth = make_llm_reranker_synth(
    n_queries=2000, candidates_per_query=20, embed_dim=16, seed=7
)

# 2. The candidate (new) reranker, as a per-decision target-policy matrix.
policy_probs = induce_reranker_policy(logs, candidate_embeddings)

# 3. Evaluate offline with MIPS.
result = evaluate_reranker_mips(logs, candidate_embeddings, policy_probs)
print(result.V_hat, result.support_health)
```

## Read the verdict, not just the number

On the synthetic dataset MIPS recovers the closed-form target value within
±2 SE — that is the *ground-truth recovery* proof. On real logs you won't have a
ground truth, so the deciding signal is the **trust verdict**:

- check **ESS**, **Pareto-k**, and the **embedding-sufficiency** diagnostic;
- `support_health == "ok"` → the offline estimate is usable decision evidence;
- otherwise → the embeddings/logs don't yet support the comparison, and the
  number is not deployment evidence.

Offline evaluation is decision *support*, not a replacement for an online A/B
test.

## Generalizing to agent / tool-selection policies

The same machinery evaluates an LLM **tool-selection / routing** policy: map
your agent traces with the [trace adapter](../weaver-stack.md)
(`skdr_eval.adapters.from_jsonl_trace`) and evaluate the candidate routing
policy — see the
[agent-routing example](https://github.com/dgenio/skdr-eval/blob/main/examples/use_cases/06_agent_routing_policy.py)
and the [offline-evaluation companion guide](../weaver-stack.md).
