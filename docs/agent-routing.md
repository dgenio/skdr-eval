# Evaluating agent routing and tool-selection policies offline

`skdr-eval` is an offline policy evaluation library, and an LLM-agent's
routing and tool-selection choices are exactly the kind of **one-shot logged
decision** it was built for. If your agent stack records which tool, model, or
route it picked and what that cost, you can ask the counterfactual question —
*"would this candidate routing policy have done better?"* — **before** you ship
it.

This page is the conceptual landing page for that use case. The runnable
version is
[`examples/use_cases/06_agent_routing_policy.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/use_cases/06_agent_routing_policy.py),
and the broader story is in the
[offline-evaluation companion guide](weaver-stack.md).

!!! warning "Offline evaluation does not replace online validation."
    A green verdict says a candidate routing policy is *worth* an online test,
    not that it is safe to ship. And for genuinely **sequential / long-horizon**
    agents (multi-step plans where each action changes the state the next one
    sees), single-step OPE is the wrong tool — see the limitations below.

## What a logged agent decision looks like

Each logged decision is a `(context, action, outcome, time)` tuple:

- **context** — request features available at decision time: prompt size,
  detected intent, user tier, conversation length, retrieved-context size.
- **action** — the discrete choice the agent made.
- **outcome / reward** — what you observed afterward.
- **time** — when the decision happened (used for time-aware splits).

### Example actions

- selected **tool** (search vs. calculator vs. code-exec);
- selected **model / route** (fast-cheap vs. smart-expensive);
- **handoff target** (which downstream agent);
- **escalation choice** (auto-resolve vs. ask-human);
- **policy branch** (allow / deny / ask, for a guardrail policy).

### Example rewards / outcomes

success, resolution, latency, cost (tokens/$), safety violation (as a
penalty), or human-correction events. Costs work as negative rewards — lower
is better — matching the policy-induction convention in
`induce_policy_from_sklearn`.

## What support / overlap means here

The estimate is only trustworthy where the *logging* agent actually explored.
If your production agent almost always picks `route_smart`, your logs carry
little signal about a candidate that leans on `route_cheap` — there is no
counterfactual evidence for the actions it would newly take. `skdr-eval`
reports this honestly as `support_health = high_risk` rather than returning a
confident number. A logging agent that explores (even a little ε-greedy
jitter) keeps every route's logged probability healthy and makes offline
evaluation possible.

## How DR/SNDR diagnostics apply before rollout

The workflow is the standard one (see the [Daily Driver guide](daily-driver.md)):

1. Map traces to logs with `skdr_eval.adapters.from_records` /
   `from_jsonl_trace`.
2. Wrap the candidate routing policy as a scikit-learn-compatible model.
3. Run `evaluate_sklearn_models` and read **support health before V̂**.
4. Use the verdict (`deploy` / `ab_test` / `insufficient_evidence` /
   `do_not_deploy`) to decide whether the candidate earns an online test.

```python
import skdr_eval

adapted = skdr_eval.adapters.from_jsonl_trace("agent_traces.jsonl", reward_col="cost")
artifact = skdr_eval.evaluate_sklearn_models(
    logs=adapted.logs,
    models={"router_v2": candidate_router},
    y_col="cost",
    fit_models=True,
    policy_train="pre_split",
)
print(artifact.warnings[["model", "estimator", "support_health"]])
```

## Relation to the agent ecosystem

In an agent stack such as the Weaver stack, the upstream components produce the
logs this page consumes:

- **contextweaver** routing logs → the `(context, action)` for model/route
  selection;
- **AgentFence** allow / deny / ask decisions → a guardrail-policy action with
  a safety-violation outcome;
- **agent-kernel** `ActionTrace` records → the per-decision trace mapped via
  `from_records`.

`skdr-eval` is the **offline evaluation layer** for those governance
decisions — it does not run the agent or route traffic; it judges a candidate
policy from the logs the stack already emits. See
[weaver-stack.md](weaver-stack.md) for the end-to-end companion story.

## When OPE is the wrong tool for agents

- **Sequential / long-horizon plans** — when an action changes the state the
  next action sees, you need reinforcement-learning OPE (SCOPE-RL / d3rlpy),
  not single-step contextual-bandit DR. See [comparisons](comparisons.md).
- **No exploration in the logs** — a near-deterministic logging agent leaves
  no overlap; gather exploratory logs first.
- **Reward you cannot measure offline** — if the outcome only exists after a
  human interacts live, offline evaluation cannot estimate it.

## See also

- [`examples/use_cases/06_agent_routing_policy.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/use_cases/06_agent_routing_policy.py)
- [Offline-evaluation companion (Weaver stack)](weaver-stack.md)
- [LLM-reranker OPE recipe](recipes/llm-reranker-ope.md)
- [The Daily Driver guide](daily-driver.md)
