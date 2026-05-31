# skdr-eval as an offline-evaluation companion

`skdr-eval` is a **standalone, MIT-licensed** off-policy evaluation library for
scikit-learn-compatible policies. It predates the Weaver agent stack, targets a
different audience (ML/DS doing offline policy evaluation), and has **no runtime
dependency on any Weaver repository**. It is a *companion*, not a member, of
that stack.

There is, however, one genuine and valuable seam between them.

## The seam

The Weaver stack produces **logged agent decisions** — routing and
tool-selection traces of the form `(context, action, reward)`. `skdr-eval`
answers a question those logs raise but the runtime cannot:

> *"Would this candidate routing/tool-selection policy actually perform better,
> from these logs, before we ship it?"*

That is exactly offline policy evaluation: estimate a candidate policy's value
from logged decisions, and — just as importantly — report **whether the logs
even support the question** (`support_health`, ESS, Pareto-k) so you don't act
on an estimate the data can't justify.

`skdr-eval` is reached **when you want to evaluate a logged agent/decision
policy offline**, not as part of the runtime request path.

## How to use it from an agent stack

1. **Map your traces.** Use the generic trace adapter to turn
   `(context, action, reward[, timestamp, propensity])` records — including
   JSONL agent traces — into a schema-valid logs frame:

   ```python
   import skdr_eval

   adapted = skdr_eval.adapters.from_jsonl_trace("agent_traces.jsonl", reward_col="cost")
   skdr_eval.validate_logs(adapted.logs, y_col=adapted.reward_col)
   ```

   See the trace-adapter entry in the [API reference](api.md) and the
   end-to-end [agent-routing example](https://github.com/dgenio/skdr-eval/blob/main/examples/use_cases/06_agent_routing_policy.py).

2. **Evaluate a candidate policy.** Feed the adapted logs to
   `evaluate_sklearn_models` and read the trust verdict first:

   ```python
   artifact = skdr_eval.evaluate_sklearn_models(
       logs=adapted.logs, models={"candidate": my_regressor}, y_col=adapted.reward_col,
   )
   print(artifact.report[["model", "estimator", "V_hat", "support_health"]])
   ```

3. **Act on the verdict, not just the number.** `support_health == "ok"` means
   the logs support the comparison; `high_risk` means improve exploration /
   logging before trusting the estimate.

## Positioning boundary

- **Standalone-first.** `skdr-eval` installs and runs with no Weaver package
  present.
- **License.** `skdr-eval` is MIT; it is intentionally *not* folded into the
  Apache-2.0 Weaver core, to keep its standalone ML credibility intact.
- **Companion, not core.** The relationship is reciprocal cross-linking, not a
  code dependency in either direction.

## Cross-links

These live in sibling repositories; coordinate the back-links there:

- agent-kernel — issue #96 (offline-evaluation hook).
- ChainWeaver — `examples/skdr_policy_eval_flow.py`.
- contextweaver — `eval_artifact_profile`.

Related building blocks in this repo: the
[logs → experiment-review card recipe](recipes/logs-to-experiment-card.md), the
[LLM-reranker OPE recipe](recipes/llm-reranker-ope.md), and the trace adapter
(`skdr_eval.adapters`).
