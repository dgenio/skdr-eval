# Recipe: natural-language evaluation summary (provider-agnostic)

**You want a one-paragraph, plain-English summary of an evaluation for
leadership — without wiring an LLM vendor into the library.**

`skdr-eval` stays CPU-only and dependency-light: it does **not** call any LLM.
Instead it emits a compact, structured *facts* payload from an
`EvaluationArtifact`, and you feed those facts into the prompt template below
with whatever model you already use (Claude, or any other provider). This keeps
the core neutral and free of API keys, cost, and heavy dependencies (#249).

> This produces a *summary of the verdict `skdr-eval` already computed*. It does
> not ask the model to judge trustworthiness — the trust diagnostics are the
> ground truth, and the model only narrates them.

## 1. Get the facts

```python
import skdr_eval

artifact = skdr_eval.evaluate_sklearn_models(...)
facts = artifact.to_summary_facts("my_model", estimator="SNDR")
```

`to_summary_facts` returns a JSON-ready dict:

| Key | Meaning |
| --- | --- |
| `model`, `estimator` | Which row is summarized. |
| `V_hat` | Estimated policy value. |
| `ci_lower`, `ci_upper` | Confidence-interval bounds (`null` if no CI was computed). |
| `baseline`, `delta_vs_baseline` | Baseline value and the candidate's delta (`null` if no baseline). |
| `verdict` | `deploy` / `ab_test` / `insufficient_evidence` / `do_not_deploy` / `null`. |
| `confidence` | `high` / `medium` / `low` / `null`. |
| `support_health` | `ok` / `caution` / `high_risk` / `null`. |
| `primary_blocker` | The single warning code that most drives the verdict, if any. |
| `warning_codes` | All diagnostic warning codes on the row. |
| `reasons` | Ordered list of `{code, message, severity}`. |

## 2. The prompt template

Copy this into your own LLM call. It is deliberately strict: the model must not
invent numbers, and must lead with the trust verdict rather than the headline
value.

```text
You are writing a short executive summary of an OFFLINE policy evaluation.
Use ONLY the JSON facts below. Do not invent numbers or claim online results.

Rules:
- Lead with the verdict and whether the result is trustworthy enough to act on.
- If support_health is "high_risk" or verdict is "do_not_deploy" /
  "insufficient_evidence", make clear the estimate should NOT be acted on yet.
- State V_hat and the confidence interval if present; say "no CI computed" if
  ci_lower/ci_upper are null.
- Mention the primary_blocker in plain language if it is set.
- Keep it to one short paragraph. No hype.

FACTS:
{facts_json}
```

Fill `{facts_json}` with `json.dumps(facts, indent=2)`.

## 3. Send it (example with the Anthropic SDK)

`skdr-eval` does not depend on any provider SDK; install and call your own:

```python
import json
import anthropic  # your dependency, not skdr-eval's

prompt = TEMPLATE.replace("{facts_json}", json.dumps(facts, indent=2))
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}],
)
print(msg.content[0].text)
```

Swap in any provider — the facts payload and template are provider-agnostic.

## Why this design

- **No lock-in, no runtime cost in the library.** The facts are portable JSON;
  the template is text. `skdr-eval` never imports an LLM client.
- **Honest by construction.** The facts carry `support_health`, `verdict`, and
  the `primary_blocker`, and the template forces the summary to lead with them —
  so a thin-support result cannot be narrated as a green light.
