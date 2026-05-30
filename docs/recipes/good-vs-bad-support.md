# Good vs bad support

The single most important skill for trustworthy offline policy evaluation is
recognising when the logged data **does not support** the policy you want to
evaluate. This tutorial shows both regimes on the *same* problem.

> **Runnable notebook:**
> [`examples/notebooks/06_good_vs_bad_support.ipynb`](https://github.com/dgenio/skdr-eval/blob/main/examples/notebooks/06_good_vs_bad_support.ipynb)
> (executed in CI via `nbmake`).

## The setup

A routing problem with three operators. The only thing that changes between the
two scenarios is **how much the logging policy explored**:

- **Good support** — the baseline policy is *epsilon-greedy with a large
  exploration rate*. Every operator is tried often, so the candidate policy's
  preferred actions are well represented in the logs.
- **Bad support** — the baseline policy is *near-deterministic* (tiny
  exploration). It almost always picks its greedy operator, so there is little
  counterfactual information about the alternatives.

## What you see

| | Good support | Bad support |
|---|---|---|
| `support_health` | `ok` | `high_risk` |
| Importance-weight tail (Pareto-k) | small (< 0.5) | large (≥ 0.7) |
| `ESS` | high (most of the sample) | collapsed |
| Warnings | none | `POOR_OVERLAP`, `HIGH_PARETO_K`, `EXTREME_CLIP` |
| Interpretation | `V_hat` is usable evidence | `V_hat` is **not** deployment evidence |

The point is not that the estimator "fails" in the bad case — it is that the
diagnostics **tell you loudly** not to trust the number. That is the trust
contract.

## The decision

- **Good support** → proceed to benchmark / A-B-test planning.
- **Bad support** → improve the *logging* before trusting any offline estimate:
  add exploration, fix the propensity model, or collect more data in the region
  the candidate policy cares about.

See [reading the report](../report-interpretation.md) for the meaning of each
warning code, and the [metrics glossary](../metrics-glossary.md) for `ESS`,
`match_rate`, Pareto-k, and ECE.
