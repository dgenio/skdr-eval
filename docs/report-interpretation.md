# Report interpretation guide: from HTML output to decision

You ran an evaluation, exported an HTML report and a stakeholder card, and now
you have a screen full of numbers. This guide takes you from *"I generated a
report"* to *"I understand what decision this supports"*.

It pairs with the [metrics glossary](metrics-glossary.md) (what each field
*means*) — this page is about the **reading order** and the **decision logic**.

> Offline policy evaluation estimates how a candidate policy *would have*
> performed on logged data. It is a pre-A/B-test filter, not a deployment
> guarantee. A good report tells you whether the estimate is trustworthy
> enough to be worth an experiment — nothing more.

## A mini scenario

You trained a new call-routing model. Offline metrics look good, but rolling it
out live risks SLA violations and operator overtime. You evaluate it with
`evaluate_pairwise_models` against last quarter's logs and get an artifact.
Below is the order to read it in.

## 1. Start with the question

Before looking at a single number, write down:

- **Which candidate** policy/model are we evaluating, and against what
  baseline (today's logged policy, or a target number)?
- **Which metric** are we trying to move (`V_hat` measures what)?
- **Is higher or lower better?** Reward-style metrics: higher. Cost/time
  metrics (e.g. service time): lower.

If you can't answer these, the report can't help you yet.

## 2. Read the headline result

Look at `V_hat`, its uncertainty (`SE_if`), and the confidence interval
(`ci_lower` / `ci_upper`) if you enabled `ci_bootstrap=True`. If you passed a
`baseline=`, read `delta_V_hat` — the decision-relevant "better or worse than
today" number. Note: `delta_V_hat` (with `delta_ci_lower` / `delta_ci_upper`)
is a column in `artifact.report` — the DataFrame/JSON. The HTML report and
stakeholder card don't render the `delta_*` columns, so read them from the
artifact programmatically.

**Do not compare models yet.** A point estimate is meaningless until you know
the logs support it.

## 3. Check support health *before acting*

Find `support_health` for each `(model, estimator)` row:

- **`ok`** — the estimate is at least diagnostically usable. Continue.
- **`caution`** — inspect the warning codes before treating it as decision
  evidence.
- **`high_risk`** — stop. This is a **data/support problem**, not a
  model-ranking result. The number on the screen is not evidence about which
  model is better.

A common, healthy-looking trap: two different candidate models showing
*identical* `V_hat` and `match_rate = 1.0`. That usually means the candidates
collapsed onto the logged action and there is nothing to compare — not that
they are equally good.

## 4. Inspect the warning codes

Each code has a one-line action (full list in the
[glossary](metrics-glossary.md#warning-codes-artifactwarnings)):

- `LOW_ESS` → collect broader logs or reduce policy shift.
- `POOR_OVERLAP` / `EXTREME_CLIP` → the candidate goes where the logs don't;
  fix logging/exploration before comparing.
- `HIGH_PARETO_K` → importance weights are unstable; the value is fragile.
- `MISCAL_PROP` / `PER_ACTION_MISCAL` → improve or recalibrate the propensity
  model.
- `RARE_ACTION_NO_SUPPORT` → the candidate relies on actions the logs barely
  contain.

## 5. Check sensitivity

In `artifact.sensitivity`, a large `V_range` or `stable = no` means the
estimate **moves a lot as you change the clip threshold** — the answer is an
artifact of a tuning choice, not a robust finding. Also check `dr_sndr_agree`:
DR and SNDR disagreeing is a red flag that the assumptions are strained.

## 6. Check calibration and overlap

In `artifact.diagnostics`: `ece` / `brier_score` describe how trustworthy the
propensity model is (lower is better), and `overlap_ratio` describes whether
the candidate's actions are represented in the logs (higher is better). Poor
calibration or low overlap undermines everything above it.

## 7. Make a decision

| What you see | Decision |
|---|---|
| `support_health = ok`, narrow CI, `stable = yes`, delta CI clears zero | **Continue** — this candidate is worth an A/B test. |
| `support_health = caution`, warnings you understand and accept | **Use as exploratory evidence**; consider a guarded experiment. |
| `support_health = high_risk`, or `stable = no`, or CIs overlap | **Rerun** with better logs / a closer candidate / a recalibrated propensity model. |
| `POOR_OVERLAP` / `RARE_ACTION_NO_SUPPORT` dominate | **Reject the comparison** — the logs cannot support this question. Fix data collection first. |

The decision is never "the model with the highest `V_hat` wins". It is
"is this estimate trustworthy enough to justify the cost of an experiment?".

## Deployment verdicts

`artifact.recommendation(model, estimator=...)` (and the card's `trust.recommendation`)
condense the reading above into a single `verdict`. There are exactly four:

| Verdict | Meaning | Typical trigger | CLI exit code |
|---|---|---|---|
| `deploy` | CI clears the baseline with no caution/high-risk flags. | clean diagnostics + a winning CI | `0` |
| `ab_test` | Directionally promising, but confirm online. | CI clears baseline *with* a caution flag, or the CI overlaps the baseline | `0` |
| `insufficient_evidence` | The logs can't answer the question yet — usually **no bootstrap CI** was computed. | run without `--ci-bootstrap` | `4` |
| `do_not_deploy` | A high-risk diagnostic actively fired. | `POOR_OVERLAP`, `HIGH_PARETO_K`, or `EXTREME_CLIP` | `3` |

The `skdr-eval evaluate` / `pairwise` commands surface this as a process exit
code for CI gates. The gate inspects **every** estimator present in the
artifact (`DR`, `SNDR`, `MRDR`, `SWITCH-DR`, `DRos`, `MIPS`, …): a single
`do_not_deploy` returns exit `3` (it takes precedence), and an
`insufficient_evidence` with no block returns exit `4`. Treating
`insufficient_evidence` as a non-zero gate is deliberate — an honest "we can't
tell" should not pass CI as green. Re-run with `--ci-bootstrap` to turn an
`insufficient_evidence` into a real `deploy` / `ab_test` / `do_not_deploy`
decision.

## 8. Stakeholder explanation template

Paste-ready for a PR description, experiment-review doc, or product chat:

> We evaluated **[candidate]** offline against **[baseline]** on **[N]** logged
> decisions, optimizing **[metric]**. Estimated value: **[V_hat]**
> (**[delta vs baseline]**, 95% CI **[ci_lower, ci_upper]**). Support health is
> **[ok/caution/high_risk]**[, with warnings: codes]. This means the offline
> estimate is **[trustworthy enough to A/B test / not yet reliable because …]**.
> Offline evaluation does not replace an online test; recommendation:
> **[continue / collect more data / reject]**.

## Where to go next

- [Metrics glossary](metrics-glossary.md) — definitions for every field.
- [Good-vs-bad support tutorial](https://github.com/dgenio/skdr-eval/blob/main/examples/known_failures/README.md) —
  runnable demos of what healthy vs unhealthy support looks like.
- [Estimand and assumptions](concepts/estimands-and-assumptions.md) — the
  formal contract behind `V_hat`.
