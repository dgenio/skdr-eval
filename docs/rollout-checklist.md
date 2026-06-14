# Production rollout checklist: offline → online

A green `skdr-eval` verdict means a candidate policy is *worth testing* — not
that it is safe to ship to 100% of traffic. This checklist is the bridge from
an offline estimate to a controlled online rollout. Work through it in order;
each gate has a clear stop condition.

!!! warning "Offline evaluation informs the rollout decision; it does not replace the online experiment."
    Every step below assumes a confirmatory online test (canary or A/B) is
    still on the path to full deployment. Skipping it is the single most
    common way offline wins fail in production.

## Gate 1 — Support health is `ok`

Before anything else, confirm the offline estimate is even trustworthy.

- [ ] `artifact.warnings` shows `support_health == "ok"` for the candidate and
      estimator you intend to act on.
- [ ] No `high_risk` warning codes (`POOR_OVERLAP`, `LOW_ESS` at the severe
      threshold, `HIGH_PARETO_K ≥ 0.7`, severe `MISCAL_PROP`).
- [ ] Match-rate is high enough that the estimate reflects the candidate's
      actual action distribution.

**Stop if** `support_health` is `high_risk`: the logs do not support the
counterfactual. Improve logging/exploration before considering rollout.

## Gate 2 — Uncertainty interval excludes "no improvement"

- [ ] The confidence interval on the V̂ delta vs. baseline is on the right
      side of zero (the candidate is better, accounting for direction —
      lower is better for cost/latency, higher for reward).
- [ ] The interval is *useful*, not just signed: a barely-significant interval
      that includes a negligible effect is not a deployment case.
- [ ] You used the influence-function SE (`SE_if`) and/or the moving-block
      bootstrap CI (`block_bootstrap_ci`), not a naive i.i.d. interval, given
      time-correlated logs.

**Stop if** the interval includes no-improvement: treat the verdict as
`insufficient_evidence` and gather more data.

## Gate 3 — Sensitivity is stable

- [ ] V̂ does not swing materially across the clip grid
      (`artifact.sensitivity` / `summarize_sensitivity`). A decision-stable
      estimate survives reasonable changes to the clip threshold.
- [ ] If two estimators disagree (e.g. DR vs. SNDR), you understand why — see
      [choosing an estimator](choosing-an-estimator.md#when-estimators-disagree).

**Stop if** the estimate is an artifact of a particular clip value.

## Gate 4 — Stakeholder review with the card

- [ ] Generate the stakeholder card (`artifact.save_card(...)` or
      `artifact.to_html(...)`) and review it with whoever owns the metric.
- [ ] The card's verdict, support health, and assumptions are acceptable to
      the decision-maker — not just the data scientist.
- [ ] The estimand and assumption tags on the card match how the policy will
      actually be deployed.

## Gate 5 — Design the online test

Use the offline estimate to size the confirmatory experiment:

- [ ] Target effect size taken from the offline V̂ delta (use the conservative
      end of the CI).
- [ ] Per-arm sample size and expected duration computed from traffic and
      reward variance (the statistical helpers `power_analysis` /
      `sample_size_calculation` give a starting point).
- [ ] Randomization unit and guardrail metrics defined.
- [ ] Start as a **canary** (small traffic slice) before a full A/B split when
      the action has real-world cost or safety implications.

## Gate 6 — Monitoring and rollback

- [ ] Primary metric and guardrail metrics instrumented for the live test.
- [ ] Support health re-checked on *fresh* logs once the candidate is serving
      a slice of traffic (the candidate's own decisions change the log
      distribution).
- [ ] Pre-committed rollback criteria: a metric threshold and a maximum
      observation window after which a non-improving or regressing arm is
      reverted.
- [ ] An owner and an on-call path for the rollback.

## Quick reference: verdict → rollout posture

| Offline verdict | Rollout posture |
|---|---|
| `deploy` | Proceed to a confirmatory A/B test, then ramp. |
| `ab_test` | Design and run an A/B test; do not ramp without it. |
| `insufficient_evidence` | Do not roll out; collect more logs or accept status quo. |
| `do_not_deploy` | Do not roll out; fix support/overlap or the candidate. |

## See also

- [The Daily Driver guide](daily-driver.md) — the workflow that produces the
  verdict this checklist consumes.
- [Report interpretation guide](report-interpretation.md)
- [What skdr-eval claims](https://github.com/dgenio/skdr-eval/blob/main/CLAIMS.md) — especially the non-claims.
