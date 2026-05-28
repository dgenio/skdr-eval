# Metrics glossary

A plain-language reference for every metric and warning code that appears in
an `EvaluationArtifact` — in `artifact.report`, `artifact.warnings`,
`artifact.sensitivity`, `artifact.diagnostics`, and in the HTML report and
stakeholder card (`artifact.to_html(...)` / `artifact.save_card(...)`).

The goal is simple: you should not need to read the source or an OPE paper to
understand what a column means at a practical level. Each entry gives the
plain meaning first, then **how to read it** — the good / caution / risky
intuition, the common failure mode, and what to do next.

> **These are trust signals, not proof.** A clean diagnostic does not prove an
> offline estimate is correct — it means the logged data does not *obviously*
> violate the assumptions the estimator needs. Offline evaluation never
> replaces online validation. See
> [`report-interpretation.md`](report-interpretation.md) for how to turn these
> numbers into a decision, and
> [`concepts/estimands-and-assumptions.md`](concepts/estimands-and-assumptions.md)
> for the formal estimand and assumptions.

## How to read this page

Metrics are grouped by what they tell you:

- **Value** — the estimate itself and its uncertainty.
- **Support / overlap** — whether the logs cover what the candidate policy
  would do (the precondition for trusting the value at all).
- **Sensitivity** — how much the estimate moves as you vary the clip grid.
- **Calibration** — how trustworthy the propensity model is.
- **Per-decision** — opt-in attribution of the value to individual rows.

## Value metrics (`artifact.report`)

| Metric | Plain meaning | How to read it |
|---|---|---|
| `V_hat` | Estimated value of the candidate policy under your metric (e.g. mean reward, or mean service time for the pairwise API). | Whether **higher or lower is better depends on your task** — reward-style metrics are "higher is better", a cost/time metric is "lower is better". Compare against a baseline, not in isolation. Common failure: reading `V_hat` when `support_health` is poor. Action: check support health *before* comparing models. |
| `SE_if` | Influence-function standard error — an uncertainty proxy for `V_hat`. | Smaller means **more precise**, not more *correct*. A tiny `SE_if` on poorly supported data is false confidence. Action: pair with `support_health` and the CI. |
| `ci_lower`, `ci_upper` | Confidence-interval bounds, present when `ci_bootstrap=True` (moving-block bootstrap). | If two models' CIs overlap heavily, treat them as **not distinguishable** from these logs. Action: do not rank models on point estimates whose CIs overlap. |
| `delta_V_hat` | `V_hat` minus the `baseline=` you passed (a float, `"logged"`, or omitted). `delta_ci_lower` / `delta_ci_upper` accompany it when CIs are enabled. | The decision-relevant quantity: "how much better/worse than what we run today?". A delta CI that crosses zero means **no demonstrated improvement**. |
| `clip` | The importance-weight clipping threshold actually used for that row. | Heavy clipping (small `clip`) trades variance for bias. If `clip` is driving the result, see `sensitivity`. |
| `ESS` | Effective sample size — roughly how many decisions actually carry the estimate after importance weighting. | Low `ESS` (relative to `n`) means **a few rows dominate** the estimate; it is fragile. Common failure: `ESS` of a few dozen on tens of thousands of rows. Action: collect broader logs or reduce policy shift; expect a `LOW_ESS` warning. |
| `match_rate` | Fraction of rows where the candidate policy's action is supported by the logged action distribution. | Higher is better. A `match_rate` of exactly `1.0` across *different* candidate models is suspicious — it usually means the candidates collapse to the logged action. Action: verify the policies actually differ. |
| `min_pscore` | Smallest logging-policy propensity encountered. | Values at the floor (e.g. `1e-8`) signal **near-deterministic logging** — the overlap the estimator needs is missing. Action: this drives `EXTREME_CLIP` / `POOR_OVERLAP`. |
| `pareto_k` | PSIS (Pareto-smoothed importance sampling) tail-shape estimate for the importance weights. | `≤ 0.7` is healthy; above it the weight distribution has a heavy tail and the estimate is unstable. `NaN` can occur when there are too few tail samples to fit — treat as "cannot certify", not "fine". Action: a high `pareto_k` triggers `HIGH_PARETO_K`. |
| `support_health` | One-word summary: `ok`, `caution`, or `high_risk`. | `ok`: diagnostically usable. `caution`: inspect warnings before using as decision evidence. `high_risk`: treat as a **data/support problem**, not a model-ranking result. |

## Warning codes (`artifact.warnings`)

Each `(model, estimator)` row carries a list of warning codes. They are
diagnostic flags — one-line meanings and the action they imply:

| Code | What it means | What to do |
|---|---|---|
| `LOW_ESS` | Effective sample size is small; a few rows dominate. | Collect broader logs or reduce the gap between the candidate and logging policies. |
| `EXTREME_CLIP` | Importance weights had to be clipped hard to stay finite. | Overlap is thin; trust the value only as exploratory. Check `sensitivity`. |
| `POOR_OVERLAP` | The candidate policy puts mass where the logs have little/none. | This is a data problem. Improve logging exploration before comparing models. |
| `LOW_MATCH_RATE` | Few rows support the candidate's actions. | The policy is far from what was logged; the estimate rests on little data. |
| `HIGH_PARETO_K` | PSIS tail estimate above the high-risk threshold. | Importance weights are unstable; do not treat `V_hat` as deployment evidence. |
| `MISCAL_PROP` | The propensity model is poorly calibrated overall (high ECE/Brier). | Improve or recalibrate the propensity model. |
| `PER_ACTION_MISCAL` | Propensity calibration is poor for at least one specific action. | Inspect `diagnostics.per_action`; the worst action drives this. |
| `RARE_ACTION_NO_SUPPORT` | A target-support action has too few logged samples to estimate. | The candidate relies on actions the logs barely contain; gather more data. |

## Sensitivity metrics (`artifact.sensitivity`)

These summarize how the value moves as the clip threshold is swept across a
grid — the OPE equivalent of a robustness check.

| Metric | Plain meaning | How to read it |
|---|---|---|
| `V_min`, `V_max` | Smallest / largest `V_hat` over the clip grid. | A wide spread means the answer **depends on the clip choice** — a fragility signal. |
| `V_range` | `V_max - V_min`. | Large relative to the mean reward ⇒ the estimate is clip-sensitive; do not over-interpret a single `V_hat`. |
| `chosen_clip` | The clip threshold selected for the headline `V_hat`. | Cross-check against `argmin_MSE_clip`. |
| `argmin_MSE_clip` | The clip that minimized estimated MSE on the grid. | If it differs a lot from `chosen_clip`, the result is sensitive to the selection rule. |
| `dr_sndr_agree` | Whether DR and SNDR land close to each other. | Disagreement is a **red flag** — the two estimators should roughly agree when assumptions hold. |
| `stable` | Overall stability flag derived from the above. | `no` ⇒ treat the value as exploratory and improve support/calibration first. |

## Calibration & overlap diagnostics (`artifact.diagnostics`)

These describe the **propensity model** that the DR/SNDR estimators rely on.

| Metric | Plain meaning | How to read it |
|---|---|---|
| `ece` (ECE) | Expected Calibration Error of the propensity model. | **Lower is better.** High ECE ⇒ predicted probabilities don't match observed frequencies ⇒ `MISCAL_PROP`. |
| `brier_score` (Brier) | Mean squared error of the probabilistic propensity predictions. | **Lower is better.** A general accuracy + calibration measure. |
| `log_loss_score` | Log loss of the propensity model. | **Lower is better.** Penalizes confident wrong predictions heavily. |
| `overlap_ratio` | How much the candidate's action distribution overlaps the logged one. | **Higher is better.** Low overlap is the root cause of most `high_risk` results. |
| `balance_ratio` | Covariate balance between logged and target-action populations. | Far from `1.0` ⇒ the populations differ ⇒ extrapolation risk. |
| `calibration_score`, `discrimination_score` | Summary calibration / discrimination quality of the propensity model. | Higher discrimination + good calibration ⇒ more trustworthy weights. |

Per-action propensity diagnostics (`diagnostics.per_action`) break ECE / Brier
/ log-loss down by action and flag actions that are `rare` or `insufficient`;
these drive `PER_ACTION_MISCAL` and `RARE_ACTION_NO_SUPPORT`.

## Per-decision attribution

| Metric | Plain meaning | How to read it |
|---|---|---|
| `contribution_to_V` | Per-row contribution to `V_hat`, exposed when you pass `keep_contributions=True`. | `contribution_to_V.mean()` equals `V_hat` by construction. A handful of rows with huge contributions is the same fragility that `ESS` summarizes — inspect the top rows. |

## One worked reading

`support_health = high_risk` together with `LOW_ESS` and `POOR_OVERLAP` means:
**do not treat the value estimate as deployment evidence yet.** The logs do not
cover what the candidate policy would do often enough to estimate its value
reliably. The right next step is to improve logging/exploration (or evaluate a
candidate closer to the logged policy), not to pick the model with the highest
`V_hat`. Walk through the full reading order in
[`report-interpretation.md`](report-interpretation.md).
