# Metrics glossary

A plain-language reference for the metrics and warning codes in an
`EvaluationArtifact` — across `artifact.report`, `artifact.warnings`,
`artifact.sensitivity`, `artifact.diagnostics`, and the HTML report and
stakeholder card (`artifact.to_html(...)` / `artifact.save_card(...)`). It
covers the headline fields plus the lower-level columns those summaries are
derived from. A few raw payload structures (e.g. the full per-action
diagnostics table) are described by shape rather than enumerated field by
field, and some `delta_*` / per-action columns live only in the
DataFrame/JSON — those are called out where they appear below.

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
| `delta_V_hat` | `V_hat` minus the `baseline=` you passed (a float, `"logged"`, or omitted). `delta_ci_lower` / `delta_ci_upper` accompany it when CIs are enabled. | The decision-relevant quantity: "how much better/worse than what we run today?". A delta CI that crosses zero means **no demonstrated improvement**. Available in `artifact.report` (DataFrame/JSON) only — the HTML report and stakeholder card don't render the `delta_*` columns, so read them programmatically. |
| `clip` | The importance-weight clipping threshold actually used for that row. | Heavy clipping (small `clip`) trades variance for bias. If `clip` is driving the result, see `sensitivity`. |
| `ESS` | Effective sample size — roughly how many decisions actually carry the estimate after importance weighting. | Low `ESS` (relative to `n`) means **a few rows dominate** the estimate; it is fragile. Common failure: `ESS` of a few dozen on tens of thousands of rows. Action: collect broader logs or reduce policy shift; expect a `LOW_ESS` warning. |
| `match_rate` | Fraction of rows where the candidate policy's action is supported by the logged action distribution. | Higher is better. A `match_rate` of exactly `1.0` across *different* candidate models is suspicious — it usually means the candidates collapse to the logged action. Action: verify the policies actually differ. |
| `min_pscore` | Smallest logging-policy propensity encountered. | Values at the floor (e.g. `1e-8`) signal **near-deterministic logging** — the overlap the estimator needs is missing. Action: this drives `EXTREME_CLIP` / `POOR_OVERLAP`. |
| `pscore_q10`, `pscore_q05`, `pscore_q01` | 10th / 5th / 1st percentiles of the logging-policy propensities on matched rows. | The low tail of the overlap distribution — the same thinness `min_pscore` flags, but as a spread. Percentiles near the propensity floor mean the worst-supported decisions are nearly deterministic in the logs. Action: low values foreshadow `EXTREME_CLIP` / `POOR_OVERLAP`. |
| `tail_mass` | Fraction of matched rows whose importance weight was clipped to zero (the zero-weight fraction). | Higher means more of the candidate's mass lands where the logs give no support. Above `extreme_clip_tail_mass` (default `0.05`) it triggers `EXTREME_CLIP`; above 2× the threshold it is the high-risk band. Action: treat the value as exploratory and check `sensitivity`. |
| `MSE_est` | Estimated mean-squared error of the estimator at the chosen clip. | A selection diagnostic, not a headline number: it is what `argmin_MSE_clip` minimizes over the grid. Lower is better. |
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

## Recommendation verdicts (`artifact.recommendation`)

`recommendation(model, estimator=...).verdict` (also surfaced as the card's
`trust.recommendation`) is the one-word decision. There are exactly four
values:

| Verdict | What it means | What to do |
|---|---|---|
| `deploy` | CI clears the baseline with no caution/high-risk flags. | Strong offline evidence; proceed to a confirmatory A/B test. |
| `ab_test` | Promising but not clean — a caution flag is present, or the CI overlaps the baseline. | Treat as exploratory; design a guarded online experiment. |
| `insufficient_evidence` | The logs can't decide yet — usually **no bootstrap CI** was computed. | Re-run with `ci_bootstrap=True` (CLI: `--ci-bootstrap`). |
| `do_not_deploy` | A high-risk diagnostic fired (`POOR_OVERLAP`, `HIGH_PARETO_K`, `EXTREME_CLIP`). | Do not deploy; fix the data/overlap problem first. |

The `skdr-eval` CLI maps these to stable exit codes for CI gates: `do_not_deploy`
→ `3` (takes precedence), `insufficient_evidence` → `4`, and `deploy`/`ab_test`
→ `0`. See [report interpretation](report-interpretation.md#deployment-verdicts).

## Sensitivity metrics (`artifact.sensitivity`)

These summarize how the value moves as the clip threshold is swept across a
grid — the OPE equivalent of a robustness check.

| Metric | Plain meaning | How to read it |
|---|---|---|
| `V_min`, `V_max` | Smallest / largest `V_hat` over the clip grid. | A wide spread means the answer **depends on the clip choice** — a fragility signal. |
| `V_range` | `V_max - V_min`. | Large relative to the mean reward ⇒ the estimate is clip-sensitive; do not over-interpret a single `V_hat`. |
| `chosen_clip` | The clip threshold selected for the headline `V_hat`. | Cross-check against `argmin_MSE_clip`. |
| `chosen_V` | The `V_hat` at `chosen_clip` — the value this sensitivity row was built around. | Should match the `V_hat` in `report`. It is the denominator for `v_range_frac`. |
| `argmin_MSE_clip` | The clip that minimized estimated MSE on the grid. | If it differs a lot from `chosen_clip`, the result is sensitive to the selection rule. |
| `dr_sndr_agree` | Whether DR and SNDR land close to each other. | Disagreement is a **red flag** — the two estimators should roughly agree when assumptions hold. |
| `v_range_frac` | `V_range / max(\|chosen_V\|, eps)` — the clip-grid spread relative to the chosen value. | The scale-free version of `V_range`. `< 0.10` is the `stable` cut-off; larger means the answer depends on the clip choice. |
| `stable` | Overall stability flag derived from the above. | `no` ⇒ treat the value as exploratory and improve support/calibration first. |
| `stability_grade` | Three-band refinement of `stable`: `stable`, `sensitive`, or `unstable`. | `stable`: `v_range_frac < 10%` **and** DR/SNDR agree. `sensitive`: tight range but DR/SNDR disagree, **or** `10% ≤ v_range_frac < 25%`. `unstable`: `≥ 25%`, or non-finite (e.g. `chosen_V = 0`). Read `dr_sndr_agree` alongside it to tell the two `sensitive` cases apart. |

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
| `reliability_curve` | Calibration curve as a list of `(mean_predicted, observed_frequency, count)` bins. | Each bin compares predicted vs observed propensity; large gaps are the miscalibration `ece` summarizes into one number. Rendered as the calibration plot in the stakeholder card. |
| `ece_n_bins` | Number of bins used to compute `ece` and the reliability curve (default `15`). | A reporting parameter, not a quality signal — it sets the granularity of the calibration check. |

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
