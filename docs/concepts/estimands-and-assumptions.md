# Target estimand and statistical assumptions

`skdr-eval` is positioned as a *practitioner-first* offline policy
evaluation (OPE) toolkit. To use it safely, the target estimand and the
assumptions under which the estimator is valid must be explicit. This page
states them in the language used by the rest of the docs and by the
machine-readable card (`EvaluationCard.estimand`).

## 1. Target estimand

For a candidate (target) policy `π(a | x)` and observed contexts
`X ∼ P_X`, the estimand reported on every report row is the **policy
value**

```
V(π) = E_X [ E_{A ∼ π(·|X)} [ Y(X, A) ] ]
```

where `Y(x, a)` is the (potential) reward that would be observed had
action `a` been taken in context `x`.

- The expectation `E_X` is taken over the *evaluation* context
  distribution — the rows of the log slice that are *not* used to fit
  policies or nuisances (see `policy_train="pre_split"` /
  `policy_train_frac=0.85`).
- The estimand is **not** the long-run online value of `π` after it is
  deployed and changes the context distribution. OPE does not predict
  feedback loops.

The slate variants (`skdr_eval.slate`) target the slate-value analogue:

```
V_slate(π) = E_X [ E_{S ∼ π(·|X)} [ Σ_{k=1..K} r_k(X, S, k) ] ]
```

with `r_k` the position-conditional reward.

## 2. Standing assumptions

OPE results are only as good as the assumptions below. The
`EvaluationArtifact.estimand` block lists them on every artifact so they
travel with the report.

1. **No unmeasured confounding (NUC) / unconfoundedness**:
   `Y(x, a) ⫫ A | X` under the logging policy. Equivalently, the
   logged propensities `e(a | x)` capture every variable that *both*
   drives the decision *and* drives the outcome.

2. **Overlap / positivity**: for every `(x, a)` that the target policy
   `π` puts non-trivial mass on, the logging policy must also put
   strictly positive mass: `π(a | x) > 0 ⇒ e(a | x) > 0`. `skdr-eval`
   surfaces overlap quality via `min_pscore`, `match_rate`, PSIS
   Pareto-k, and the `POOR_OVERLAP` / `HIGH_PARETO_K` warnings.

3. **Stable Unit Treatment Value Assumption (SUTVA)**: one unit's
   action does not change another unit's outcome. This is *not* the
   right tool when actions interact (network effects, marketplace
   spillovers).

4. **Correctly specified propensity OR outcome model (double
   robustness)**: DR / SNDR are consistent if *either* the propensity
   model OR the outcome model is correctly specified. Both wrong → bias.
   `skdr-eval` calibrates propensities (`fit_propensity_timecal`) and
   cross-fits the outcome model (`fit_outcome_crossfit`) to mitigate
   this.

5. **Logged decisions are non-deterministic where the target wants to
   act**: if the logging policy is argmax-deterministic, the IPS weight
   is `∞` everywhere `π` disagrees, and DR collapses to the direct
   method (which loses its DR consistency property). The
   `EXTREME_CLIP` warning fires when more than 5 % of weights hit the
   clip floor.

6. **Bounded variance of the importance weights**: the IPS / DR
   estimators are unbiased only if `E[w(X, A)] < ∞`. PSIS Pareto-k > 0.7
   is the gate `skdr-eval` uses to declare *variance does not exist*
   (Vehtari et al. 2024).

7. **Time structure is respected**: for time-correlated logs, the
   evaluator uses `TimeSeriesSplit` with a configurable `gap` (default
   `gap=1`) and the moving-block bootstrap. If the data violates
   stationarity beyond what the block length absorbs, the CI under-
   covers — see `docs/statistical-validation-matrix.md`.

## 3. What the card carries

Every `EvaluationCard` produced after card schema version `1.1.0`
includes an `estimand` block summarizing 1 and 2 in human-readable form
plus a machine-readable list of assumption tags
(`unconfoundedness`, `overlap`, `sutva`, `double_robustness`,
`stochastic_logging`, `bounded_weight_variance`,
`time_structure_respected`). The block is rendered at the top of the
HTML card so the assumptions travel with the headline.

## 4. When NOT to interpret `V_hat`

- The target policy chooses actions that are never (or almost never)
  taken under the logging policy — overlap is the binding constraint,
  not estimator choice.
- A subgroup, peak-hour slice, or rare-action arm shows
  `support_health = high_risk` while the global row is `ok`.
- The logs span a regime change (new product launch, traffic shift)
  and the train / eval slices straddle it.
- The user is trying to bound the *online* value of `π` after deployment
  rather than its retrospective counterfactual value.

In those settings the report's job is to **block deployment** rather
than to provide a deployable point estimate.

## 5. References

- Robins, J. M., Rotnitzky, A. & Zhao, L. P. (1994). *Estimation of
  regression coefficients when some regressors are not always
  observed.* JASA.
- Dudík, M., Langford, J. & Li, L. (2011). *Doubly robust policy
  evaluation and learning.* ICML.
- Kang, J. D. Y. & Schafer, J. L. (2007). *Demystifying double
  robustness.* Statistical Science.
- Owen, A. B. (2013). *Monte Carlo theory, methods and examples.*
  §9.4 ESS.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. (2024).
  *Pareto smoothed importance sampling.* JMLR.
