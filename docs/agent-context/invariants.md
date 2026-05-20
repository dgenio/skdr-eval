# Hard Invariants and Forbidden Shortcuts

## Mathematical Invariants (Do Not "Simplify")
1. **Cross-fitting is Sacred**: Never remove or bypass cross-fitting loops to improve execution speed. Without cross-fitting, evaluating a model on the same data used to fit its nuisance parameters introduces severe bias.
2. **Propensity Clipping / Stabilization**: Do not remove the propensity score clipping/bounding logic (e.g., the `dr_value_with_clip` function and `w_clip` computation in `src/skdr_eval/core.py`). Reciprocals of near-zero propensities will cause variance to explode or result in `NaN`s. It may look like arbitrary data manipulation, but it is mathematically required.

## Implementation Guardrails
- **Thresholds**: Constants like `DIRECT_STRATEGY_THRESHOLD` exist to prevent Out-Of-Memory (OOM) errors during heavy dataframe operations. Do not blindly increase these limits or remove fallback strategies without explicit memory profiling.

## Strategy Seam (Issue #86)
- **Protocol-only seam**: `skdr_eval.estimators.WeightTransform` and `skdr_eval.estimators.OutcomeLoss` are the only sanctioned extension points for new DR-family estimators. Do not fork `dr_value_with_clip` to add a variant; build a `WeightTransform` (or `OutcomeLoss` for variance-reducing q̂ refits) and a named `EstimatorStrategy` instead.
- **Clip transform is the floor**: `ClipTransform` remains the default for DR/SNDR. Any new strategy that bypasses the propensity bound entirely (i.e. returns ``1/pi_obs`` with no cap) must document the variance trade-off and ship with a simulation proof that bounded variance is preserved under the target DGP.
- **MIPS bias**: MIPS is biased when the action embedding is not a sufficient statistic for the reward distribution. Code that exposes MIPS through `evaluate_sklearn_models` / `evaluate_pairwise_models` must also surface the `embedding_sufficiency_diagnostic` to the user (the standard `quickstart_mips.py` example shows the wiring).

## Domain Vocabulary (Hard Separation)
- **treatment**: The action taken in the *historical* data (actual logs).
- **policy**: The action recommended by the *model* currently being evaluated.
- Confusing these in variable names, parameter names, or logic will invert or completely invalidate evaluation results.

## Governance
- **Update Triggers**: If a new statistical stabilization method (e.g., a new clipping heuristic or bounds check) is added anywhere in `src/skdr_eval/`, it must be abstracted and listed here as a forbidden shortcut and rationale noted.
- **Changed Statistical Defaults**: Adding or changing the default value of a kwarg on any statistical entry point (`evaluate_sklearn_models`, `evaluate_pairwise_models`, `fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`, `dr_value_with_clip`) is a **numerical-behaviour change** even when the public signature stays backwards-compatible. Such a change requires (a) a `### Changed` entry in `CHANGELOG.md` naming the parameter, the old default, the new default, and the one-line revert; (b) a simulation proof that recovers a known ground-truth parameter under the new default; and (c) a corresponding pattern in [`lessons-learned.md`](./lessons-learned.md) if the design rationale generalises. The fold-gap default (`gap=1`, was implicit `gap=0` via sklearn's `TimeSeriesSplit`) is the canonical example — see PR #79.