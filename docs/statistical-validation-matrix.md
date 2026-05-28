# Statistical validation matrix

This page is the maintainer-level evidence map for the estimators
shipped in `skdr-eval`. It pairs each estimator with the assumptions
under which it is consistent, the simulation evidence we ship as a
proof, the known failure modes, and the tests that exercise them.

It is the structured complement to
[`docs/concepts/estimands-and-assumptions.md`](concepts/estimands-and-assumptions.md)
(prose) and [`docs/methods.md`](methods.md) (positioning).

## How to read this table

- **Assumptions**: what must be (approximately) true for the estimator
  to be unbiased. A tag in this column maps to a section of the
  estimand/assumptions doc.
- **Simulation evidence**: the test file that recovers a known
  ground-truth `V*` under a benign DGP. The repository contract
  (`.claude/CLAUDE.md` §2 / `docs/agent-context/review-checklist.md`)
  requires every statistical change to ship one of these.
- **Failure modes**: regimes where the estimator is known to fail, and
  the test that demonstrates that failure (or the warning code that
  fires).

## Estimator rows

| Estimator | Assumptions | Simulation (recovers V*) | Known failure modes | Warning/test surfacing the failure |
|---|---|---|---|---|
| **DR** (`dr_value_with_clip`) | Unconfoundedness; overlap; *one* of propensity OR outcome correct; bounded weight variance | `tests/test_estimator_recovery_simulation.py::test_dr_recovers_ground_truth` | (a) Both nuisances wrong; (b) overlap collapse → IPS leg blows up; (c) cross-fitting bypassed → optimistic bias | `tests/sim_studies/test_dr_misspecification.py`; warnings `POOR_OVERLAP`, `HIGH_PARETO_K`, `EXTREME_CLIP` |
| **SNDR** (`dr_value_with_clip`, SNDR strategy) | Same as DR; finite weight ratio | `tests/test_estimator_recovery_simulation.py::test_dr_recovers_ground_truth` (DR and SNDR are returned together by `dr_value_with_clip` and both checked in this test) | Ratio bias when `Σ w` is small (low ESS); under-coverage when q̂ is wildly wrong | `tests/sim_studies/test_overlap_failure.py`; `LOW_ESS` warning |
| **MRDR** (`build_strategy("MRDR")`) | Same as DR + weighted-MSE q̂ fit converges | `tests/test_estimator_recovery_simulation.py::test_mrdr_recovers_ground_truth` | Re-weighting amplifies tail decisions if `(π/e)^2` is heavy-tailed | Pareto-k of `(π/e)^2`-reweighted residuals; `HIGH_PARETO_K` |
| **SWITCH-DR** (`SwitchTauTransform`) | DR assumptions + threshold `τ` separates "trust IPS" from "trust q̂" | `tests/test_estimator_recovery_simulation.py::test_switch_dr_recovers_ground_truth` | All weights above `τ` → degenerates to direct method (DM bias); all below → identity → vanilla IPS | `tests/sim_studies/test_overlap_failure.py` (DM tail) |
| **DRos** (`DRosShrinkTransform`) | DR assumptions + shrinkage `λ` reduces tail variance at cost of small bias | `tests/test_estimator_recovery_simulation.py::test_dros_recovers_ground_truth` | `λ → 0` collapses to DM; `λ → ∞` recovers raw IPS variance | `EXTREME_CLIP` |
| **MIPS** (`MIPSTransform`) | DR assumptions on embedding space; **embedding sufficiency** | `tests/test_estimator_recovery_simulation.py::test_mips_recovers_when_embedding_sufficient` | Embedding loses action-specific reward signal → bias | `embedding_sufficiency_diagnostic()` + `INSUFFICIENT_EMBEDDING` warning (logged via `core.py`) |
| **Slate / Cascade-DR** (`slate_cascade_dr`) | Cascade click model holds; per-position propensities estimable | `tests/test_estimator_recovery_simulation.py::test_slate_estimators_recover_uniform_target` + `tests/test_estimator_recovery_simulation.py::test_slate_cascade_dr_lower_variance_than_ips` | Click model misspecified (e.g. PBM data scored as cascade) → bias; K-large slates with rare combinations | (tracked in #135 follow-up; manual inspection of `slate.SlateResult.diagnostics`) |
| **Slate / Pseudo-Inverse IPS** (`pseudo_inverse_ips`) | Linear scoring assumption; rank-K marginal propensities estimable | `tests/test_estimator_recovery_simulation.py::test_slate_estimators_recover_uniform_target` (covers all three slate estimators against a uniform target) | Non-linear position interactions (e.g. context-dependent diversity) | Same as Cascade-DR |
| **Slate / Reward-Interaction IPS** (`reward_interaction_ips`) | Cascade clicks; first-order interactions only | `tests/test_estimator_recovery_simulation.py::test_slate_estimators_recover_uniform_target` (covers all three slate estimators against a uniform target) | Higher-order reward interactions in slate | Same as Cascade-DR |

## Diagnostics rows

| Diagnostic | What it estimates | Simulation proof | Failure mode |
|---|---|---|---|
| **PSIS Pareto-k** (`psis_pareto_k`) | Tail shape of importance weights (GPD `k`) | `tests/test_diagnostics_trust.py::test_pareto_k_recovers_known_tail_simulation` | `n < 25` → unreliable; flagged via `_MIN_SAMPLES_PARETO_K` |
| **ECE** (`compute_propensity_ece`) | Expected Calibration Error of propensity model | `tests/test_diagnostics_trust.py::test_ece_zero_under_perfect_calibration_simulation` | Bin under-population at small `n` (`_MIN_SAMPLES_RELIABILITY = 30`) |
| **Per-action calibration** (`per_action_propensity_diagnostics`) | Per-action ECE + match rate + log-loss | `tests/test_per_action_diagnostics.py::test_per_action_ece_zero_under_perfect_calibration_simulation` + `tests/test_per_action_diagnostics.py::test_per_action_ece_detects_one_bad_arm_simulation` | Rare actions with `< _MIN_ACTION_COUNT_DISC` samples → reported as `INSUFFICIENT_ACTION` |
| **Moving-block bootstrap CI** (`block_bootstrap_ci`) | CI under temporal correlation | `tests/sim_studies/test_bootstrap_validity.py` + `skdr_eval._simulation.simulate_coverage` | iid bootstrap under temporal correlation → CI under-covers; very small `n` → unstable block length |
| **Decision stability** (`summarize_sensitivity` + `stability_grade`) | Whether the recommended decision flips across clipping / folds / estimators | `tests/test_stability_grade.py` | Single-axis stability hides multi-axis flips — surfaced via `stability_grade ∈ {stable, sensitive, unstable}` |

## Failure-mode tutorials

The repo ships three intentionally failing example scripts so users see
what a *bad* offline evaluation looks like before they ship one:

| Tutorial | Failure regime | What the user should learn |
|---|---|---|
| [`examples/known_failures/poor_overlap.py`](../examples/known_failures/poor_overlap.py) | Logging policy is near-argmax; target policy disagrees | `support_health = high_risk`; do not trust V_hat |
| [`examples/known_failures/misspecified_q.py`](../examples/known_failures/misspecified_q.py) | Outcome model is severely under-fit | DR survives via the IPS leg; DM is biased — illustrates *double* robustness |
| [`examples/known_failures/non_stationary.py`](../examples/known_failures/non_stationary.py) | Reward distribution drifts between fold 1 and fold N | Moving-block bootstrap absorbs short-range dependence; long-range drift still under-covers — confirms the assumption boundary |

## Update policy

- **Adding an estimator**: add a row above with the assumption tags, a
  simulation-proof test file, and at least one named failure mode.
- **Adding a diagnostic**: add a row in the diagnostics table with the
  reference test file.
- **Adding a tutorial**: extend the tutorials table; do NOT add the
  script to `examples/use_cases/` (those are the *happy-path* gallery
  and must stay `support_health = ok`).
