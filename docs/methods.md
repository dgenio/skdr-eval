# skdr-eval: methods note (draft outline)

> Status: outline. After the Zenodo DOI is minted (see
> [`zenodo.md`](zenodo.md)), this note can be polished and deposited on
> arXiv as a 4–6 page methods note.

## 1. Problem setting

Off-policy evaluation (OPE) of a *target* decision policy `π(a | x)` from
*logged* decisions made under a different *behavior* policy `π_b(a | x)`,
where the action `a` is chosen for a context `x` and a reward `Y` is
observed. The goal is to estimate the policy value
`V(π) = E_x[ E_{a ∼ π(·|x)}[ E[Y | x, a] ] ]`
*without* deploying `π` to live traffic.

skdr-eval focuses on the practical subset of OPE that ships in production
ML pipelines: time-correlated logs, sklearn-compatible estimators,
propensities estimated from the same logs, and a deliverable HTML card a
PM can read.

## 2. Estimators

- **Doubly Robust (DR)** — Robins, Rotnitzky & Zhao (1994). Per-decision:
  `ψ_i = q̂(x_i, a_i^π) + w_i · (Y_i − q̂(x_i, a_i))` where
  `w_i = π(a_i | x_i) / π̂_b(a_i | x_i)` is the calibrated importance
  weight and `q̂` is the cross-fitted outcome model.
- **Stabilized DR (SNDR)** — self-normalized variant that rescales the
  residual term by `n / Σ_i w_i`. Lower variance when weights are large;
  small bias from the ratio estimator.

## 3. Trust-mode contributions

`skdr-eval` does not invent new estimators. It adds, *around* the
classical DR/SNDR machinery, a small toolkit that survives a real-world
deployment review:

- **Time-aware cross-fitting** with a configurable `gap` between train
  and test folds (default `gap=1`), to defuse adjacent-row leakage.
- **Calibrated propensities** (`fit_propensity_timecal`) — isotonic /
  Platt calibration on a rolling validation slice; ECE and Brier scores
  exposed on the artifact.
- **PSIS Pareto-k support-health diagnostic** — Vehtari, Simpson, Gelman,
  Yao & Gabry (JMLR 2024). Generalized-Pareto shape parameter of the
  unclipped importance-weight tail; `k ≥ 0.5` is a caution, `k ≥ 0.7` is
  a high-risk gate.
- **Moving-block bootstrap CIs** that preserve time-series correlation
  structure; a coverage-probability simulation harness (PR #100 / issue
  #81) proves nominal coverage on iid, AR(1), and seasonal DGPs.
- **Stakeholder evaluation card** — single HTML page bundling `V̂`,
  CI, ESS, match_rate, clip-grid sparkline, support-health banner,
  Pareto-k, ECE, top contributors / detractors.
- **`EvaluationArtifact`** — typed (Pydantic v2) carrier for everything
  above; round-trips to JSON and HTML; versioned schema (`SCHEMA_VERSION`).

## 4. Relation to existing ecosystems

- **Open Bandit Pipeline (OBP)** — shares the OPE goal; OBP is broader
  (slate / cascade-DR / MIPS / pseudo-inverse out-of-the-box) but does
  not focus on calibrated propensities, time-aware folds, or the
  stakeholder card. `skdr-eval` does not currently ship slate estimators
  (issue #75); MIPS is tracked in #85.
- **SCOPE-RL** — full RL OPE / OPL toolkit. Different audience.
  `skdr-eval` deliberately stays in contextual-bandit territory.
- **banditml** — focused on the bandit *deployment* loop; complementary,
  not overlapping.

## 5. Reproducibility

- Synthetic DGPs (`make_synth_logs`, `make_pairwise_synth`) are seeded
  and ship with every install.
- Simulation proofs of statistical correctness live under
  `tests/test_*_simulation.py`; required by the project's
  `docs/agent-context/review-checklist.md` for any change to a
  statistical primitive.
- `examples/notebooks/` and `examples/use_cases/` are exercised in CI
  via `nbmake` and the existing `examples-smoke` job, so the
  reproducibility surface does not silently rot.

## 6. Open work

Tracked on the public issue tracker and grouped by milestone:

- **M2 surface** — Typer CLI (#89), `doctor` probe (#91),
  per-group/slice evaluation (#87), Pydantic `EvaluationCard` schema
  (#88), tracker protocol (#93).
- **M2 estimators** — MIPS (#85), MRDR / SWITCH-DR / DRos composable
  strategies (#86).
- **M3 ecosystem** — slate / top-K estimators (#75), benchmark harness
  (#94), OBP-style adapters (#35), public-dataset loaders (#70),
  LLM-reranker recipe (#95).

## References

- Robins, Rotnitzky & Zhao (1994). Estimation of regression coefficients
  when some regressors are not always observed. *JASA*.
- Dudík, Langford & Li (2011). Doubly Robust Policy Evaluation and
  Learning. *ICML*.
- Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto Smoothed
  Importance Sampling. *JMLR* 25:1–58.
- Naeini, Cooper & Hauskrecht (2015). Obtaining well calibrated
  probabilities using Bayesian binning. *AAAI*.
- Chernozhukov et al. (2018). Double / debiased machine learning for
  treatment and structural parameters. *Econometrics Journal*.
