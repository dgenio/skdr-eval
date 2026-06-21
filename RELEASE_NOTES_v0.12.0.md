# skdr-eval v0.12.0 Release Notes

**Decision-Layer & CLI-First Reliability Release — Gated Verdicts, Explainability, and All-Estimator CI Coverage**

This release hardens the deployment decision layer and expands the CLI surface: `insufficient_evidence` is now a first-class gated verdict with its own exit code, the decision engine is cleanly extracted into `skdr_eval.recommendation`, the deploy-gate audits **all** estimators (not just DR/SNDR), the conditional-logit RNG is isolated for concurrent-safe fitting, and the CLI gains `explain`, `quickstart`, and `capabilities` commands.

**There are no breaking API changes.** All existing import paths continue to work.

---

## New Features

### CLI-first diagnostics & verdict explainability (#164, #201, #207, #215, #246)

Four new CLI subcommands and a stronger `doctor`:

- **`skdr-eval explain artifact.json --model M [--estimator SNDR] [--json]`** — narrates *why* a saved artifact got its verdict without re-running the evaluation. Each gating reason is shown with its measured value and threshold.
- **`skdr-eval quickstart`** — runs synthetic logs → `doctor` → `evaluate` → card → `explain` in one command. An onboarding demo that always exits `0` on success.
- **`skdr-eval capabilities`** — reports which optional extras (`viz`/`speed`/`cli`/`boosting`/`mlflow`/`wandb`/`aim`) are installed and what each unlocks.
- **`doctor` enhancements** — time-ordering and column-missingness checks, plus a privacy-safe `DataProfile` (column names/dtypes/shape only, never cell values).
- **`skdr-eval doctor --repro`** — emits a data-free minimal-reproduction snippet to attach to bug reports.

### `insufficient_evidence` as first-class gated verdict (#197)

The CLI now returns exit code `4` when every estimator's verdict is either `deploy` or `insufficient_evidence`, and at least one is `insufficient_evidence`. Previously, "the logs can't decide" silently passed a CI gate as green (exit `0`).

| Exit | Meaning |
|------|---------|
| `0` | All estimators recommend `deploy` |
| `3` | At least one estimator says `do_not_deploy` |
| `4` | No `do_not_deploy`, but at least one says `insufficient_evidence` |
| `5` | Evaluation error or unclassifiable state |

### `skdr_eval.recommendation` module (#235)

`Recommendation`, `RecommendationPolicy`, `Reason`, `GateResult`, `DiagnosticGate`, and `gate_diagnostics` are now in their own module, extracted from `reporting.py`. Existing imports (`skdr_eval.*` and `skdr_eval.reporting.*`) continue to work unchanged.

### Practitioner guides and API stability policy (#192)

Three new docs pages:
- `docs/getting-started/practitioner-guide.md` — real-world workflow from logs to deploy decision
- `docs/concepts/api-stability.md` — what "stable" means for public symbols
- `docs/concepts/estimator-extension.md` — how to add a new estimator strategy

---

## Fixed

### CLI deploy-gate now covers every estimator (#196)

**Critical:** `_verdict_exit_code` previously inspected only `DR`/`SNDR` report rows and swallowed recommendation errors with `except Exception: continue`. A `do_not_deploy` from MRDR, SWITCH-DR, DRos, or MIPS produced a **false-green exit `0`**.

The gate now scans **all** estimators present in the artifact and logs (rather than silently dropping) any recommendation error.

> **Behaviour change:** CI pipelines that previously passed because a non-DR/SNDR `do_not_deploy` was ignored will now correctly fail with exit `3`.

### Conditional-logit RNG isolated from global NumPy state (#193)

`fit_conditional_logit`, `sample_negative_pairs`, and `fit_conditional_logit_with_sampling` previously called `np.random.seed(...)`, silently reseeding a caller's RNG and breaking concurrent evaluations.

They now use a local `np.random.default_rng(...)`. `random_state` accepts `int`, `np.random.Generator`, or `None`. Determinism is preserved for integer seeds; only the exact initialization draw sequence shifts at the last digits.

---

## Upgrade Notes

- No API changes. All pre-v0.12.0 code continues to run unchanged.
- If you pin numeric test snapshots from `fit_conditional_logit` calls with a fixed integer `random_state`, expect last-digit shifts (the weight matrix is identical; only the sampling order on tie-breaking differs).
- If you run `skdr-eval` in CI with MRDR / SWITCH-DR / DRos / MIPS and previously saw green exit `0` on `do_not_deploy`, your pipeline will now correctly fail. This is the intended fix.
