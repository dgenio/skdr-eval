# Lessons Learned & Failure Patterns

*(This file captures reusable patterns and traps to avoid, not a timeline of one-off incidents.)*

## Workflow: Promoting Lessons
When agents or maintainers identify a recurring trap or misunderstanding in reviews/implementation, extract the underlying rule and record it here. If the rule is a hard operational boundary, move it directly to [`docs/agent-context/invariants.md`](./invariants.md).

## Promotion Criteria
A lesson is promoted here only when:
- It has recurred at least twice in code review or implementation, or
- It is systemic and affects the safety/correctness of the library

One-off incidents do not belong in durable guidance.

## Durable Patterns

### Pattern 1: Aspirational Documentation
Markdown-based documentation in this repository (such as `README.md` and `DEVELOPMENT.md`) sometimes describes planned features or unverified processes, rather than facts grounded in code and CI.

**The Lesson**: When in doubt about an API call, workflow step, or process, verify against the source code (`src/`), the Makefile, the CI configuration, and the working scripts in `examples/`. Do not assume that a README example is correct or that a workflow described in narrative form is currently used.

**Implication for Docs**: After each code change, review whether `examples/` scripts and readme examples still match the current API.

### Pattern 2: Statistical Defaults Are Behavior Changes
When a default value on a statistical entry point (`evaluate_sklearn_models`, `evaluate_pairwise_models`, `fit_propensity_timecal`, `fit_outcome_crossfit`, `estimate_propensity_pairwise`) is added or changed, callers who do not pass the parameter get a numerically different result on the same data and seed. This breaks reproducibility of prior baselines even when no public-API signature has changed.

**Concrete instance**: the fold-gap default switched from sklearn's `TimeSeriesSplit` implicit `gap=0` to `skdr_eval`'s `gap=1` (conservative adjacent-row leakage guard). Existing callers who do not pass `gap` now get a different fold layout and slightly different DR/SNDR estimates than they got before the change. The fix is documentation discipline, not a code rollback — the new default is statistically sounder.

**The Lesson**: Treat a default-value change on a statistical entry point as a **Changed** item in `CHANGELOG.md`, not as **Added**. The changelog entry must:

1. Name the parameter and the entry points it affects.
2. State the old default and the new default.
3. Give the exact one-line migration to recover the prior behaviour (e.g. "pass `gap=0` to restore the pre-PR fold layout").
4. State whether the simulation proof still recovers the ground-truth parameter under both old and new defaults, or only under the new one.

**The Lesson (reviewer-facing)**: When auditing a PR that adds a kwarg with a non-zero default to any of the above entry points, check that the changelog framing matches the rubric above. A bullet under **Added** for a defaults change hides the numerical-behaviour shift from upgraders.