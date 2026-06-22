<!--
Focused checklist for skdr-eval (a pure-Python statistics library). Delete the
sections that don't apply; keep the boxes you check honest.
-->

## What & why

<!-- One or two sentences: what this changes and the motivation. -->

Fixes #<!-- issue number, or "Related to #" / remove if none -->

**Type:** <!-- bug fix · feature · refactor · docs · tests · build/CI -->

## How verified

<!-- The exact commands you ran. `make ci-local` is the CI-faithful pass;
`make check` is the fast inner loop. -->

```bash
make ci-local   # or: make check  +  the targeted tests for this change
```

## Checklist

- [ ] Tests added/updated and passing locally (new behaviour is covered).
- [ ] `ruff check`, `ruff format`, and `mypy src/skdr_eval/` are clean
      (`make check`).
- [ ] Added a changelog fragment under `changelog.d/` (`<issue>.<type>.md`) —
      do **not** edit `CHANGELOG.md` directly (see `changelog.d/README.md`).
- [ ] Public API change? Recorded the name in `docs/api-stability.md`
      (a test enforces this) — or N/A.
- [ ] Docs/examples updated to match behaviour — or N/A.

## Statistical change? (delete if not)

A change to evaluation/estimator/gating logic (or a default flip on a
statistical entry point) must clear the bar in
[`docs/agent-context/review-checklist.md`](../docs/agent-context/review-checklist.md):

- [ ] **Simulation proof**: a test under `tests/sim_studies/` recovers a known
      ground-truth parameter (template in `CONTRIBUTING.md`).
- [ ] **Default flip**: any changed default on a statistical entry point has a
      `changed` changelog fragment calling it out.
- [ ] **Invariants**: re-checked the rules in
      [`docs/agent-context/invariants.md`](../docs/agent-context/invariants.md)
      (treatment vs policy, no math simplified without proof).
