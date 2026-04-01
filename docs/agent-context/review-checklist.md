# Review Configuration and Definition of Done

Use this checklist for self-validation before proposing changes, and for reviewing inbound Pull Requests.

## Statistical & Logic Changes
- [ ] **Simulation Proof**: Does the PR modify, add, or "optimize" any statistical evaluation logic? **REQUIREMENT**: The PR must include a simulation or test that mathematically proves the new logic accurately recovers a known ground-truth parameter.
- [ ] **Invariant Check**: Did this change bypass cross-fitting or propensity stabilization? (If yes, reject. See [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md).)

## Code Quality & CI Readiness
- [ ] **Validation**: Does \make validate\ pass locally with zero errors?

## Documentation Freshness & Governance
- [ ] **API Sync**: If a public function signature in \src/\ changed, are the scripts in \examples/\ updated?
- [ ] **README Sync**: If a public API changed, is the quickstart example in \README.md\ updated to match the working examples?
- [ ] **Workflow Sync**: If CI or toolchains changed, is the \Makefile\ updated?

## Governance
- **Update Triggers**: This checklist must be updated whenever the project adopts a new mandatory tool (e.g., an architectural linter) or experiences a severe regression that a new checklist item would have caught.