# Source-Specific Instructions for skdr-eval

This file provides scoped instructions for AI agents working specifically on `src/skdr_eval/`.

## 1. Statistical Invariants
- **Cross-fitting**: Never remove or bypass cross-fitting loops. Without cross-fitting, evaluating a model on the same data used to fit its nuisance parameters introduces severe bias.
- **Propensity Clipping**: Do not remove or "simplify" the propensity score clipping logic. Reciprocals of near-zero propensities will cause variance to explode or result in `NaN`s.

## 2. Code Review Guardrails
- **Simulation Proofs**: Any change to statistical evaluation logic must include a simulation that proves the code recovers a known ground-truth parameter.
- **Invariant Checks**: Ensure that cross-fitting and propensity stabilization are not bypassed. See [`docs/agent-context/invariants.md`](../../docs/agent-context/invariants.md).

## 3. Documentation Governance
- **API Sync**: If a public function signature changes, update the `examples/` scripts and `README.md` quick-start example to match.
- **Workflow Sync**: If CI/CD or toolchains change, ensure the `Makefile` and `scripts/validate_contribution.py` are updated to match.