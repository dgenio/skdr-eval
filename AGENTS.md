# Agent Instructions: skdr-eval

This is the primary routing and canonical guidance file for AI/coding agents interacting with \skdr-eval\.

## 1. Purpose and Architectural Intent
\skdr-eval\ is a library for doubly robust (DR) and stabilized doubly robust (SNDR) evaluation of machine learning models.
**Core Intent:** Statistical Correctness > Performance/Simplicity. Never compromise the math for execution speed. See [\docs/agent-context/architecture.md\](docs/agent-context/architecture.md).

## 2. Practical Repo Map
- \src/skdr_eval/\: Core evaluation logic. Contains path-specific statistical invariants.
- \examples/\: **Most trustworthy usage guide.** Script examples here are more reliable than markdown docs.
- \scripts/\: Internal tools, notably \validate_contribution.py\.
- \tests/\: Source of truth for edge cases and feature support.
- \docs/agent-context/\: Deep context and architectural rules for agents.

## 3. Domain Vocabulary
- **\treatment\**: The action taken in the *historical* data logs.
- **\policy\**: The action recommended by the *model* currently being evaluated.
*Warning: Confusing these will invert or invalidate evaluation results.*

## 4. Invariants & Pitfalls
- **Trust Code > Docs**: Markdown files (like \README.md\ or \DEVELOPMENT.md\) are often aspirational and drift from reality. Always verify APIs against \src/\ type signatures and \examples/\.
- Do not "simplify" complex logic without mathematical proof. See [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md).

## 5. Preferred Workflows & Commands
- The \Makefile\ is the absolute authority for workflows. Do not run ad-hoc tool commands.
- Run \make validate\ before finalizing any PR.
- See [\docs/agent-context/workflows.md\](docs/agent-context/workflows.md).

## 6. Definition of Done
- Any change to statistical logic REQUIRES a simulation-based proof recovering a ground-truth parameter.
- See [\docs/agent-context/review-checklist.md\](docs/agent-context/review-checklist.md).

## 7. Documentation Governance & Map
- **Update Policy**: If code behavior changes, update the \examples/\ scripts first and the \README.md\ second to ensure they match. 
- For specific topic rules, consult the canonical files:
  - Architecture: [\docs/agent-context/architecture.md\](docs/agent-context/architecture.md)
  - Workflows: [\docs/agent-context/workflows.md\](docs/agent-context/workflows.md)
  - Constraints: [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md)
  - Pattern Capture: [\docs/agent-context/lessons-learned.md\](docs/agent-context/lessons-learned.md)
  - PR/Review: [\docs/agent-context/review-checklist.md\](docs/agent-context/review-checklist.md)