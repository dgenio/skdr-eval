# Code Review Instructions for skdr-eval

This file governs code review behavior, especially for GitHub Copilot and AI agents. For broader context and deeper guidance, see [\AGENTS.md\](../../AGENTS.md) and [\docs/agent-context/\](../../docs/agent-context/).

## Review-Critical Rules

### 1. Statistical Integrity is Non-Negotiable
- **Never accept PRs that remove or bypass cross-fitting, propensity stabilization, or other statistical safeguards**, even if justified by performance or simplification claims.
- **Any change to statistical evaluation logic MUST include a simulation that proves the code recovers a known ground-truth parameter.** This is the definition of done.
- See [\.github/instructions/src.instructions.md\](./instructions/src.instructions.md) for specific guardrails.

### 2. Code and Documentation Must Be Reviewed Together
- If a PR modifies a public API or function signature, verify that:
  - The \examples/\ scripts are updated to match.
  - The \README.md\ quick-start example is current.
  - See [\docs/agent-context/review-checklist.md\](../../docs/agent-context/review-checklist.md).

### 3. Workflow, Invariant, and Architecture Changes Trigger Documentation Review
- If a PR modifies CI/CD (\.github/workflows/\, \pyproject.toml\), the \Makefile\ and \scripts/validate_contribution.py\ must stay in sync.