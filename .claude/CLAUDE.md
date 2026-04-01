# Claude-Specific Operational Guidance

This file provides operational guidance for Claude agents working on \skdr-eval\.

## 1. Canonical Guidance
- **Primary Reference**: The canonical shared guidance for all agents is in [\AGENTS.md\](../../AGENTS.md).
- **Deep Context**: For architectural, workflow, and invariant rules, see [\docs/agent-context/\](../../docs/agent-context/).

## 2. Claude-Specific Rules
- **Statistical Integrity**: Claude agents must enforce all statistical invariants listed in [\docs/agent-context/invariants.md\](../../docs/agent-context/invariants.md).
- **Simulation Proofs**: Any change to statistical evaluation logic must include a simulation that proves the code recovers a known ground-truth parameter.

## 3. Workflow Integration
- **Makefile Authority**: Always use \Makefile\ targets for workflows. Do not run ad-hoc commands.
- **Validation**: Run \make validate\ before finalizing any PR.

## 4. Documentation Governance
- **Update Policy**: If code behavior changes, update the \examples/\ scripts first and the \README.md\ second to ensure they match.