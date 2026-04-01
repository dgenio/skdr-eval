# Workflows and Authoritative Commands

## Single Source of Truth
The \Makefile\ is the canonical source for all developer commands. Do not run ad-hoc \pytest\, \ruff\, or \mypy\ commands in isolation when a \Make\ target exists. Do not trust descriptive markdown text (\DEVELOPMENT.md\) if it contradicts the \Makefile\.

## Core Commands
- **Full PR Check**: \make validate\ (runs \scripts/validate_contribution.py\). This is the authoritative pre-submission check.
- **Install (Dev)**: \make install-dev\
- **Test with Coverage**: \make test-cov\

## Standard Tasks
### Adding a Dependency
1. Add the package to \pyproject.toml\ (under \dependencies\ or \optional-dependencies.dev\ as appropriate).
2. Run \make install-dev\ to rebuild the environment.

## Documentation Governance for Workflows
- **Update Triggers**: If CI/CD steps in \.github/workflows/ci.yml\ change, you MUST update the corresponding \Makefile\ targets and/or \scripts/validate_contribution.py\ to match. The local \Make\ experience must mirror CI.