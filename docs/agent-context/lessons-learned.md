# Lessons Learned & Failure Patterns

*(This file captures reusable patterns and traps to avoid, not a timeline of one-off incidents.)*

## Workflow: Promoting Lessons
When agents or maintainers identify a recurring trap or misunderstanding in reviews/implementation, extract the underlying rule and record it here. If the rule is a hard operational boundary, move it directly to [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md).

## Promotion Criteria
A lesson is promoted here only when:
- It has recurred at least twice in code review or implementation, or
- It is systemic and affects the safety/correctness of the library

One-off incidents do not belong in durable guidance.

## Durable Patterns

### Pattern 1: Aspirational Documentation
Markdown-based documentation in this repository (such as \README.md\ and \DEVELOPMENT.md\) sometimes describes planned features or unverified processes, rather than facts grounded in code and CI.

**The Lesson**: When in doubt about an API call, workflow step, or process, verify against the source code (\src/\), the Makefile, the CI configuration, and the working scripts in \examples/\. Do not assume that a README example is correct or that a workflow described in narrative form is currently used.

**Implication for Docs**: After each code change, review whether \examples/\ scripts and readme examples still match the current API.