# skdr-eval roadmap

This roadmap is intentionally narrow. GitHub issues contain historical ideas,
investigations and deferred work; an open or closed issue is not automatically a
current product commitment.

The current objective is to earn a defensible validated-v1 claim and prove that
real practitioners have the required data/workflow before expanding breadth.

## NOW — validated-v1 evidence path

These are the critical path. Feature breadth does not outrank them.

1. **Freeze the supported claim/envelope** — #297.
2. **Explicit target policy/action distribution** — #279.
3. **Genuine OOF nuisance predictions and explicit evaluation population** — #280.
4. **Logged behavior propensities for the initial validated path** — #167.
5. **Correct multi-action DR/SNDR outcome semantics** — #58.
6. **Estimator-specific independent/reference agreement** — #281.
7. **End-to-end inference + false-reassurance validation** — #62.
8. **Fail-closed diagnostic/evidence states** — #259.
9. **Evidence-status semantics, not deploy authorization** — #245.
10. **Implementation maturity separate from run evidence quality** — #282.
11. **Release validation and historical-correctness communication** — #295 / #300.

No estimator/configuration becomes strongly validated merely because its code is
merged. Promotion requires the validation evidence defined by the claim boundary.

## NEXT — prove usefulness outside the maintainer loop

Work here runs in parallel only where it does not distract from correctness.

- **Public reproducible validation lab** — #223.
- **Independent human methodological review** — #298.
- **Data-ready design-partner / market validation** — #299.
- **Stable evidence artifact/provenance surface** — #202 / #205 / #212.
- **Determinism on the validated public surface** — #232.
- **Public repository/adoption surface cleanup** — #301.
- **Decide whether the project name still fits before 1.0** — #310.

The important adoption signals are real evaluation decisions, repeat use,
independent review and external contributions — not raw star count.

## DEFERRED BY DEFAULT

These ideas may be useful later, but they are not current commitments unless
repeated external demand or validation needs pull them forward:

- new estimator families / broad estimator breadth;
- estimated-propensity promotion beyond the initial logged-propensity path;
- sequential/offline-RL workflows;
- slate/top-K expansion beyond separately validated experimental work;
- multi-objective policy evaluation;
- rolling/post-deploy monitoring;
- A/B power planning;
- interactive report workspaces;
- LLM-generated summaries and badges;
- plugin ecosystems;
- broad MLflow/W&B/Aim integrations;
- general LLM/agent-routing positioning;
- broad performance/scale work that is not a demonstrated validation or adoption
  bottleneck.

Good ideas can remain documented without consuming active-roadmap attention.

## Product decisions that can change the roadmap

The strategy is deliberately falsifiable.

- If the statistical/evidence-health validation fails, **narrow or redesign**;
  do not weaken the validation bar.
- If target teams usually lack behavior propensities/action provenance, consider
  an **OPE-readiness/instrumentation** pivot.
- If teams trust established estimator backends but value skdr-eval's evidence
  workflow, make the **evidence layer over external backends** the product.
- If data-ready teams do not use the artifact in real decisions, reconsider the
  artifact/product thesis rather than adding more presentation features.

See `CLAIMS.md` and `docs/strategy/evidence-first-v1.md` for the full claim,
validation ladder, kill criteria and effort allocation.
