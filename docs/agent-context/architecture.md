# Architectural Intent and Design Rationale

## Why Doubly Robust Estimation Matters
Doubly robust (DR) and stabilized doubly robust (SNDR) estimation are inherently complex algorithms. They are not "over-engineered" — they are the product of statistical research into bias-variance tradeoffs under model misspecification. Understand this before attempting "simplification," which may introduce severe bias or exploding variance.

## Why the Library Has Chosen This Boundary
- **Input Layer**: Standard data structures (Pandas DataFrames, NumPy arrays). This follows scikit-learn conventions and reduces friction for users.
- **Core Logic**: The cross-fitting and propensity-stabilization patterns are statistically necessary, not organizational preferences. See [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md) for what must not be altered.

## Why Complex-Looking Code Exists
Functions like propensity score clipping (e.g., \_clip_propensities\) may look like arbitrary heuristics. They are required for variance stabilization in DR/SNDR and are grounded in statistical theory, not in ad-hoc tuning.

*(For specific implementation constraints that enforce this design, consult [\docs/agent-context/invariants.md\](docs/agent-context/invariants.md).)*