# Write your own estimator

`skdr-eval`'s estimator family is built on a strategy seam so a new
doubly-robust variant is a small dataclass, not a fork of the core. This guide
walks from a blank `WeightTransform` to a working, tested estimator. The
worked code lives in
[`examples/extending/custom_estimator.py`](https://github.com/dgenio/skdr-eval/blob/main/examples/extending/custom_estimator.py)
and is exercised by `tests/test_extending_example.py`, so it cannot rot.

Read the [architecture tour](../architecture.md) first if you have not — this
guide assumes you know where the estimator stage sits in the pipeline.

## The seam

An estimator is an `EstimatorStrategy`: a named pairing of two protocols.

- **`WeightTransform`** — maps the raw inverse-propensity weight `1/π` into the
  *working* importance weight used inside the DR pseudo-outcome. It receives a
  `TransformContext` (logging propensity `pi_obs`, the `matched` mask,
  `policy_probs`, observed actions `A`, eligibility, …) and returns a
  non-negative `(n,)` array with zeros on unmatched rows.
- **`OutcomeLoss`** — emits the per-sample weights for the outcome-model
  cross-fit. Constant weights recover ordinary MSE; `w²` recovers MRDR.

Both are *structural* protocols: you do not subclass anything, you just
implement `__call__`.

## Step 1 — implement the weight transform

Our example estimator, **SoftClipDR**, replaces the hard clip `min(1/π, c)`
with a smooth saturating transform `c · tanh(w / c)`:

```python
from dataclasses import dataclass
import numpy as np
from skdr_eval.estimators import TransformContext


@dataclass(frozen=True)
class SoftClipTransform:
    """Smooth saturating importance weight: c * tanh(w / c)."""

    clip: float = 10.0
    name: str = "soft_clip"

    def __call__(self, context: TransformContext) -> np.ndarray:
        n = context.pi_obs.shape[0]
        pi_target_obs = context.policy_probs[np.arange(n), context.A.astype(int)]
        w_raw = np.zeros_like(context.pi_obs, dtype=np.float64)
        safe = context.matched & (context.pi_obs > 0)
        w_raw[safe] = pi_target_obs[safe] / context.pi_obs[safe]
        return self.clip * np.tanh(w_raw / self.clip)
```

The two rules the protocol requires: **non-negative output** and **zeros on
unmatched rows** (where `matched` is `False` or `pi_obs == 0`). Everything
else is your estimator's design.

## Step 2 — pair it into a strategy

```python
from skdr_eval.estimators import EstimatorStrategy, MSEOutcomeLoss

strategy = EstimatorStrategy(
    name="SoftClipDR",                    # appears in the report 'estimator' column
    weight_transform=SoftClipTransform(clip=10.0),
    outcome_loss=MSEOutcomeLoss(),        # standard MSE cross-fit
    self_normalised=False,                # True would make it self-normalized
)
```

## Step 3 — run it through the estimator core

Feed the strategy the same nuisance matrices the high-level evaluator builds:

```python
import skdr_eval
from skdr_eval.estimators import dr_value_with_strategy
from sklearn.ensemble import HistGradientBoostingRegressor

logs, ops_all, _ = skdr_eval.make_synth_logs(n=4000, n_ops=4, seed=0)
design = skdr_eval.build_design(logs)
propensities, _ = skdr_eval.fit_propensity_timecal(
    design.X_phi, design.A, design.ts, n_splits=3, random_state=0
)
q_hat, _ = skdr_eval.fit_outcome_crossfit(
    design.X_obs, design.Y, n_splits=3, random_state=0
)
model = HistGradientBoostingRegressor(random_state=0).fit(design.X_obs, design.Y)
policy_probs = skdr_eval.induce_policy_from_sklearn(
    model, design.X_base, list(ops_all), design.elig
)

result = dr_value_with_strategy(
    propensities=propensities,
    policy_probs=policy_probs,
    Y=design.Y,
    q_hat=q_hat,
    A=design.A,
    elig=design.elig,
    strategy=strategy,
)
print(result.V_hat, result.SE_if, result.ESS)
```

That is a complete, working estimator — no library edit required.

## Step 4 (optional) — make it resolvable by name

To let `build_strategy("SoftClipDR")` (and therefore the high-level
evaluators) resolve your estimator, add a branch to
`skdr_eval.estimators.build_strategy` — a one-clause library change:

```python
# in src/skdr_eval/estimators/__init__.py, inside build_strategy(...)
if canonical == "SOFTCLIPDR":
    return EstimatorStrategy(
        name="SoftClipDR",
        weight_transform=SoftClipTransform(clip=clip),
        outcome_loss=MSEOutcomeLoss(),
        self_normalised=False,
    )
```

Keep the shipped estimator list curated: only add names you intend to support
and validate. A community estimator can live in `examples/` indefinitely
without being registered.

## Step 5 — the simulation proof (required for library inclusion)

Per `docs/agent-context/invariants.md`, **any change to estimation logic must
ship a simulation that recovers a known ground-truth value.** If you are
proposing your estimator for the library, add a proof under
`tests/sim_studies/` that constructs a DGP with a known policy value and
asserts your estimator recovers it within tolerance, mirroring the existing
estimator-recovery simulations. The bar is the same one the shipped DR/SNDR/
MRDR/SWITCH-DR/DRos estimators clear.

## Checklist for a library-grade estimator

- [ ] `WeightTransform` (and/or `OutcomeLoss`) implemented as a frozen
      dataclass with a `name`.
- [ ] Non-negative weights, zeros on unmatched rows.
- [ ] Registered in `build_strategy` (if it should resolve by name).
- [ ] Simulation proof in `tests/sim_studies/` recovering a known value.
- [ ] Glossary / methods entry and a row in
      [choosing an estimator](../choosing-an-estimator.md).
- [ ] Public names added to the [API inventory](../api-stability.md).

## See also

- [Architecture tour](../architecture.md)
- [Choosing an estimator](../choosing-an-estimator.md)
- [API stability & inventory](../api-stability.md)
