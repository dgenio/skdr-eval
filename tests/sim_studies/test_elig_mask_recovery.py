"""Eligibility-mask value-type recovery simulation (#155, #158).

The equivalence tests in ``tests/test_frame_inputs.py`` prove that ``list`` /
``set`` / ``np.ndarray`` eligibility masks agree, but agreement alone can hide a
wrong ``list`` path. This is the ground-truth recovery proof the repo's
statistical-integrity rules require for a change that alters eligibility
interpretation: on a controlled pairwise DGP with an analytic *constrained*
optimum ``V*_restricted`` (the value of routing each client to its best
**eligible** operator), the induced-policy DR estimate recovers
``V*_restricted`` — and does so identically whether the mask is a ``list``, a
``set``, or an ``np.ndarray``.

The DGP is deliberately eligibility-sensitive: each client's eligible set is a
strict random subset, so the unconstrained optimum ``V*_full`` (best operator
overall) is strictly better (lower service time) than ``V*_restricted``. An
evaluator that ignored eligibility — the pre-fix bug, which treated
``set``/``ndarray`` masks as "every operator eligible" — would induce the
*unconstrained* argmin policy and pull the estimate toward ``V*_full``. So
"closer to ``V*_restricted`` than to ``V*_full``" is simultaneously the recovery
signal and the guard against the bug (verified to fail when the normalization is
removed).

A near-oracle outcome model (the true ``mu``, exposed as a fitted sklearn
regressor) is used only to *induce* the policy, so the induced policy is exactly
"best eligible operator" with an analytic value; the DR estimate itself still
relies on the evaluator's own estimated propensities and cross-fit outcome
model.

Gated by ``SIM_REPS`` (default 10 for CI, since each rep runs a full pairwise
evaluation; set ``SIM_REPS=50`` locally for a thorough check).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from skdr_eval import evaluate_pairwise_models

SIM_REPS = int(os.environ.get("SIM_REPS", "10"))

# Outcome surface: mu(client, operator) = BASE - SPREAD * (x . z). SPREAD sets
# operator differentiation; large enough that constrained vs unconstrained
# routing are clearly different policies.
BASE = 50.0
SPREAD = 8.0
N = 2000
N_OPS = 5


class _OracleRegressor(RegressorMixin, BaseEstimator):
    """Exposes the true ``mu`` as a fitted regressor (induction only).

    The pairwise evaluator builds candidate-pair features as
    ``[cli_x0, cli_x1, op_z0, op_z1]``; this returns the exact mean outcome so
    the induced (argmin, ``direction="min"``) policy routes each client to its
    lowest-``mu`` **eligible** operator. ``__sklearn_is_fitted__`` lets it pass
    the evaluator's ``check_is_fitted`` guard under ``fit_models=False`` without
    a real fit step.
    """

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X: object, y: object = None) -> _OracleRegressor:
        return self

    def predict(self, X: object) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return BASE - SPREAD * (arr[:, 0] * arr[:, 2] + arr[:, 1] * arr[:, 3])


def _build_problem(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Build ``(logs_df, op_daily_df, V*_restricted, V*_full)`` for one rep.

    One day, ``N_OPS`` operators, continuous ``service_time`` to minimize.
    Each client gets a strict random eligible subset (each operator eligible
    w.p. 0.6, at least two), and is logged uniformly over that subset — so the
    excluded operators are genuinely chosen by *other* clients, making the
    propensity estimate sensitive to whether eligibility is honored.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 2))  # client features
    z = rng.normal(size=(N_OPS, 2))  # per-operator features (fixed for the day)
    mu = BASE - SPREAD * (x @ z.T)  # (N, N_OPS) mean outcome

    op_ids = [f"op_{j:03d}" for j in range(N_OPS)]
    client_ids = [f"client_{i:06d}" for i in range(N)]

    elig_bool = rng.random((N, N_OPS)) < 0.6
    for i in range(N):
        if elig_bool[i].sum() < 2:
            elig_bool[i, rng.choice(N_OPS, size=2, replace=False)] = True

    # Logging policy: uniform over each client's eligible set (full overlap on
    # the eligible support, the canonical OPE recovery setup).
    actions = np.array([rng.choice(np.flatnonzero(elig_bool[i])) for i in range(N)])
    y = mu[np.arange(N), actions] + rng.normal(scale=1.0, size=N)

    # Analytic optima. V*_restricted routes to the best *eligible* operator;
    # V*_full to the best operator overall (always at least as good → lower).
    masked = np.where(elig_bool, mu, np.inf)
    best_elig = masked.argmin(axis=1)
    v_star_restricted = float(mu[np.arange(N), best_elig].mean())
    v_star_full = float(mu[np.arange(N), mu.argmin(axis=1)].mean())

    elig_names = [[op_ids[j] for j in np.flatnonzero(elig_bool[i])] for i in range(N)]
    logs_df = pd.DataFrame(
        {
            "arrival_day": "day_00",
            "client_id": client_ids,
            "operator_id": [op_ids[j] for j in actions],
            "cli_x0": x[:, 0],
            "cli_x1": x[:, 1],
            "op_z0": z[actions, 0],
            "op_z1": z[actions, 1],
            "elig_mask": elig_names,
            "service_time": y,
        }
    )
    op_daily_df = pd.DataFrame(
        {
            "operator_id": op_ids,
            "arrival_day": "day_00",
            "op_z0": z[:, 0],
            "op_z1": z[:, 1],
        }
    )
    return logs_df, op_daily_df, v_star_restricted, v_star_full


def _v_hat(logs_df: pd.DataFrame, op_daily_df: pd.DataFrame, seed: int) -> float:
    artifact = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models={"oracle": _OracleRegressor()},
        metric_col="service_time",
        task_type="regression",
        direction="min",
        n_splits=3,
        fit_models=False,
        policy_train="all",
        propensity="multinomial",
        random_state=seed,
    )
    report = artifact.report
    row = report[(report["model"] == "oracle") & (report["estimator"] == "DR")]
    assert len(row) == 1
    return float(row["V_hat"].iloc[0])


@pytest.mark.slow
def test_elig_mask_recovers_constrained_value_across_value_types() -> None:
    """list/set/ndarray masks all recover the analytic constrained value V*.

    Three checks:

    1. **Eligibility sensitivity:** the DGP's constrained optimum is strictly
       worse than the unconstrained one, so eligibility genuinely matters.
    2. **Recovery (direction-robust):** the induced-policy DR estimate is closer
       to ``V*_restricted`` than to ``V*_full`` (an evaluator that dropped the
       mask would induce the unconstrained argmin and fail this), and the median
       relative bias vs ``V*_restricted`` is small.
    3. **Value-type equivalence:** ``list`` / ``set`` / ``np.ndarray`` masks
       produce a byte-identical estimate, so all three recover the same V*.
    """
    rel_biases: list[float] = []
    for rep, seed in enumerate(range(70_000, 70_000 + SIM_REPS)):
        logs_df, op_daily_df, v_restricted, v_full = _build_problem(seed)

        # 1. The DGP is eligibility-sensitive (restricting raises the optimum).
        assert v_restricted > v_full + 0.5

        v_list = _v_hat(logs_df, op_daily_df, seed)
        rel_biases.append((v_list - v_restricted) / abs(v_restricted))

        # 2. Recovery direction: closer to the constrained optimum than to the
        #    unconstrained one (the pre-fix bug pulled this toward V*_full).
        assert abs(v_list - v_restricted) < abs(v_list - v_full)

        # 3. Value-type equivalence (checked on the first rep to keep the
        #    simulation's wall-clock dominated by the recovery loop).
        if rep == 0:
            logs_set = logs_df.copy()
            logs_set["elig_mask"] = logs_set["elig_mask"].map(set)
            logs_arr = logs_df.copy()
            logs_arr["elig_mask"] = logs_arr["elig_mask"].map(np.asarray)

            v_set = _v_hat(logs_set, op_daily_df, seed)
            v_arr = _v_hat(logs_arr, op_daily_df, seed)
            assert v_set == v_list
            assert v_arr == v_list
            # Each value-type also recovers the constrained optimum.
            assert abs(v_set - v_restricted) < abs(v_set - v_full)
            assert abs(v_arr - v_restricted) < abs(v_arr - v_full)

    # Aggregate recovery: median bias small for an extreme argmin target under
    # the full estimated-propensity + cross-fit-outcome pipeline.
    assert abs(float(np.median(rel_biases))) < 0.15, float(np.median(rel_biases))
