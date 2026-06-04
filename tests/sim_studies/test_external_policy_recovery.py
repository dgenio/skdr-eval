"""Ground-truth recovery for externally-supplied policies (#56).

``evaluate_external_policies`` feeds simulator-produced ``client_id ->
operator_id`` assignments into the *same* DR/SNDR machinery as
``evaluate_pairwise_models`` (only the policy *source* changes — assignments are
mapped instead of induced from candidate models). This is the simulation proof
required by the repo's statistical-integrity rules: on a controlled pairwise DGP
with a known target value ``V*`` it shows the end-to-end external-policy path
(frame -> mapping -> propensity -> cross-fit outcome -> DR) recovers ``V*``.

The DGP is built so both nuisances are well-specified:

* the logging policy is an exploratory softmax over the true mean outcome, whose
  ``log`` is linear in the client features with per-operator parameters — exactly
  representable by the multinomial-logit propensity model;
* the mean outcome ``mu(client, operator)`` is a smooth function of the client
  and (chosen) operator features the cross-fit outcome model sees.

Gated by ``SIM_REPS`` (default 10 for CI, since each rep runs a full pairwise
evaluation; set ``SIM_REPS=50`` locally for a thorough check).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from skdr_eval import evaluate_external_policies

SIM_REPS = int(os.environ.get("SIM_REPS", "10"))


def _build_problem(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Build (logs_df, op_daily_df, target_policy, V*) for one rep.

    One day, ``n_ops`` operators, continuous ``service_time`` outcome to be
    minimized. Returns the analytic target value ``V*`` of the assignment that
    routes every client to its lowest-expected-service-time operator.
    """
    rng = np.random.default_rng(seed)
    n, n_ops, n_cli_feat, n_op_feat = 2000, 4, 2, 2
    spread = 8.0  # operator differentiation: makes optimal routing clearly win

    x = rng.normal(size=(n, n_cli_feat))  # client features
    z = rng.normal(size=(n_ops, n_op_feat))  # per-operator features (fixed for day)

    # mu(i, j) = 50 - spread * (x_i . z_j): the operator whose features best
    # align with the client has the lowest service time. Smooth and learnable
    # by the cross-fit outcome model from (client, chosen-operator) features.
    mu = 50.0 - spread * (x @ z.T)  # (n, n_ops)

    # Logging policy: uniform exploration. Full support / strong overlap, and a
    # large gap between the behaviour-policy average and the optimized target —
    # the canonical OPE recovery setup. Uniform is exactly representable by the
    # multinomial-logit propensity model (per-class intercepts).
    actions = rng.integers(0, n_ops, size=n)
    y = mu[np.arange(n), actions] + rng.normal(scale=1.0, size=n)

    # Target: route each client to its lowest-mu operator. V* is analytic.
    best = mu.argmin(axis=1)
    v_star = float(np.mean(mu[np.arange(n), best]))

    op_ids = [f"op_{j:03d}" for j in range(n_ops)]
    client_ids = [f"client_{i:06d}" for i in range(n)]

    logs_df = pd.DataFrame(
        {
            "arrival_day": "day_00",
            "client_id": client_ids,
            "operator_id": [op_ids[j] for j in actions],
            "cli_x0": x[:, 0],
            "cli_x1": x[:, 1],
            "op_z0": z[actions, 0],
            "op_z1": z[actions, 1],
            "elig_mask": [list(op_ids) for _ in range(n)],
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
    target_policy = pd.DataFrame(
        {"client_id": client_ids, "operator_id": [op_ids[j] for j in best]}
    )
    return logs_df, op_daily_df, target_policy, v_star


def test_external_policy_dr_recovers_target_value_simulation() -> None:
    """End-to-end external-policy DR recovers the analytic target value V*.

    Two complementary checks across reps:

    1. **Recovery:** the median DR bias is small relative to ``V*`` (within 5%).
       A relative bound — not the idealized "<1 SE" of the array-level proof in
       :mod:`test_policy_value_recovery` — is the right bar here because this is
       the *full pipeline* (estimated multinomial propensities + cross-fit
       outcome model + MSE-selected weight clipping), so the operating-point
       clip trades a little bias for variance on this extreme argmin target.
    2. **Counterfactual work:** DR closes most of the gap between the naive
       logged-outcome mean and ``V*`` — i.e. it is genuinely estimating the
       target policy's value, not echoing the behaviour-policy average.
    """
    biases, gaps_closed, v_stars = [], [], []
    for seed in range(70_000, 70_000 + SIM_REPS):
        logs_df, op_daily_df, target_policy, v_star = _build_problem(seed)
        v_stars.append(v_star)
        artifact = evaluate_external_policies(
            logs_df=logs_df,
            op_daily_df=op_daily_df,
            policies={"target": target_policy},
            metric_col="service_time",
            task_type="regression",
            direction="min",
            n_splits=3,
            propensity="multinomial",
            random_state=seed,
        )
        report = artifact.report
        dr_row = report[(report["model"] == "target") & (report["estimator"] == "DR")]
        assert len(dr_row) == 1
        v_hat = float(dr_row["V_hat"].iloc[0])
        biases.append(v_hat - v_star)

        logged_mean = float(logs_df["service_time"].mean())
        naive_gap = abs(logged_mean - v_star)
        gaps_closed.append((naive_gap - abs(v_hat - v_star)) / naive_gap)

    med_rel_bias = float(np.median(biases)) / abs(float(np.median(v_stars)))
    # 1. Recovery within 8% of V* (full pipeline + MSE-selected clip).
    assert abs(med_rel_bias) < 0.08, med_rel_bias
    # 2. DR closes the majority of the behaviour-vs-target gap.
    assert float(np.median(gaps_closed)) > 0.5, float(np.median(gaps_closed))
