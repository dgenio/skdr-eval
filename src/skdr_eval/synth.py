"""Synthetic data generator for offline policy evaluation."""

from typing import Tuple

import numpy as np
import pandas as pd


def make_synth_logs(
    n: int = 5000, n_ops: int = 5, seed: int = 0
) -> Tuple[pd.DataFrame, pd.Index, np.ndarray]:
    """Generate synthetic service logs for offline policy evaluation.

    Creates realistic synthetic data with:
    - Client features (cli_*) and service-time features (st_*)
    - Operator eligibility and actions
    - Service times with realistic dependencies
    - Time-ordered arrival timestamps

    Parameters
    ----------
    n : int, default=5000
        Number of log entries to generate.
    n_ops : int, default=5
        Number of operators in the system.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    logs : pd.DataFrame
        Synthetic log data with columns:
        - arrival_ts: timestamp of request arrival
        - cli_*: client features
        - st_*: service-time features
        - op_*_elig: eligibility indicators for each operator
        - action: chosen operator
        - service_time: observed service time
    ops_all : pd.Index
        All operator names in the system.
    true_q : np.ndarray
        True expected service times for each (context, operator) pair.

    Examples
    --------
    >>> logs, ops_all, true_q = make_synth_logs(n=1000, n_ops=3, seed=42)
    >>> print(logs.columns.tolist())
    ['arrival_ts', 'cli_urgency', 'cli_complexity', 'st_load', 'st_time_of_day',
     'op_A_elig', 'op_B_elig', 'op_C_elig', 'action', 'service_time']
    """
    rng = np.random.RandomState(seed)

    # Generate operator names
    ops_all = pd.Index([f"op_{chr(65 + i)}" for i in range(n_ops)])

    # Generate timestamps (sorted)
    base_time = pd.Timestamp("2024-01-01")
    time_deltas = rng.exponential(scale=60, size=n)  # minutes between arrivals
    arrival_ts = base_time + pd.to_timedelta(np.cumsum(time_deltas), unit="min")

    # Client features
    cli_urgency = rng.beta(2, 5, size=n)  # skewed toward lower urgency
    cli_complexity = rng.gamma(2, 2, size=n)  # complexity score

    # Service-time features
    st_load = rng.exponential(scale=1.0, size=n)  # system load
    st_time_of_day = np.sin(2 * np.pi * arrival_ts.hour / 24)  # time of day effect

    # Generate eligibility (each request has 2-4 eligible operators)
    eligibility = np.zeros((n, n_ops), dtype=bool)
    for i in range(n):
        n_eligible = rng.randint(2, min(n_ops + 1, 5))
        eligible_ops = rng.choice(n_ops, size=n_eligible, replace=False)
        eligibility[i, eligible_ops] = True

    # True service time model (ground truth)
    # Each operator has different efficiency for different types of requests
    op_efficiency = rng.uniform(0.5, 1.5, size=n_ops)
    op_urgency_sensitivity = rng.uniform(0.8, 1.2, size=n_ops)
    op_complexity_penalty = rng.uniform(1.0, 2.0, size=n_ops)

    # Compute true expected service times
    true_q = np.zeros((n, n_ops))
    for j in range(n_ops):
        base_time = 10.0 * op_efficiency[j]  # base service time
        urgency_effect = cli_urgency * 5.0 * op_urgency_sensitivity[j]
        complexity_effect = cli_complexity * 3.0 * op_complexity_penalty[j]
        load_effect = st_load * 2.0
        time_effect = st_time_of_day * 1.5  # night shift slower
        
        true_q[:, j] = (
            base_time + urgency_effect + complexity_effect + load_effect + time_effect
        )
        # Set ineligible operators to very high service time
        true_q[~eligibility[:, j], j] = 1000.0

    # Generate actions using a realistic policy
    # Policy tends to choose operators with lower expected service time + some noise
    action_probs = np.zeros((n, n_ops))
    for i in range(n):
        eligible_mask = eligibility[i]
        if eligible_mask.sum() == 0:
            continue
        
        # Softmax over negative service times (prefer lower service time)
        eligible_q = true_q[i, eligible_mask]
        logits = -eligible_q / 2.0  # temperature = 2.0
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        action_probs[i, eligible_mask] = probs

    # Sample actions
    actions = np.array([
        rng.choice(n_ops, p=action_probs[i]) if action_probs[i].sum() > 0 else 0
        for i in range(n)
    ])

    # Generate observed service times with noise
    observed_service_times = np.array([
        true_q[i, actions[i]] + rng.normal(0, 1.0) for i in range(n)
    ])
    observed_service_times = np.maximum(observed_service_times, 0.1)  # minimum service time

    # Create DataFrame
    data = {
        "arrival_ts": arrival_ts,
        "cli_urgency": cli_urgency,
        "cli_complexity": cli_complexity,
        "st_load": st_load,
        "st_time_of_day": st_time_of_day,
    }

    # Add eligibility columns
    for j, op in enumerate(ops_all):
        data[f"{op}_elig"] = eligibility[:, j]

    # Add action and outcome
    data["action"] = [ops_all[a] for a in actions]
    data["service_time"] = observed_service_times

    logs = pd.DataFrame(data)
    
    return logs, ops_all, true_q
