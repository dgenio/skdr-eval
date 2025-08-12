"""Synthetic data generator for offline policy evaluation."""

import numpy as np
import pandas as pd

# Constants
# Priority levels: 0=low, 1=medium, 2=high
HIGH_PRIORITY = 2


def make_synth_logs(
    n: int = 5000, n_ops: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, pd.Index, np.ndarray]:
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
    actions = np.array(
        [
            rng.choice(n_ops, p=action_probs[i]) if action_probs[i].sum() > 0 else 0
            for i in range(n)
        ]
    )

    # Generate observed service times with noise
    observed_service_times = np.array(
        [true_q[i, actions[i]] + rng.normal(0, 1.0) for i in range(n)]
    )
    observed_service_times = np.maximum(
        observed_service_times, 0.1
    )  # minimum service time

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


def make_pairwise_synth(
    n_days: int = 14,
    n_clients_day: int = 2000,
    n_ops: int = 200,
    seed: int = 0,
    binary: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic pairwise data for offline policy evaluation.

    Creates realistic synthetic data with:
    - Client features (cli_*) varying by client
    - Operator features (op_*) varying by operator and day
    - Eligibility masks restricting operator choices
    - Observed decisions and outcomes (service time or binary success)

    Parameters
    ----------
    n_days : int, default=14
        Number of days to generate
    n_clients_day : int, default=2000
        Number of clients per day
    n_ops : int, default=200
        Number of operators in the system
    seed : int, default=0
        Random seed for reproducibility
    binary : bool, default=False
        If True, generate binary outcomes (success/failure)
        If False, generate continuous outcomes (service times)

    Returns
    -------
    logs_df : pd.DataFrame
        Observed decisions with columns:
        - arrival_day: day identifier
        - arrival_ts: timestamp within day
        - client_id: unique client identifier
        - operator_id: chosen operator
        - cli_*: client features
        - op_*: operator features at decision time
        - elig_mask: list of eligible operators
        - target: outcome metric (service_time or success)
    op_daily_df : pd.DataFrame
        Daily operator snapshots with columns:
        - operator_id: operator identifier
        - arrival_day: day identifier
        - op_*: operator features for that day

    Examples
    --------
    >>> logs_df, op_daily_df = make_pairwise_synth(n_days=3, n_clients_day=100, n_ops=10)
    >>> print(logs_df.shape)
    (300, 12)  # 3 days * 100 clients, ~12 columns
    >>> print(op_daily_df.shape)
    (30, 5)    # 3 days * 10 ops, ~5 columns
    """
    rng = np.random.RandomState(seed)

    # Generate operator names
    op_names = [f"op_{i:03d}" for i in range(n_ops)]

    # Generate days
    days = [f"day_{i:02d}" for i in range(n_days)]

    # Generate operator daily data
    op_daily_rows = []
    for day in days:
        for op_name in op_names:
            # Operator features that vary by day
            op_skill = rng.normal(0.5, 0.2)  # Base skill level
            op_load = rng.exponential(0.3)  # Daily workload
            op_efficiency = rng.beta(2, 2)  # Efficiency factor
            op_availability = rng.uniform(0.7, 1.0)  # Availability fraction

            op_daily_rows.append(
                {
                    "operator_id": op_name,
                    "arrival_day": day,
                    "op_skill": np.clip(op_skill, 0, 1),
                    "op_load": np.clip(op_load, 0, 2),
                    "op_efficiency": op_efficiency,
                    "op_availability": op_availability,
                }
            )

    op_daily_df = pd.DataFrame(op_daily_rows)

    # Generate client decisions
    logs_rows = []
    client_counter = 0

    for day_idx, day in enumerate(days):
        # Get operators for this day
        day_ops = op_daily_df[op_daily_df["arrival_day"] == day].copy()

        for _client_idx in range(n_clients_day):
            client_counter += 1
            client_id = f"client_{client_counter:06d}"

            # Client features
            cli_urgency = rng.uniform(0, 1)
            cli_complexity = rng.exponential(0.5)
            cli_priority = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # Low, medium, high
            cli_location = rng.uniform(0, 1)

            # Eligibility: ~80% of operators are eligible on average
            eligible_ops = rng.choice(
                op_names, size=rng.binomial(n_ops, 0.8), replace=False
            ).tolist()

            if len(eligible_ops) == 0:  # Ensure at least one eligible
                eligible_ops = [rng.choice(op_names)]

            # Choose operator based on realistic selection process
            # Higher skill and availability, lower load -> higher selection probability
            elig_day_ops = day_ops[day_ops["operator_id"].isin(eligible_ops)]

            if len(elig_day_ops) == 0:
                # Fallback to first operator
                chosen_op = op_names[0]
                chosen_op_data = day_ops[day_ops["operator_id"] == chosen_op].iloc[0]
            else:
                # Compute selection probabilities
                selection_scores = (
                    elig_day_ops["op_skill"] * 2.0
                    + elig_day_ops["op_availability"] * 1.5
                    - elig_day_ops["op_load"] * 1.0
                    + elig_day_ops["op_efficiency"] * 1.0
                    + rng.normal(0, 0.3, len(elig_day_ops))  # Random noise
                )

                # Softmax selection
                exp_scores = np.exp(selection_scores - np.max(selection_scores))
                probs = exp_scores / np.sum(exp_scores)

                chosen_idx = rng.choice(len(elig_day_ops), p=probs)
                chosen_op_data = elig_day_ops.iloc[chosen_idx]
                chosen_op = str(chosen_op_data["operator_id"])

            # Generate outcome based on client and operator features
            if binary:
                # Binary outcome (success probability)
                success_logit = (
                    2.0 * chosen_op_data["op_skill"]
                    + 1.0 * chosen_op_data["op_efficiency"]
                    - 1.5 * chosen_op_data["op_load"]
                    + 0.5 * chosen_op_data["op_availability"]
                    - 1.0 * cli_complexity
                    - 0.5 * cli_urgency
                    + 1.0 * (cli_priority == HIGH_PRIORITY)  # High priority bonus
                    + rng.normal(0, 0.5)  # Noise
                )
                success_prob = 1 / (1 + np.exp(-success_logit))
                outcome = rng.binomial(1, success_prob)
                target_col = "success"
            else:
                # Continuous outcome (service time)
                base_time = 30  # Base service time in minutes
                time_multiplier = (
                    1.0
                    + 0.5 * cli_complexity
                    + 0.3 * cli_urgency
                    + 0.2 * (cli_priority == 0)  # Low priority penalty
                    - 0.3 * chosen_op_data["op_skill"]
                    - 0.2 * chosen_op_data["op_efficiency"]
                    + 0.4 * chosen_op_data["op_load"]
                )

                service_time = base_time * time_multiplier * rng.lognormal(0, 0.3)
                outcome = np.clip(service_time, 5, 300)  # 5 min to 5 hours
                target_col = "service_time"

            # Arrival timestamp within day
            arrival_ts = pd.Timestamp(f"2024-01-{day_idx + 1:02d}") + pd.Timedelta(
                minutes=rng.uniform(0, 24 * 60)
            )

            logs_rows.append(
                {
                    "arrival_day": day,
                    "arrival_ts": arrival_ts,
                    "client_id": client_id,
                    "operator_id": chosen_op,
                    "cli_urgency": cli_urgency,
                    "cli_complexity": np.clip(cli_complexity, 0, 3),
                    "cli_priority": cli_priority,
                    "cli_location": cli_location,
                    "op_skill": chosen_op_data["op_skill"],
                    "op_load": chosen_op_data["op_load"],
                    "op_efficiency": chosen_op_data["op_efficiency"],
                    "op_availability": chosen_op_data["op_availability"],
                    "elig_mask": eligible_ops,
                    target_col: outcome,
                }
            )

    logs_df = pd.DataFrame(logs_rows)

    # Sort by arrival time
    logs_df = logs_df.sort_values("arrival_ts").reset_index(drop=True)

    return logs_df, op_daily_df
