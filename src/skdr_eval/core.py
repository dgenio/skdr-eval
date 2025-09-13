"""Core implementation of DR and Stabilized DR for offline policy evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .choice import (
    SCIPY_AVAILABLE,
    fit_conditional_logit_with_sampling,
    predict_proba_condlogit,
)
from .pairwise import PairwiseDesign, induce_policy

logger = logging.getLogger("skdr_eval")


# Type definitions for better type safety
class EstimatorProtocol(Protocol):
    """Protocol for sklearn estimators."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> Any: ...


class ClassifierProtocol(Protocol):
    """Protocol for sklearn classifiers."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> Any: ...


class RegressorProtocol(Protocol):
    """Protocol for sklearn regressors."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> Any: ...


@dataclass
class Design:
    """Design matrix for offline policy evaluation.

    Attributes
    ----------
    X_base : np.ndarray
        Base features (context without action).
    X_obs : np.ndarray
        Observed features including action one-hot.
    X_phi : np.ndarray
        Propensity features (excludes action, includes standardized time).
    A : np.ndarray
        Action indices.
    Y : np.ndarray
        Outcomes (service times).
    ts : np.ndarray
        Timestamps for time-aware splitting.
    ops_all : List[str]
        All operator names.
    elig : np.ndarray
        Eligibility matrix (n_samples, n_ops).
    idx : Dict[str, int]
        Mapping from operator names to indices.
    """

    X_base: np.ndarray
    X_obs: np.ndarray
    X_phi: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    ts: np.ndarray
    ops_all: list[str]
    elig: np.ndarray
    idx: dict[str, int]


@dataclass
class DRResult:
    """Results from DR/SNDR evaluation.

    Attributes
    ----------
    clip : float
        Selected clipping threshold.
    V_hat : float
        Estimated policy value.
    SE_if : float
        Standard error from influence function.
    ESS : float
        Effective sample size.
    tail_mass : float
        Mass in clipped tail.
    MSE_est : float
        Estimated MSE (bias^2 + variance).
    match_rate : float
        Fraction of samples with positive propensity.
    min_pscore : float
        Minimum propensity score in matched set.
    pscore_q10 : float
        10th percentile of propensity scores.
    pscore_q05 : float
        5th percentile of propensity scores.
    pscore_q01 : float
        1st percentile of propensity scores.
    grid : pd.DataFrame
        Full grid of results across clipping thresholds.
    """

    clip: float
    V_hat: float
    SE_if: float
    ESS: float
    tail_mass: float
    MSE_est: float
    match_rate: float
    min_pscore: float
    pscore_q10: float
    pscore_q05: float
    pscore_q01: float
    grid: pd.DataFrame


def build_design(
    logs: pd.DataFrame, cli_pref: str = "cli_", st_pref: str = "st_"
) -> Design:
    """Build design matrices from logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Log data with columns: arrival_ts, cli_*, st_*, op_*_elig, action, service_time.
    cli_pref : str, default="cli_"
        Prefix for client features.
    st_pref : str, default="st_"
        Prefix for service-time features.

    Returns
    -------
    Design
        Design matrices and metadata.
    """
    # Extract operators from eligibility columns
    elig_cols = [col for col in logs.columns if col.endswith("_elig")]
    ops_all = [col.replace("_elig", "") for col in elig_cols]
    idx = {op: i for i, op in enumerate(ops_all)}

    # Base features (context)
    cli_cols = [col for col in logs.columns if col.startswith(cli_pref)]
    st_cols = [col for col in logs.columns if col.startswith(st_pref)]
    base_cols = cli_cols + st_cols
    X_base = logs[base_cols].values

    # Eligibility matrix
    elig = logs[elig_cols].values

    # Action indices
    A = np.array([idx[action] for action in logs["action"]])

    # Observed features (base + action one-hot)
    action_onehot = np.zeros((len(logs), len(ops_all)))
    action_onehot[np.arange(len(logs)), A] = 1
    X_obs = np.column_stack([X_base, action_onehot])

    # Propensity features (base + standardized time, no action)
    scaler = StandardScaler()
    ts_norm = scaler.fit_transform(logs[["arrival_ts"]].values.astype(float))
    X_phi = np.column_stack([X_base, ts_norm])

    # Outcomes and timestamps
    Y: np.ndarray = logs["service_time"].values.astype(np.float64)
    ts: np.ndarray = logs["arrival_ts"].values.astype(np.float64)

    return Design(
        X_base=X_base,
        X_obs=X_obs,
        X_phi=X_phi,
        A=A,
        Y=Y,
        ts=ts,
        ops_all=ops_all,
        elig=elig,
        idx=idx,
    )


def fit_propensity_timecal(
    X_phi: np.ndarray,
    A: np.ndarray,
    ts: Optional[np.ndarray] = None,
    n_splits: int = 3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit propensity model with time-aware cross-validation and calibration.

    Parameters
    ----------
    X_phi : np.ndarray
        Propensity features.
    A : np.ndarray
        Action indices.
    ts : np.ndarray, optional
        Timestamps for time-aware sorting. If None, assumes data is already sorted.
    n_splits : int, default=3
        Number of time-series splits.
    random_state : int, default=0
        Random seed.

    Returns
    -------
    propensities : np.ndarray
        Calibrated propensity scores (n_samples, n_actions).
    fold_indices : np.ndarray
        Fold assignment for each sample.
    """
    n_samples, _ = X_phi.shape
    n_actions = A.max() + 1

    # Sort by timestamp if provided to ensure proper time-series ordering
    if ts is not None:
        time_order = np.argsort(ts)
        X_phi_sorted = X_phi[time_order]
        A_sorted = A[time_order]
        # Keep track of original indices for mapping back
        inverse_order = np.empty_like(time_order)
        inverse_order[time_order] = np.arange(len(time_order))
    else:
        X_phi_sorted = X_phi
        A_sorted = A
        time_order = np.arange(n_samples)
        inverse_order = np.arange(n_samples)

    # Time-series split on sorted data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    propensities = np.zeros((n_samples, n_actions))
    fold_indices = np.full(n_samples, -1)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_phi_sorted)):
        # Map sorted indices back to original order for fold assignment
        original_test_idx = time_order[test_idx]
        fold_indices[original_test_idx] = fold

        X_train, X_test = X_phi_sorted[train_idx], X_phi_sorted[test_idx]
        A_train, _A_test = A_sorted[train_idx], A_sorted[test_idx]

        # Fit base classifier with robustness for single class
        try:
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
            clf.fit(X_train, A_train)

            # Get uncalibrated predictions - ensure we have all actions
            if hasattr(clf, "classes_") and len(clf.classes_) < n_actions:
                # Handle case where not all actions are in training data
                pred_proba_full = np.zeros((len(X_test), n_actions))
                pred_proba_partial = clf.predict_proba(X_test)
                for i, class_idx in enumerate(clf.classes_):
                    pred_proba_full[:, class_idx] = pred_proba_partial[:, i]
                # Add small uniform probability for missing classes
                missing_mass = 1.0 - pred_proba_full.sum(axis=1, keepdims=True)
                missing_classes = np.setdiff1d(np.arange(n_actions), clf.classes_)
                if len(missing_classes) > 0:
                    pred_proba_full[:, missing_classes] = missing_mass / len(
                        missing_classes
                    )
                pred_proba = pred_proba_full
            else:
                pred_proba = clf.predict_proba(X_test)

        except ValueError as e:
            if "only one class" in str(e):
                # Handle single class case - assign uniform probabilities
                pred_proba = np.ones((len(X_test), n_actions)) / n_actions
                clf = None  # Mark as failed
            else:
                raise

        # Simple calibration using CalibratedClassifierCV approach
        try:
            if clf is not None and len(np.unique(A_train)) > 1:
                # Use calibrated classifier for better probability estimates
                cal_clf = CalibratedClassifierCV(clf, method="isotonic", cv=2)
                cal_clf.fit(X_train, A_train)

                # Get calibrated predictions
                if hasattr(cal_clf, "classes_") and len(cal_clf.classes_) < n_actions:
                    # Handle missing classes
                    cal_proba_full = np.zeros((len(X_test), n_actions))
                    cal_proba_partial = cal_clf.predict_proba(X_test)
                    for i, class_idx in enumerate(cal_clf.classes_):
                        cal_proba_full[:, class_idx] = cal_proba_partial[:, i]
                    # Add small uniform probability for missing classes
                    missing_mass = 1.0 - cal_proba_full.sum(axis=1, keepdims=True)
                    missing_classes = np.setdiff1d(
                        np.arange(n_actions), cal_clf.classes_
                    )
                    if len(missing_classes) > 0:
                        cal_proba_full[:, missing_classes] = missing_mass / len(
                            missing_classes
                        )
                    pred_proba = cal_proba_full
                else:
                    pred_proba = cal_clf.predict_proba(X_test)
        except (ValueError, RuntimeError, AttributeError):
            # Fallback to uncalibrated predictions if calibration fails
            # This can happen with edge cases in the calibration process
            pass

        # Ensure probabilities sum to 1 and are positive
        row_sums = pred_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        pred_proba = pred_proba / row_sums

        # Add small epsilon to avoid zero probabilities
        epsilon = 1e-8
        pred_proba = pred_proba + epsilon
        pred_proba = pred_proba / pred_proba.sum(axis=1, keepdims=True)

        propensities[original_test_idx] = pred_proba

    # Handle samples not assigned to any fold (shouldn't happen with TimeSeriesSplit but be safe)
    unassigned_mask = fold_indices == -1
    if np.any(unassigned_mask):
        # Assign uniform probabilities to unassigned samples
        propensities[unassigned_mask] = 1.0 / n_actions
        # Assign them to the last fold
        fold_indices[unassigned_mask] = n_splits - 1

    return propensities, fold_indices


def fit_outcome_crossfit(
    X_obs: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    estimator: Union[str, Callable[[], Any]] = "hgb",
    random_state: int = 0,
) -> tuple[np.ndarray, list[tuple[Any, np.ndarray, np.ndarray]]]:
    """Fit outcome model with cross-fitting.

    Parameters
    ----------
    X_obs : np.ndarray
        Observed features including action one-hot.
    Y : np.ndarray
        Outcomes.
    n_splits : int, default=3
        Number of cross-fitting splits.
    estimator : str or callable, default="hgb"
        Estimator type or factory function.
    random_state : int, default=0
        Random seed.

    Returns
    -------
    predictions : np.ndarray
        Cross-fitted predictions.
    models_info : List[Tuple[Any, np.ndarray, np.ndarray]]
        List of (model, train_idx, test_idx) for each fold.
    """
    n_samples = X_obs.shape[0]
    predictions = np.zeros(n_samples)
    models_info = []

    # Get estimator
    if estimator == "hgb":

        def est_factory() -> HistGradientBoostingRegressor:
            return HistGradientBoostingRegressor(random_state=random_state)
    elif estimator == "ridge":

        def est_factory() -> Ridge:
            return Ridge(random_state=random_state)
    elif estimator == "rf":

        def est_factory() -> RandomForestRegressor:
            return RandomForestRegressor(random_state=random_state)
    elif callable(estimator):
        est_factory = estimator
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(X_obs):
        X_train, X_test = X_obs[train_idx], X_obs[test_idx]
        Y_train = Y[train_idx]

        # Fit model
        model = est_factory()
        model.fit(X_train, Y_train)

        # Predict
        predictions[test_idx] = model.predict(X_test)
        models_info.append((model, train_idx, test_idx))

    return predictions, models_info


def induce_policy_from_sklearn(
    model: Any,
    X_base: np.ndarray,
    ops_all: list[str],
    elig: np.ndarray,
    idx: dict[str, int],  # noqa: ARG001
) -> np.ndarray:
    """Induce policy from sklearn model by predicting service times.

    Parameters
    ----------
    model : Any
        Trained sklearn model.
    X_base : np.ndarray
        Base features (context without action).
    ops_all : List[str]
        All operator names.
    elig : np.ndarray
        Eligibility matrix.
    idx : Dict[str, int]
        Operator name to index mapping.

    Returns
    -------
    policy_probs : np.ndarray
        Policy probabilities (n_samples, n_ops).
    """
    n_samples, _ = X_base.shape
    n_ops = len(ops_all)
    policy_probs = np.zeros((n_samples, n_ops))

    for i in range(n_samples):
        eligible_ops = np.where(elig[i])[0]
        pred_times: list[float] = []

        # Predict service time for each eligible operator
        for op_idx in eligible_ops:
            # Create feature vector with this operator's one-hot
            action_onehot = np.zeros(n_ops)
            action_onehot[op_idx] = 1
            x_with_action = np.concatenate([X_base[i], action_onehot])

            # Predict service time
            pred_time = model.predict(x_with_action.reshape(1, -1))[0]
            pred_times.append(pred_time)

        # Convert to probabilities (lower time = higher probability)
        if len(pred_times) > 0:
            pred_times_array = np.array(pred_times)
            policy_probs[i, eligible_ops] = 1.0 / (pred_times_array + 1e-8)
            policy_probs[i] /= policy_probs[i].sum()

    result: np.ndarray = np.array(policy_probs, dtype=np.float64)
    return result


def dr_value_with_clip(
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    min_ess_frac: float = 0.02,
) -> dict[str, DRResult]:
    """Compute DR and SNDR values with clipping threshold selection.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    policy_probs : np.ndarray
        Policy probabilities (n_samples, n_actions).
    Y : np.ndarray
        Outcomes.
    q_hat : np.ndarray
        Outcome predictions.
    A : np.ndarray
        Action indices.
    elig : np.ndarray
        Eligibility matrix.
    clip_grid : tuple[float, ...], default=(2, 5, 10, 20, 50, inf)
        Clipping thresholds to evaluate.
    min_ess_frac : float, default=0.02
        Minimum ESS fraction for DR clip selection.

    Returns
    -------
    results : dict[str, DRResult]
        Results for "DR" and "SNDR" estimators.
    """
    n_samples = len(Y)
    results_grid = []

    # Compute policy value under each operator
    q_pi = np.sum(policy_probs * q_hat.reshape(n_samples, -1), axis=1)

    # Get propensity scores for observed actions
    pi_obs = propensities[np.arange(n_samples), A]

    # Compute importance weights and matched set
    # Ensure A is integer type for indexing and elig is boolean for bitwise ops
    A_int: np.ndarray = A.astype(int)
    elig_bool: np.ndarray = elig.astype(bool)
    matched = (pi_obs > 0) & elig_bool[np.arange(n_samples), A_int]

    if matched.sum() == 0:
        raise ValueError("No matched samples found")

    # Diagnostics on matched set
    pi_matched = pi_obs[matched]
    match_rate = matched.mean()
    min_pscore = pi_matched.min()
    pscore_q01 = np.percentile(pi_matched, 1)
    pscore_q05 = np.percentile(pi_matched, 5)
    pscore_q10 = np.percentile(pi_matched, 10)

    for clip_val in clip_grid:
        # Compute clipped weights with safe division
        if clip_val == float("inf"):
            w_clip = np.where(pi_obs > 0, 1.0 / pi_obs, 0.0)
            w_clip[~matched] = 0
        else:
            w_clip = np.where(pi_obs > 0, np.minimum(1.0 / pi_obs, clip_val), 0.0)
            w_clip[~matched] = 0

        # DR estimate
        dr_contrib = q_pi + w_clip * (Y - q_hat)
        V_dr = dr_contrib.mean()

        # SNDR estimate
        if w_clip.sum() > 0:
            V_sndr = q_pi.mean() + (w_clip * (Y - q_hat)).sum() / w_clip.sum()
        else:
            V_sndr = q_pi.mean()

        # Effective sample size
        ess = w_clip.sum() ** 2 / (w_clip**2).sum() if w_clip.sum() > 0 else 0

        # Tail mass
        if clip_val == float("inf"):
            tail_mass = 0.0
        else:
            tail_mass = (pi_obs[matched] < 1.0 / clip_val).mean()

        # Variance estimates (simplified)
        se_dr = np.std(dr_contrib) / np.sqrt(n_samples)
        se_sndr = se_dr  # Simplified

        # MSE proxy (bias^2 + variance)
        mse_dr = se_dr**2  # Simplified, ignoring bias
        mse_sndr = se_sndr**2

        results_grid.append(
            {
                "clip": clip_val,
                "V_DR": V_dr,
                "V_SNDR": V_sndr,
                "SE_DR": se_dr,
                "SE_SNDR": se_sndr,
                "ESS": ess,
                "tail_mass": tail_mass,
                "MSE_DR": mse_dr,
                "MSE_SNDR": mse_sndr,
            }
        )

    grid_df = pd.DataFrame(results_grid)

    # Select DR clip: minimize MSE with ESS floor
    min_ess = min_ess_frac * n_samples
    valid_dr = grid_df["ESS"] >= min_ess
    if valid_dr.sum() == 0:
        # Fallback to highest ESS
        dr_idx = grid_df["ESS"].idxmax()
    else:
        dr_idx = int(grid_df.loc[valid_dr, "MSE_DR"].idxmin())

    # Select SNDR clip: minimize |SNDR - DR| + MSE
    dr_value = grid_df.loc[dr_idx, "V_DR"]
    sndr_criterion = np.abs(grid_df["V_SNDR"] - dr_value) + grid_df["MSE_SNDR"]
    sndr_idx = sndr_criterion.idxmin()

    # Create results
    def _extract_scalar(value: object) -> float:
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value)  # type: ignore[arg-type]

    dr_result = DRResult(
        clip=_extract_scalar(grid_df.loc[dr_idx, "clip"]),
        V_hat=_extract_scalar(grid_df.loc[dr_idx, "V_DR"]),
        SE_if=_extract_scalar(grid_df.loc[dr_idx, "SE_DR"]),
        ESS=_extract_scalar(grid_df.loc[dr_idx, "ESS"]),
        tail_mass=_extract_scalar(grid_df.loc[dr_idx, "tail_mass"]),
        MSE_est=_extract_scalar(grid_df.loc[dr_idx, "MSE_DR"]),
        match_rate=match_rate,
        min_pscore=min_pscore,
        pscore_q10=float(pscore_q10),
        pscore_q05=float(pscore_q05),
        pscore_q01=float(pscore_q01),
        grid=grid_df,
    )

    sndr_result = DRResult(
        clip=grid_df.loc[sndr_idx, "clip"],
        V_hat=grid_df.loc[sndr_idx, "V_SNDR"],
        SE_if=grid_df.loc[sndr_idx, "SE_SNDR"],
        ESS=grid_df.loc[sndr_idx, "ESS"],
        tail_mass=grid_df.loc[sndr_idx, "tail_mass"],
        MSE_est=grid_df.loc[sndr_idx, "MSE_SNDR"],
        match_rate=match_rate,
        min_pscore=min_pscore,
        pscore_q10=float(pscore_q10),
        pscore_q05=float(pscore_q05),
        pscore_q01=float(pscore_q01),
        grid=grid_df,
    )

    return {"DR": dr_result, "SNDR": sndr_result}


def block_bootstrap_ci(
    values_num: np.ndarray,
    values_den: Optional[np.ndarray],
    base_mean: np.ndarray,  # noqa: ARG001
    n_boot: int = 400,
    block_len: Optional[int] = None,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float]:
    """Compute confidence interval using moving-block bootstrap.

    Parameters
    ----------
    values_num : np.ndarray
        Numerator values for bootstrap.
    values_den : np.ndarray, optional
        Denominator values for ratio estimation.
    base_mean : np.ndarray
        Base mean for centering.
    n_boot : int, default=400
        Number of bootstrap samples.
    block_len : int, optional
        Block length. If None, uses sqrt(n).
    alpha : float, default=0.05
        Significance level (1-alpha confidence).
    random_state : int, default=0
        Random seed.

    Returns
    -------
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1) or if values_num is empty.
    """
    # Parameter validation
    if len(values_num) == 0:
        raise ValueError("values_num cannot be empty")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    rng = np.random.RandomState(random_state)
    n = len(values_num)

    if block_len is None:
        block_len = max(1, int(np.sqrt(n)))

    # Ensure block_len doesn't exceed data length
    block_len = min(block_len, n)

    bootstrap_stats_list: list[float] = []

    for _ in range(n_boot):
        # Generate block bootstrap sample
        n_blocks = int(np.ceil(n / block_len))
        boot_indices: list[int] = []

        for _ in range(n_blocks):
            start_idx = rng.randint(0, n - block_len + 1)
            boot_indices.extend(range(start_idx, min(start_idx + block_len, n)))

        boot_indices = boot_indices[:n]  # Trim to original length

        # Compute bootstrap statistic
        boot_num = values_num[boot_indices]
        if values_den is not None:
            boot_den = values_den[boot_indices]
            boot_stat = boot_num.sum() / boot_den.sum() if boot_den.sum() > 0 else 0.0
        else:
            boot_stat = boot_num.mean()

        bootstrap_stats_list.append(boot_stat)

    bootstrap_stats = np.array(bootstrap_stats_list)

    # Compute percentile confidence interval
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)


def evaluate_sklearn_models(
    logs: pd.DataFrame,
    models: dict[str, Any],
    fit_models: bool = True,
    n_splits: int = 3,
    outcome_estimator: Union[str, Callable[[], Any]] = "hgb",
    random_state: int = 0,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    policy_train: str = "all",
    policy_train_frac: float = 0.85,
) -> tuple[pd.DataFrame, dict[str, dict[str, DRResult]]]:
    """Evaluate sklearn models using DR and SNDR estimators.

    Parameters
    ----------
    logs : pd.DataFrame
        Log data.
    models : Dict[str, Any]
        Dictionary of model name -> model instance.
    fit_models : bool, default=True
        Whether to fit models or use pre-fitted ones.
    n_splits : int, default=3
        Number of cross-validation splits.
    outcome_estimator : str or callable, default="hgb"
        Outcome model estimator.
    random_state : int, default=0
        Random seed.
    clip_grid : Tuple[float, ...], default=(2, 5, 10, 20, 50, inf)
        Clipping thresholds.
    ci_bootstrap : bool, default=False
        Whether to compute bootstrap confidence intervals.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    policy_train : str, default="all"
        Training data for policy ("all" or "pre_split").
    policy_train_frac : float, default=0.85
        Fraction of data for policy training if policy_train="pre_split".

    Returns
    -------
    report : pd.DataFrame
        Summary report with evaluation metrics.
    detailed_results : Dict[str, Dict[str, DRResult]]
        Detailed results for each model and estimator.
    """
    # Build design
    design = build_design(logs)

    # Split data for policy training if needed
    if policy_train == "pre_split":
        n_train = int(len(logs) * policy_train_frac)
        train_design = Design(
            X_base=design.X_base[:n_train],
            X_obs=design.X_obs[:n_train],
            X_phi=design.X_phi[:n_train],
            A=design.A[:n_train],
            Y=design.Y[:n_train],
            ts=design.ts[:n_train],
            ops_all=design.ops_all,
            elig=design.elig[:n_train],
            idx=design.idx,
        )
        eval_design = Design(
            X_base=design.X_base[n_train:],
            X_obs=design.X_obs[n_train:],
            X_phi=design.X_phi[n_train:],
            A=design.A[n_train:],
            Y=design.Y[n_train:],
            ts=design.ts[n_train:],
            ops_all=design.ops_all,
            elig=design.elig[n_train:],
            idx=design.idx,
        )
    else:
        train_design = design
        eval_design = design

    # Fit propensity model
    propensities, _ = fit_propensity_timecal(
        eval_design.X_phi,
        eval_design.A,
        eval_design.ts,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Fit outcome model
    q_hat, _ = fit_outcome_crossfit(
        eval_design.X_obs,
        eval_design.Y,
        n_splits=n_splits,
        estimator=outcome_estimator,
        random_state=random_state,
    )

    # Evaluate each model
    report_rows = []
    detailed_results = {}

    for model_name, model in models.items():
        if fit_models:
            # Fit model on training data
            model.fit(train_design.X_obs, train_design.Y)

        # Induce policy
        policy_probs = induce_policy_from_sklearn(
            model,
            eval_design.X_base,
            eval_design.ops_all,
            eval_design.elig,
            eval_design.idx,
        )

        # Compute DR/SNDR values
        results = dr_value_with_clip(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=eval_design.Y,
            q_hat=q_hat,
            A=eval_design.A,
            elig=eval_design.elig,
            clip_grid=clip_grid,
        )

        detailed_results[model_name] = results

        # Add to report
        for estimator_name, result in results.items():
            row = {
                "model": model_name,
                "estimator": estimator_name,
                "V_hat": result.V_hat,
                "SE_if": result.SE_if,
                "clip": result.clip,
                "ESS": result.ESS,
                "tail_mass": result.tail_mass,
                "MSE_est": result.MSE_est,
                "match_rate": result.match_rate,
                "min_pscore": result.min_pscore,
                "pscore_q10": result.pscore_q10,
                "pscore_q05": result.pscore_q05,
                "pscore_q01": result.pscore_q01,
            }

            # Add confidence intervals if requested
            if ci_bootstrap:
                # Use proper block bootstrap for time-series data
                try:
                    # Recompute DR contributions for bootstrap
                    q_pi = np.sum(
                        policy_probs * q_hat.reshape(len(eval_design.Y), -1), axis=1
                    )
                    pi_obs = propensities[np.arange(len(eval_design.Y)), eval_design.A]
                    A_int: np.ndarray = eval_design.A.astype(int)
                    elig_bool: np.ndarray = eval_design.elig.astype(bool)
                    matched = (pi_obs > 0) & elig_bool[
                        np.arange(len(eval_design.Y)), A_int
                    ]

                    if matched.sum() > 0:
                        # Compute clipped weights
                        if result.clip == float("inf"):
                            w_clip = np.where(pi_obs > 0, 1.0 / pi_obs, 0.0)
                        else:
                            w_clip = np.where(
                                pi_obs > 0, np.minimum(1.0 / pi_obs, result.clip), 0.0
                            )
                        w_clip[~matched] = 0

                        # Create bootstrap values from DR contributions
                        dr_contrib = q_pi + w_clip * (eval_design.Y - q_hat)
                        ci_lower, ci_upper = block_bootstrap_ci(
                            values_num=dr_contrib,
                            values_den=None,
                            base_mean=np.array([result.V_hat]),
                            n_boot=400,
                            alpha=alpha,
                            random_state=random_state,
                        )
                    else:
                        # Fallback if no matched samples
                        ci_lower, ci_upper = (
                            result.V_hat - 1.96 * result.SE_if,
                            result.V_hat + 1.96 * result.SE_if,
                        )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    # Fallback to normal approximation if bootstrap fails
                    # This can happen with numerical issues in bootstrap calculations
                    ci_lower, ci_upper = (
                        result.V_hat - 1.96 * result.SE_if,
                        result.V_hat + 1.96 * result.SE_if,
                    )
                row["ci_lower"] = ci_lower
                row["ci_upper"] = ci_upper

            report_rows.append(row)

    report = pd.DataFrame(report_rows)

    return report, detailed_results


def _get_outcome_estimator(
    estimator: Union[str, Callable[[], Any]], task_type: str
) -> Any:
    """Get outcome estimator based on task type."""
    if callable(estimator):
        result = estimator()
        # Basic validation that the result has the expected methods
        if not hasattr(result, "fit") or not hasattr(result, "predict"):
            raise TypeError(
                f"Callable estimator must return an object with 'fit' and 'predict' methods, "
                f"got {type(result).__name__}"
            )
        # For binary classification, also check for predict_proba
        if task_type == "binary" and not hasattr(result, "predict_proba"):
            logger.warning(
                f"Binary classifier {type(result).__name__} missing 'predict_proba' method. "
                "This may cause issues in propensity estimation."
            )
        return result

    if task_type == "regression":
        if estimator == "hgb":
            return HistGradientBoostingRegressor(random_state=0)
        elif estimator == "ridge":
            return Ridge(random_state=0)
        elif estimator == "rf":
            return RandomForestRegressor(random_state=0)
        else:
            raise ValueError(f"Unknown regression estimator: {estimator}")
    elif task_type == "binary":
        if estimator == "hgb":
            return HistGradientBoostingClassifier(random_state=0)
        elif estimator == "logistic":
            return LogisticRegression(random_state=0, max_iter=1000)
        else:
            raise ValueError(f"Unknown binary estimator: {estimator}")
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def estimate_propensity_pairwise(
    design: PairwiseDesign,
    strategy: Literal["condlogit", "multinomial"] = "multinomial",
    method: Literal["condlogit", "multinomial"] = "condlogit",
    neg_per_pos: int = 5,
    n_splits: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate propensity scores for pairwise evaluation.

    Parameters
    ----------
    design : PairwiseDesign
        Pairwise design object
    strategy : Literal["auto", "condlogit", "multinomial"]
        Strategy for propensity estimation
    method : Literal["condlogit", "multinomial"]
        Method to use (condlogit requires scipy)
    neg_per_pos : int
        Negative samples per positive for conditional logit
    n_splits : int
        Number of time series splits
    random_state : int
        Random seed

    Returns
    -------
    propensities : np.ndarray
        Propensity scores (n_decisions, n_max_operators)
    """

    # Validate parameters
    if strategy not in ["auto", "condlogit", "multinomial"]:
        raise ValueError(
            f"Unknown strategy: {strategy}. Must be 'auto', 'condlogit', or 'multinomial'"
        )

    n_decisions = len(design.logs_df)
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())
    propensities: np.ndarray = np.zeros((n_decisions, max_ops), dtype=np.float64)

    # Use the provided method directly since strategy is now constrained

    if method == "condlogit" and not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, falling back to multinomial")
        method = "multinomial"

    if method == "condlogit":
        # Build pairwise training data with time-forward splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        days_sorted = sorted(design.ops_all_by_day.keys())

        # Create day-to-index mapping for time splits
        day_indices = {}
        for _i, day in enumerate(days_sorted):
            day_mask = design.logs_df[design.day_col] == day
            day_indices[day] = design.logs_df[day_mask].index.tolist()

        all_indices_list = []
        for day in days_sorted:
            all_indices_list.extend(day_indices[day])
        all_indices = np.array(all_indices_list)

        # Fit conditional logit with cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(all_indices)):
            train_decisions = all_indices[train_idx]
            test_decisions = all_indices[test_idx]

            # Build training pairs
            train_pairs = []
            train_choice_ids = []
            train_y = []

            for decision_idx in train_decisions:
                decision_row = design.logs_df.loc[decision_idx]
                day = decision_row[design.day_col]
                chosen_op = decision_row[design.operator_id_col]

                if day not in design.day_to_op_df:
                    continue

                # Get eligible operators
                if design.elig_col and design.elig_col in decision_row:
                    elig_ops = decision_row[design.elig_col]
                    if isinstance(elig_ops, (list, tuple)):
                        day_ops = design.day_to_op_df[day]
                        eligible_ops_df = day_ops[
                            day_ops[design.operator_id_col].isin(elig_ops)
                        ]
                    else:
                        eligible_ops_df = design.day_to_op_df[day]
                else:
                    eligible_ops_df = design.day_to_op_df[day]

                # Create pairs
                for _, op_row in eligible_ops_df.iterrows():
                    pair_features = []
                    for feat in design.cli_features:
                        pair_features.append(decision_row[feat])
                    for feat in design.op_features:
                        pair_features.append(op_row[feat])

                    train_pairs.append(pair_features)
                    train_choice_ids.append(decision_idx)
                    train_y.append(
                        1 if op_row[design.operator_id_col] == chosen_op else 0
                    )

            if not train_pairs:
                continue

            X_train = np.array(train_pairs, dtype=np.float32)
            choice_ids_train = np.array(train_choice_ids)
            y_train = np.array(train_y)

            # Fit conditional logit
            try:
                coef, intercept, temp = fit_conditional_logit_with_sampling(
                    X_train,
                    choice_ids_train,
                    y_train,
                    neg_per_pos,
                    random_state=random_state,
                )

                # Build test pairs and predict
                test_pairs = []
                test_choice_ids = []
                test_decision_to_ops = {}

                for decision_idx in test_decisions:
                    decision_row = design.logs_df.loc[decision_idx]
                    day = decision_row[design.day_col]

                    if day not in design.day_to_op_df:
                        continue

                    # Get eligible operators
                    if design.elig_col and design.elig_col in decision_row:
                        elig_ops = decision_row[design.elig_col]
                        if isinstance(elig_ops, (list, tuple)):
                            day_ops = design.day_to_op_df[day]
                            eligible_ops_df = day_ops[
                                day_ops[design.operator_id_col].isin(elig_ops)
                            ]
                        else:
                            eligible_ops_df = design.day_to_op_df[day]
                    else:
                        eligible_ops_df = design.day_to_op_df[day]

                    ops_list = []
                    for _, op_row in eligible_ops_df.iterrows():
                        pair_features = []
                        for feat in design.cli_features:
                            pair_features.append(decision_row[feat])
                        for feat in design.op_features:
                            pair_features.append(op_row[feat])

                        test_pairs.append(pair_features)
                        test_choice_ids.append(decision_idx)
                        ops_list.append(op_row[design.operator_id_col])

                    test_decision_to_ops[decision_idx] = ops_list

                if test_pairs:
                    X_test = np.array(test_pairs, dtype=np.float32)
                    choice_ids_test = np.array(test_choice_ids)

                    # Predict probabilities
                    probs = predict_proba_condlogit(
                        X_test, choice_ids_test, coef, intercept, temp
                    )

                    # Assign to propensities matrix
                    pair_idx = 0
                    for decision_idx in test_decisions:
                        if decision_idx in test_decision_to_ops:
                            ops_list = test_decision_to_ops[decision_idx]
                            for _i, op in enumerate(ops_list):
                                # Find operator index in global operator list
                                day = design.logs_df.loc[decision_idx, design.day_col]
                                if day in design.ops_all_by_day:
                                    try:
                                        op_idx = design.ops_all_by_day[day].index(op)
                                        propensities[decision_idx, op_idx] = probs[
                                            pair_idx
                                        ]
                                    except ValueError:
                                        pass
                                pair_idx += 1

            except Exception as e:
                logger.error(f"Error fitting conditional logit for fold {fold}: {e}")

    else:  # multinomial method
        logger.info("Using multinomial propensity estimation")

        # Build client + time features
        client_features = []
        actions = []

        for _, row in design.logs_df.iterrows():
            features = []
            for feat in design.cli_features:
                features.append(row[feat])
            # Add time features (day as numeric)
            try:
                day_numeric = pd.to_datetime(row[design.day_col]).dayofyear
                features.append(day_numeric)
            except (ValueError, TypeError):
                features.append(0)

            client_features.append(features)
            actions.append(row[design.operator_id_col])

        X_client = np.array(client_features, dtype=np.float32)

        # Fit multinomial logistic regression with time-forward CV
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_idx, test_idx in tscv.split(X_client):
            X_train, X_test = X_client[train_idx], X_client[test_idx]
            y_train = np.array([actions[i] for i in train_idx])

            # Fit multinomial model
            model = LogisticRegression(
                multi_class="multinomial", random_state=random_state, max_iter=1000
            )
            try:
                model.fit(X_train, y_train)

                # Predict probabilities
                probs = model.predict_proba(X_test)
                classes = model.classes_

                # Assign to propensities matrix
                for i, test_decision_idx in enumerate(test_idx):
                    for j, op in enumerate(classes):
                        day = design.logs_df.iloc[test_decision_idx][design.day_col]
                        if (
                            day in design.ops_all_by_day
                            and op in design.ops_all_by_day[day]
                        ):
                            op_idx = design.ops_all_by_day[day].index(op)
                            propensities[test_decision_idx, op_idx] = probs[i, j]

            except Exception as e:
                logger.error(f"Error fitting multinomial model: {e}")

    # Normalize propensities and handle eligibility
    for i, row in design.logs_df.iterrows():
        day = row[design.day_col]
        if day not in design.ops_all_by_day:
            continue

        # Get eligible operators
        if design.elig_col and design.elig_col in row:
            elig_ops = row[design.elig_col]
            if isinstance(elig_ops, (list, tuple)):
                elig_mask = np.array(
                    [op in elig_ops for op in design.ops_all_by_day[day]]
                )
            else:
                elig_mask = np.ones(len(design.ops_all_by_day[day]), dtype=bool)
        else:
            elig_mask = np.ones(len(design.ops_all_by_day[day]), dtype=bool)

        # Zero out ineligible operators
        day_probs = propensities[i, : len(design.ops_all_by_day[day])]
        day_probs[~elig_mask] = 0

        # Renormalize
        if np.sum(day_probs) > 0:
            day_probs = day_probs / np.sum(day_probs)
        else:
            # Uniform over eligible
            day_probs[elig_mask] = 1.0 / np.sum(elig_mask)

        propensities[i, : len(design.ops_all_by_day[day])] = day_probs

    return propensities


def evaluate_pairwise_models(
    logs_df: pd.DataFrame,
    op_daily_df: pd.DataFrame,
    models: dict[str, Any],
    metric_col: str,
    task_type: Literal["regression", "binary"],
    direction: Literal["min", "max"],
    n_splits: int = 3,
    strategy: Literal["auto", "direct", "stream", "stream_topk"] = "auto",
    propensity: Literal["condlogit", "multinomial"] = "condlogit",
    topk: int = 20,
    neg_per_pos: int = 5,
    chunk_pairs: int = 2_000_000,
    min_ess_frac: float = 0.02,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    day_col: str = "arrival_day",
    client_id_col: str = "client_id",
    operator_id_col: str = "operator_id",
    elig_col: Optional[str] = "elig_mask",
    random_state: int = 0,
    outcome_estimator: Union[str, Callable[[], Any]] = "hgb",
) -> tuple[pd.DataFrame, dict[str, dict[str, DRResult]]]:
    """Evaluate pairwise models using autoscale strategy.

    Parameters
    ----------
    logs_df : pd.DataFrame
        Observed decisions with required columns
    op_daily_df : pd.DataFrame
        Daily operator snapshots
    models : Dict[str, Any]
        Dictionary of model_name -> fitted model
    metric_col : str
        Target metric column name
    task_type : Literal["regression", "binary"]
        Type of prediction task
    direction : Literal["min", "max"]
        Whether to minimize or maximize metric
    n_splits : int
        Number of cross-validation splits
    strategy : Literal["auto", "direct", "stream", "stream_topk"]
        Policy induction strategy
    propensity : Literal["auto", "condlogit", "multinomial"]
        Propensity estimation method
    topk : int
        Top-K for stream_topk strategy
    neg_per_pos : int
        Negative samples per positive for conditional logit
    chunk_pairs : int
        Chunk size for streaming
    min_ess_frac : float
        Minimum ESS fraction for clipping
    clip_grid : Tuple[float, ...]
        Clipping thresholds
    ci_bootstrap : bool
        Whether to compute bootstrap CIs
    alpha : float
        Significance level for CIs
    day_col : str
        Day column name
    client_id_col : str
        Client ID column name
    operator_id_col : str
        Operator ID column name
    elig_col : Optional[str]
        Eligibility column name
    random_state : int
        Random seed
    outcome_estimator : Union[str, Callable[[], Any]]
        Outcome model estimator

    Returns
    -------
    report : pd.DataFrame
        Summary report
    detailed_results : Dict[str, Dict[str, DRResult]]
        Detailed results per model and estimator
    """

    logger.info("Starting pairwise evaluation")

    # Validate parameters
    if task_type not in ["regression", "binary"]:
        raise ValueError(
            f"Unknown task_type: {task_type}. Must be 'regression' or 'binary'"
        )
    if direction not in ["min", "max"]:
        raise ValueError(f"Unknown direction: {direction}. Must be 'min' or 'max'")

    # Create pairwise design
    design = PairwiseDesign.from_dataframes(
        logs_df, op_daily_df, day_col, client_id_col, operator_id_col, elig_col
    )

    # Log statistics
    stats = design.get_stats()
    logger.info(
        f"Dataset stats: {stats['n_rows']:,} decisions, "
        f"{stats['candidate_pairs']:,} candidate pairs, "
        f"{stats['memory_gb']:.2f} GB estimated memory"
    )

    # Induce policies
    policies = induce_policy(models, design, strategy, direction, topk, chunk_pairs)

    # Estimate propensity scores
    propensities = estimate_propensity_pairwise(
        design, propensity, propensity, neg_per_pos, n_splits, random_state
    )

    # Fit outcome models with cross-fitting
    Y = logs_df[metric_col].values

    # Create observed features (client + chosen operator features)
    X_obs_list = []
    A_list = []  # Action indices

    for _i, row in logs_df.iterrows():
        day = row[day_col]
        chosen_op = row[operator_id_col]

        # Client features
        obs_features: list[float] = []
        for feat in design.cli_features:
            obs_features.append(float(row[feat]))

        # Chosen operator features
        if str(day) in design.day_to_op_df:
            op_df = design.day_to_op_df[str(day)]
            op_row = op_df[op_df[operator_id_col] == chosen_op]
            if len(op_row) > 0:
                for feat in design.op_features:
                    obs_features.append(float(op_row.iloc[0][feat]))
            else:
                # Fallback to zeros
                for _feat in design.op_features:
                    obs_features.append(0.0)
        else:
            # Fallback to zeros
            for _feat in design.op_features:
                obs_features.append(0.0)

        X_obs_list.append(obs_features)

        # Action index
        if (
            str(day) in design.ops_all_by_day
            and str(chosen_op) in design.ops_all_by_day[str(day)]
        ):
            A_list.append(design.ops_all_by_day[str(day)].index(str(chosen_op)))
        else:
            A_list.append(0)  # Fallback

    X_obs = np.array(X_obs_list, dtype=np.float32)
    A = np.array(A_list)

    # Fit outcome model
    estimator_obj = _get_outcome_estimator(outcome_estimator, task_type)
    q_hat, _ = fit_outcome_crossfit(
        X_obs, Y.astype(np.float64), n_splits, lambda: estimator_obj, random_state
    )

    # Convert policies to policy probabilities
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())

    detailed_results = {}
    report_rows = []

    for model_name, policy_decisions in policies.items():
        # Convert policy decisions to probabilities
        policy_probs = np.zeros((len(logs_df), max_ops))

        for i, (_idx, row) in enumerate(logs_df.iterrows()):
            day_str = str(row[day_col])
            if day_str in design.ops_all_by_day:
                chosen_op_str = str(policy_decisions[i])
                if chosen_op_str in design.ops_all_by_day[day_str]:
                    op_idx = design.ops_all_by_day[day_str].index(chosen_op_str)
                    policy_probs[i, op_idx] = 1.0

        # Create eligibility matrix
        elig = np.zeros((len(logs_df), max_ops))
        for idx, row in logs_df.iterrows():
            i = int(idx) if isinstance(idx, int) else 0
            day_str = str(row[day_col])
            if day_str in design.ops_all_by_day:
                if elig_col and elig_col in row:
                    elig_ops = row[elig_col]
                    # Handle pandas Series or direct list/tuple values
                    if hasattr(elig_ops, "iloc"):
                        # It's a pandas Series, get the actual value
                        elig_value = elig_ops.iloc[0] if len(elig_ops) > 0 else []
                    else:
                        elig_value = elig_ops
                    if isinstance(elig_value, (list, tuple)):
                        for op in elig_value:
                            if str(op) in design.ops_all_by_day[day_str]:
                                op_idx = design.ops_all_by_day[day_str].index(str(op))
                                elig[i, op_idx] = 1.0
                    else:
                        elig[i, : len(design.ops_all_by_day[day_str])] = 1.0
                else:
                    elig[i, : len(design.ops_all_by_day[day_str])] = 1.0

        # Compute DR values
        try:
            results = dr_value_with_clip(
                propensities,
                policy_probs,
                Y.astype(np.float64),
                q_hat,
                A,
                elig,
                clip_grid,
                min_ess_frac,
            )

            detailed_results[model_name] = results

            # Add to report
            for estimator_name, result in results.items():
                report_row: dict[str, object] = {
                    "model": model_name,
                    "estimator": estimator_name,
                    "clip": result.clip,
                    "V_hat": result.V_hat,
                    "SE_if": result.SE_if,
                    "ESS": result.ESS,
                    "tail_mass": result.tail_mass,
                    "MSE_est": result.MSE_est,
                    "match_rate": result.match_rate,
                    "min_pscore": result.min_pscore,
                    "pscore_q10": result.pscore_q10,
                    "pscore_q05": result.pscore_q05,
                    "pscore_q01": result.pscore_q01,
                }

                if ci_bootstrap:
                    # Use proper block bootstrap for time-series data
                    try:
                        # Recompute DR contributions for bootstrap
                        q_pi = np.sum(policy_probs * q_hat.reshape(len(Y), -1), axis=1)
                        pi_obs = propensities[np.arange(len(Y)), A]
                        A_int: np.ndarray = A.astype(int)
                        elig_bool: np.ndarray = elig.astype(bool)
                        matched = (pi_obs > 0) & elig_bool[np.arange(len(Y)), A_int]

                        if matched.sum() > 0:
                            # Compute clipped weights
                            if result.clip == float("inf"):
                                w_clip = np.where(pi_obs > 0, 1.0 / pi_obs, 0.0)
                            else:
                                w_clip = np.where(
                                    pi_obs > 0,
                                    np.minimum(1.0 / pi_obs, result.clip),
                                    0.0,
                                )
                            w_clip[~matched] = 0

                            # Create bootstrap values from DR contributions
                            dr_contrib = q_pi + w_clip * (Y - q_hat)
                            ci_lower, ci_upper = block_bootstrap_ci(
                                values_num=dr_contrib,
                                values_den=None,
                                base_mean=np.array([result.V_hat]),
                                n_boot=400,
                                alpha=alpha,
                                random_state=random_state,
                            )
                        else:
                            # Fallback if no matched samples
                            ci_lower, ci_upper = (
                                result.V_hat - 1.96 * result.SE_if,
                                result.V_hat + 1.96 * result.SE_if,
                            )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        # Fallback to normal approximation if bootstrap fails
                        # This can happen with numerical issues in bootstrap calculations
                        ci_lower, ci_upper = (
                            result.V_hat - 1.96 * result.SE_if,
                            result.V_hat + 1.96 * result.SE_if,
                        )
                    report_row["ci_lower"] = ci_lower
                    report_row["ci_upper"] = ci_upper

                report_rows.append(report_row)

        except Exception as e:
            logger.error(f"Error computing DR values for model {model_name}: {e}")

    report = pd.DataFrame(report_rows)

    logger.info(f"Completed pairwise evaluation for {len(models)} models")

    return report, detailed_results
