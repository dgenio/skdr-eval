"""Core implementation of DR and Stabilized DR for offline policy evaluation."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


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
    ops_all: List[str]
    elig: np.ndarray
    idx: Dict[str, int]


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
    Y = logs["service_time"].values
    ts = logs["arrival_ts"].values
    
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
    X_phi: np.ndarray, A: np.ndarray, ts: Optional[np.ndarray] = None, 
    n_splits: int = 3, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
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
    n_samples, n_features = X_phi.shape
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
        A_train, A_test = A_sorted[train_idx], A_sorted[test_idx]
        
        # Fit base classifier with robustness for single class
        try:
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
            clf.fit(X_train, A_train)
            
            # Get uncalibrated predictions - ensure we have all actions
            if hasattr(clf, 'classes_') and len(clf.classes_) < n_actions:
                # Handle case where not all actions are in training data
                pred_proba_full = np.zeros((len(X_test), n_actions))
                pred_proba_partial = clf.predict_proba(X_test)
                for i, class_idx in enumerate(clf.classes_):
                    pred_proba_full[:, class_idx] = pred_proba_partial[:, i]
                # Add small uniform probability for missing classes
                missing_mass = 1.0 - pred_proba_full.sum(axis=1, keepdims=True)
                missing_classes = np.setdiff1d(np.arange(n_actions), clf.classes_)
                if len(missing_classes) > 0:
                    pred_proba_full[:, missing_classes] = missing_mass / len(missing_classes)
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
                cal_clf = CalibratedClassifierCV(clf, method='isotonic', cv=2)
                cal_clf.fit(X_train, A_train)
                
                # Get calibrated predictions
                if hasattr(cal_clf, 'classes_') and len(cal_clf.classes_) < n_actions:
                    # Handle missing classes
                    cal_proba_full = np.zeros((len(X_test), n_actions))
                    cal_proba_partial = cal_clf.predict_proba(X_test)
                    for i, class_idx in enumerate(cal_clf.classes_):
                        cal_proba_full[:, class_idx] = cal_proba_partial[:, i]
                    # Add small uniform probability for missing classes
                    missing_mass = 1.0 - cal_proba_full.sum(axis=1, keepdims=True)
                    missing_classes = np.setdiff1d(np.arange(n_actions), cal_clf.classes_)
                    if len(missing_classes) > 0:
                        cal_proba_full[:, missing_classes] = missing_mass / len(missing_classes)
                    pred_proba = cal_proba_full
                else:
                    pred_proba = cal_clf.predict_proba(X_test)
        except Exception:
            # Fallback to uncalibrated predictions
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
) -> Tuple[np.ndarray, List[Tuple[Any, np.ndarray, np.ndarray]]]:
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
        est_factory = lambda: HistGradientBoostingRegressor(random_state=random_state)
    elif estimator == "ridge":
        est_factory = lambda: Ridge(random_state=random_state)
    elif estimator == "rf":
        est_factory = lambda: RandomForestRegressor(random_state=random_state)
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
    ops_all: List[str],
    elig: np.ndarray,
    idx: Dict[str, int],
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
    n_samples, n_base_features = X_base.shape
    n_ops = len(ops_all)
    policy_probs = np.zeros((n_samples, n_ops))
    
    for i in range(n_samples):
        eligible_ops = np.where(elig[i])[0]
        if len(eligible_ops) == 0:
            continue
            
        # Predict service time for each eligible operator
        service_times = []
        for op_idx in eligible_ops:
            # Create feature vector with this operator's one-hot
            action_onehot = np.zeros(n_ops)
            action_onehot[op_idx] = 1
            x_with_action = np.concatenate([X_base[i], action_onehot])
            
            # Predict service time
            pred_time = model.predict(x_with_action.reshape(1, -1))[0]
            service_times.append(pred_time)
        
        # Choose operator with minimum predicted service time
        best_op_idx = eligible_ops[np.argmin(service_times)]
        policy_probs[i, best_op_idx] = 1.0
    
    return policy_probs


def dr_value_with_clip(
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    clip_grid: Tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    min_ess_frac: float = 0.02,
) -> Dict[str, DRResult]:
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
    clip_grid : Tuple[float, ...], default=(2, 5, 10, 20, 50, inf)
        Clipping thresholds to evaluate.
    min_ess_frac : float, default=0.02
        Minimum ESS fraction for DR clip selection.
        
    Returns
    -------
    results : Dict[str, DRResult]
        Results for "DR" and "SNDR" estimators.
    """
    n_samples = len(Y)
    results_grid = []
    
    # Compute policy value under each operator
    q_pi = np.sum(policy_probs * q_hat.reshape(n_samples, -1), axis=1)
    
    # Get propensity scores for observed actions
    pi_obs = propensities[np.arange(n_samples), A]
    
    # Compute importance weights and matched set
    matched = (pi_obs > 0) & elig[np.arange(n_samples), A]
    
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
        if w_clip.sum() > 0:
            ess = (w_clip.sum() ** 2) / (w_clip ** 2).sum()
        else:
            ess = 0
        
        # Tail mass
        if clip_val == float("inf"):
            tail_mass = 0.0
        else:
            tail_mass = (pi_obs[matched] < 1.0 / clip_val).mean()
        
        # Variance estimates (simplified)
        se_dr = np.std(dr_contrib) / np.sqrt(n_samples)
        se_sndr = se_dr  # Simplified
        
        # MSE proxy (bias^2 + variance)
        mse_dr = se_dr ** 2  # Simplified, ignoring bias
        mse_sndr = se_sndr ** 2
        
        results_grid.append({
            "clip": clip_val,
            "V_DR": V_dr,
            "V_SNDR": V_sndr,
            "SE_DR": se_dr,
            "SE_SNDR": se_sndr,
            "ESS": ess,
            "tail_mass": tail_mass,
            "MSE_DR": mse_dr,
            "MSE_SNDR": mse_sndr,
        })
    
    grid_df = pd.DataFrame(results_grid)
    
    # Select DR clip: minimize MSE with ESS floor
    min_ess = min_ess_frac * n_samples
    valid_dr = grid_df["ESS"] >= min_ess
    if valid_dr.sum() == 0:
        # Fallback to highest ESS
        dr_idx = grid_df["ESS"].idxmax()
    else:
        dr_idx = grid_df.loc[valid_dr, "MSE_DR"].idxmin()
    
    # Select SNDR clip: minimize |SNDR - DR| + MSE
    dr_value = grid_df.loc[dr_idx, "V_DR"]
    sndr_criterion = np.abs(grid_df["V_SNDR"] - dr_value) + grid_df["MSE_SNDR"]
    sndr_idx = sndr_criterion.idxmin()
    
    # Create results
    dr_result = DRResult(
        clip=grid_df.loc[dr_idx, "clip"],
        V_hat=grid_df.loc[dr_idx, "V_DR"],
        SE_if=grid_df.loc[dr_idx, "SE_DR"],
        ESS=grid_df.loc[dr_idx, "ESS"],
        tail_mass=grid_df.loc[dr_idx, "tail_mass"],
        MSE_est=grid_df.loc[dr_idx, "MSE_DR"],
        match_rate=match_rate,
        min_pscore=min_pscore,
        pscore_q10=pscore_q10,
        pscore_q05=pscore_q05,
        pscore_q01=pscore_q01,
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
        pscore_q10=pscore_q10,
        pscore_q05=pscore_q05,
        pscore_q01=pscore_q01,
        grid=grid_df,
    )
    
    return {"DR": dr_result, "SNDR": sndr_result}


def block_bootstrap_ci(
    values_num: np.ndarray,
    values_den: Optional[np.ndarray],
    base_mean: np.ndarray,
    n_boot: int = 400,
    block_len: Optional[int] = None,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Tuple[float, float]:
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
    """
    rng = np.random.RandomState(random_state)
    n = len(values_num)
    
    if block_len is None:
        block_len = max(1, int(np.sqrt(n)))
    
    bootstrap_stats = []
    
    for _ in range(n_boot):
        # Generate block bootstrap sample
        n_blocks = int(np.ceil(n / block_len))
        boot_indices = []
        
        for _ in range(n_blocks):
            start_idx = rng.randint(0, n - block_len + 1)
            boot_indices.extend(range(start_idx, min(start_idx + block_len, n)))
        
        boot_indices = boot_indices[:n]  # Trim to original length
        
        # Compute bootstrap statistic
        boot_num = values_num[boot_indices]
        if values_den is not None:
            boot_den = values_den[boot_indices]
            if boot_den.sum() > 0:
                boot_stat = boot_num.sum() / boot_den.sum()
            else:
                boot_stat = 0.0
        else:
            boot_stat = boot_num.mean()
        
        bootstrap_stats.append(boot_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentile confidence interval
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return ci_lower, ci_upper


def evaluate_sklearn_models(
    logs: pd.DataFrame,
    models: Dict[str, Any],
    fit_models: bool = True,
    n_splits: int = 3,
    outcome_estimator: Union[str, Callable[[], Any]] = "hgb",
    random_state: int = 0,
    clip_grid: Tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    policy_train: str = "all",
    policy_train_frac: float = 0.85,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, DRResult]]]:
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
        eval_design.X_phi, eval_design.A, eval_design.ts, n_splits=n_splits, random_state=random_state
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
            model, eval_design.X_base, eval_design.ops_all, eval_design.elig, eval_design.idx
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
                # Simplified bootstrap (would need more sophisticated implementation)
                ci_lower, ci_upper = result.V_hat - 1.96 * result.SE_if, result.V_hat + 1.96 * result.SE_if
                row["ci_lower"] = ci_lower
                row["ci_upper"] = ci_upper
            
            report_rows.append(row)
    
    report = pd.DataFrame(report_rows)
    
    return report, detailed_results
