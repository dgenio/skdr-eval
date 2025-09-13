"""Conditional logit model for propensity estimation."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("skdr_eval")

# Try to import scipy, but make it optional
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "SciPy not available. Conditional logit will fall back to multinomial."
    )


def fit_conditional_logit(
    X: np.ndarray,
    choice_ids: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    maxiter: int = 200,
    random_state: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Fit conditional logit model using scipy optimization.

    Parameters
    ----------
    X : np.ndarray
        Pairwise features (n_pairs, n_features)
    choice_ids : np.ndarray
        Choice set identifiers (n_pairs,)
    y : np.ndarray
        Binary outcomes (1 for chosen, 0 for not chosen)
    l2 : float
        L2 regularization strength
    maxiter : int
        Maximum optimization iterations
    random_state : int
        Random seed for initialization

    Returns
    -------
    coef : np.ndarray
        Fitted coefficients
    intercept : float
        Fitted intercept
    temp : float
        Temperature parameter (1.0 initially, calibrated later)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for conditional logit. Install with: pip install skdr-eval[choice]"
        )

    np.random.seed(random_state)
    n_features = X.shape[1]

    # Initialize parameters
    initial_params = np.random.normal(0, 0.01, n_features + 1)  # +1 for intercept

    def objective(params: np.ndarray) -> float:
        coef = params[:-1]
        intercept = params[-1]

        # Compute utilities
        utilities = X @ coef + intercept

        # Compute log-likelihood for each choice set
        log_likelihood = 0.0
        unique_choice_ids = np.unique(choice_ids)

        for choice_id in unique_choice_ids:
            mask = choice_ids == choice_id
            choice_utilities = utilities[mask]
            choice_y = y[mask]

            if len(choice_utilities) == 0 or np.sum(choice_y) == 0:
                continue

            # Softmax probabilities
            max_utility: float = np.max(choice_utilities)
            exp_utils = np.exp(choice_utilities - max_utility)  # numerical stability
            probs = exp_utils / np.sum(exp_utils)

            # Add to log-likelihood
            log_likelihood += np.sum(choice_y * np.log(probs + 1e-15))

        # Add L2 regularization
        regularization = l2 * np.sum(coef**2)

        # Return negative log-likelihood (for minimization)
        return float(-(log_likelihood - regularization))

    # Optimize
    try:
        result = minimize(
            objective, initial_params, method="L-BFGS-B", options={"maxiter": maxiter}
        )

        if not result.success:
            logger.warning(
                f"Conditional logit optimization did not converge: {result.message}"
            )

        coef = result.x[:-1]
        intercept = result.x[-1]
        temp = 1.0  # Initial temperature, will be calibrated later

        return coef, intercept, temp

    except Exception as e:
        logger.error(f"Error fitting conditional logit: {e}")
        # Return zero coefficients as fallback
        return np.zeros(n_features), 0.0, 1.0


def predict_proba_condlogit(
    X: np.ndarray,
    choice_ids: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    temp: float = 1.0,
) -> np.ndarray:
    """Predict choice probabilities using conditional logit.

    Parameters
    ----------
    X : np.ndarray
        Pairwise features (n_pairs, n_features)
    choice_ids : np.ndarray
        Choice set identifiers (n_pairs,)
    coef : np.ndarray
        Fitted coefficients
    intercept : float
        Fitted intercept
    temp : float
        Temperature parameter for calibration

    Returns
    -------
    probs : np.ndarray
        Choice probabilities (n_pairs,)
    """
    # Compute utilities
    utilities = (X @ coef + intercept) / temp

    # Compute probabilities for each choice set
    probs: np.ndarray = np.zeros(len(X), dtype=np.float64)
    unique_choice_ids = np.unique(choice_ids)

    for choice_id in unique_choice_ids:
        mask = choice_ids == choice_id
        choice_utilities = utilities[mask]

        if len(choice_utilities) == 0:
            continue

        # Softmax probabilities
        max_utility: float = float(np.max(choice_utilities))
        exp_utils = np.exp(choice_utilities - max_utility)
        choice_probs = exp_utils / np.sum(exp_utils)

        probs[mask] = choice_probs

    return probs


def calibrate_temperature(
    X_val: np.ndarray,
    choice_ids_val: np.ndarray,
    y_val: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    temp_grid: Optional[np.ndarray] = None,
) -> float:
    """Calibrate temperature parameter on validation set.

    Parameters
    ----------
    X_val : np.ndarray
        Validation features
    choice_ids_val : np.ndarray
        Validation choice set identifiers
    y_val : np.ndarray
        Validation outcomes
    coef : np.ndarray
        Fitted coefficients
    intercept : float
        Fitted intercept
    temp_grid : np.ndarray, optional
        Temperature values to try

    Returns
    -------
    best_temp : float
        Best temperature parameter
    """
    if temp_grid is None:
        temp_grid = np.logspace(-1, 1, 21)  # 0.1 to 10

    best_temp = 1.0
    best_nll = float("inf")

    for temp in temp_grid:
        try:
            probs = predict_proba_condlogit(
                X_val, choice_ids_val, coef, intercept, temp
            )

            # Compute negative log-likelihood
            log_probs = np.log(probs + 1e-15)
            weighted_log_probs = y_val * log_probs
            nll_val: float = float(np.sum(weighted_log_probs))
            nll = -nll_val

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError):
            # Skip this temperature if numerical issues occur
            continue

    return best_temp


def sample_negative_pairs(
    X: np.ndarray,
    choice_ids: np.ndarray,
    y: np.ndarray,
    neg_per_pos: int = 5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample negative pairs to reduce computational cost.

    Parameters
    ----------
    X : np.ndarray
        Pairwise features
    choice_ids : np.ndarray
        Choice set identifiers
    y : np.ndarray
        Binary outcomes
    neg_per_pos : int
        Number of negative samples per positive
    random_state : int
        Random seed

    Returns
    -------
    X_sampled : np.ndarray
        Sampled features
    choice_ids_sampled : np.ndarray
        Sampled choice set identifiers
    y_sampled : np.ndarray
        Sampled outcomes
    """
    np.random.seed(random_state)

    sampled_indices = []
    unique_choice_ids = np.unique(choice_ids)

    for choice_id in unique_choice_ids:
        mask = choice_ids == choice_id
        choice_indices = np.where(mask)[0]
        choice_y = y[mask]

        # Get positive indices
        pos_indices = choice_indices[choice_y == 1]
        neg_indices = choice_indices[choice_y == 0]

        # Always keep all positives
        sampled_indices.extend(pos_indices)

        # Sample negatives
        if len(neg_indices) > 0:
            n_neg_to_sample = min(len(neg_indices), len(pos_indices) * neg_per_pos)
            sampled_neg = np.random.choice(neg_indices, n_neg_to_sample, replace=False)
            sampled_indices.extend(sampled_neg)

    sampled_indices_array = np.array(sampled_indices)

    return (
        X[sampled_indices_array],
        choice_ids[sampled_indices_array],
        y[sampled_indices_array],
    )


def fit_conditional_logit_with_sampling(
    X: np.ndarray,
    choice_ids: np.ndarray,
    y: np.ndarray,
    neg_per_pos: int = 5,
    l2: float = 1.0,
    maxiter: int = 200,
    random_state: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Fit conditional logit with negative sampling for efficiency.

    Parameters
    ----------
    X : np.ndarray
        Pairwise features
    choice_ids : np.ndarray
        Choice set identifiers
    y : np.ndarray
        Binary outcomes
    neg_per_pos : int
        Number of negative samples per positive
    l2 : float
        L2 regularization strength
    maxiter : int
        Maximum optimization iterations
    random_state : int
        Random seed

    Returns
    -------
    coef : np.ndarray
        Fitted coefficients
    intercept : float
        Fitted intercept
    temp : float
        Temperature parameter
    """
    logger.info(f"Fitting conditional logit with neg_per_pos={neg_per_pos}")

    # Sample negative pairs
    X_sampled, choice_ids_sampled, y_sampled = sample_negative_pairs(
        X, choice_ids, y, neg_per_pos, random_state
    )

    logger.info(f"Sampled {len(X_sampled):,} pairs from {len(X):,} total")

    # Fit model on sampled data
    return fit_conditional_logit(
        X_sampled, choice_ids_sampled, y_sampled, l2, maxiter, random_state
    )
