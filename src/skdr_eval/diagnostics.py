"""Propensity score diagnostics for skdr-eval library."""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from .exceptions import ConfigurationError, DataValidationError, InsufficientDataError

logger = logging.getLogger("skdr_eval")

# Minimum sample thresholds used across diagnostic functions
_MIN_SAMPLES_SMALL = 5  # statistics, log loss, balance stats
_MIN_SAMPLES_MEDIUM = 10  # overlap and balance checking
_MIN_SAMPLES_LARGE = 20  # calibration, discrimination, comprehensive
_MIN_ACTION_COUNT = 2  # min samples in an action group (basic metrics)
_MIN_ACTION_COUNT_DISC = 5  # min samples in a group for discrimination
_PER_ACTION_NDIM = 2  # propensities are (n_samples, n_actions)
_MIN_UNIQUE_LABELS = 2  # min unique binary labels for AUC
_ZERO_VARIANCE_TOL = 1e-10  # tolerance for zero-variance equality check

# ECE / Brier (#84) — 15-bin standard from Naeini et al. 2015 and Brier 1950.
_ECE_DEFAULT_N_BINS = 15  # ECE bin count; matches torch.distributions and BBT 2015
_MIN_ECE_N_BINS = 2  # minimum bin count; below this ECE is degenerate
_MIN_SAMPLES_RELIABILITY = (
    30  # min samples for ECE/Brier (more than _LARGE due to bin count)
)
_MIN_PARETO_K_TAIL_FRAC = (
    0.2  # tail fraction targeted by PSIS (Vehtari et al. 2024, eq. 2.3)
)
_MAX_PARETO_K_TAIL_LEN = 200  # PSIS caps the tail length at 3 * sqrt(n) — we cap here
_MIN_SAMPLES_PARETO_K = 25  # below this we cannot estimate a 5-point tail reliably
_MIN_TAIL_LEN_FOR_GPD_FIT = 2  # GPD profile-likelihood is undefined below 2 exceedances


@dataclass
class PerActionDiagnostics:
    """Per-action propensity diagnostics (#131).

    Surfaces calibration, support, and miscalibration evidence at the
    *action* level so rare or strategically-important actions cannot hide
    behind a healthy global ECE / Brier score.

    Attributes
    ----------
    action : int
        Action index.
    n : int
        Number of logged decisions where this action was taken.
    logged_frac : float
        ``n / n_total`` — empirical mass of the action under the logging
        policy. Values below ``rare_action_floor`` flag the action as rare.
    mean_pscore_taken : float
        Mean propensity assigned to this action *on rows where it was
        actually taken*.
    mean_pscore_global : float
        Mean propensity assigned to this action across the full sample.
    ece : float
        Expected Calibration Error of the binary "action == this one"
        problem, computed on the per-action propensity. ``nan`` when
        ``n < _MIN_ACTION_COUNT_DISC``.
    brier : float
        Brier score for the same binary problem.
    log_loss : float
        Cross-entropy for the same binary problem.
    insufficient : bool
        True when ``n < _MIN_ACTION_COUNT_DISC``; downstream
        diagnostics (ECE, Brier, log-loss) are unreliable.
    rare : bool
        True when ``logged_frac < rare_action_floor`` *and* the action
        is in the target policy support.
    """

    action: int
    n: int
    logged_frac: float
    mean_pscore_taken: float
    mean_pscore_global: float
    ece: float
    brier: float
    log_loss: float
    insufficient: bool
    rare: bool


@dataclass
class PropensityDiagnostics:
    """Container for propensity score diagnostic results.

    Notes
    -----
    The trust-additions fields (``ece``, ``brier_score``, ``reliability_curve``,
    ``ece_n_bins``) were introduced for issue #84 and are populated by
    :func:`comprehensive_propensity_diagnostics`. They default to ``nan`` /
    empty for backward compatibility when constructed directly.

    Per-action diagnostics (``per_action``) were added in #131 and surface
    calibration / rare-action / support quality at the *action* level so
    a global ECE cannot hide an under-supported arm.
    """

    overlap_ratio: float
    balance_ratio: float
    calibration_score: float
    discrimination_score: float
    log_loss_score: float
    statistics: dict[str, float]
    balance_stats: dict[str, float]
    calibration_curve: list[tuple[float, float]]
    roc_curve: list[tuple[float, float]]
    quantiles: dict[str, float] = field(default_factory=dict)
    # --- Trust additions (#84): Expected Calibration Error + Brier score. ---
    ece: float = float("nan")
    brier_score: float = float("nan")
    reliability_curve: list[tuple[float, float, int]] = field(default_factory=list)
    ece_n_bins: int = _ECE_DEFAULT_N_BINS
    # --- Trust additions (#131): per-action calibration + rare-action checks.
    per_action: list[PerActionDiagnostics] = field(default_factory=list)
    rare_action_floor: float = 0.01
    n_rare_actions: int = 0
    n_insufficient_actions: int = 0
    # Number of actions that are *both* rare AND insufficient. This is the
    # gate for the ``RARE_ACTION_NO_SUPPORT`` warning (#131): rare-but-supported
    # actions and insufficient-but-not-rare-under-target actions independently
    # are not high-risk on their own — only their conjunction is.
    n_rare_and_insufficient_actions: int = 0
    max_per_action_ece: float = float("nan")


def check_propensity_overlap(propensities: np.ndarray, actions: np.ndarray) -> float:
    """Check propensity score overlap between actions.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    float
        Overlap ratio (0-1, higher is better).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_MEDIUM:
        raise InsufficientDataError("Need at least 10 samples for overlap analysis")

    n_actions = propensities.shape[1]
    overlap_scores = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < _MIN_ACTION_COUNT:
            continue

        action_props = propensities[action_mask, action]
        other_props = propensities[~action_mask, action]

        if len(other_props) == 0:
            continue

        # Distributional overlap coefficient: Σ min(p_i, q_i) over a shared histogram.
        # Returns 0 when distributions are disjoint, 1 when identical.
        n_hist_bins = 20
        hist_a, _ = np.histogram(action_props, bins=n_hist_bins, range=(0.0, 1.0))
        hist_b, _ = np.histogram(other_props, bins=n_hist_bins, range=(0.0, 1.0))
        total_a = hist_a.sum()
        total_b = hist_b.sum()
        if total_a == 0 or total_b == 0:
            continue
        p = hist_a / total_a
        q = hist_b / total_b
        overlap = float(np.sum(np.minimum(p, q)))
        overlap_scores.append(overlap)

    return float(np.mean(overlap_scores)) if overlap_scores else 0.0


def check_propensity_balance(propensities: np.ndarray, actions: np.ndarray) -> float:
    """Check propensity score balance across actions.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    float
        Balance ratio (0-1, higher is better).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_MEDIUM:
        raise InsufficientDataError("Need at least 10 samples for balance analysis")

    n_actions = propensities.shape[1]
    balance_scores = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < _MIN_ACTION_COUNT:
            continue

        action_props = propensities[action_mask, action]
        other_props = propensities[~action_mask, action]

        if len(other_props) == 0:
            continue

        # Standardized Mean Difference (SMD): industry-standard causal-inference
        # balance measure. SMD=0 → perfect balance; higher → worse.
        # balance = max(0, 1 - SMD) so 1 = perfect, 0 = completely unbalanced.
        pooled_std = float(
            ((action_props.std() ** 2 + other_props.std() ** 2) / 2) ** 0.5
        )
        if pooled_std > 0:
            smd = (
                abs(float(action_props.mean()) - float(other_props.mean())) / pooled_std
            )
            balance = max(0.0, 1.0 - smd)
        else:
            # Both groups have zero variance: balance depends on whether means match.
            balance = (
                1.0
                if abs(float(action_props.mean()) - float(other_props.mean()))
                < _ZERO_VARIANCE_TOL
                else 0.0
            )
        balance_scores.append(balance)

    return float(np.mean(balance_scores)) if balance_scores else 0.0


def assess_propensity_calibration(
    propensities: np.ndarray, actions: np.ndarray, n_bins: int = 10
) -> tuple[float, list[tuple[float, float]]]:
    """Assess propensity score calibration via reliability curve.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).
    n_bins : int, default=10
        Number of bins for the reliability curve.

    Returns
    -------
    float
        Calibration score (0-1, higher is better; 1 - mean absolute error).
    List[Tuple[float, float]]
        Reliability curve as a list of exactly n_bins (mean_predicted, actual_fraction)
        tuples. Empty bins use (bin_center, 0.0).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_LARGE:
        raise InsufficientDataError("Need at least 20 samples for calibration analysis")

    n_actions = propensities.shape[1]
    calibration_scores = []
    calibration_curves = []

    for action in range(n_actions):
        action_binary = (actions == action).astype(float)
        action_probs = propensities[
            :, action
        ]  # predicted P(A=action|X) for ALL samples

        # Bin all samples by their predicted probability and compare to actual
        # action frequency per bin (reliability / calibration diagram).
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_center_arr = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_points = []

        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask = (action_probs >= bin_edges[i]) & (
                    action_probs <= bin_edges[i + 1]
                )
            else:
                bin_mask = (action_probs >= bin_edges[i]) & (
                    action_probs < bin_edges[i + 1]
                )

            if bin_mask.sum() > 0:
                mean_predicted = float(action_probs[bin_mask].mean())
                actual_fraction = float(action_binary[bin_mask].mean())
            else:
                mean_predicted = float(bin_center_arr[i])
                actual_fraction = 0.0
            bin_points.append((mean_predicted, actual_fraction))

        mae = float(np.mean([abs(pred - act) for pred, act in bin_points]))
        calibration_scores.append(max(0.0, 1.0 - mae))
        calibration_curves.append(bin_points)

    avg_score = float(np.mean(calibration_scores)) if calibration_scores else 0.0
    first_curve = calibration_curves[0] if calibration_curves else []

    return avg_score, first_curve


def assess_propensity_discrimination(
    propensities: np.ndarray, actions: np.ndarray
) -> tuple[float, list[tuple[float, float]]]:
    """Assess propensity score discrimination.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    float
        Discrimination score (AUC, 0-1, higher is better).
    List[Tuple[float, float]]
        ROC curve (fpr, tpr).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_LARGE:
        raise InsufficientDataError(
            "Need at least 20 samples for discrimination analysis"
        )

    n_actions = propensities.shape[1]
    discrimination_scores = []
    roc_curves = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < _MIN_ACTION_COUNT_DISC:
            continue

        # Create binary labels for this action
        y_true = action_mask.astype(int)
        y_scores = propensities[:, action]

        if len(np.unique(y_true)) < _MIN_UNIQUE_LABELS:
            continue

        try:
            auc = roc_auc_score(y_true, y_scores)
            discrimination_scores.append(auc)

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_curves.append(list(zip(fpr, tpr, strict=False)))
        except ValueError:
            continue

    # Return average AUC and first ROC curve
    avg_auc = float(np.mean(discrimination_scores)) if discrimination_scores else 0.0
    first_roc = roc_curves[0] if roc_curves else []

    return avg_auc, first_roc


def compute_propensity_statistics(
    propensities: np.ndarray, actions: np.ndarray
) -> dict[str, float]:
    """Compute comprehensive propensity score statistics.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    Dict[str, float]
        Dictionary of statistics.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_SMALL:
        raise InsufficientDataError("Need at least 5 samples for statistics")

    # Get propensity scores for observed actions
    observed_props = propensities[np.arange(len(actions)), actions]

    stats = {
        "min_pscore": float(observed_props.min()),
        "max_pscore": float(observed_props.max()),
        "mean_pscore": float(observed_props.mean()),
        "std_pscore": float(observed_props.std()),
        "median_pscore": float(np.median(observed_props)),
    }

    # Add quantiles
    quantiles = [1, 5, 10, 25, 75, 90, 95, 99]
    for q in quantiles:
        stats[f"pscore_q{q}"] = float(np.percentile(observed_props, q))

    return stats


def compute_balance_statistics(
    propensities: np.ndarray, actions: np.ndarray
) -> dict[str, float]:
    """Compute balance statistics for each action.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    Dict[str, float]
        Dictionary of balance statistics.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_SMALL:
        raise InsufficientDataError("Need at least 5 samples for balance statistics")

    n_actions = propensities.shape[1]
    balance_stats = {}

    for action in range(n_actions):
        action_mask = actions == action
        action_props = propensities[action_mask, action]

        balance_stats[f"action_{action}_count"] = float(action_mask.sum())
        balance_stats[f"action_{action}_mean_pscore"] = (
            float(action_props.mean()) if len(action_props) > 0 else 0.0
        )
        balance_stats[f"action_{action}_std_pscore"] = (
            float(action_props.std()) if len(action_props) > 0 else 0.0
        )

    return balance_stats


def compute_propensity_log_loss(propensities: np.ndarray, actions: np.ndarray) -> float:
    """Compute log loss for propensity scores.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    float
        Log loss score (lower is better).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_SMALL:
        raise InsufficientDataError("Need at least 5 samples for log loss")

    # Create one-hot encoded labels
    n_actions = propensities.shape[1]
    y_true = np.zeros((len(actions), n_actions))
    y_true[np.arange(len(actions)), actions] = 1

    return float(log_loss(y_true, propensities))


def compute_propensity_ece(
    propensities: np.ndarray,
    actions: np.ndarray,
    n_bins: int = _ECE_DEFAULT_N_BINS,
) -> float:
    """Compute Expected Calibration Error (ECE) for the propensity model.

    Implements the *top-1 confidence* ECE definition from Naeini, Cooper, &
    Hauskrecht (2015) "Obtaining Well Calibrated Probabilities Using Bayesian
    Binning" — the same form used by Guo et al. (2017) "On Calibration of
    Modern Neural Networks":

        conf[i]    = max_a pi(a | x_i)
        pred[i]    = argmax_a pi(a | x_i)
        correct[i] = 1 if pred[i] == a_i else 0
        ECE        = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where samples are binned by ``conf``, ``acc(B)`` is the empirical accuracy
    inside the bin, and ``conf(B)`` is the mean predicted top-1 probability.
    Under a DGP where actions ~ Categorical(π(· | x)), this ECE collapses to
    0 in expectation because P(predᵢ = aᵢ | x_i, predᵢ = a) = π(a | x_i),
    which is the confidence by definition.

    Parameters
    ----------
    propensities : np.ndarray
        Predicted propensities (n_samples, n_actions). Rows need not sum to 1
        exactly, but each row should be a valid probability vector.
    actions : np.ndarray
        Observed action indices (n_samples,). Each entry must be a valid
        action in ``range(n_actions)``.
    n_bins : int, default=15
        Number of equal-width probability bins. The 15-bin default matches the
        Naeini et al. 2015 and Guo et al. 2017 conventions.

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better; 0 means perfectly calibrated. Returns
        ``nan`` if the sample is too small (``< _MIN_SAMPLES_RELIABILITY``).

    Notes
    -----
    Bin edges span [0, 1]. The last bin is right-closed so samples with
    ``max(π) = 1.0`` are not silently dropped. Empty bins contribute zero
    to the sum (per the formula). For binary problems (n_actions=2) this
    reduces to standard binary classifier ECE.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )
    if n_bins < _MIN_ECE_N_BINS:
        raise ConfigurationError(f"n_bins must be >= {_MIN_ECE_N_BINS}, got {n_bins}")

    if len(propensities) < _MIN_SAMPLES_RELIABILITY:
        return float("nan")

    n = len(actions)
    actions_int = actions.astype(int)
    confidences = propensities.max(axis=1)
    predictions = propensities.argmax(axis=1)
    correctness = (predictions == actions_int).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correctness[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_propensity_brier(propensities: np.ndarray, actions: np.ndarray) -> float:
    """Compute the multiclass Brier score for the propensity model.

    Multiclass Brier score (Brier 1950; Stanford-Yates form):

        BS = (1/n) · Σ_i Σ_a (π(a | x_i) - 1[a_i = a])²

    Bounded in ``[0, 2]`` for K ≥ 2 actions. Lower is better; 0 means perfect.

    Parameters
    ----------
    propensities : np.ndarray
        Predicted propensities (n_samples, n_actions).
    actions : np.ndarray
        Observed action indices (n_samples,).

    Returns
    -------
    float
        Brier score. Returns ``nan`` if the sample is too small.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_SMALL:
        return float("nan")

    n, k = propensities.shape
    y_onehot = np.zeros((n, k), dtype=float)
    y_onehot[np.arange(n), actions.astype(int)] = 1.0
    sq = (propensities - y_onehot) ** 2
    return float(sq.sum(axis=1).mean())


def compute_propensity_reliability_curve(
    propensities: np.ndarray,
    actions: np.ndarray,
    n_bins: int = _ECE_DEFAULT_N_BINS,
) -> list[tuple[float, float, int]]:
    """Compute the top-1 reliability curve underlying :func:`compute_propensity_ece`.

    Returns one row per bin: ``(bin_mean_confidence, bin_accuracy, bin_count)``.
    Empty bins are returned with ``bin_mean_confidence`` set to the bin centre,
    ``bin_accuracy = nan``, and ``bin_count = 0`` so plotting code can decide
    whether to skip them.

    Parameters
    ----------
    propensities, actions, n_bins : see :func:`compute_propensity_ece`.

    Returns
    -------
    list of (float, float, int)
        Per-bin reliability triples ``(mean_predicted_top1, empirical_accuracy, count)``.
        Returns an empty list when ``propensities`` has zero rows.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )
    if n_bins < _MIN_ECE_N_BINS:
        raise ConfigurationError(f"n_bins must be >= {_MIN_ECE_N_BINS}, got {n_bins}")

    n = len(actions)
    if n == 0:
        return []

    actions_int = actions.astype(int)
    confidences = propensities.max(axis=1)
    predictions = propensities.argmax(axis=1)
    correctness = (predictions == actions_int).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    rows: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            rows.append((float(centres[i]), float("nan"), 0))
            continue
        mean_pred = float(confidences[mask].mean())
        bin_acc = float(correctness[mask].mean())
        rows.append((mean_pred, bin_acc, count))

    return rows


def psis_pareto_k(weights: np.ndarray, min_tail_len: int = 5) -> float:
    """Estimate the Pareto shape parameter ``k`` of the importance-weight tail.

    Implements the PSIS Pareto-k diagnostic from Vehtari, Simpson, Gelman,
    Yao, & Gabry (2024) "Pareto Smoothed Importance Sampling" (JMLR 25:1-58),
    §2.2-2.3:

    1. Sort raw importance weights in ascending order.
    2. Take the top ``M = min(0.2·n, 3·sqrt(n))`` weights as the tail,
       capped at ``_MAX_PARETO_K_TAIL_LEN`` (200) to keep the GPD fit
       fast on very large samples.
    3. Subtract the threshold (the largest weight *below* the tail).
    4. Fit a Generalized Pareto Distribution to those tail exceedances via
       the profile-likelihood estimator of Zhang & Stephens (2009).

    Returns ``k`` (the GPD shape parameter). The interpretation in the PSIS
    paper:

    - ``k < 0.5``: importance-sampling estimator has finite variance — trust it.
    - ``0.5 ≤ k < 0.7``: variance is finite but slow to converge — caution.
    - ``0.7 ≤ k < 1.0``: variance does not exist — high risk; results likely
      driven by a few decisions.
    - ``k ≥ 1.0``: mean of the importance weights does not exist; the
      estimator is not reliable.

    Parameters
    ----------
    weights : np.ndarray
        Raw (unclipped) importance weights ``1 / π(a_observed | x)`` for the
        matched subset.  Must be 1-D (or column-vector ``(n, 1)``),
        non-negative, and finite.
    min_tail_len : int, default=5
        Minimum required tail length to attempt the GPD fit. Below this we
        return ``nan`` (statistically meaningless).

    Returns
    -------
    float
        Estimated Pareto-k shape parameter, or ``nan`` if the weights are
        degenerate (all equal, too few non-zero, or too few samples).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim > 2 or (w.ndim == 2 and w.shape[1] != 1):  # noqa: PLR2004
        raise DataValidationError(
            f"psis_pareto_k: weights must be 1-D or (n, 1), got shape {w.shape}"
        )
    w = w.ravel()
    if w.size == 0:
        return float("nan")
    if not np.isfinite(w).all():
        # The matched-set construction already drops zero-prob rows; any inf
        # here means an upstream caller forgot to clip — surface that loudly.
        raise DataValidationError("psis_pareto_k: weights contain non-finite values")
    if (w < 0).any():
        raise DataValidationError("psis_pareto_k: weights must be non-negative")

    n = w.size
    if n < _MIN_SAMPLES_PARETO_K:
        return float("nan")

    sorted_w = np.sort(w)
    # Tail length: min(0.2 * n, 3 * sqrt(n)), capped at _MAX_PARETO_K_TAIL_LEN
    # to keep the GPD fit fast on very large samples (Vehtari et al. §2.3).
    tail_len = int(min(_MIN_PARETO_K_TAIL_FRAC * n, 3.0 * math.sqrt(n)))
    tail_len = min(tail_len, _MAX_PARETO_K_TAIL_LEN, n - 1)
    if tail_len < min_tail_len:
        return float("nan")

    threshold = sorted_w[-(tail_len + 1)]
    tail = sorted_w[-tail_len:] - threshold

    # Degenerate tail (e.g., all weights equal at top) — k is undefined.
    if tail[-1] <= 0:
        return float("nan")

    return float(_gpd_fit_zhang_stephens(tail))


def _gpd_fit_zhang_stephens(exceedances: np.ndarray) -> float:
    """Estimate the GPD shape parameter k via Zhang & Stephens (2009).

    Implements the profile-likelihood / empirical-Bayes estimator from
    Zhang & Stephens (2009) "A New and Efficient Estimation Method for the
    Generalized Pareto Distribution" (Technometrics, 51:3, 316-325). This is
    the estimator recommended by Vehtari et al. (2024) for PSIS because it is
    bias-corrected for small samples and stable when ``k -> 0``.

    Notation matches the paper: the exceedances are ``x_1 <= ... <= x_n``, the
    profile grid is over ``theta`` (with theta = k/sigma) not over ``k``, and
    ``k`` is recovered from theta_hat via ``k = -mean(log(1 - theta_hat * x_i))``.

    Parameters
    ----------
    exceedances : np.ndarray
        Strictly positive tail exceedances (already threshold-subtracted),
        sorted ascending.

    Returns
    -------
    float
        Estimated shape parameter ``k``. Sign convention: ``k > 0`` corresponds
        to a heavy upper tail (the case PSIS warns about). Returns ``nan`` if
        the fit is numerically degenerate (<= 1 exceedance, all-equal exceedances,
        or anchor failure).
    """
    x = np.asarray(exceedances, dtype=float)
    n = x.size
    # Degenerate-input bailout — kept as one return-nan branch so we stay
    # under ruff's PLR0911 6-return cap.  Conditions: too few exceedances;
    # threshold-subtraction left no positive value; or the lower-quartile
    # anchor is zero (the grid would degenerate).
    if n < _MIN_TAIL_LEN_FOR_GPD_FIT or x[-1] <= 0:
        return float("nan")

    # Zhang & Stephens (2009) profile-likelihood grid: build m candidate values
    # of theta and take the posterior-weighted mean.  ``m = 20 + floor(sqrt(n))``
    # follows the paper's default (Section 3, equation 7).  math.floor / math.ceil
    # already return ints, so the int(...) wrapping is unnecessary (RUF046).
    m = 20 + math.floor(math.sqrt(n))
    # Anchor ``x_star`` per ArviZ / PyMC: x_{ceil(n/4)} (lower quartile of the
    # tail).  The original paper uses x_{n - floor(sqrt(n))} for the upper
    # anchor; the lower-quartile variant is numerically more stable because
    # most theta_j collapse onto 1/x_max with the upper anchor.
    star_idx = max(1, math.ceil(n / 4)) - 1  # 0-based
    x_star = x[star_idx]
    x_max = x[-1]
    if x_star <= 0:
        # Fold into the same nan path as the top-of-function guard.
        return float("nan")

    # Build the theta grid: theta_j = 1/x_max + (1 - sqrt(m/(j-0.5))) / (3*x_star).
    j = np.arange(1, m + 1, dtype=float)
    theta = 1.0 / x_max + (1.0 - np.sqrt(m / (j - 0.5))) / (3.0 * x_star)

    # k(theta) = -mean(log(1 - theta * x_i))   (Zhang & Stephens profile MLE for k)
    # log L(theta) = n * (log(theta / k(theta)) + k(theta) - 1)
    #                derived by concentrating out sigma; equivalent to Zhang &
    #                Stephens equation 5 after collecting sigma_hat = k_hat/theta.
    # In the Zhang & Stephens GPD parametrization ``k < 0`` corresponds to a
    # heavy upper tail (unbounded support); we negate at the end so the
    # returned value matches the PSIS convention ``k > 0 = heavy tail``
    # (Vehtari et al. 2024, Section 2.2).
    one_minus_tx = 1.0 - np.outer(theta, x)
    invalid = (one_minus_tx <= 0).any(axis=1)
    one_minus_tx_safe = np.where(one_minus_tx > 0, one_minus_tx, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_theta = -np.log(one_minus_tx_safe).mean(axis=1)
        # theta and k_theta always carry the same sign at the MLE (both
        # negative for heavy tail, both positive for bounded tail) because
        # k_theta(theta) = -E[log(1 - theta * x)] is a monotone function of
        # theta passing through 0 at theta = 0.  Their ratio is therefore
        # non-negative and log is well-defined except at theta = 0 (excluded
        # from the grid for n >= 2).
        ratio = np.where(k_theta != 0, theta / k_theta, np.nan)
        log_lik = n * (np.log(np.where(ratio > 0, ratio, np.nan)) + k_theta - 1.0)
    log_lik = np.where(invalid, -np.inf, log_lik)
    log_lik = np.where(np.isfinite(log_lik), log_lik, -np.inf)

    # Combine the "no valid log-likelihood" branches into one nan-return path
    # so the function stays under ruff's PLR0911 6-return cap.  All three
    # failure modes — non-finite max, zero total weight, post-fit inside <= 0
    # — leave theta_hat undefined.
    log_lik_max = log_lik.max()
    theta_hat: float | None = None
    if np.isfinite(log_lik_max):
        weights = np.exp(log_lik - log_lik_max)
        total = weights.sum()
        if total > 0:
            theta_hat = float((theta * weights).sum() / total)
    if theta_hat is None:
        return float("nan")

    if theta_hat == 0.0:
        return 0.0
    inside = 1.0 - theta_hat * x
    if (inside <= 0).any():
        return float("nan")
    # Zhang & Stephens convention: k_hat negative iff heavy upper tail.
    # Convert to PSIS convention (k > 0 iff heavy tail).
    k_hat_zs = float(-np.log(inside).mean())
    return -k_hat_zs


def per_action_propensity_diagnostics(
    propensities: np.ndarray,
    actions: np.ndarray,
    *,
    target_actions: np.ndarray | None = None,
    rare_action_floor: float = 0.01,
    n_bins: int = _ECE_DEFAULT_N_BINS,
) -> list[PerActionDiagnostics]:
    """Per-action propensity diagnostics (#131).

    Global ECE / Brier can hide an under-supported arm: a model can look
    well-calibrated on the marginal "max-prob argmax" view while being
    biased on a rare-but-strategically-important action. This routine
    re-frames the multi-class problem as ``n_actions`` independent binary
    problems and computes per-action calibration, support fractions, and
    rare-action flags.

    Parameters
    ----------
    propensities : np.ndarray
        Predicted propensities of shape ``(n_samples, n_actions)``.
    actions : np.ndarray
        Observed action indices of shape ``(n_samples,)``.
    target_actions : np.ndarray, optional
        Optional ``(n_samples,)`` array of actions the *target* policy
        would take. Used only to refine the ``rare`` flag: a logged action
        below the rarity floor is reported as ``rare=True`` only if the
        target policy puts non-zero mass on it. Without this argument
        every below-floor action is flagged.
    rare_action_floor : float, default 0.01
        Logged frequency below which the action is reported as ``rare``.
    n_bins : int, default 15
        ECE bin count.

    Returns
    -------
    list[PerActionDiagnostics]
        One entry per action in ``range(n_actions)``. Insufficient-sample
        actions report ``nan`` for ECE/Brier/log-loss and ``insufficient=True``.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions"
            f" length {len(actions)}"
        )
    if propensities.ndim != _PER_ACTION_NDIM:
        raise DataValidationError(
            f"Expected propensities of shape (n, n_actions); got {propensities.shape}"
        )

    n_total = len(actions)
    n_actions = propensities.shape[1]
    actions_int = actions.astype(int)
    target_support: set[int] | None = (
        {int(x) for x in np.unique(target_actions)}
        if target_actions is not None
        else None
    )

    rows: list[PerActionDiagnostics] = []
    for a in range(n_actions):
        mask = actions_int == a
        n_a = int(mask.sum())
        logged_frac = float(n_a / n_total) if n_total else 0.0
        p_col = propensities[:, a].astype(float)
        mean_pscore_global = float(p_col.mean()) if n_total else float("nan")
        mean_pscore_taken = float(p_col[mask].mean()) if n_a else float("nan")
        insufficient = n_a < _MIN_ACTION_COUNT_DISC

        # Binary calibration for "is this action chosen?"
        y_a = mask.astype(float)
        if insufficient or n_total < _MIN_SAMPLES_RELIABILITY:
            ece_a = float("nan")
            brier_a = float("nan")
            ll_a = float("nan")
        else:
            ece_a = _binary_ece(p_col, y_a, n_bins=n_bins)
            brier_a = float(np.mean((p_col - y_a) ** 2))
            eps = 1e-12
            ll_a = float(
                -np.mean(
                    y_a * np.log(p_col + eps) + (1 - y_a) * np.log(1 - p_col + eps)
                )
            )

        rare = logged_frac < rare_action_floor
        if rare and target_support is not None and a not in target_support:
            # Below floor but the target policy never recommends it →
            # not strategically rare; just an unused arm.
            rare = False

        rows.append(
            PerActionDiagnostics(
                action=a,
                n=n_a,
                logged_frac=logged_frac,
                mean_pscore_taken=mean_pscore_taken,
                mean_pscore_global=mean_pscore_global,
                ece=ece_a,
                brier=brier_a,
                log_loss=ll_a,
                insufficient=insufficient,
                rare=rare,
            )
        )
    return rows


def _binary_ece(probs: np.ndarray, labels: np.ndarray, *, n_bins: int) -> float:
    """Binary-classifier ECE for a single propensity column.

    Helper for :func:`per_action_propensity_diagnostics`. Returns ``nan``
    on degenerate inputs (constant predictions or empty bins everywhere).
    """
    if len(probs) != len(labels):
        return float("nan")
    n = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    used_mass = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        used_mass += count / n
        ece += (count / n) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    if used_mass == 0.0:
        return float("nan")
    return float(ece)


def comprehensive_propensity_diagnostics(
    propensities: np.ndarray,
    actions: np.ndarray,
    n_bins: int = 10,
    *,
    target_actions: np.ndarray | None = None,
) -> PropensityDiagnostics:
    """Run comprehensive propensity score diagnostics.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).
    n_bins : int, default=10
        Reliability-curve bin count.
    target_actions : np.ndarray, optional
        If provided, the per-action ``rare`` flag is gated on whether the
        target policy can pick the action (per #131's "target-support"
        qualifier). When omitted, every logged action is treated as in
        target support — the warning then conservatively flags any rare +
        insufficient action regardless of whether the target policy can
        pick it.

    Returns
    -------
    PropensityDiagnostics
        Comprehensive diagnostic results.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < _MIN_SAMPLES_LARGE:
        raise InsufficientDataError(
            "Need at least 20 samples for comprehensive diagnostics"
        )

    # Run all diagnostics
    overlap_ratio = check_propensity_overlap(propensities, actions)
    balance_ratio = check_propensity_balance(propensities, actions)
    calibration_score, calibration_curve = assess_propensity_calibration(
        propensities, actions, n_bins
    )
    discrimination_score, roc_curve = assess_propensity_discrimination(
        propensities, actions
    )
    log_loss_score = compute_propensity_log_loss(propensities, actions)
    statistics = compute_propensity_statistics(propensities, actions)
    balance_stats = compute_balance_statistics(propensities, actions)

    # Trust additions (#84): industry-standard ECE + multiclass Brier.
    # ECE uses 15 bins (BBT 2015 convention); Brier is the standard multiclass
    # form.  Both fail gracefully (nan) on samples smaller than _MIN_SAMPLES_RELIABILITY
    # rather than raising — diagnostics should never break an otherwise valid run.
    ece = compute_propensity_ece(propensities, actions, n_bins=_ECE_DEFAULT_N_BINS)
    brier_score = compute_propensity_brier(propensities, actions)
    reliability_curve = compute_propensity_reliability_curve(
        propensities, actions, n_bins=_ECE_DEFAULT_N_BINS
    )

    # #131 — Per-action calibration / rare-action / support map. ``target_actions``
    # threads through so the rare-action flag respects target-policy support.
    per_action = per_action_propensity_diagnostics(
        propensities, actions, target_actions=target_actions
    )
    finite_eces = [r.ece for r in per_action if math.isfinite(r.ece)]
    max_per_action_ece = max(finite_eces) if finite_eces else float("nan")

    return PropensityDiagnostics(
        overlap_ratio=overlap_ratio,
        balance_ratio=balance_ratio,
        calibration_score=calibration_score,
        discrimination_score=discrimination_score,
        log_loss_score=log_loss_score,
        statistics=statistics,
        balance_stats=balance_stats,
        calibration_curve=calibration_curve,
        roc_curve=roc_curve,
        ece=ece,
        brier_score=brier_score,
        reliability_curve=reliability_curve,
        ece_n_bins=_ECE_DEFAULT_N_BINS,
        per_action=per_action,
        n_rare_actions=sum(1 for r in per_action if r.rare),
        n_insufficient_actions=sum(1 for r in per_action if r.insufficient),
        n_rare_and_insufficient_actions=sum(
            1 for r in per_action if r.rare and r.insufficient
        ),
        max_per_action_ece=max_per_action_ece,
    )


def generate_propensity_report(
    diagnostics: PropensityDiagnostics, output_format: str = "text"
) -> str:
    """Generate a human-readable propensity score diagnostic report.

    Parameters
    ----------
    diagnostics : PropensityDiagnostics
        Diagnostic results.
    output_format : str, default="text"
        Output format ("text" or "markdown").

    Returns
    -------
    str
        Formatted report.
    """
    if output_format == "text":
        return _generate_text_report(diagnostics)
    elif output_format == "markdown":
        return _generate_markdown_report(diagnostics)
    else:
        raise ConfigurationError(f"Unknown output format: {output_format}")


def _generate_text_report(diagnostics: PropensityDiagnostics) -> str:
    """Generate text format report."""
    report = []
    report.append("=" * 50)
    report.append("PROPENSITY SCORE DIAGNOSTICS REPORT")
    report.append("=" * 50)
    report.append("")

    # Summary scores
    report.append("SUMMARY SCORES:")
    report.append(f"  Overlap Ratio:     {diagnostics.overlap_ratio:.3f}")
    report.append(f"  Balance Ratio:     {diagnostics.balance_ratio:.3f}")
    report.append(f"  Calibration Score: {diagnostics.calibration_score:.3f}")
    report.append(f"  Discrimination:    {diagnostics.discrimination_score:.3f}")
    report.append(f"  Log Loss:          {diagnostics.log_loss_score:.3f}")
    report.append("")

    # Statistics
    report.append("PROPENSITY SCORE STATISTICS:")
    stats = diagnostics.statistics
    report.append(f"  Min:    {stats['min_pscore']:.4f}")
    report.append(f"  Max:    {stats['max_pscore']:.4f}")
    report.append(f"  Mean:   {stats['mean_pscore']:.4f}")
    report.append(f"  Std:    {stats['std_pscore']:.4f}")
    report.append(f"  Median: {stats['median_pscore']:.4f}")
    report.append("")

    # Quantiles
    report.append("QUANTILES:")
    for q in [1, 5, 10, 25, 75, 90, 95, 99]:
        if f"pscore_q{q}" in stats:
            report.append(f"  {q:2d}th percentile: {stats[f'pscore_q{q}']:.4f}")
    report.append("")

    # Per-action balance statistics
    report.append("PER-ACTION BALANCE STATISTICS:")
    balance_stats = diagnostics.balance_stats
    action_idx = 0
    while f"action_{action_idx}_count" in balance_stats:
        count = int(balance_stats[f"action_{action_idx}_count"])
        mean_ps = balance_stats[f"action_{action_idx}_mean_pscore"]
        std_ps = balance_stats[f"action_{action_idx}_std_pscore"]
        report.append(
            f"  Action {action_idx}: count={count}, mean={mean_ps:.4f}, std={std_ps:.4f}"
        )
        action_idx += 1
    report.append("")

    return "\n".join(report)


def _generate_markdown_report(diagnostics: PropensityDiagnostics) -> str:
    """Generate markdown format report."""
    report = []
    report.append("# Propensity Score Diagnostics Report")
    report.append("")

    # Summary scores
    report.append("## Summary Scores")
    report.append("")
    report.append("| Metric | Score |")
    report.append("|--------|-------|")
    report.append(f"| Overlap Ratio | {diagnostics.overlap_ratio:.3f} |")
    report.append(f"| Balance Ratio | {diagnostics.balance_ratio:.3f} |")
    report.append(f"| Calibration Score | {diagnostics.calibration_score:.3f} |")
    report.append(f"| Discrimination | {diagnostics.discrimination_score:.3f} |")
    report.append(f"| Log Loss | {diagnostics.log_loss_score:.3f} |")
    report.append("")

    # Statistics
    report.append("## Propensity Score Statistics")
    report.append("")
    stats = diagnostics.statistics
    report.append("| Statistic | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Min | {stats['min_pscore']:.4f} |")
    report.append(f"| Max | {stats['max_pscore']:.4f} |")
    report.append(f"| Mean | {stats['mean_pscore']:.4f} |")
    report.append(f"| Std | {stats['std_pscore']:.4f} |")
    report.append(f"| Median | {stats['median_pscore']:.4f} |")
    report.append("")

    # Per-action balance statistics
    report.append("## Per-Action Balance Statistics")
    report.append("")
    report.append("| Action | Count | Mean P-score | Std P-score |")
    report.append("|--------|-------|--------------|-------------|")
    balance_stats = diagnostics.balance_stats
    action_idx = 0
    while f"action_{action_idx}_count" in balance_stats:
        count = int(balance_stats[f"action_{action_idx}_count"])
        mean_ps = balance_stats[f"action_{action_idx}_mean_pscore"]
        std_ps = balance_stats[f"action_{action_idx}_std_pscore"]
        report.append(f"| {action_idx} | {count} | {mean_ps:.4f} | {std_ps:.4f} |")
        action_idx += 1
    report.append("")

    return "\n".join(report) + "\n"
