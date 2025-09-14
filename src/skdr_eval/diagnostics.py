"""Propensity score diagnostics for skdr-eval library."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .exceptions import DataValidationError, InsufficientDataError

logger = logging.getLogger("skdr_eval")


@dataclass
class PropensityDiagnostics:
    """Container for propensity score diagnostic results."""

    overlap_ratio: float
    balance_ratio: float
    calibration_score: float
    discrimination_score: float
    log_loss_score: float
    statistics: Dict[str, float]
    balance_stats: Dict[str, float]
    calibration_curve: List[Tuple[float, float]]
    roc_curve: List[Tuple[float, float]]


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

    if len(propensities) < 10:
        raise InsufficientDataError("Need at least 10 samples for overlap analysis")

    n_actions = propensities.shape[1]
    overlap_scores = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < 2:
            continue

        action_props = propensities[action_mask, action]
        other_props = propensities[~action_mask, action]

        if len(other_props) == 0:
            continue

        # Compute overlap as the proportion of other actions that fall within
        # the range of this action's propensity scores
        min_prop = action_props.min()
        max_prop = action_props.max()
        overlap = np.mean((other_props >= min_prop) & (other_props <= max_prop))
        overlap_scores.append(overlap)

    return np.mean(overlap_scores) if overlap_scores else 0.0


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

    if len(propensities) < 10:
        raise InsufficientDataError("Need at least 10 samples for balance analysis")

    n_actions = propensities.shape[1]
    balance_scores = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < 2:
            continue

        action_props = propensities[action_mask, action]
        other_props = propensities[~action_mask, action]

        if len(other_props) == 0:
            continue

        # Compute balance as 1 - normalized difference in means
        mean_diff = abs(action_props.mean() - other_props.mean())
        max_diff = max(action_props.mean(), other_props.mean())
        balance = 1.0 - (mean_diff / max_diff) if max_diff > 0 else 1.0
        balance_scores.append(balance)

    return np.mean(balance_scores) if balance_scores else 0.0


def assess_propensity_calibration(
    propensities: np.ndarray, actions: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Tuple[float, float]]]:
    """Assess propensity score calibration.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).
    n_bins : int, default=10
        Number of bins for calibration curve.

    Returns
    -------
    float
        Calibration score (0-1, higher is better).
    List[Tuple[float, float]]
        Calibration curve (bin_centers, bin_means).
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < 20:
        raise InsufficientDataError("Need at least 20 samples for calibration analysis")

    n_actions = propensities.shape[1]
    calibration_scores = []
    calibration_curves = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < 5:
            continue

        action_props = propensities[action_mask, action]
        if len(action_props) < 5:
            continue

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = []

        for i in range(n_bins):
            bin_mask = (action_props >= bin_edges[i]) & (action_props < bin_edges[i + 1])
            if i == n_bins - 1:  # Include the last edge
                bin_mask = (action_props >= bin_edges[i]) & (action_props <= bin_edges[i + 1])

            if bin_mask.sum() > 0:
                bin_means.append(action_props[bin_mask].mean())
            else:
                bin_means.append(0.0)

        # Compute calibration score as 1 - mean absolute error
        mae = np.mean(np.abs(np.array(bin_means) - bin_centers))
        calibration_score = 1.0 - mae
        calibration_scores.append(calibration_score)
        calibration_curves.append(list(zip(bin_centers, bin_means)))

    # Return average calibration score and first curve
    avg_score = np.mean(calibration_scores) if calibration_scores else 0.0
    first_curve = calibration_curves[0] if calibration_curves else []

    return avg_score, first_curve


def assess_propensity_discrimination(
    propensities: np.ndarray, actions: np.ndarray
) -> Tuple[float, List[Tuple[float, float]]]:
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

    if len(propensities) < 20:
        raise InsufficientDataError("Need at least 20 samples for discrimination analysis")

    n_actions = propensities.shape[1]
    discrimination_scores = []
    roc_curves = []

    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() < 5:
            continue

        # Create binary labels for this action
        y_true = action_mask.astype(int)
        y_scores = propensities[:, action]

        if len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_scores)
            discrimination_scores.append(auc)

            # Compute ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_curves.append(list(zip(fpr, tpr)))
        except ValueError:
            continue

    # Return average AUC and first ROC curve
    avg_auc = np.mean(discrimination_scores) if discrimination_scores else 0.0
    first_roc = roc_curves[0] if roc_curves else []

    return avg_auc, first_roc


def compute_propensity_statistics(propensities: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
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

    if len(propensities) < 5:
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


def compute_balance_statistics(propensities: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
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

    if len(propensities) < 5:
        raise InsufficientDataError("Need at least 5 samples for balance statistics")

    n_actions = propensities.shape[1]
    balance_stats = {}

    for action in range(n_actions):
        action_mask = actions == action
        action_props = propensities[action_mask, action]

        balance_stats[f"action_{action}_count"] = float(action_mask.sum())
        balance_stats[f"action_{action}_mean_pscore"] = float(action_props.mean()) if len(action_props) > 0 else 0.0
        balance_stats[f"action_{action}_std_pscore"] = float(action_props.std()) if len(action_props) > 0 else 0.0

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

    if len(propensities) < 5:
        raise InsufficientDataError("Need at least 5 samples for log loss")

    # Create one-hot encoded labels
    n_actions = propensities.shape[1]
    y_true = np.zeros((len(actions), n_actions))
    y_true[np.arange(len(actions)), actions] = 1

    return float(log_loss(y_true, propensities))


def comprehensive_propensity_diagnostics(
    propensities: np.ndarray, actions: np.ndarray, n_bins: int = 10
) -> PropensityDiagnostics:
    """Run comprehensive propensity score diagnostics.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).

    Returns
    -------
    PropensityDiagnostics
        Comprehensive diagnostic results.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < 20:
        raise InsufficientDataError("Need at least 20 samples for comprehensive diagnostics")

    # Run all diagnostics
    overlap_ratio = check_propensity_overlap(propensities, actions)
    balance_ratio = check_propensity_balance(propensities, actions)
    calibration_score, calibration_curve = assess_propensity_calibration(propensities, actions, n_bins)
    discrimination_score, roc_curve = assess_propensity_discrimination(propensities, actions)
    log_loss_score = compute_propensity_log_loss(propensities, actions)
    statistics = compute_propensity_statistics(propensities, actions)
    balance_stats = compute_balance_statistics(propensities, actions)

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
        raise ValueError(f"Unknown output format: {output_format}")


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

    return "\n".join(report)