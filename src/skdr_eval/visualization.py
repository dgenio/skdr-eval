"""Visualization tools for skdr-eval library."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .exceptions import DataValidationError, InsufficientDataError

logger = logging.getLogger("skdr_eval")

# Set default style
plt.style.use("default")
sns.set_palette("husl")


def plot_propensity_distribution(
    propensities: np.ndarray,
    actions: np.ndarray,
    action_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot propensity score distributions by action.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).
    action_names : List[str], optional
        Names for actions. If None, uses indices.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < 10:
        raise InsufficientDataError("Need at least 10 samples for visualization")

    n_actions = propensities.shape[1]
    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Propensity Score Distributions", fontsize=16, fontweight="bold")

    # Plot 1: Histogram of propensity scores by action
    ax1 = axes[0, 0]
    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() > 0:
            action_props = propensities[action_mask, action]
            ax1.hist(
                action_props,
                bins=20,
                alpha=0.7,
                label=action_names[action],
                density=True,
            )
    ax1.set_xlabel("Propensity Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution by Action")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot of propensity scores
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() > 0:
            action_props = propensities[action_mask, action]
            box_data.append(action_props)
            box_labels.append(action_names[action])
    
    if box_data:
        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_ylabel("Propensity Score")
        ax2.set_title("Box Plot by Action")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Overlap visualization
    ax3 = axes[1, 0]
    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() > 0:
            action_props = propensities[action_mask, action]
            ax3.hist(
                action_props,
                bins=20,
                alpha=0.5,
                label=action_names[action],
                density=True,
            )
    ax3.set_xlabel("Propensity Score")
    ax3.set_ylabel("Density")
    ax3.set_title("Overlap Visualization")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Propensity score heatmap
    ax4 = axes[1, 1]
    # Sample a subset for heatmap if too many samples
    n_samples_heatmap = min(100, len(propensities))
    sample_indices = np.random.choice(len(propensities), n_samples_heatmap, replace=False)
    heatmap_data = propensities[sample_indices]
    
    im = ax4.imshow(heatmap_data.T, aspect="auto", cmap="viridis")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Action")
    ax4.set_title("Propensity Score Heatmap")
    ax4.set_yticks(range(n_actions))
    ax4.set_yticklabels(action_names)
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_dr_results(
    results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot DR/SNDR evaluation results.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Results dictionary with model names as keys and metrics as values.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if not results:
        raise DataValidationError("Results dictionary cannot be empty")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("DR/SNDR Evaluation Results", fontsize=16, fontweight="bold")

    # Extract data
    models = list(results.keys())
    dr_values = [results[model].get("V_hat", 0) for model in models]
    dr_errors = [results[model].get("SE_if", 0) for model in models]
    ess_values = [results[model].get("ESS", 0) for model in models]
    clip_values = [results[model].get("clip", 0) for model in models]

    # Plot 1: DR values with error bars
    ax1 = axes[0, 0]
    x_pos = np.arange(len(models))
    ax1.bar(x_pos, dr_values, yerr=dr_errors, capsize=5, alpha=0.7)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("DR Value")
    ax1.set_title("DR Values with Standard Errors")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effective Sample Size
    ax2 = axes[0, 1]
    ax2.bar(x_pos, ess_values, alpha=0.7, color="orange")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Effective Sample Size")
    ax2.set_title("Effective Sample Size")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Clipping thresholds
    ax3 = axes[1, 0]
    ax3.bar(x_pos, clip_values, alpha=0.7, color="green")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Clipping Threshold")
    ax3.set_title("Selected Clipping Thresholds")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model comparison
    ax4 = axes[1, 1]
    metrics = ["V_hat", "ESS", "clip"]
    x = np.arange(len(metrics))
    width = 0.35

    for i, model in enumerate(models):
        values = [results[model].get(metric, 0) for metric in metrics]
        # Normalize values for comparison
        values = np.array(values)
        if values.max() > 0:
            values = values / values.max()
        ax4.bar(x + i * width, values, width, label=model, alpha=0.7)

    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Normalized Value")
    ax4.set_title("Model Comparison (Normalized)")
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_calibration_curve(
    calibration_curve: List[Tuple[float, float]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot propensity score calibration curve.

    Parameters
    ----------
    calibration_curve : List[Tuple[float, float]]
        Calibration curve data (bin_centers, bin_means).
    figsize : Tuple[int, int], default=(8, 6)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if not calibration_curve:
        raise DataValidationError("Calibration curve data cannot be empty")

    fig, ax = plt.subplots(figsize=figsize)
    
    bin_centers, bin_means = zip(*calibration_curve)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)

    # Plot calibration curve
    ax.plot(bin_centers, bin_means, "o-", label="Calibration Curve", linewidth=2, markersize=6)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Propensity Score Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_roc_curve(
    roc_curve: List[Tuple[float, float]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot ROC curve for propensity score discrimination.

    Parameters
    ----------
    roc_curve : List[Tuple[float, float]]
        ROC curve data (fpr, tpr).
    figsize : Tuple[int, int], default=(8, 6)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if not roc_curve:
        raise DataValidationError("ROC curve data cannot be empty")

    fig, ax = plt.subplots(figsize=figsize)
    
    fpr, tpr = zip(*roc_curve)
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, "b-", label="ROC Curve", linewidth=2)
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", alpha=0.7)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve for Propensity Score Discrimination")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_diagnostics_summary(
    diagnostics: "PropensityDiagnostics",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot comprehensive diagnostics summary.

    Parameters
    ----------
    diagnostics : PropensityDiagnostics
        Diagnostic results.
    figsize : Tuple[int, int], default=(15, 10)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("Propensity Score Diagnostics Summary", fontsize=16, fontweight="bold")

    # Plot 1: Summary scores
    ax1 = axes[0, 0]
    scores = [
        diagnostics.overlap_ratio,
        diagnostics.balance_ratio,
        diagnostics.calibration_score,
        diagnostics.discrimination_score,
    ]
    score_names = ["Overlap", "Balance", "Calibration", "Discrimination"]
    colors = ["skyblue", "lightgreen", "orange", "pink"]
    
    bars = ax1.bar(score_names, scores, color=colors, alpha=0.7)
    ax1.set_ylabel("Score")
    ax1.set_title("Summary Scores")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

    # Plot 2: Log loss
    ax2 = axes[0, 1]
    ax2.bar(["Log Loss"], [diagnostics.log_loss_score], color="red", alpha=0.7)
    ax2.set_ylabel("Log Loss")
    ax2.set_title("Log Loss Score")
    ax2.text(0, diagnostics.log_loss_score + 0.01, f'{diagnostics.log_loss_score:.3f}',
             ha='center', va='bottom')

    # Plot 3: Propensity score statistics
    ax3 = axes[0, 2]
    stats = diagnostics.statistics
    stat_names = ["Min", "Max", "Mean", "Std", "Median"]
    stat_values = [
        stats.get("min_pscore", 0),
        stats.get("max_pscore", 0),
        stats.get("mean_pscore", 0),
        stats.get("std_pscore", 0),
        stats.get("median_pscore", 0),
    ]
    ax3.bar(stat_names, stat_values, color="lightcoral", alpha=0.7)
    ax3.set_ylabel("Propensity Score")
    ax3.set_title("Propensity Score Statistics")
    ax3.tick_params(axis="x", rotation=45)

    # Plot 4: Calibration curve
    ax4 = axes[1, 0]
    if diagnostics.calibration_curve:
        bin_centers, bin_means = zip(*diagnostics.calibration_curve)
        ax4.plot(bin_centers, bin_means, "o-", label="Calibration", linewidth=2)
        ax4.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax4.set_xlabel("Mean Predicted")
    ax4.set_ylabel("Fraction of Positives")
    ax4.set_title("Calibration Curve")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: ROC curve
    ax5 = axes[1, 1]
    if diagnostics.roc_curve:
        fpr, tpr = zip(*diagnostics.roc_curve)
        ax5.plot(fpr, tpr, "b-", label="ROC", linewidth=2)
        ax5.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax5.set_xlabel("False Positive Rate")
    ax5.set_ylabel("True Positive Rate")
    ax5.set_title("ROC Curve")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Quantiles
    ax6 = axes[1, 2]
    quantiles = diagnostics.quantiles
    if quantiles:
        q_names = list(quantiles.keys())
        q_values = list(quantiles.values())
        ax6.plot(q_names, q_values, "o-", linewidth=2, markersize=6)
    ax6.set_ylabel("Propensity Score")
    ax6.set_title("Quantiles")
    ax6.tick_params(axis="x", rotation=45)
    ax6.grid(True, alpha=0.3)

    # Plot 7-9: Balance statistics
    ax7 = axes[2, 0]
    balance_stats = diagnostics.balance_stats
    if balance_stats:
        action_counts = [v for k, v in balance_stats.items() if k.endswith("_count")]
        action_names = [f"Action {i}" for i in range(len(action_counts))]
        ax7.bar(action_names, action_counts, alpha=0.7)
    ax7.set_ylabel("Count")
    ax7.set_title("Action Counts")
    ax7.tick_params(axis="x", rotation=45)

    ax8 = axes[2, 1]
    if balance_stats:
        mean_scores = [v for k, v in balance_stats.items() if k.endswith("_mean_pscore")]
        action_names = [f"Action {i}" for i in range(len(mean_scores))]
        ax8.bar(action_names, mean_scores, alpha=0.7, color="lightblue")
    ax8.set_ylabel("Mean Propensity Score")
    ax8.set_title("Mean Propensity by Action")
    ax8.tick_params(axis="x", rotation=45)

    ax9 = axes[2, 2]
    if balance_stats:
        std_scores = [v for k, v in balance_stats.items() if k.endswith("_std_pscore")]
        action_names = [f"Action {i}" for i in range(len(std_scores))]
        ax9.bar(action_names, std_scores, alpha=0.7, color="lightgreen")
    ax9.set_ylabel("Std Propensity Score")
    ax9.set_title("Std Propensity by Action")
    ax9.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def create_dashboard(
    propensities: np.ndarray,
    actions: np.ndarray,
    results: Optional[Dict[str, Dict[str, float]]] = None,
    diagnostics: Optional["PropensityDiagnostics"] = None,
    action_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 15),
    save_path: Optional[str] = None,
) -> Figure:
    """Create a comprehensive dashboard with all visualizations.

    Parameters
    ----------
    propensities : np.ndarray
        Propensity scores (n_samples, n_actions).
    actions : np.ndarray
        Action indices (n_samples,).
    results : Dict[str, Dict[str, float]], optional
        DR/SNDR evaluation results.
    diagnostics : PropensityDiagnostics, optional
        Diagnostic results.
    action_names : List[str], optional
        Names for actions.
    figsize : Tuple[int, int], default=(20, 15)
        Figure size.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if len(propensities) != len(actions):
        raise DataValidationError(
            f"Propensities length {len(propensities)} doesn't match actions length {len(actions)}"
        )

    if len(propensities) < 10:
        raise InsufficientDataError("Need at least 10 samples for dashboard")

    fig = plt.figure(figsize=figsize)
    fig.suptitle("skdr-eval Comprehensive Dashboard", fontsize=20, fontweight="bold")

    # Create subplots
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Propensity distributions
    ax1 = fig.add_subplot(gs[0, :2])
    n_actions = propensities.shape[1]
    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]
    
    for action in range(n_actions):
        action_mask = actions == action
        if action_mask.sum() > 0:
            action_props = propensities[action_mask, action]
            ax1.hist(action_props, bins=20, alpha=0.7, label=action_names[action], density=True)
    ax1.set_xlabel("Propensity Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Propensity Score Distributions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Summary scores (if diagnostics available)
    ax2 = fig.add_subplot(gs[0, 2:])
    if diagnostics:
        scores = [
            diagnostics.overlap_ratio,
            diagnostics.balance_ratio,
            diagnostics.calibration_score,
            diagnostics.discrimination_score,
        ]
        score_names = ["Overlap", "Balance", "Calibration", "Discrimination"]
        bars = ax2.bar(score_names, scores, alpha=0.7)
        ax2.set_ylabel("Score")
        ax2.set_title("Diagnostic Scores")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", rotation=45)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, "No diagnostics available", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Diagnostic Scores")

    # Plot 3: DR results (if available)
    ax3 = fig.add_subplot(gs[1, :2])
    if results:
        models = list(results.keys())
        dr_values = [results[model].get("V_hat", 0) for model in models]
        dr_errors = [results[model].get("SE_if", 0) for model in models]
        
        x_pos = np.arange(len(models))
        ax3.bar(x_pos, dr_values, yerr=dr_errors, capsize=5, alpha=0.7)
        ax3.set_xlabel("Model")
        ax3.set_ylabel("DR Value")
        ax3.set_title("DR Values with Standard Errors")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models, rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No results available", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("DR Results")

    # Plot 4: Calibration curve (if diagnostics available)
    ax4 = fig.add_subplot(gs[1, 2:])
    if diagnostics and diagnostics.calibration_curve:
        bin_centers, bin_means = zip(*diagnostics.calibration_curve)
        ax4.plot(bin_centers, bin_means, "o-", label="Calibration", linewidth=2)
        ax4.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax4.set_xlabel("Mean Predicted")
        ax4.set_ylabel("Fraction of Positives")
        ax4.set_title("Calibration Curve")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No calibration data", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Calibration Curve")

    # Plot 5: ROC curve (if diagnostics available)
    ax5 = fig.add_subplot(gs[2, :2])
    if diagnostics and diagnostics.roc_curve:
        fpr, tpr = zip(*diagnostics.roc_curve)
        ax5.plot(fpr, tpr, "b-", label="ROC", linewidth=2)
        ax5.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax5.set_xlabel("False Positive Rate")
        ax5.set_ylabel("True Positive Rate")
        ax5.set_title("ROC Curve")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No ROC data", ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("ROC Curve")

    # Plot 6: Propensity score heatmap
    ax6 = fig.add_subplot(gs[2, 2:])
    n_samples_heatmap = min(100, len(propensities))
    sample_indices = np.random.choice(len(propensities), n_samples_heatmap, replace=False)
    heatmap_data = propensities[sample_indices]
    
    im = ax6.imshow(heatmap_data.T, aspect="auto", cmap="viridis")
    ax6.set_xlabel("Sample Index")
    ax6.set_ylabel("Action")
    ax6.set_title("Propensity Score Heatmap")
    ax6.set_yticks(range(n_actions))
    ax6.set_yticklabels(action_names)
    plt.colorbar(im, ax=ax6)

    # Plot 7: Balance statistics (if diagnostics available)
    ax7 = fig.add_subplot(gs[3, :2])
    if diagnostics and diagnostics.balance_stats:
        balance_stats = diagnostics.balance_stats
        mean_scores = [v for k, v in balance_stats.items() if k.endswith("_mean_pscore")]
        action_names_short = [f"Action {i}" for i in range(len(mean_scores))]
        ax7.bar(action_names_short, mean_scores, alpha=0.7, color="lightblue")
        ax7.set_ylabel("Mean Propensity Score")
        ax7.set_title("Mean Propensity by Action")
        ax7.tick_params(axis="x", rotation=45)
    else:
        ax7.text(0.5, 0.5, "No balance data", ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title("Balance Statistics")

    # Plot 8: Summary statistics
    ax8 = fig.add_subplot(gs[3, 2:])
    if diagnostics:
        stats = diagnostics.statistics
        stat_names = ["Min", "Max", "Mean", "Std", "Median"]
        stat_values = [
            stats.get("min_pscore", 0),
            stats.get("max_pscore", 0),
            stats.get("mean_pscore", 0),
            stats.get("std_pscore", 0),
            stats.get("median_pscore", 0),
        ]
        ax8.bar(stat_names, stat_values, alpha=0.7, color="lightcoral")
        ax8.set_ylabel("Propensity Score")
        ax8.set_title("Propensity Score Statistics")
        ax8.tick_params(axis="x", rotation=45)
    else:
        ax8.text(0.5, 0.5, "No statistics available", ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title("Summary Statistics")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Dashboard saved to {save_path}")

    return fig