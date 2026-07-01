"""Core implementation of DR and Stabilized DR for offline policy evaluation."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .reporting import EvaluationArtifact, SupportHealthThresholds
    from .trackers import Tracker

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ._frames import coerce_to_pandas
from .choice import (
    SCIPY_AVAILABLE,
    fit_conditional_logit_with_sampling,
    predict_proba_condlogit,
)
from .diagnostics import (
    PropensityDiagnostics,
    comprehensive_propensity_diagnostics,
    generate_propensity_report,
)
from .exceptions import (
    ConvergenceError,
    DataValidationError,
    InsufficientDataError,
    InsufficientOverlapError,
    ModelValidationError,
    OutcomeModelError,
    PolicyInductionError,
    PropensityScoreError,
)
from .pairwise import (
    LARGE_DATA_ROW_THRESHOLD,
    PairwiseDesign,
    build_eval_arrays_vectorized,
    build_policy_probs_vectorized,
    induce_policy,
    map_external_policies,
)
from .validation import (
    validate_dataframe,
    validate_finite_values,
    validate_models_dict,
    validate_numpy_array,
    validate_positive_integer,
    validate_probabilities,
    validate_random_state,
    validate_sklearn_estimator,
    validate_string_choice,
)

logger = logging.getLogger("skdr_eval")


def _autolog_tracker(
    tracker: "Tracker | None",
    artifact: "EvaluationArtifact",
    *,
    evaluator: str,
) -> None:
    """Push metrics, tags, and one card per model to ``tracker`` (#93).

    Silently returns when ``tracker is None`` (the default). Errors from the
    tracker are logged but not re-raised — instrumentation must never break a
    finished evaluation.
    """
    if tracker is None:
        return
    try:
        tracker.set_tag("evaluator", evaluator)
        version = artifact.metadata.get("skdr_eval_version")
        if version is not None:
            tracker.set_tag("skdr_eval_version", str(version))
        seed = artifact.metadata.get("random_state")
        if seed is not None:
            tracker.set_tag("random_state", str(seed))
        n_splits_meta = artifact.metadata.get("n_splits")
        if n_splits_meta is not None:
            tracker.set_tag("n_splits", str(n_splits_meta))
        for _, row in artifact.report.iterrows():
            prefix = f"{row['model']}/{row['estimator']}"
            for col in (
                "V_hat",
                "ci_lower",
                "ci_upper",
                "ESS",
                "match_rate",
                "pareto_k",
            ):
                value = row.get(col)
                if value is None:
                    continue
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fvalue):
                    continue
                tracker.log_metric(f"{prefix}/{col}", fvalue)
        for model_name in artifact.detailed:
            for estimator in ("DR", "SNDR"):
                if estimator not in artifact.detailed[model_name]:
                    continue
                try:
                    card = artifact.card_schema(model_name, estimator=estimator)
                    tracker.log_card(card)
                except Exception as exc:
                    logger.warning(
                        "tracker.log_card failed for %s/%s: %s",
                        model_name,
                        estimator,
                        exc,
                    )
    except Exception as exc:
        logger.warning("tracker auto-log failed: %s", exc)


def _make_time_series_split(
    n_samples: int,
    n_splits: int,
    *,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
) -> TimeSeriesSplit:
    """Build a ``TimeSeriesSplit`` with validated temporal-split controls.

    ``gap``, ``test_size``, and ``max_train_size`` are passed straight through
    to :class:`sklearn.model_selection.TimeSeriesSplit`; we only enforce
    pre-conditions that would otherwise surface as opaque sklearn errors deep
    in fold iteration.

    The default ``gap=1`` is a conservative leakage guard for adjacent
    observations in time-ordered data — it prevents the fold immediately
    following the training cut from touching the last training row.
    """
    # n_splits >= 2 is a TimeSeriesSplit precondition; surface it eagerly
    # with our typed error rather than letting sklearn raise deep in .split().
    _MIN_SPLITS = 2
    if not isinstance(n_splits, int) or n_splits < _MIN_SPLITS:
        raise DataValidationError(
            f"n_splits must be an integer >= {_MIN_SPLITS}, got {n_splits!r}"
        )
    if not isinstance(gap, int) or gap < 0:
        raise DataValidationError(f"gap must be a non-negative integer, got {gap!r}")
    if test_size is not None and (not isinstance(test_size, int) or test_size <= 0):
        raise DataValidationError(
            f"test_size must be a positive integer or None, got {test_size!r}"
        )
    if max_train_size is not None and (
        not isinstance(max_train_size, int) or max_train_size <= 0
    ):
        raise DataValidationError(
            f"max_train_size must be a positive integer or None, got {max_train_size!r}"
        )

    # Feasibility: sklearn's TimeSeriesSplit needs n_samples > n_splits *
    # test_size + gap. We also enforce the n_splits * 2 floor so every fold
    # contains at least one train sample even when test_size is implicit.
    effective_test = test_size if test_size is not None else 1
    min_required = max(
        n_splits * 2,
        n_splits * effective_test + gap + 1,
    )
    if n_samples < min_required:
        raise InsufficientDataError(
            f"Need at least {min_required} samples for "
            f"n_splits={n_splits}, gap={gap}, test_size={test_size}; "
            f"got {n_samples}"
        )

    return TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
    )


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
        Fraction of decisions in the DR overlap set — where the behavior and
        the target policy both put positive probability on the observed action
        (within the eligibility mask). For a deterministic target this is the
        familiar "policy agrees with the log" rate; for a full-support
        stochastic target it is the behavior-support rate.
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
    pareto_k : float, default ``nan``
        PSIS Pareto-k shape parameter (#80) estimated on the *unclipped*
        importance weights of the matched subset. ``k < 0.5`` is reliable,
        ``0.5 ≤ k < 0.7`` is caution, ``k ≥ 0.7`` is high-risk. ``nan`` when
        the matched subset is too small to fit a tail.
    contributions : dict[str, np.ndarray] | None, optional
        Per-decision contributions captured at the selected clip. ``None``
        unless the evaluator was called with ``keep_contributions=True`` or
        ``ci_bootstrap=True``. Keys: ``decision_id`` (int row position into
        the evaluated slice), ``q_pi``, ``q_hat``, ``weight``, ``reward``,
        ``contribution_to_V``. By construction
        ``contribution_to_V.mean() == V_hat`` to float64 precision for both
        DR and SNDR (issue #92).
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
    pareto_k: float = float("nan")
    contributions: dict[str, np.ndarray] | None = field(default=None)


def build_design(
    logs: pd.DataFrame,
    cli_pref: str = "cli_",
    st_pref: str = "st_",
    y_col: str = "service_time",
) -> Design:
    """Build design matrices from logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Log data with columns: arrival_ts, cli_*, st_*, op_*_elig, action, and
        the reward column ``y_col`` (``service_time`` by default).
    cli_pref : str, default="cli_"
        Prefix for client features.
    st_pref : str, default="st_"
        Prefix for service-time features.
    y_col : str, default="service_time"
        Name of the reward/outcome column read into ``Design.Y``. Defaults to
        ``"service_time"`` for backward compatibility.

    Returns
    -------
    Design
        Design matrices and metadata.

    Raises
    ------
    DataValidationError
        If input data is invalid
    InsufficientDataError
        If there's insufficient data for estimation
    """
    try:
        # Validate input DataFrame
        required_columns = ["arrival_ts", "action", y_col]
        validate_dataframe(logs, "logs", required_columns, min_rows=10)

        # Validate prefixes
        validate_string_choice(cli_pref, "cli_pref", ["cli_", "client_", ""])
        validate_string_choice(st_pref, "st_pref", ["st_", "service_", ""])

        # Extract operators from eligibility columns
        elig_cols = [col for col in logs.columns if col.endswith("_elig")]
        if not elig_cols:
            raise DataValidationError(
                "No eligibility columns found (ending with '_elig')"
            )

        ops_all = [col.replace("_elig", "") for col in elig_cols]
        idx = {op: i for i, op in enumerate(ops_all)}

        # Validate actions are valid
        invalid_actions = set(logs["action"]) - set(ops_all)
        if invalid_actions:
            raise DataValidationError(
                f"Invalid actions found: {sorted(invalid_actions)}. "
                f"Valid actions: {sorted(ops_all)}"
            )

        # Base features (context)
        cli_cols = [col for col in logs.columns if col.startswith(cli_pref)]
        st_cols = [col for col in logs.columns if col.startswith(st_pref)]
        base_cols = cli_cols + st_cols

        if not base_cols:
            raise DataValidationError(
                f"No base features found with prefixes '{cli_pref}' or '{st_pref}'"
            )

        X_base = logs[base_cols].values
        validate_numpy_array(X_base, "X_base", min_size=1)
        validate_finite_values(X_base, "X_base")

        # Eligibility matrix
        elig = logs[elig_cols].values
        validate_numpy_array(elig, "elig", expected_shape=(len(logs), len(ops_all)))

        # Check eligibility values are 0 or 1
        if not np.all(np.isin(elig, [0, 1])):
            raise DataValidationError("Eligibility matrix must contain only 0s and 1s")

        # Action indices
        A = np.array([idx[action] for action in logs["action"]])
        validate_numpy_array(A, "A", min_size=1)
        validate_finite_values(A, "A")

        # Observed features (base + action one-hot)
        action_onehot = np.zeros((len(logs), len(ops_all)))
        action_onehot[np.arange(len(logs)), A] = 1
        X_obs = np.column_stack([X_base, action_onehot])
        validate_finite_values(X_obs, "X_obs")

        # Propensity features (base + standardized time, no action)
        scaler = StandardScaler()
        ts_norm = scaler.fit_transform(logs[["arrival_ts"]].values.astype(float))
        X_phi = np.column_stack([X_base, ts_norm])
        validate_finite_values(X_phi, "X_phi")

        # Outcomes and timestamps
        Y: np.ndarray = logs[y_col].values.astype(np.float64)
        ts: np.ndarray = logs["arrival_ts"].values.astype(np.float64)

        validate_numpy_array(Y, "Y", min_size=1)
        validate_finite_values(Y, "Y")
        validate_numpy_array(ts, "ts", min_size=1)
        validate_finite_values(ts, "ts")

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

    except Exception as e:
        if isinstance(e, (DataValidationError, InsufficientDataError)):
            raise
        raise DataValidationError(f"Error building design: {e!s}") from e


def _check_n_jobs(n_jobs: int) -> None:
    """Validate a joblib ``n_jobs`` before it reaches :class:`joblib.Parallel`.

    joblib rejects ``0`` with an opaque error; skdr-eval fails loudly with a
    clear message instead (#178 review). ``1`` is serial, ``-1`` uses all
    cores, and other negatives follow joblib's ``n_cpus + 1 + n_jobs`` rule.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero (1 = serial, -1 = all cores)")


def fit_propensity_timecal(
    X_phi: np.ndarray,
    A: np.ndarray,
    ts: np.ndarray | None = None,
    n_splits: int = 3,
    random_state: int = 0,
    *,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
    n_jobs: int = 1,
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
    gap : int, default=1
        Number of samples to exclude between the end of each training fold and
        the start of the corresponding test fold. The default of ``1`` is a
        conservative guard against adjacent-row temporal leakage; set to ``0``
        to recover the unbuffered sklearn default.
    test_size : int, optional
        Per-fold test-window size in samples. ``None`` (default) defers to
        sklearn's automatic sizing.
    max_train_size : int, optional
        Cap the training-fold size in samples (sliding-window CV). ``None``
        (default) uses an expanding window.
    n_jobs : int, default=1
        Number of joblib workers for the cross-validation folds (#178). Folds
        are independent fits writing disjoint test indices, so the result is
        identical to the serial path for any ``n_jobs``. ``-1`` uses all cores.

    Returns
    -------
    propensities : np.ndarray
        Calibrated propensity scores (n_samples, n_actions).
    fold_indices : np.ndarray
        Fold assignment for each sample.

    Raises
    ------
    DataValidationError
        If input data is invalid
    PropensityScoreError
        If propensity score estimation fails
    ConvergenceError
        If optimization fails to converge
    """
    # Validate n_jobs outside the try so the message isn't rewrapped as a
    # PropensityScoreError (#178 review).
    _check_n_jobs(n_jobs)
    try:
        # Validate inputs
        validate_numpy_array(X_phi, "X_phi", min_size=1)
        validate_finite_values(X_phi, "X_phi")

        validate_numpy_array(A, "A", min_size=1)
        validate_finite_values(A, "A")

        if ts is not None:
            validate_numpy_array(ts, "ts", min_size=1)
            validate_finite_values(ts, "ts")
            if len(ts) != len(X_phi):
                raise DataValidationError(
                    f"ts length {len(ts)} doesn't match X_phi length {len(X_phi)}"
                )

        validate_positive_integer(n_splits, "n_splits")
        validate_random_state(random_state)

        if len(X_phi) != len(A):
            raise DataValidationError(
                f"X_phi length {len(X_phi)} doesn't match A length {len(A)}"
            )

        n_samples, _ = X_phi.shape
        n_actions = A.max() + 1

        if n_actions <= 1:
            raise InsufficientDataError(f"Need at least 2 actions, got {n_actions}")

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
        tscv = _make_time_series_split(
            n_samples,
            n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )
        propensities = np.zeros((n_samples, n_actions))
        fold_indices = np.full(n_samples, -1)

        def _fit_one_fold(
            fold: int, train_idx: np.ndarray, test_idx: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, int]:
            # Map sorted indices back to original order for fold assignment
            original_test_idx = time_order[test_idx]

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
                    cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
                    cal_clf.fit(X_train, A_train)

                    # Get calibrated predictions
                    if (
                        hasattr(cal_clf, "classes_")
                        and len(cal_clf.classes_) < n_actions
                    ):
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

            return original_test_idx, pred_proba, fold

        folds = list(enumerate(tscv.split(X_phi_sorted)))
        # Folds are independent fits writing disjoint test indices, so threading
        # is bit-identical to the serial loop (#178).
        if n_jobs == 1:
            fold_results = [_fit_one_fold(f, tr, te) for f, (tr, te) in folds]
        else:
            fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one_fold)(f, tr, te) for f, (tr, te) in folds
            )
        for original_test_idx, pred_proba, fold in fold_results:
            propensities[original_test_idx] = pred_proba
            fold_indices[original_test_idx] = fold

        # Handle samples not assigned to any fold (shouldn't happen with TimeSeriesSplit but be safe)
        unassigned_mask = fold_indices == -1
        if np.any(unassigned_mask):
            # Assign uniform probabilities to unassigned samples
            propensities[unassigned_mask] = 1.0 / n_actions
            # Assign them to the last fold
            fold_indices[unassigned_mask] = n_splits - 1

        # Validate output
        validate_probabilities(propensities, "propensities")
        validate_finite_values(fold_indices, "fold_indices")

        return propensities, fold_indices

    except Exception as e:
        if isinstance(
            e,
            (
                DataValidationError,
                InsufficientDataError,
                PropensityScoreError,
                ConvergenceError,
            ),
        ):
            raise
        raise PropensityScoreError(f"Error fitting propensity model: {e!s}") from e


def fit_outcome_crossfit(
    X_obs: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 3,
    estimator: str | Callable[[], Any] = "hgb",
    random_state: int = 0,
    *,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
    sample_weight: np.ndarray | None = None,
    n_jobs: int = 1,
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
    gap : int, default=1
        Number of samples skipped between train and test windows; see
        :func:`fit_propensity_timecal` for full semantics.
    test_size : int, optional
        Per-fold test-window size in samples.
    max_train_size : int, optional
        Cap on training-fold size in samples.
    sample_weight : np.ndarray, optional
        Per-row non-negative sample weights forwarded to the underlying
        estimator's ``fit(..., sample_weight=...)``. Required by MRDR (#86)
        to recover the variance-minimising outcome model under DR. ``None``
        (default) means unweighted MSE.
    n_jobs : int, default=1
        Number of joblib workers for the cross-fitting folds (#178). Folds are
        independent fits writing disjoint test indices, so the result is
        identical to the serial path for any ``n_jobs``. ``-1`` uses all cores.

    Returns
    -------
    predictions : np.ndarray
        Cross-fitted predictions.
    models_info : List[Tuple[Any, np.ndarray, np.ndarray]]
        List of (model, train_idx, test_idx) for each fold.

    Raises
    ------
    DataValidationError
        If input data is invalid
    OutcomeModelError
        If outcome model fitting fails
    ModelValidationError
        If estimator is invalid
    """
    # Validate n_jobs outside the try so the message isn't rewrapped as an
    # OutcomeModelError (#178 review).
    _check_n_jobs(n_jobs)
    try:
        # Validate inputs
        validate_numpy_array(X_obs, "X_obs", min_size=1)
        validate_finite_values(X_obs, "X_obs")

        validate_numpy_array(Y, "Y", min_size=1)
        validate_finite_values(Y, "Y")

        validate_positive_integer(n_splits, "n_splits")
        validate_random_state(random_state)

        if len(X_obs) != len(Y):
            raise DataValidationError(
                f"X_obs length {len(X_obs)} doesn't match Y length {len(Y)}"
            )

        n_samples = X_obs.shape[0]
        predictions = np.zeros(n_samples)
        models_info = []

        # Get estimator factory
        estimator_factories = {
            "hgb": lambda: HistGradientBoostingRegressor(random_state=random_state),
            "ridge": lambda: Ridge(random_state=random_state),
            "rf": lambda: RandomForestRegressor(random_state=random_state),
        }

        if isinstance(estimator, str) and estimator in estimator_factories:
            est_factory = estimator_factories[estimator]
        elif callable(estimator):
            est_factory = estimator
            # Validate the estimator
            try:
                test_est = est_factory()
                validate_sklearn_estimator(test_est, "estimator", ["fit", "predict"])
            except Exception as e:
                raise ModelValidationError(f"Invalid estimator factory: {e!s}") from e
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        # Time-series split
        tscv = _make_time_series_split(
            n_samples,
            n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )

        if sample_weight is not None:
            validate_numpy_array(sample_weight, "sample_weight", min_size=1)
            validate_finite_values(sample_weight, "sample_weight")
            if sample_weight.shape != (n_samples,):
                raise DataValidationError(
                    f"sample_weight shape {sample_weight.shape} must be "
                    f"(n_samples,) = ({n_samples},), matching X_obs.shape[0]"
                )
            if np.any(sample_weight < 0):
                raise DataValidationError(
                    "sample_weight must be non-negative; received negatives"
                )

        def _fit_one_fold(
            fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray
        ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
            try:
                X_train, X_test = X_obs[train_idx], X_obs[test_idx]
                Y_train = Y[train_idx]

                # Fit model — pass sample_weight when supplied. Falls back to
                # unweighted fit when the underlying estimator does not accept
                # the kwarg (rare in sklearn but defensive for callable
                # factories returning custom regressors).
                model = est_factory()
                if sample_weight is not None:
                    sw_train = sample_weight[train_idx]
                    try:
                        model.fit(X_train, Y_train, sample_weight=sw_train)
                    except TypeError:
                        # Underlying estimator does not accept sample_weight;
                        # fall back rather than silently ignore.
                        raise OutcomeModelError(
                            f"Estimator {model!r} does not accept "
                            "sample_weight=; MRDR / weighted outcome losses "
                            "are unavailable for this estimator."
                        ) from None
                else:
                    model.fit(X_train, Y_train)

                # Predict
                pred = model.predict(X_test)
                validate_finite_values(pred, f"predictions_fold_{fold_idx}")
            except Exception as e:
                raise OutcomeModelError(f"Error in fold {fold_idx}: {e!s}") from e
            return model, train_idx, test_idx, pred

        folds = list(enumerate(tscv.split(X_obs)))
        # Folds are independent fits writing disjoint test indices, so threading
        # is bit-identical to the serial loop (#178). Reassemble in fold order
        # to keep ``models_info`` deterministic.
        if n_jobs == 1:
            fold_results = [_fit_one_fold(fi, tr, te) for fi, (tr, te) in folds]
        else:
            fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_one_fold)(fi, tr, te) for fi, (tr, te) in folds
            )
        for model, train_idx, test_idx, pred in fold_results:
            predictions[test_idx] = pred
            models_info.append((model, train_idx, test_idx))

        # Validate output
        validate_finite_values(predictions, "predictions")

        return predictions, models_info

    except Exception as e:
        if isinstance(
            e,
            (
                DataValidationError,
                InsufficientDataError,
                OutcomeModelError,
                ModelValidationError,
            ),
        ):
            raise
        raise OutcomeModelError(f"Error fitting outcome model: {e!s}") from e


def induce_policy_from_sklearn(
    model: Any,
    X_base: np.ndarray,
    ops_all: list[str],
    elig: np.ndarray,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Induce policy from sklearn model by predicting service times.

    Vectorized: builds a single stacked ``(sum_i n_elig_i, n_features + n_ops)``
    feature matrix and issues **one** ``model.predict`` call instead of the
    per-(sample, eligible-op) python loop (closes #46).

    When ``chunk_size`` is set, the stacked feature matrix — the dominant
    transient allocation on large logs — is built and predicted in row-blocks
    of at most ``chunk_size`` eligible (sample, op) pairs, bounding peak memory
    without changing the result (predictions are row-independent, so the output
    is bit-identical to the single-shot path; this backs ``execution_mode=
    "large_data"`` in :func:`evaluate_sklearn_models`, #210).

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

    Returns
    -------
    policy_probs : np.ndarray
        Policy probabilities (n_samples, n_ops).

    Raises
    ------
    DataValidationError
        If input data is invalid
    PolicyInductionError
        If policy induction fails
    ModelValidationError
        If model is invalid
    """
    try:
        # Validate inputs
        validate_sklearn_estimator(model, "model", ["predict"])

        validate_numpy_array(X_base, "X_base", min_size=1)
        validate_finite_values(X_base, "X_base")

        if not ops_all:
            raise DataValidationError("ops_all cannot be empty")

        validate_numpy_array(elig, "elig", min_size=1)
        if not np.all(np.isin(elig, [0, 1])):
            raise DataValidationError("Eligibility matrix must contain only 0s and 1s")

        if len(X_base) != len(elig):
            raise DataValidationError(
                f"X_base length {len(X_base)} doesn't match elig length {len(elig)}"
            )

        if elig.shape[1] != len(ops_all):
            raise DataValidationError(
                f"elig width {elig.shape[1]} doesn't match ops_all length {len(ops_all)}"
            )

        n_samples, _ = X_base.shape
        n_ops = len(ops_all)
        elig_bool = elig.astype(bool)

        # Sample / op index pairs for every eligible (sample, op) cell. Order
        # is row-major: sample_idx_flat == np.repeat(arange(n), elig.sum(1)).
        sample_idx_flat, op_idx_flat = np.nonzero(elig_bool)

        policy_probs = np.zeros((n_samples, n_ops), dtype=np.float64)

        if sample_idx_flat.size > 0:
            n_pairs = sample_idx_flat.size

            if chunk_size is not None and chunk_size <= 0:
                raise DataValidationError(
                    f"chunk_size must be positive, got {chunk_size}"
                )

            def _stack_and_predict(s_idx: np.ndarray, o_idx: np.ndarray) -> np.ndarray:
                # Build [X_base[sample] | one_hot(op)] for this block of pairs.
                X_repeated = X_base[s_idx]
                onehot = np.zeros((s_idx.size, n_ops), dtype=X_base.dtype)
                onehot[np.arange(s_idx.size), o_idx] = 1
                X_stacked = np.concatenate([X_repeated, onehot], axis=1)
                return np.asarray(model.predict(X_stacked))

            if chunk_size is None or chunk_size >= n_pairs:
                # Single stacked allocation — the unchanged fast path.
                preds = _stack_and_predict(sample_idx_flat, op_idx_flat)
            else:
                # Memory-bounded path (#210): predict in row-blocks and scatter
                # into a preallocated vector. Pairs are processed in the same
                # row-major order, so ``preds`` is identical to the single-shot
                # array and all downstream handling below is unchanged.
                preds = np.empty(n_pairs, dtype=np.float64)
                for start in range(0, n_pairs, chunk_size):
                    stop = min(start + chunk_size, n_pairs)
                    preds[start:stop] = _stack_and_predict(
                        sample_idx_flat[start:stop], op_idx_flat[start:stop]
                    )

            if not np.all(np.isfinite(preds)):
                bad = int(np.argmax(~np.isfinite(preds)))
                raise PolicyInductionError(
                    f"Non-finite prediction for sample "
                    f"{int(sample_idx_flat[bad])}, operator "
                    f"{int(op_idx_flat[bad])}"
                )
            if np.any(preds < 0):
                logger.warning(
                    "Negative predictions encountered; using absolute values"
                )
                preds = np.abs(preds)

            # Scatter 1 / (pred + eps) back into the dense policy matrix.
            policy_probs[sample_idx_flat, op_idx_flat] = 1.0 / (preds + 1e-8)

        # Samples with no eligible operators get a uniform distribution
        # across *all* ops (matches the prior scalar behavior).
        no_elig = ~elig_bool.any(axis=1)
        if no_elig.any():
            policy_probs[no_elig] = 1.0 / n_ops

        # Row-normalize. Rows with at least one eligible op sum to >0 because
        # 1/(pred+eps) is strictly positive after the abs() above; rows that
        # got the uniform fallback already sum to 1. Use in-place division so
        # mypy keeps the ndarray dtype (`a / b` would surface as Any under the
        # current numpy stubs and trip ``no-any-return`` on the return below).
        row_sums = policy_probs.sum(axis=1, keepdims=True)
        policy_probs /= row_sums

        # Validate output
        validate_probabilities(policy_probs, "policy_probs")

        return policy_probs

    except Exception as e:
        if isinstance(
            e, (DataValidationError, PolicyInductionError, ModelValidationError)
        ):
            raise
        raise PolicyInductionError(f"Error inducing policy: {e!s}") from e


def _dr_weight_components(
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared building blocks for the DR importance weights (#106).

    The doubly-robust importance ratio for the *observed* action is

        ``w_raw = π(A_i | x_i) / e(A_i | x_i)``

    — the **target-policy** probability over the **behavior** (logging)
    propensity. The earlier implementation used only ``1 / e(A_i | x_i)`` and
    so dropped the ``π(A_i | x_i)`` numerator. That made every estimate
    independent of the policy being evaluated, so different candidate models
    collapsed to an identical ``V_hat`` and the correction term no longer
    targeted any well-defined policy value. See issue #106.

    Parameters
    ----------
    propensities : np.ndarray
        Behavior propensities ``e(a | x)`` (n_samples, n_actions).
    policy_probs : np.ndarray
        Target policy ``π(a | x)`` (n_samples, n_actions).
    A : np.ndarray
        Observed action indices (n_samples,).
    elig : np.ndarray
        Eligibility matrix (n_samples, n_actions).

    Returns
    -------
    pi_obs : np.ndarray
        Behavior propensity of the observed action, ``e(A_i | x_i)``.
    w_raw : np.ndarray
        Unclipped DR importance ratio ``π(A_i | x_i) / e(A_i | x_i)``; zero
        where the behavior propensity is zero.
    matched : np.ndarray
        Boolean DR overlap set: rows where the observed action is eligible and
        both the behavior **and** the target put positive probability on it.
        ``match_rate`` is ``matched.mean()`` — for a deterministic target this
        is the familiar "policy agrees with the log" rate; for a full-support
        stochastic target it is the behavior-support rate.
    """
    n = len(A)
    idx = np.arange(n)
    A_int = A.astype(int)
    pi_obs: np.ndarray = propensities[idx, A_int]
    pi_target_obs: np.ndarray = policy_probs[idx, A_int]
    elig_bool = elig.astype(bool)
    matched: np.ndarray = (pi_obs > 0) & (pi_target_obs > 0) & elig_bool[idx, A_int]
    w_raw: np.ndarray = np.divide(
        pi_target_obs,
        pi_obs,
        out=np.zeros(n, dtype=np.float64),
        where=pi_obs > 0,
    )
    return pi_obs, w_raw, matched


def _clip_weights(w_raw: np.ndarray, matched: np.ndarray, clip: float) -> np.ndarray:
    """Clip the raw DR importance ratio and zero it outside the overlap set."""
    w_clip: np.ndarray
    if clip == float("inf"):
        w_clip = np.where(matched, w_raw, 0.0)
    else:
        w_clip = np.where(matched, np.minimum(w_raw, clip), 0.0)
    return w_clip


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

    # Compute policy-weighted outcome model prediction.
    #
    # Design note (q_pi == q_hat simplification):
    # q_hat is 1D (n_samples,) from fit_outcome_crossfit — one prediction per
    # observation, conditioned on the observed features but not on the action.
    # The reshape to (n, 1) broadcasts across policy_probs (n, n_actions) and
    # the sum over axis=1 yields  q_hat[i] * Σ_a π(a|x_i) = q_hat[i] * 1.
    # So q_pi == q_hat by construction.
    #
    # This is intentional: the outcome model is a marginal predictor E[Y|X],
    # not a per-action model E[Y|X, A=a].  A per-action outcome model would
    # produce a 2D q_hat (n, n_actions), and the reshape + sum would then
    # compute the textbook  Σ_a π(a|x) q̂(x, a).  The code is written to
    # support both shapes without changes — only fit_outcome_crossfit would
    # need to return a 2D array.  See issue #58 for discussion.
    #
    # With a marginal (1D) q_hat the *direct-method* term is policy-independent,
    # but the DR estimate as a whole is NOT: the policy enters through the
    # importance weight π(A|x)/e(A|x) below (the correction term). With a
    # per-action q_hat the direct term would carry the policy as well. The
    # earlier code dropped the π(A|x) numerator from the weight, which made the
    # whole estimate collapse to a policy-independent value — see issue #106.
    q_pi = np.sum(policy_probs * q_hat.reshape(n_samples, -1), axis=1)

    # DR importance ratio for the observed action: π(A|x) / e(A|x). The
    # target-policy probability π(A|x) is essential — without it every
    # candidate policy collapses to the same estimate (see #106).
    pi_obs, w_raw, matched = _dr_weight_components(propensities, policy_probs, A, elig)

    if matched.sum() == 0:
        raise ValueError("No matched samples found")

    # Diagnostics on matched set
    pi_matched = pi_obs[matched]
    match_rate = matched.mean()
    min_pscore = pi_matched.min()
    pscore_q01 = np.percentile(pi_matched, 1)
    pscore_q05 = np.percentile(pi_matched, 5)
    pscore_q10 = np.percentile(pi_matched, 10)

    # PSIS Pareto-k (#80): fit on the *unclipped* importance ratios for the
    # matched subset.  This is the standard PSIS usage — the diagnostic is
    # supposed to tell you whether the raw weights have a problematic tail,
    # independent of any clipping the user might apply.  See Vehtari et al.
    # 2024 (JMLR 25:1-58).  Defer the import to avoid a top-of-module circular
    # cycle with reporting.py.
    from .diagnostics import psis_pareto_k  # noqa: PLC0415

    raw_weights_matched = w_raw[matched]
    pareto_k = psis_pareto_k(raw_weights_matched)

    for clip_val in clip_grid:
        # Clip the DR importance ratio π(A|x)/e(A|x) at clip_val.
        w_clip = _clip_weights(w_raw, matched, clip_val)

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

        # Tail mass: fraction of overlap-set decisions whose raw DR weight
        # π(A|x)/e(A|x) exceeds the clip and is therefore truncated.
        if clip_val == float("inf"):
            tail_mass = 0.0
        else:
            tail_mass = float((w_raw[matched] > clip_val).mean())

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
        pareto_k=float(pareto_k),
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
        pareto_k=float(pareto_k),
    )

    return {"DR": dr_result, "SNDR": sndr_result}


def _resolve_baseline(
    baseline: float | str | None, eval_Y: np.ndarray
) -> tuple[str | None, float | None]:
    """Resolve the ``baseline=`` evaluator kwarg to ``(kind, value)`` (#132).

    Accepts:

    - ``None`` (default): no baseline; no delta columns are emitted.
    - ``float``: a fixed scalar baseline (e.g. an SLA target or a
      previously-published number); ``kind = "scalar"``.
    - ``"logged"``: the empirical mean of the *evaluation slice* of the
      logged reward; ``kind = "logged"``. This is the right baseline for
      "does my candidate policy beat the logged policy on this slice?".

    Unknown strings raise :class:`DataValidationError` rather than
    silently treating them as a no-op so typos surface immediately.
    """
    if baseline is None:
        return None, None
    # ``bool`` is a subclass of ``int`` in Python, so it would silently coerce
    # to 0.0 / 1.0 via the scalar branch below. That is almost never what a
    # caller passing ``baseline=some_flag`` meant — reject it loudly.
    if isinstance(baseline, bool):
        from .exceptions import DataValidationError  # noqa: PLC0415

        raise DataValidationError(
            f"baseline must be float, 'logged', or None; got bool {baseline!r}."
        )
    if isinstance(baseline, str):
        if baseline != "logged":
            from .exceptions import DataValidationError  # noqa: PLC0415

            raise DataValidationError(
                f"Unknown baseline string {baseline!r}; expected"
                " 'logged' or a numeric value."
            )
        return "logged", float(np.asarray(eval_Y, dtype=float).mean())
    return "scalar", float(baseline)


# Names of extra estimators that the strategy seam (#86, #85) recognises.
# These are evaluated *in addition* to the historical ``("DR", "SNDR")`` pair.
EXTRA_ESTIMATORS = ("MRDR", "SWITCH-DR", "DRos", "MIPS")

# Action-signal gap (output of ``embedding_sufficiency_diagnostic``) above which
# the MIPS path warns that the embedding may be insufficient and MIPS biased.
# Mirrors the diagnostic's own "approximately sufficient" band (gap < 1%).
_MIPS_SUFFICIENCY_WARN_GAP = 0.01

# A per-action (2D) ``q_hat`` carries one column per action; the observed-action
# prediction is then a gather along axis 1.
_PER_ACTION_NDIM = 2


def _canonical_estimator_name(name: str) -> str:
    """Canonicalise estimator names so case / separators don't drift."""
    canonical = name.strip().upper().replace("_", "-")
    if canonical == "DROS":
        return "DRos"
    if canonical in {"DR", "SNDR", "MRDR", "SWITCH-DR", "MIPS"}:
        return canonical
    raise ValueError(
        f"Unknown estimator {name!r}. Known: DR, SNDR, MRDR, SWITCH-DR, "
        "DRos, MIPS (case-insensitive)."
    )


def _resolve_action_embedding(
    action_embedding: "np.ndarray | str | None",
    logs: pd.DataFrame,
    actions: np.ndarray,
    n_actions: int,
    *,
    allow_column_name: bool = True,
) -> np.ndarray | None:
    """Resolve ``action_embedding`` to an ``(n_actions, embed_dim)`` array (#136).

    Accepts three forms:

    * ``None`` — returned unchanged (MIPS then falls back to SNDR).
    * ``np.ndarray`` — validated to be 2D and returned as float64.
    * ``str`` — a column name in ``logs`` holding a per-row embedding vector.
      Action-level embeddings are reconstructed by averaging each action's
      logged row embeddings, which is exact when the embedding is a fixed
      per-action feature and robust (mean) when it carries per-row noise.
    """
    if action_embedding is None:
        return None
    if isinstance(action_embedding, str):
        if not allow_column_name:
            raise DataValidationError(
                "action_embedding column names are not supported for the "
                "pairwise evaluator: the action index 'A' is a day-relative "
                "operator index, so averaging logged-row embeddings by it would "
                "mix different operators across days and silently produce "
                "incorrect per-action embeddings. Pass an explicit "
                "(n_actions, embed_dim) array in the same action-indexing "
                "scheme as 'A'/'policy_probs' instead."
            )
        if action_embedding not in logs.columns:
            raise DataValidationError(
                f"action_embedding column {action_embedding!r} not found in logs; "
                f"available columns: {list(logs.columns)}"
            )
        rows = [np.asarray(v, dtype=np.float64).ravel() for v in logs[action_embedding]]
        if len(rows) != len(actions):
            raise DataValidationError(
                f"action_embedding column {action_embedding!r} has {len(rows)} rows "
                f"but the evaluated design has {len(actions)} logged rows; pass an "
                "(n_actions, embed_dim) array instead when the row alignment is "
                "ambiguous (e.g. the pairwise path)."
            )
        widths = {r.shape[0] for r in rows}
        if len(widths) != 1:
            raise DataValidationError(
                f"action_embedding column {action_embedding!r} has ragged vectors "
                f"with widths {sorted(widths)}; every row must share one embed_dim."
            )
        per_row = np.asarray(rows, dtype=np.float64)
        embed_dim = per_row.shape[1]
        emb = np.zeros((n_actions, embed_dim), dtype=np.float64)
        a_int = np.asarray(actions).astype(int)
        for a in range(n_actions):
            sel = per_row[a_int == a]
            if sel.shape[0] > 0:
                emb[a] = sel.mean(axis=0)
        return emb
    arr = np.asarray(action_embedding, dtype=np.float64)
    if arr.ndim != _PER_ACTION_NDIM:
        raise DataValidationError(
            f"action_embedding array must be 2D (n_actions, embed_dim), got "
            f"ndim={arr.ndim}"
        )
    return arr


def _apply_extra_estimators(
    *,
    estimators: tuple[str, ...],
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    X_obs: np.ndarray,
    selected_clip: float,
    action_embedding: np.ndarray | None,
    switch_tau: float,
    dros_lam: float,
    mips_bandwidth: float | str,
    mips_kernel: "str | Callable[[np.ndarray], np.ndarray]",
    outcome_estimator: "str | Callable[[], Any]",
    n_splits: int,
    random_state: int,
    gap: int,
    test_size: int | None,
    max_train_size: int | None,
) -> dict[str, "DRResult"]:
    """Compute MRDR / SWITCH-DR / DRos / MIPS values via the strategy seam.

    Returns a ``{estimator_name: DRResult}`` mapping that the caller merges
    into ``detailed_results[model_name]`` alongside the base ``DR`` /
    ``SNDR`` rows. The ``selected_clip`` argument should be the DR clip
    chosen by :func:`dr_value_with_clip` — MRDR / SWITCH-DR / DRos reuse it
    so the operating point stays consistent with the base run.
    """
    from .estimators import (  # noqa: PLC0415
        MSEOutcomeLoss,
        build_strategy,
        dr_value_with_strategy,
        embedding_sufficiency_diagnostic,
    )
    from .estimators.protocols import EstimatorStrategy  # noqa: PLC0415
    from .estimators.weight_transforms import ClipTransform  # noqa: PLC0415

    extras: dict[str, DRResult] = {}
    # MRDR uses a sample-weighted outcome refit; defer the refit until we
    # know whether MRDR is actually requested (cost amortises per model).
    mrdr_q_hat: np.ndarray | None = None
    # Preserve the caller's estimator order (deduplicated) so the report rows
    # are deterministic — a set iterates in hash-seed-dependent order.
    seen_canonical: set[str] = set()
    canonical_estimators: list[str] = []
    for name in estimators:
        canonical_name = _canonical_estimator_name(name)
        if canonical_name not in seen_canonical:
            seen_canonical.add(canonical_name)
            canonical_estimators.append(canonical_name)
    for canonical in canonical_estimators:
        if canonical in {"DR", "SNDR"}:
            continue
        if canonical == "MIPS" and action_embedding is None:
            # #136 / #85: graceful fallback to SNDR with a clear warning rather
            # than a hard failure. The MIPS row in the report carries the SNDR
            # value so the report stays rectangular and the user is told why.
            warnings.warn(
                "MIPS estimator was requested but action_embedding= was not "
                "supplied; falling back to SNDR for the 'MIPS' row. Pass an "
                "(n_actions, embed_dim) array (or a logs column name) via "
                "action_embedding= to compute true MIPS.",
                UserWarning,
                stacklevel=2,
            )
            fallback = EstimatorStrategy(
                name="MIPS",
                weight_transform=ClipTransform(
                    clip=float(selected_clip)
                    if np.isfinite(selected_clip)
                    else float("inf")
                ),
                outcome_loss=MSEOutcomeLoss(),
                self_normalised=True,
            )
            extras["MIPS"] = dr_value_with_strategy(
                propensities=propensities,
                policy_probs=policy_probs,
                Y=Y,
                q_hat=q_hat,
                A=A,
                elig=elig,
                strategy=fallback,
            )
            continue

        strategy = build_strategy(
            canonical,
            clip=float(selected_clip) if np.isfinite(selected_clip) else float("inf"),
            tau=switch_tau,
            lam=dros_lam,
            action_embedding=action_embedding,
            bandwidth=mips_bandwidth,
            kernel=mips_kernel,
        )

        if canonical == "MRDR":
            if mrdr_q_hat is None:
                sw = strategy.outcome_loss(
                    pi_obs=propensities[np.arange(len(Y)), A.astype(int)],
                    policy_probs=policy_probs,
                    A=A,
                )
                mrdr_q_hat, _ = fit_outcome_crossfit(
                    X_obs,
                    Y,
                    n_splits=n_splits,
                    estimator=outcome_estimator,
                    random_state=random_state,
                    gap=gap,
                    test_size=test_size,
                    max_train_size=max_train_size,
                    sample_weight=sw,
                )
            q_hat_for_strategy = mrdr_q_hat
        else:
            q_hat_for_strategy = q_hat

        if canonical == "MIPS":
            # invariants.md: MIPS is biased when the embedding is not a
            # sufficient statistic for the reward, so surface the
            # embedding-sufficiency diagnostic to the user. ``q_hat`` may be
            # per-action (2D); the diagnostic wants the observed-action slice.
            assert action_embedding is not None  # guarded by the raise above
            q_hat_obs = (
                q_hat[np.arange(len(Y)), A.astype(int)]
                if q_hat.ndim == _PER_ACTION_NDIM
                else q_hat
            )
            sufficiency = embedding_sufficiency_diagnostic(
                Y=Y, q_hat=q_hat_obs, A=A, action_embedding=action_embedding
            )
            if sufficiency.r2_action >= _MIPS_SUFFICIENCY_WARN_GAP:
                warnings.warn(
                    "MIPS embedding may be insufficient (action-signal gap="
                    f"{sufficiency.r2_action:.3f}): {sufficiency.notes} MIPS can "
                    "be biased; inspect skdr_eval.embedding_sufficiency_diagnostic().",
                    UserWarning,
                    stacklevel=2,
                )

        result = dr_value_with_strategy(
            propensities=propensities,
            policy_probs=policy_probs,
            Y=Y,
            q_hat=q_hat_for_strategy,
            A=A,
            elig=elig,
            strategy=strategy,
            action_embedding=action_embedding,
        )
        extras[strategy.name] = result
    return extras


# Maximum number of evaluated decisions for which per-decision contributions
# may be captured. At 5 float64 columns x 100M decisions x 2 estimators this is
# ~8 GB; a higher value is a deliberate opt-in.
DEFAULT_MAX_KEPT_CONTRIBUTIONS = 100_000_000


def _compute_contributions(
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    clip: float,
    *,
    eval_log_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute per-decision DR ingredients at a selected clip.

    Returns ``(q_pi, w_clip, dr_contrib, decision_id, matched)`` arrays of
    length ``n = len(Y)``. ``dr_contrib`` is the textbook DR pseudo-outcome
    ``q_pi + w_clip * (Y - q_hat)``; ``mean(dr_contrib) == V_DR`` to float64
    precision. ``decision_id`` is ``eval_log_indices`` if supplied (so
    callers can join contributions back to the original logs after a
    pre-split), else ``arange(n)``.

    This helper exists so the new ``keep_contributions`` path (issue #92)
    and the existing ``ci_bootstrap`` path share one code path — otherwise
    the identity test ``mean(contribution_to_V) == V_hat`` could drift from
    the value the bootstrap CI conditions on.
    """
    n = len(Y)
    # q_pi == q_hat when q_hat is 1D; see design note in dr_value_with_clip.
    q_pi = np.sum(policy_probs * q_hat.reshape(n, -1), axis=1)
    # DR importance ratio π(A|x)/e(A|x) over the overlap set (#106).
    _pi_obs, w_raw, matched = _dr_weight_components(propensities, policy_probs, A, elig)
    w_clip = _clip_weights(w_raw, matched, clip)

    dr_contrib = q_pi + w_clip * (Y - q_hat)

    decision_id: np.ndarray
    if eval_log_indices is None:
        decision_id = np.arange(n, dtype=np.int64)
    else:
        decision_id = np.asarray(eval_log_indices, dtype=np.int64)
        if len(decision_id) != n:
            raise DataValidationError(
                f"eval_log_indices length {len(decision_id)} != n_decisions {n}",
            )

    return q_pi, w_clip, dr_contrib, decision_id, matched


def _build_contributions_payload(
    results: dict[str, "DRResult"],
    propensities: np.ndarray,
    policy_probs: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
    A: np.ndarray,
    elig: np.ndarray,
    *,
    eval_log_indices: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build ``{estimator: contributions_dict}`` for every entry in ``results``.

    DR uses the textbook pseudo-outcome ``q_pi + w*(Y - q_hat)`` whose mean
    is ``V_DR``. SNDR rescales the residual term by ``n / Σw`` so that the
    per-decision values average to the ratio estimator's ``V_SNDR``
    (``q_pi.mean() + Σw*(Y-q_hat)/Σw``).
    """
    payload: dict[str, dict[str, np.ndarray]] = {}
    for estimator_name, result in results.items():
        q_pi, w_clip, dr_contrib, decision_id, _matched = _compute_contributions(
            propensities,
            policy_probs,
            Y,
            q_hat,
            A,
            elig,
            result.clip,
            eval_log_indices=eval_log_indices,
        )

        if estimator_name == "SNDR":
            w_sum = float(w_clip.sum())
            n = len(Y)
            if w_sum > 0:
                contribution_to_V = q_pi + (n * w_clip / w_sum) * (Y - q_hat)
            else:
                # SNDR fallback path matches dr_value_with_clip: q_pi.mean().
                # Unreachable from the public API because dr_value_with_clip
                # raises ValueError("No matched samples found") before
                # returning when w_clip.sum() would be 0.
                contribution_to_V = q_pi.copy()  # pragma: no cover
        else:
            contribution_to_V = dr_contrib

        payload[estimator_name] = {
            "decision_id": decision_id,
            "q_pi": q_pi,
            "q_hat": np.asarray(q_hat, dtype=np.float64),
            "weight": w_clip,
            "reward": np.asarray(Y, dtype=np.float64),
            "contribution_to_V": contribution_to_V,
        }
    return payload


def _sndr_bootstrap_values(
    q_pi: np.ndarray,
    w_clip: np.ndarray,
    Y: np.ndarray,
    q_hat: np.ndarray,
) -> np.ndarray:
    """Compute SNDR pseudo-outcomes for bootstrap CI.

    Uses the self-normalised formula: q_pi + (n/Σw)*w*(Y - q_hat)
    so that mean(bootstrap_values) == V_SNDR.
    """
    w_sum = float(w_clip.sum())
    n = len(Y)
    if w_sum > 0:
        result: np.ndarray = q_pi + (n * w_clip / w_sum) * (Y - q_hat)
        return result
    copied: np.ndarray = q_pi.copy()
    return copied


def block_bootstrap_ci(
    values_num: np.ndarray,
    values_den: np.ndarray | None,
    base_mean: np.ndarray,  # noqa: ARG001
    n_boot: int = 400,
    block_len: int | None = None,
    alpha: float = 0.05,
    random_state: int = 0,
    n_jobs: int = 1,
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
    n_jobs : int, default=1
        Number of joblib workers for the bootstrap replicates (#178). Each
        replicate draws from an **independent** child of
        ``np.random.SeedSequence(random_state)``, so the returned interval is
        deterministic for a given ``random_state`` *regardless* of ``n_jobs``
        — running serial (``n_jobs=1``) and parallel produces bit-identical
        CIs. ``-1`` uses all available cores.

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

    _check_n_jobs(n_jobs)

    n = len(values_num)

    if block_len is None:
        block_len = max(1, int(np.sqrt(n)))

    # Ensure block_len doesn't exceed data length
    block_len = min(block_len, n)
    n_blocks = int(np.ceil(n / block_len))

    # One independent, order-stable RNG per replicate. Spawning from a single
    # SeedSequence decouples the result from execution order, so parallel and
    # serial runs agree exactly and the CI stays reproducible under any
    # ``n_jobs`` (#178).
    seeds = np.random.SeedSequence(random_state).spawn(n_boot)

    def _replicate(seed: "np.random.SeedSequence") -> float:
        rng = np.random.default_rng(seed)
        # Moving-block resample: n_blocks blocks, each a contiguous run of up to
        # block_len rows starting at a random offset, concatenated and trimmed
        # to the original length.
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        boot_indices = np.concatenate(
            [np.arange(s, min(s + block_len, n)) for s in starts]
        )[:n]

        boot_num = values_num[boot_indices]
        if values_den is not None:
            boot_den = values_den[boot_indices]
            den_sum = boot_den.sum()
            return float(boot_num.sum() / den_sum) if den_sum > 0 else 0.0
        return float(boot_num.mean())

    if n_jobs == 1:
        bootstrap_stats = np.array([_replicate(s) for s in seeds])
    else:
        # Threads: the per-replicate work is numpy indexing / reductions that
        # release the GIL, so we avoid pickling the (shared, read-only) value
        # arrays to worker processes.
        bootstrap_stats = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_replicate)(s) for s in seeds
            )
        )

    # Compute percentile confidence interval
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)


def _requires_overlap_precheck(
    design: "Design",
    *,
    random_state: int,
    overlap_floor: float,
    min_match_rate: float,
) -> None:
    """Cheap positivity/overlap gate run *before* the expensive nuisance fit.

    Estimates a coarse propensity with a single (un-cross-fitted, uncalibrated)
    :class:`~sklearn.linear_model.LogisticRegression` and inspects the logged
    data's own support: the eligibility match rate and the smallest estimated
    probability of an observed action. When either is below its floor the logs
    cannot support *any* candidate policy, so we raise
    :class:`InsufficientOverlapError` immediately instead of paying for
    cross-fitting and the per-model bootstrap (#206).

    This is a fast-fail guard, not the authoritative diagnostic — the full
    ``support_health`` / ``gate_diagnostics`` path on the returned artifact
    remains the source of truth for borderline cases.
    """
    A = design.A.astype(int)
    elig = design.elig
    n = len(A)

    # Match rate: fraction of rows whose observed action is eligible — the
    # overlap set the DR estimator can actually use.
    matched = elig[np.arange(n), A] == 1
    match_rate = float(matched.mean()) if n else 0.0
    if match_rate < min_match_rate:
        raise InsufficientOverlapError(
            f"Overlap precheck failed: only {match_rate:.1%} of rows have their "
            f"observed action marked eligible (min_match_rate={min_match_rate:.1%}). "
            "The logged data does not overlap the candidate policies enough for "
            "off-policy evaluation. Fix eligibility/action encoding or supply "
            "logs with adequate support."
        )

    # Coarse, single-shot propensity: skip when there is only one observed
    # action (the downstream estimator raises a clearer 'need >=2 actions').
    if len(np.unique(A)) < 2:  # noqa: PLR2004
        return
    try:
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(design.X_phi, A)
        proba = clf.predict_proba(design.X_phi)
        class_to_col = {int(c): i for i, c in enumerate(clf.classes_)}
        obs_cols = np.array([class_to_col.get(int(a), -1) for a in A])
        seen = obs_cols >= 0
        if not seen.any():
            return
        pi_obs = proba[np.arange(n)[seen], obs_cols[seen]]
        min_pscore = float(pi_obs.min())
    except (ValueError, RuntimeError):
        # A failed coarse fit is not itself evidence of no overlap; defer to the
        # full path rather than raising a misleading precheck error.
        return

    if min_pscore < overlap_floor:
        raise InsufficientOverlapError(
            f"Overlap precheck failed: smallest estimated propensity of an "
            f"observed action is {min_pscore:.2e} (overlap_floor={overlap_floor:.2e}). "
            "Some logged actions are taken in regions where they are essentially "
            "never chosen, so importance weights would be unbounded and no OPE "
            "method can rescue the estimate. Trim those rows or collect logs with "
            "better positivity."
        )


def evaluate_sklearn_models(
    logs: pd.DataFrame,
    models: dict[str, Any],
    fit_models: bool = True,
    n_splits: int = 3,
    outcome_estimator: str | Callable[[], Any] = "hgb",
    random_state: int = 0,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    policy_train: str | None = None,
    policy_train_frac: float = 0.85,
    support_thresholds: "SupportHealthThresholds | None" = None,
    *,
    y_col: str = "service_time",
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
    keep_contributions: bool = False,
    max_kept_contributions: int = DEFAULT_MAX_KEPT_CONTRIBUTIONS,
    estimators: tuple[str, ...] = ("DR", "SNDR"),
    action_embedding: "np.ndarray | str | None" = None,
    switch_tau: float = 5.0,
    dros_lam: float = 1.0,
    mips_bandwidth: float | str = 1.0,
    mips_kernel: "str | Callable[[np.ndarray], np.ndarray]" = "rbf",
    tracker: "Tracker | None" = None,
    baseline: float | str | None = None,
    n_jobs: int = 1,
    execution_mode: Literal["auto", "standard", "large_data"] = "auto",
    chunk_size: int = 100_000,
    requires_overlap: bool = False,
    overlap_floor: float = 1e-3,
    min_match_rate: float = 0.05,
) -> "EvaluationArtifact":
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
        Whether to compute bootstrap confidence intervals via
        ``block_bootstrap_ci`` on the per-decision DR contributions. Note that
        this is a **conditional bootstrap**: ``q_hat``, propensities, and the
        policy itself are held fixed across replicates, so the resulting CI
        captures sampling variability of the influence function *given* the
        nuisances but not variability *of* the nuisances. May under-cover when
        nuisance estimation error is non-negligible.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    policy_train : str, optional
        Training data for policy when ``fit_models=True``. ``"pre_split"``
        (default) fits on the first ``policy_train_frac`` of the data and
        evaluates on the held-out tail — the statistically-safe choice.
        ``"all"`` fits and evaluates on the same data; retained for backward
        compatibility but introduces training-on-test bias when ``fit_models=True``.
        Passing ``None`` (or omitting the argument) uses ``"pre_split"`` and
        emits a :class:`DeprecationWarning`; pass an explicit value to suppress
        the warning. The default will become a required keyword in a future
        release.
    policy_train_frac : float, default=0.85
        Fraction of data for policy training if policy_train="pre_split".
    support_thresholds : SupportHealthThresholds, optional
        Thresholds for support-health warnings attached to the returned
        artifact. See :class:`skdr_eval.reporting.SupportHealthThresholds`.
    y_col : str, default="service_time"
        Name of the reward/outcome column in ``logs``. Defaults to
        ``"service_time"`` for backward compatibility; set it for
        general-purpose OPE logs whose reward is named e.g. ``"reward"``,
        ``"click"``, or ``"revenue"``.
    keep_contributions : bool, default=False
        If True, attach per-decision DR/SNDR contributions to each
        :class:`DRResult` under ``contributions`` and make them queryable
        via :meth:`EvaluationArtifact.contributions` (issue #92). Default
        is False to preserve the memory profile. Independent of
        ``ci_bootstrap``: enabling bootstrap CIs does not attach a payload.
    max_kept_contributions : int, default=100_000_000
        Hard cap on the number of evaluated decisions for which per-decision
        contributions may be captured. Raises :class:`DataValidationError`
        if ``n_eval`` exceeds this value while contributions are being kept.
    estimators : tuple[str, ...], default=("DR", "SNDR")
        Estimators to compute. Supported: ``"DR"``, ``"SNDR"``, ``"MRDR"``,
        ``"SWITCH-DR"``, ``"DRos"``, ``"MIPS"``. Each maps to a built-in
        strategy via :func:`skdr_eval.estimators.build_strategy`.
    action_embedding : np.ndarray, optional
        Per-action embedding of shape ``(n_actions, embed_dim)``. **Required
        when** ``"MIPS"`` is requested; ignored by the other estimators.
    switch_tau : float, default=5.0
        SWITCH-DR importance-weight threshold: weights above ``switch_tau``
        fall back to the direct method. Used only for ``"SWITCH-DR"``.
    dros_lam : float, default=1.0
        DRos optimistic-shrinkage parameter ``lambda``. Used only for
        ``"DRos"``.
    mips_bandwidth : float, default=1.0
        Gaussian-kernel bandwidth for the MIPS embedding marginal. Used only
        for ``"MIPS"``.
    n_jobs : int, default=1
        Number of parallel workers (#178). Applies to three independent axes:
        the candidate-model loop, the cross-fitting folds of the propensity and
        outcome nuisances, and the bootstrap replicates when ``ci_bootstrap=
        True``. Results are deterministic and **independent of ``n_jobs``** (the
        bootstrap reseeds per replicate), so a parallel run reproduces the
        serial numbers exactly. ``1`` (default) runs serially; ``-1`` uses all
        cores. joblib's thread backend is used, so candidate models are still
        fit in place.
    execution_mode : {"auto", "standard", "large_data"}, default="auto"
        Memory profile of policy induction (#210). ``"large_data"`` builds and
        predicts the stacked induction feature matrix in ``chunk_size`` blocks,
        bounding peak memory on large logs; it is **numerically identical** to
        ``"standard"`` (parity-tested to <1e-10). ``"auto"`` selects
        ``"large_data"`` once ``len(logs)`` reaches
        :data:`skdr_eval.pairwise.LARGE_DATA_ROW_THRESHOLD`, else ``"standard"``.
    chunk_size : int, default=100_000
        Maximum number of eligible (sample, operator) pairs materialised at once
        during chunked induction (``execution_mode="large_data"``). Ignored in
        ``"standard"`` mode.
    requires_overlap : bool, default=False
        When True, run a cheap positivity/overlap precheck on the evaluation
        slice *before* the expensive cross-fitting and bootstrap, raising
        :class:`InsufficientOverlapError` if the logs cannot support OPE at all
        (#206). Off by default so existing callers are unaffected.
    overlap_floor : float, default=1e-3
        Smallest coarse propensity of an observed action tolerated by the
        precheck. Only consulted when ``requires_overlap=True``.
    min_match_rate : float, default=0.05
        Smallest eligibility match rate tolerated by the precheck. Only
        consulted when ``requires_overlap=True``.

    Returns
    -------
    EvaluationArtifact
        Bundled result containing ``report``, ``detailed``, ``warnings``,
        ``sensitivity``, ``diagnostics``, and ``metadata``. **Breaking change
        in 0.6.0**: previously returned ``(report, detailed)``; migrate by
        unpacking ``artifact.report`` and ``artifact.detailed``.

    Raises
    ------
    DataValidationError
        If ``models`` is not a non-empty dict of ``{name: estimator}``, or if
        ``keep_contributions=True`` is set with more than
        ``max_kept_contributions`` evaluated decisions.
    """
    validate_models_dict(models)
    # Accept polars / pyarrow logs at the boundary; convert once (#72).
    logs = coerce_to_pandas(logs, name="logs")

    # Resolve policy_train sentinel: None means "pre_split" + emit warning.
    if policy_train is None:
        warnings.warn(
            "evaluate_sklearn_models: policy_train was not specified and now "
            "defaults to 'pre_split' (was 'all'). The 'all' mode fits and "
            "evaluates on the same data, introducing training-on-test bias "
            "when fit_models=True. Pass policy_train='pre_split' to suppress "
            "this warning, or policy_train='all' to retain the old behavior. "
            "See #60 and #82.",
            DeprecationWarning,
            stacklevel=2,
        )
        policy_train = "pre_split"
    if policy_train not in ("all", "pre_split"):
        raise ValueError(
            f"Unknown policy_train: {policy_train!r}. Must be 'all' or 'pre_split'"
        )

    _check_n_jobs(n_jobs)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if execution_mode not in ("auto", "standard", "large_data"):
        raise ValueError(
            f"Unknown execution_mode: {execution_mode!r}. "
            "Must be 'auto', 'standard', or 'large_data'."
        )
    # Overlap-precheck thresholds are probabilities/fractions; reject values
    # that would make the gate meaningless (#206 review). NaN fails both.
    if not 0.0 < overlap_floor <= 1.0:
        raise ValueError(f"overlap_floor must be in (0, 1], got {overlap_floor}")
    if not 0.0 <= min_match_rate <= 1.0:
        raise ValueError(f"min_match_rate must be in [0, 1], got {min_match_rate}")

    # Build design
    design = build_design(logs, y_col=y_col)

    # Resolve execution_mode (#210). "auto" switches to the memory-bounded
    # chunked induction once the log is large; "large_data" forces it. The
    # chunked path is numerically identical, so induction_chunk_size is the only
    # thing that changes downstream.
    resolved_execution_mode = execution_mode
    if execution_mode == "auto":
        resolved_execution_mode = (
            "large_data" if len(logs) >= LARGE_DATA_ROW_THRESHOLD else "standard"
        )
    induction_chunk_size = (
        chunk_size if resolved_execution_mode == "large_data" else None
    )

    # Resolve a column-name embedding to an action-level array up front (#136),
    # using the full design so every action's embedding is covered.
    resolved_action_embedding = _resolve_action_embedding(
        action_embedding, logs, design.A, int(design.elig.shape[1])
    )

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
        n_train = 0

    # Original-log row positions for the evaluated slice. Carried through to
    # per-decision contributions so users can join back to ``logs`` (#92).
    eval_log_indices = np.arange(n_train, n_train + len(eval_design.Y), dtype=np.int64)

    # Contributions capture is opt-in via keep_contributions. The
    # ci_bootstrap path keeps its own local arrays (see below) and does not
    # attach a payload to DRResult, so existing CI callers see no change in
    # retained memory.
    if keep_contributions and len(eval_log_indices) > max_kept_contributions:
        raise DataValidationError(
            f"keep_contributions requested for {len(eval_log_indices)} "
            f"decisions, exceeding max_kept_contributions="
            f"{max_kept_contributions}. Raise the cap explicitly or "
            f"disable keep_contributions.",
        )

    # Fast-fail overlap precheck (#206): run the cheap positivity gate before
    # the expensive cross-fitting + bootstrap so hopeless logs abort in
    # ~one logistic fit instead of the full pipeline. Opt-in via
    # requires_overlap=True.
    if requires_overlap:
        _requires_overlap_precheck(
            eval_design,
            random_state=random_state,
            overlap_floor=overlap_floor,
            min_match_rate=min_match_rate,
        )

    # Fit propensity + outcome models on the evaluation slice. When
    # policy_train="pre_split", the slice is only (1 - policy_train_frac) of
    # the input, so a too-small-data failure reports the post-split count and
    # confuses users who passed more rows (#114). Enrich the error with the
    # pre_split context while leaving the policy_train="all" message untouched.
    try:
        propensities, _ = fit_propensity_timecal(
            eval_design.X_phi,
            eval_design.A,
            eval_design.ts,
            n_splits=n_splits,
            random_state=random_state,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
            n_jobs=n_jobs,
        )

        # Fit outcome model
        q_hat, _ = fit_outcome_crossfit(
            eval_design.X_obs,
            eval_design.Y,
            n_splits=n_splits,
            estimator=outcome_estimator,
            random_state=random_state,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
            n_jobs=n_jobs,
        )
    except InsufficientDataError as exc:
        if policy_train == "pre_split":
            raise InsufficientDataError(
                f"{exc} -- after policy_train='pre_split' reserved "
                f"{policy_train_frac:.0%} of your {len(logs)} input rows for "
                f"policy training, only {len(eval_design.Y)} evaluation rows "
                "remain. Lower policy_train_frac, pass policy_train='all', or "
                "provide more data."
            ) from exc
        raise

    # Evaluate each model. The per-model work (induction → DR/SNDR → strategy
    # estimators → optional bootstrap) is independent across models, so it can
    # run in parallel (#178). To avoid nested oversubscription we hand n_jobs to
    # whichever axis dominates: the model loop when there are several models, or
    # the inner bootstrap when there is a single model.
    model_n_jobs = n_jobs if len(models) > 1 else 1
    bootstrap_n_jobs = n_jobs if len(models) == 1 else 1

    def _evaluate_one_model(
        model_name: str, model: Any
    ) -> tuple[str, dict[str, "DRResult"], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        if fit_models:
            # Fit model on training data
            model.fit(train_design.X_obs, train_design.Y)

        # Induce policy. ``induction_chunk_size`` is None in standard mode and
        # the configured chunk in large_data mode (#210) — both give the same
        # policy_probs.
        policy_probs = induce_policy_from_sklearn(
            model,
            eval_design.X_base,
            eval_design.ops_all,
            eval_design.elig,
            chunk_size=induction_chunk_size,
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

        # Extra estimators (#85 #86): MRDR / SWITCH-DR / DRos / MIPS run via
        # the strategy seam and are merged into the per-model detailed dict.
        # The DR clip selected above is reused as the operating point so the
        # comparison is apples-to-apples.
        dr_selected_clip = float(results["DR"].clip)
        extras = _apply_extra_estimators(
            estimators=estimators,
            propensities=propensities,
            policy_probs=policy_probs,
            Y=eval_design.Y,
            q_hat=q_hat,
            A=eval_design.A,
            elig=eval_design.elig,
            X_obs=eval_design.X_obs,
            selected_clip=dr_selected_clip,
            action_embedding=resolved_action_embedding,
            switch_tau=switch_tau,
            dros_lam=dros_lam,
            mips_bandwidth=mips_bandwidth,
            mips_kernel=mips_kernel,
            outcome_estimator=outcome_estimator,
            n_splits=n_splits,
            random_state=random_state,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )
        results.update(extras)

        # Attach per-decision contributions (#92). Opt-in only; does not
        # interact with the ci_bootstrap path, which uses its own local
        # arrays inline to preserve historical CI values.
        if keep_contributions:
            contribs_payload = _build_contributions_payload(
                {k: v for k, v in results.items() if k in {"DR", "SNDR"}},
                propensities,
                policy_probs,
                eval_design.Y,
                q_hat,
                eval_design.A,
                eval_design.elig,
                eval_log_indices=eval_log_indices,
            )
            for est_name, est_payload in contribs_payload.items():
                results[est_name].contributions = est_payload

        # Add to report — emit a row per requested estimator that we actually
        # produced. The strategy-seam estimators report ``clip=nan`` (the
        # operating clip is implicit in the strategy).
        requested_canon = [_canonical_estimator_name(n) for n in estimators]
        emit_order = [n for n in requested_canon if n in results]
        for estimator_name in emit_order:
            result = results[estimator_name]
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
                "pareto_k": result.pareto_k,
            }

            # Add confidence intervals if requested
            if ci_bootstrap:
                # Extra estimators (MRDR/SWITCH-DR/DRos/MIPS) fall back to the
                # IF-based normal CI here. Their bootstrap definition is
                # strategy-specific and is tracked as a follow-up; producing
                # the normal-approximation interval keeps the report
                # rectangular and the CI conservative.
                if estimator_name not in {"DR", "SNDR"}:
                    z = norm.ppf(1 - alpha / 2)
                    ci_lower, ci_upper = (
                        result.V_hat - z * result.SE_if,
                        result.V_hat + z * result.SE_if,
                    )
                    row["ci_lower"] = ci_lower
                    row["ci_upper"] = ci_upper
                    rows.append(row)
                    continue
                # Use proper block bootstrap for time-series data
                try:
                    # Recompute estimator-specific pseudo-outcomes for bootstrap.
                    # q_pi == q_hat when q_hat is 1D; see design note in
                    # dr_value_with_clip. For SNDR we use the normalised
                    # residual form so the bootstrap CI is anchored to the
                    # same point estimate as V_hat (fixes #58).
                    q_pi = np.sum(
                        policy_probs * q_hat.reshape(len(eval_design.Y), -1), axis=1
                    )
                    # DR importance ratio π(A|x)/e(A|x) over the overlap set (#106).
                    _pi_obs, w_raw, matched = _dr_weight_components(
                        propensities, policy_probs, eval_design.A, eval_design.elig
                    )

                    if matched.sum() > 0:
                        w_clip = _clip_weights(w_raw, matched, result.clip)

                        # Estimator-specific pseudo-outcome for the bootstrap:
                        # DR: q_pi + w*(Y - q_hat)  # noqa: ERA001
                        # SNDR: q_pi + (n/Σw)*w*(Y - q_hat) — normalised so
                        #       mean(bootstrap_values) == V_SNDR.
                        if estimator_name == "SNDR":
                            bootstrap_values = _sndr_bootstrap_values(
                                q_pi, w_clip, eval_design.Y, q_hat
                            )
                        else:
                            bootstrap_values = q_pi + w_clip * (eval_design.Y - q_hat)

                        ci_lower, ci_upper = block_bootstrap_ci(
                            values_num=bootstrap_values,
                            values_den=None,
                            base_mean=np.array([result.V_hat]),
                            n_boot=400,
                            alpha=alpha,
                            random_state=random_state,
                            n_jobs=bootstrap_n_jobs,
                        )
                    else:
                        # Fallback if no matched samples
                        z = norm.ppf(1 - alpha / 2)
                        ci_lower, ci_upper = (
                            result.V_hat - z * result.SE_if,
                            result.V_hat + z * result.SE_if,
                        )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    # Fallback to normal approximation if bootstrap fails
                    z = norm.ppf(1 - alpha / 2)
                    ci_lower, ci_upper = (
                        result.V_hat - z * result.SE_if,
                        result.V_hat + z * result.SE_if,
                    )
                row["ci_lower"] = ci_lower
                row["ci_upper"] = ci_upper

            rows.append(row)

        return model_name, results, rows

    # Run the per-model evaluations (serial or threaded) and reassemble in the
    # caller's model order so the report is deterministic regardless of n_jobs.
    model_items = list(models.items())
    # Threaded fitting mutates each estimator in place, so two keys pointing at
    # the *same* instance would race under n_jobs != 1 and yield nondeterministic
    # results — serial is unaffected. Fail fast with a clear message (#178 review).
    if model_n_jobs != 1 and fit_models:
        first_seen: dict[int, str] = {}
        for name, mdl in model_items:
            prior = first_seen.get(id(mdl))
            if prior is not None:
                raise DataValidationError(
                    f"models[{name!r}] and models[{prior!r}] are the same "
                    "estimator instance; parallel fitting (n_jobs != 1, "
                    "fit_models=True) would race on the shared object. Pass "
                    "distinct instances (e.g. via sklearn.clone) or use n_jobs=1."
                )
            first_seen[id(mdl)] = name
    if model_n_jobs == 1:
        computed = [_evaluate_one_model(name, mdl) for name, mdl in model_items]
    else:
        computed = Parallel(n_jobs=model_n_jobs, prefer="threads")(
            delayed(_evaluate_one_model)(name, mdl) for name, mdl in model_items
        )

    report_rows = []
    detailed_results = {}
    for model_name, results, rows in computed:
        detailed_results[model_name] = results
        report_rows.extend(rows)

    report = pd.DataFrame(report_rows)

    from .reporting import build_evaluation_artifact  # noqa: PLC0415

    baseline_kind, baseline_value = _resolve_baseline(baseline, eval_design.Y)

    artifact = build_evaluation_artifact(
        report=report,
        detailed=detailed_results,
        n_samples=len(eval_design.Y),
        propensities=propensities,
        actions=eval_design.A,
        thresholds=support_thresholds,
        evaluator="evaluate_sklearn_models",
        random_state=random_state,
        alpha=alpha if ci_bootstrap else None,
        extra_metadata={
            "policy_train": policy_train,
            "policy_train_frac": float(policy_train_frac),
            "n_splits": int(n_splits),
            "clip_grid": [None if not np.isfinite(c) else float(c) for c in clip_grid],
            "ci_bootstrap": bool(ci_bootstrap),
            "estimators": list(estimators),
            "n_jobs": int(n_jobs),
            "execution_mode": resolved_execution_mode,
            "requires_overlap": bool(requires_overlap),
        },
        baseline_kind=baseline_kind,
        baseline_value=baseline_value,
    )
    _autolog_tracker(tracker, artifact, evaluator="evaluate_sklearn_models")
    return artifact


def _get_outcome_estimator(estimator: str | Callable[[], Any], task_type: str) -> Any:
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
    *,
    method: Literal["auto", "condlogit", "multinomial"] = "auto",
    neg_per_pos: int = 5,
    n_splits: int = 3,
    random_state: int = 0,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
) -> np.ndarray:
    """Estimate propensity scores for pairwise evaluation.

    Parameters
    ----------
    design : PairwiseDesign
        Pairwise design object
    method : Literal["auto", "condlogit", "multinomial"]
        Method to use. ``"auto"`` selects ``"condlogit"`` when SciPy is
        available and falls back to ``"multinomial"`` otherwise.
        ``"condlogit"`` requires SciPy and also falls back to
        ``"multinomial"`` when SciPy is unavailable.
    neg_per_pos : int
        Negative samples per positive for conditional logit
    n_splits : int
        Number of time series splits
    random_state : int
        Random seed
    gap : int, default=1
        Samples skipped between train and test windows in the time-series CV
        used to fit the propensity classifier.
    test_size : int, optional
        Per-fold test-window size in samples; ``None`` defers to sklearn.
    max_train_size : int, optional
        Cap on training-fold size in samples; ``None`` uses expanding window.

    Returns
    -------
    propensities : np.ndarray
        Propensity scores (n_decisions, n_max_operators)
    """

    # Validate parameters
    if method not in ["auto", "condlogit", "multinomial"]:
        raise ValueError(
            f"Unknown method: {method}. Must be 'auto', 'condlogit', or 'multinomial'"
        )

    n_decisions = len(design.logs_df)
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())
    propensities: np.ndarray = np.zeros((n_decisions, max_ops), dtype=np.float64)

    if method == "auto":
        method = "condlogit" if SCIPY_AVAILABLE else "multinomial"

    if method == "condlogit" and not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, falling back to multinomial")
        method = "multinomial"

    if method == "condlogit":
        # Build pairwise training data with time-forward splits
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

        tscv = _make_time_series_split(
            len(all_indices),
            n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )

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
        tscv = _make_time_series_split(
            len(X_client),
            n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )

        for train_idx, test_idx in tscv.split(X_client):
            X_train, X_test = X_client[train_idx], X_client[test_idx]
            y_train = np.array([actions[i] for i in train_idx])

            # Fit multinomial model
            model = LogisticRegression(random_state=random_state, max_iter=1000)
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
    propensity: Literal["auto", "condlogit", "multinomial"] = "auto",
    topk: int = 20,
    neg_per_pos: int = 5,
    chunk_pairs: int = 2_000_000,
    min_ess_frac: float = 0.02,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    fit_models: bool = False,
    policy_train: Literal["all", "pre_split"] | None = None,
    policy_train_frac: float = 0.85,
    surrogate_model: str = "hgb",
    day_col: str = "arrival_day",
    client_id_col: str = "client_id",
    operator_id_col: str = "operator_id",
    elig_col: str | None = "elig_mask",
    random_state: int = 0,
    outcome_estimator: str | Callable[[], Any] = "hgb",
    support_thresholds: "SupportHealthThresholds | None" = None,
    *,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
    keep_contributions: bool = False,
    max_kept_contributions: int = DEFAULT_MAX_KEPT_CONTRIBUTIONS,
    estimators: tuple[str, ...] = ("DR", "SNDR"),
    action_embedding: "np.ndarray | str | None" = None,
    switch_tau: float = 5.0,
    dros_lam: float = 1.0,
    mips_bandwidth: float | str = 1.0,
    mips_kernel: "str | Callable[[np.ndarray], np.ndarray]" = "rbf",
    tracker: "Tracker | None" = None,
    baseline: float | str | None = None,
    external_policies: "dict[str, pd.DataFrame] | None" = None,
    execution_mode: Literal["auto", "standard", "large_data"] = "auto",
) -> "EvaluationArtifact":
    """Evaluate pairwise models using autoscale strategy.

    Parameters
    ----------
    logs_df : pd.DataFrame
        Observed decisions with required columns
    op_daily_df : pd.DataFrame
        Daily operator snapshots
    models : Dict[str, Any]
        Dictionary of model_name -> model instance (fitted or unfitted).
    metric_col : str
        Target metric column name
    task_type : Literal["regression", "binary"]
        Type of prediction task
    direction : Literal["min", "max"]
        Whether to minimize or maximize metric
    fit_models : bool, default=False
        If True, fit each model on the observed (cli_*, op_*) features from
        logs_df against metric_col before inducing policies. Set to True when
        passing unfitted model instances; set to False when passing pre-fitted
        models.
    policy_train : Literal["all", "pre_split"], optional
        How to split data for fitting policy models when ``fit_models=True``.
        ``"pre_split"`` (default) fits on the first ``policy_train_frac`` of
        the data and evaluates on the held-out tail — the statistically-safe
        choice. ``"all"`` fits and evaluates on the same data; retained for
        backward compatibility but introduces training-on-test bias when
        ``fit_models=True``. Passing ``None`` (or omitting the argument) uses
        ``"pre_split"`` and emits a :class:`DeprecationWarning`. Ignored when
        ``fit_models=False``.

        - ``"all"``: Fit on all data and evaluate on all data. Faster but
          biased — included for backward compatibility.
        - ``"pre_split"``: Fit on the first ``policy_train_frac`` of the data
          (sorted by day) and evaluate only on the held-out tail. Removes
          training-on-test leakage; the statistically safer choice when
          fitting via ``fit_models=True``.
        Ignored when ``fit_models=False``.
    policy_train_frac : float, default=0.85
        Fraction of data (chronologically) used to fit policy models when
        ``policy_train="pre_split"``. Remaining tail is used for evaluation.
    surrogate_model : str, default="hgb"
        Surrogate model used by ``stream_topk`` strategy to prefilter operators.
        - ``"hgb"`` (default): HistGradientBoostingRegressor; non-linear,
          captures cli x op interactions automatically.
        - ``"ridge_interaction"``: Ridge with explicit cli x op outer-product
          interaction terms; cheaper but still per-client personalized.
        Plain ``"ridge"`` is rejected because it produces day-global top-K.
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
        Whether to compute bootstrap CIs via ``block_bootstrap_ci`` on the
        per-decision DR contributions. **Conditional bootstrap**: ``q_hat``,
        propensities, and the policy are fixed across replicates, so the CI
        does not reflect nuisance estimation error and may under-cover when
        that error is non-negligible.
    alpha : float
        Significance level for CIs
    day_col : str
        Day column name
    client_id_col : str
        Client ID column name
    operator_id_col : str
        Operator ID column name
    elig_col : str | None
        Eligibility column name
    random_state : int
        Random seed
    outcome_estimator : str | Callable[[], Any]
        Outcome model estimator
    support_thresholds : SupportHealthThresholds, optional
        Thresholds for support-health warnings attached to the returned
        artifact. See :class:`skdr_eval.reporting.SupportHealthThresholds`.
    keep_contributions : bool, default=False
        If True, attach per-decision DR/SNDR contributions to each
        :class:`DRResult` under ``contributions`` and make them queryable
        via :meth:`EvaluationArtifact.contributions` (issue #92). Default
        is False to preserve the memory profile; ``ci_bootstrap=True``
        auto-enables capture (the bootstrap path computes the same arrays).
    max_kept_contributions : int, default=100_000_000
        Hard cap on the number of evaluated decisions for which per-decision
        contributions may be captured. Raises :class:`DataValidationError`
        if ``n_eval`` exceeds this value while contributions are being kept.
    estimators : tuple[str, ...], default=("DR", "SNDR")
        Estimators to compute. Supported: ``"DR"``, ``"SNDR"``, ``"MRDR"``,
        ``"SWITCH-DR"``, ``"DRos"``, ``"MIPS"``. Each maps to a built-in
        strategy via :func:`skdr_eval.estimators.build_strategy`.
    action_embedding : np.ndarray, optional
        Per-action embedding of shape ``(n_actions, embed_dim)``. **Required
        when** ``"MIPS"`` is requested; ignored by the other estimators.
    switch_tau : float, default=5.0
        SWITCH-DR importance-weight threshold: weights above ``switch_tau``
        fall back to the direct method. Used only for ``"SWITCH-DR"``.
    dros_lam : float, default=1.0
        DRos optimistic-shrinkage parameter ``lambda``. Used only for
        ``"DRos"``.
    mips_bandwidth : float, default=1.0
        Gaussian-kernel bandwidth for the MIPS embedding marginal. Used only
        for ``"MIPS"``.
    external_policies : dict[str, pd.DataFrame], optional
        Externally-supplied policies to evaluate instead of inducing them from
        ``models`` (issue #56). Maps ``policy_name -> DataFrame`` where each
        frame carries at least the ``client_id`` and ``operator_id`` columns —
        the assignment schema a routing/queueing simulator emits. When given,
        the ``models``-based policy induction (and ``fit_models`` /
        ``policy_train`` pre-split) is skipped and every logged decision is
        evaluated; ``models`` may be an empty dict. Assignments are keyed by
        ``client_id`` (one operator per client). Prefer the thin wrapper
        :func:`evaluate_external_policies` for this workflow.
    execution_mode : Literal["auto", "standard", "large_data"], default="auto"
        Execution path for building the per-decision arrays (issue #33).
        ``"standard"`` uses the row-wise reference implementation;
        ``"large_data"`` uses a vectorized builder that avoids the per-row
        ``iterrows()`` loop and is **numerically identical** to ``"standard"``
        (parity-tested to <1e-10). ``"auto"`` selects ``"large_data"`` once the
        number of evaluation decisions reaches
        :data:`skdr_eval.pairwise.LARGE_DATA_ROW_THRESHOLD` and ``"standard"``
        otherwise. Does not change the induction strategy (see ``strategy`` /
        ``chunk_pairs`` for the memory-aware induction controls).

    Returns
    -------
    EvaluationArtifact
        Bundled result containing ``report``, ``detailed``, ``warnings``,
        ``sensitivity``, ``diagnostics``, and ``metadata``. **Breaking change
        in 0.6.0**: previously returned ``(report, detailed_results)``;
        migrate by unpacking ``artifact.report`` and ``artifact.detailed``.

    Raises
    ------
    DataValidationError
        If ``models`` is not a non-empty dict of ``{name: estimator}``, or if
        ``keep_contributions=True`` is set with more than
        ``max_kept_contributions`` evaluated decisions.
    """

    logger.info("Starting pairwise evaluation")

    # Validate parameters
    use_external = external_policies is not None
    if not use_external:
        validate_models_dict(models)
    # Accept polars / pyarrow frames at the boundary; convert once (#72, #236).
    # Every user-supplied frame on the pairwise path — including the external
    # policy tables — flows through the single ``coerce_to_pandas`` seam so the
    # downstream NumPy code only ever sees pandas.
    logs_df = coerce_to_pandas(logs_df, name="logs_df")
    op_daily_df = coerce_to_pandas(op_daily_df, name="op_daily_df")
    if external_policies is not None:
        external_policies = {
            name: coerce_to_pandas(frame, name=f"external_policies[{name!r}]")
            for name, frame in external_policies.items()
        }
    if execution_mode not in ("auto", "standard", "large_data"):
        raise ValueError(
            f"Unknown execution_mode: {execution_mode}. "
            "Must be 'auto', 'standard', or 'large_data'."
        )
    if task_type not in ["regression", "binary"]:
        raise ValueError(
            f"Unknown task_type: {task_type}. Must be 'regression' or 'binary'"
        )
    if direction not in ["min", "max"]:
        raise ValueError(f"Unknown direction: {direction}. Must be 'min' or 'max'")

    # Resolve policy_train sentinel: None means "pre_split" + emit warning.
    # External policies (#56) are evaluated against every logged decision; there
    # are no policy models to fit, so the pre_split machinery is skipped and the
    # deprecation warning is irrelevant.
    if use_external:
        if fit_models:
            warnings.warn(
                "evaluate_pairwise_models: external_policies was provided, so "
                "fit_models=True is ignored — externally-supplied policies are "
                "evaluated directly, without fitting models or splitting data.",
                stacklevel=2,
            )
        policy_train = "all"
    elif policy_train is None:
        warnings.warn(
            "evaluate_pairwise_models: policy_train was not specified and now "
            "defaults to 'pre_split' (was 'all'). The 'all' mode fits and "
            "evaluates on the same data, introducing training-on-test bias "
            "when fit_models=True. Pass policy_train='pre_split' to suppress "
            "this warning, or policy_train='all' to retain the old behavior. "
            "See #60 and #82.",
            DeprecationWarning,
            stacklevel=2,
        )
        policy_train = "pre_split"

    # Validate policy_train parameters
    if policy_train not in ("all", "pre_split"):
        raise ValueError(
            f"Unknown policy_train: {policy_train}. Must be 'all' or 'pre_split'"
        )
    if not 0.0 < policy_train_frac < 1.0:
        raise ValueError(
            f"policy_train_frac must be in (0, 1), got {policy_train_frac}"
        )

    # Optional pre_split: fit policy models on training portion, evaluate on
    # held-out tail. This avoids training-on-test leakage when fit_models=True.
    eval_logs_df = logs_df
    # Track original-log positions for the eval slice so per-decision
    # contributions can be joined back to the user's input ``logs_df`` via the
    # documented ``logs.reset_index(names="decision_id")`` pattern. For
    # ``policy_train="all"``, the eval slice IS the input, so the identity
    # mapping is correct.
    eval_log_indices_pair: np.ndarray = np.arange(len(logs_df), dtype=np.int64)
    if fit_models and policy_train == "pre_split":
        # Sort by day to respect time order, then split chronologically.
        # Capture the permutation so the eval-slice rows still carry their
        # positional indices into the user's input ``logs_df`` (issue #92,
        # acceptance criterion: original log columns are joinable back).
        positions = logs_df.reset_index(drop=True)
        sorted_with_pos = positions.sort_values(by=day_col, kind="stable")
        sort_order = sorted_with_pos.index.to_numpy()
        sorted_logs = sorted_with_pos.reset_index(drop=True)
        n_total = len(sorted_logs)
        n_train = max(1, int(n_total * policy_train_frac))
        train_logs = sorted_logs.iloc[:n_train].reset_index(drop=True)
        eval_logs_df = sorted_logs.iloc[n_train:].reset_index(drop=True)
        eval_log_indices_pair = sort_order[n_train:].astype(np.int64)
        if len(eval_logs_df) == 0:
            raise ValueError(
                f"pre_split left 0 evaluation rows (n_total={n_total}, "
                f"policy_train_frac={policy_train_frac}); reduce the fraction"
            )
        logger.info(
            f"pre_split: fitting policy models on {len(train_logs):,} train rows, "
            f"evaluating on {len(eval_logs_df):,} held-out rows"
        )

        # Fit on training portion only
        train_design = PairwiseDesign.from_dataframes(
            train_logs, op_daily_df, day_col, client_id_col, operator_id_col, elig_col
        )
        feature_cols = train_design.cli_features + train_design.op_features
        X_fit = train_design.logs_df[feature_cols].values.astype(np.float32)
        y_fit = train_design.logs_df[metric_col].values
        for model in models.values():
            model.fit(X_fit, y_fit)
        logger.info(f"Fitted {len(models)} policy model(s) on {len(X_fit)} train pairs")

    # Create pairwise design from the (possibly held-out) evaluation logs
    design = PairwiseDesign.from_dataframes(
        eval_logs_df,
        op_daily_df,
        day_col,
        client_id_col,
        operator_id_col,
        elig_col,
    )

    # Log statistics
    stats = design.get_stats()
    logger.info(
        f"Dataset stats: {stats['n_rows']:,} decisions, "
        f"{stats['candidate_pairs']:,} candidate pairs, "
        f"{stats['memory_gb']:.2f} GB estimated memory"
    )

    # Fit on all data (legacy behavior, biased but supported for compatibility)
    if fit_models and policy_train == "all" and not use_external:
        feature_cols = design.cli_features + design.op_features
        X_fit = design.logs_df[feature_cols].values.astype(np.float32)
        y_fit = design.logs_df[metric_col].values
        for model in models.values():
            model.fit(X_fit, y_fit)
        logger.info(
            f"Fitted {len(models)} policy model(s) on {len(X_fit)} pairs (policy_train='all')"
        )

    # Obtain policies: either map externally-supplied assignments (#56) or
    # induce them from ``models`` (held-out for pre_split, full data for "all").
    if use_external:
        assert external_policies is not None  # narrowed by use_external
        policies = map_external_policies(external_policies, design)
        logger.info(f"Evaluating {len(policies)} external policy(ies)")
    else:
        policies = induce_policy(
            models,
            design,
            strategy,
            direction,
            topk,
            chunk_pairs,
            metric_col,
            surrogate_model=surrogate_model,
            random_state=random_state,
        )

    # Mirror eval_logs_df into the local variable used downstream
    logs_df = design.logs_df

    # Resolve execution_mode (#33). "auto" selects the vectorized large-data
    # path once the evaluation set is large enough; the vectorized path is
    # numerically identical to the standard row-wise path.
    resolved_execution_mode = execution_mode
    if execution_mode == "auto":
        resolved_execution_mode = (
            "large_data" if len(logs_df) >= LARGE_DATA_ROW_THRESHOLD else "standard"
        )
    use_vectorized = resolved_execution_mode == "large_data"
    logger.info(f"Execution mode: {resolved_execution_mode}")

    # Estimate propensity scores
    propensities = estimate_propensity_pairwise(
        design,
        method=propensity,
        neg_per_pos=neg_per_pos,
        n_splits=n_splits,
        random_state=random_state,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
    )

    # Fit outcome models with cross-fitting
    Y = logs_df[metric_col].values

    # Create observed features (client + chosen operator features). The
    # vectorized large-data path (#33) builds the same X_obs / A / eligibility
    # arrays without a per-row iterrows() loop and is numerically identical.
    elig_shared: np.ndarray | None = None
    if use_vectorized:
        X_obs, A, elig_shared, _max_ops_vec = build_eval_arrays_vectorized(
            design, metric_col
        )
    else:
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
        X_obs,
        Y.astype(np.float64),
        n_splits,
        lambda: estimator_obj,
        random_state,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
    )

    # Convert policies to policy probabilities
    max_ops = max(len(ops) for ops in design.ops_all_by_day.values())

    # Per-decision contributions plumbing (#92). Opt-in only; ci_bootstrap
    # keeps its own local arrays and does not attach a payload to DRResult.
    # ``eval_log_indices_pair`` was established up at the pre_split block and
    # carries each eval-slice row's positional index into the *original*
    # ``logs_df`` so users can ``merge(frame, logs.reset_index(names=
    # "decision_id"))`` against their full input even after a pre_split.
    if keep_contributions and len(logs_df) > max_kept_contributions:
        raise DataValidationError(
            f"keep_contributions requested for {len(logs_df)} decisions, "
            f"exceeding max_kept_contributions={max_kept_contributions}. "
            f"Raise the cap explicitly or disable keep_contributions.",
        )
    # Defensive sanity check: ``logs_df`` was rebound to ``design.logs_df``
    # above; its length must match the slice ``eval_log_indices_pair`` was
    # built for. Mismatch would indicate an unexpected re-ordering inside
    # PairwiseDesign.from_dataframes.
    if len(eval_log_indices_pair) != len(logs_df):
        raise DataValidationError(
            f"internal: eval_log_indices_pair length "
            f"{len(eval_log_indices_pair)} != design.logs_df length "
            f"{len(logs_df)}; contributions decision_id would be wrong"
        )

    detailed_results = {}
    report_rows = []

    for model_name, policy_decisions in policies.items():
        # Convert policy decisions to probabilities and build the eligibility
        # matrix. The vectorized path (#33) reuses the day-grouped builders;
        # ``elig`` is policy-independent so it is built once above.
        if use_vectorized:
            assert elig_shared is not None  # set together with use_vectorized
            policy_probs = build_policy_probs_vectorized(
                design, policy_decisions, max_ops
            )
            elig = elig_shared
        else:
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
                                    op_idx = design.ops_all_by_day[day_str].index(
                                        str(op)
                                    )
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

            # Extra estimators (#85 #86) for pairwise — operate on the same
            # propensities / policy / outcome predictions; MRDR triggers a
            # sample-weighted outcome refit. The DR clip is reused as the
            # operating point.
            dr_selected_clip = float(results["DR"].clip)
            extras = _apply_extra_estimators(
                estimators=estimators,
                propensities=propensities,
                policy_probs=policy_probs,
                Y=Y.astype(np.float64),
                q_hat=q_hat,
                A=A,
                elig=elig,
                X_obs=X_obs,
                selected_clip=dr_selected_clip,
                action_embedding=_resolve_action_embedding(
                    action_embedding,
                    logs_df,
                    A,
                    int(elig.shape[1]),
                    allow_column_name=False,
                ),
                switch_tau=switch_tau,
                dros_lam=dros_lam,
                mips_bandwidth=mips_bandwidth,
                mips_kernel=mips_kernel,
                outcome_estimator=outcome_estimator,
                n_splits=n_splits,
                random_state=random_state,
                gap=gap,
                test_size=test_size,
                max_train_size=max_train_size,
            )
            results.update(extras)

            detailed_results[model_name] = results

            # Attach per-decision contributions (#92). Opt-in only.
            if keep_contributions:
                contribs_payload = _build_contributions_payload(
                    {k: v for k, v in results.items() if k in {"DR", "SNDR"}},
                    propensities,
                    policy_probs,
                    Y.astype(np.float64),
                    q_hat,
                    A,
                    elig,
                    eval_log_indices=eval_log_indices_pair,
                )
                for est_name, est_payload in contribs_payload.items():
                    results[est_name].contributions = est_payload

            # Add to report — emit a row per requested estimator that we
            # actually produced; preserves historical row order for the
            # default ("DR", "SNDR") request.
            requested_canon = [_canonical_estimator_name(n) for n in estimators]
            emit_order = [n for n in requested_canon if n in results]
            for estimator_name in emit_order:
                result = results[estimator_name]
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
                    "pareto_k": result.pareto_k,
                }

                if ci_bootstrap:
                    # Extra estimators (MRDR/SWITCH-DR/DRos/MIPS) get the
                    # IF-based normal CI here; strategy-specific block
                    # bootstrap is tracked as a follow-up.
                    if estimator_name not in {"DR", "SNDR"}:
                        z = norm.ppf(1 - alpha / 2)
                        report_row["ci_lower"] = result.V_hat - z * result.SE_if
                        report_row["ci_upper"] = result.V_hat + z * result.SE_if
                        report_rows.append(report_row)
                        continue
                    # Use proper block bootstrap for time-series data
                    try:
                        # Recompute estimator-specific pseudo-outcomes for
                        # bootstrap.  q_pi == q_hat when q_hat is 1D; see
                        # design note in dr_value_with_clip. For SNDR we use
                        # the normalised residual form so the bootstrap CI
                        # is anchored to the same point estimate as V_hat
                        # (fixes #58).
                        q_pi = np.sum(policy_probs * q_hat.reshape(len(Y), -1), axis=1)
                        # DR importance ratio π(A|x)/e(A|x) over overlap set (#106).
                        _pi_obs, w_raw, matched = _dr_weight_components(
                            propensities, policy_probs, A, elig
                        )

                        if matched.sum() > 0:
                            w_clip = _clip_weights(w_raw, matched, result.clip)

                            # Estimator-specific pseudo-outcome for bootstrap:
                            # DR: q_pi + w*(Y - q_hat)  # noqa: ERA001
                            # SNDR: q_pi + (n/Σw)*w*(Y - q_hat)  # noqa: ERA001
                            if estimator_name == "SNDR":
                                bootstrap_values = _sndr_bootstrap_values(
                                    q_pi, w_clip, Y, q_hat
                                )
                            else:
                                bootstrap_values = q_pi + w_clip * (Y - q_hat)

                            ci_lower, ci_upper = block_bootstrap_ci(
                                values_num=bootstrap_values,
                                values_den=None,
                                base_mean=np.array([result.V_hat]),
                                n_boot=400,
                                alpha=alpha,
                                random_state=random_state,
                            )
                        else:
                            # Fallback if no matched samples
                            z = norm.ppf(1 - alpha / 2)
                            ci_lower, ci_upper = (
                                result.V_hat - z * result.SE_if,
                                result.V_hat + z * result.SE_if,
                            )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        # Fallback to normal approximation if bootstrap fails
                        z = norm.ppf(1 - alpha / 2)
                        ci_lower, ci_upper = (
                            result.V_hat - z * result.SE_if,
                            result.V_hat + z * result.SE_if,
                        )
                    report_row["ci_lower"] = ci_lower
                    report_row["ci_upper"] = ci_upper

                report_rows.append(report_row)

        except Exception as e:
            logger.error(f"Error computing DR values for model {model_name}: {e}")

    report = pd.DataFrame(report_rows)

    logger.info(f"Completed pairwise evaluation for {len(models)} models")

    from .reporting import build_evaluation_artifact  # noqa: PLC0415

    baseline_kind, baseline_value = _resolve_baseline(baseline, np.asarray(Y))

    artifact = build_evaluation_artifact(
        report=report,
        detailed=detailed_results,
        n_samples=len(Y),
        propensities=propensities,
        actions=np.asarray(A, dtype=int),
        thresholds=support_thresholds,
        evaluator="evaluate_pairwise_models",
        random_state=random_state,
        alpha=alpha if ci_bootstrap else None,
        extra_metadata={
            "task_type": task_type,
            "direction": direction,
            "strategy": strategy,
            "propensity": propensity,
            "metric_col": metric_col,
            "policy_train": policy_train,
            "policy_train_frac": float(policy_train_frac),
            "n_splits": int(n_splits),
            "clip_grid": [None if not np.isfinite(c) else float(c) for c in clip_grid],
            "ci_bootstrap": bool(ci_bootstrap),
            "estimators": list(estimators),
            "execution_mode": resolved_execution_mode,
            "external_policies": bool(use_external),
        },
        baseline_kind=baseline_kind,
        baseline_value=baseline_value,
    )
    _autolog_tracker(tracker, artifact, evaluator="evaluate_pairwise_models")
    return artifact


def evaluate_external_policies(
    logs_df: pd.DataFrame,
    op_daily_df: pd.DataFrame,
    policies: dict[str, pd.DataFrame],
    metric_col: str,
    task_type: Literal["regression", "binary"],
    direction: Literal["min", "max"],
    **eval_kwargs: Any,
) -> "EvaluationArtifact":
    """Evaluate externally-supplied pairwise policies with DR/SNDR (issue #56).

    Use this when the assignments come from an **external decision process** —
    e.g. a discrete-event call-centre simulator that accounts for queues, shift
    schedules and sequential dependencies — rather than from a per-client greedy
    ``induce_policy`` over candidate operators. Each policy is a DataFrame of
    ``client_id -> operator_id`` assignments; the DR/SNDR estimators then score
    those assignments against the logged outcomes, with the same trust
    diagnostics and confidence intervals as :func:`evaluate_pairwise_models`.

    This is a thin wrapper over :func:`evaluate_pairwise_models` with
    ``external_policies=policies`` (so policy induction and model fitting are
    skipped and every logged decision is evaluated). All other keyword arguments
    accepted by :func:`evaluate_pairwise_models` — ``propensity``, ``n_splits``,
    ``clip_grid``, ``ci_bootstrap``, ``estimators``, ``execution_mode``, the
    column-name overrides, etc. — are forwarded unchanged via ``eval_kwargs``.

    Parameters
    ----------
    logs_df : pd.DataFrame
        Observed decisions (one row per logged call) with the required columns.
    op_daily_df : pd.DataFrame
        Daily operator snapshots.
    policies : dict[str, pd.DataFrame]
        Mapping of policy name to an assignment frame carrying at least the
        ``client_id`` and ``operator_id`` columns. One operator per client.
    metric_col : str
        Outcome column to evaluate (e.g. service time).
    task_type : Literal["regression", "binary"]
        Outcome type.
    direction : Literal["min", "max"]
        Whether lower or higher outcomes are better.
    **eval_kwargs : Any
        Additional keyword arguments forwarded to
        :func:`evaluate_pairwise_models`.

    Returns
    -------
    EvaluationArtifact
        The same bundled artifact returned by :func:`evaluate_pairwise_models`.

    Examples
    --------
    >>> import pandas as pd
    >>> from skdr_eval import evaluate_external_policies
    >>> sim_a = pd.DataFrame({"client_id": [...], "operator_id": [...]})
    >>> sim_b = pd.DataFrame({"client_id": [...], "operator_id": [...]})
    >>> artifact = evaluate_external_policies(  # doctest: +SKIP
    ...     logs_df=logs_df,
    ...     op_daily_df=op_daily_df,
    ...     policies={"simulator_a": sim_a, "simulator_b": sim_b},
    ...     metric_col="service_time",
    ...     task_type="regression",
    ...     direction="min",
    ... )
    """
    for reserved in ("models", "external_policies", "fit_models", "policy_train"):
        if reserved in eval_kwargs:
            raise TypeError(
                f"evaluate_external_policies() does not accept {reserved!r}; "
                "external policies are evaluated directly without model fitting."
            )
    return evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models={},
        metric_col=metric_col,
        task_type=task_type,
        direction=direction,
        external_policies=policies,
        **eval_kwargs,
    )


def evaluate_propensity_diagnostics(
    propensities: np.ndarray,
    actions: np.ndarray,
    output_format: str = "text",
) -> tuple[PropensityDiagnostics, str]:
    """Evaluate propensity score diagnostics and generate a report.

    This function provides comprehensive diagnostics for propensity scores including
    overlap analysis, balance assessment, calibration evaluation, and discrimination
    analysis.

    Parameters
    ----------
    propensities : np.ndarray
        Array of propensity scores with shape (n_samples, n_actions)
    actions : np.ndarray
        Array of action indices with shape (n_samples,)
    output_format : str, default="text"
        Output format for the report ("text" or "markdown")

    Returns
    -------
    tuple[PropensityDiagnostics, str]
        A tuple containing the diagnostics object and the generated report

    Raises
    ------
    DataValidationError
        If input data is invalid
    InsufficientDataError
        If there's insufficient data for evaluation
    ConfigurationError
        If output_format is not a recognized format
    """
    # Run comprehensive diagnostics
    diagnostics = comprehensive_propensity_diagnostics(propensities, actions)

    # Generate report
    report = generate_propensity_report(diagnostics, output_format=output_format)

    return diagnostics, report
