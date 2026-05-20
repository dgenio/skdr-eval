"""Core implementation of DR and Stabilized DR for offline policy evaluation."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .reporting import EvaluationArtifact, SupportHealthThresholds

import numpy as np
import pandas as pd
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
    ModelValidationError,
    OutcomeModelError,
    PolicyInductionError,
    PropensityScoreError,
)
from .pairwise import PairwiseDesign, induce_policy
from .validation import (
    validate_dataframe,
    validate_finite_values,
    validate_numpy_array,
    validate_positive_integer,
    validate_probabilities,
    validate_random_state,
    validate_sklearn_estimator,
    validate_string_choice,
)

logger = logging.getLogger("skdr_eval")


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

    Raises
    ------
    DataValidationError
        If input data is invalid
    InsufficientDataError
        If there's insufficient data for estimation
    """
    try:
        # Validate input DataFrame
        required_columns = ["arrival_ts", "action", "service_time"]
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
        Y: np.ndarray = logs["service_time"].values.astype(np.float64)
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

            propensities[original_test_idx] = pred_proba

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
                    f"sample_weight shape {sample_weight.shape} must match "
                    f"X_obs ({n_samples},)"
                )
            if np.any(sample_weight < 0):
                raise DataValidationError(
                    "sample_weight must be non-negative; received negatives"
                )

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_obs)):
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
                predictions[test_idx] = pred
                models_info.append((model, train_idx, test_idx))

            except Exception as e:
                raise OutcomeModelError(f"Error in fold {fold_idx}: {e!s}") from e

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
) -> np.ndarray:
    """Induce policy from sklearn model by predicting service times.

    Vectorized: builds a single stacked ``(sum_i n_elig_i, n_features + n_ops)``
    feature matrix and issues **one** ``model.predict`` call instead of the
    per-(sample, eligible-op) python loop (closes #46).

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
            # Build [X_base[sample] | one_hot(op)] in a single allocation.
            X_repeated = X_base[sample_idx_flat]
            onehot = np.zeros((sample_idx_flat.size, n_ops), dtype=X_base.dtype)
            onehot[np.arange(sample_idx_flat.size), op_idx_flat] = 1
            X_stacked = np.concatenate([X_repeated, onehot], axis=1)

            # ONE predict call covers every eligible (sample, op) pair.
            preds = np.asarray(model.predict(X_stacked))
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

    # PSIS Pareto-k (#80): fit on the *unclipped* importance ratios for the
    # matched subset.  This is the standard PSIS usage — the diagnostic is
    # supposed to tell you whether the raw weights have a problematic tail,
    # independent of any clipping the user might apply.  See Vehtari et al.
    # 2024 (JMLR 25:1-58).  Defer the import to avoid a top-of-module circular
    # cycle with reporting.py.
    from .diagnostics import psis_pareto_k  # noqa: PLC0415

    raw_weights_matched = 1.0 / pi_matched
    pareto_k = psis_pareto_k(raw_weights_matched)

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


# Names of extra estimators that the strategy seam (#86, #85) recognises.
# These are evaluated *in addition* to the historical ``("DR", "SNDR")`` pair.
EXTRA_ESTIMATORS = ("MRDR", "SWITCH-DR", "DRos", "MIPS")


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
    mips_bandwidth: float,
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
    from .estimators import build_strategy, dr_value_with_strategy  # noqa: PLC0415

    extras: dict[str, DRResult] = {}
    # MRDR uses a sample-weighted outcome refit; defer the refit until we
    # know whether MRDR is actually requested (cost amortises per model).
    mrdr_q_hat: np.ndarray | None = None
    canonical_set = {_canonical_estimator_name(n) for n in estimators}
    for canonical in canonical_set:
        if canonical in {"DR", "SNDR"}:
            continue
        if canonical == "MIPS" and action_embedding is None:
            raise ValueError(
                "MIPS estimator was requested but action_embedding= was not "
                "supplied to evaluate_sklearn_models / evaluate_pairwise_models. "
                "Pass an (n_actions, embed_dim) array via action_embedding=."
            )

        strategy = build_strategy(
            canonical,
            clip=float(selected_clip) if np.isfinite(selected_clip) else float("inf"),
            tau=switch_tau,
            lam=dros_lam,
            action_embedding=action_embedding,
            bandwidth=mips_bandwidth,
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
    pi_obs = propensities[np.arange(n), A]
    A_int = A.astype(int)
    elig_bool = elig.astype(bool)
    matched = (pi_obs > 0) & elig_bool[np.arange(n), A_int]

    if clip == float("inf"):
        w_clip = np.where(pi_obs > 0, 1.0 / pi_obs, 0.0)
    else:
        w_clip = np.where(pi_obs > 0, np.minimum(1.0 / pi_obs, clip), 0.0)
    w_clip[~matched] = 0

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
    return q_pi.copy()


def block_bootstrap_ci(
    values_num: np.ndarray,
    values_den: np.ndarray | None,
    base_mean: np.ndarray,  # noqa: ARG001
    n_boot: int = 400,
    block_len: int | None = None,
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
    outcome_estimator: str | Callable[[], Any] = "hgb",
    random_state: int = 0,
    clip_grid: tuple[float, ...] = (2, 5, 10, 20, 50, float("inf")),
    ci_bootstrap: bool = False,
    alpha: float = 0.05,
    policy_train: str | None = None,
    policy_train_frac: float = 0.85,
    support_thresholds: "SupportHealthThresholds | None" = None,
    *,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
    keep_contributions: bool = False,
    max_kept_contributions: int = DEFAULT_MAX_KEPT_CONTRIBUTIONS,
    estimators: tuple[str, ...] = ("DR", "SNDR"),
    action_embedding: np.ndarray | None = None,
    switch_tau: float = 5.0,
    dros_lam: float = 1.0,
    mips_bandwidth: float = 1.0,
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

    Returns
    -------
    EvaluationArtifact
        Bundled result containing ``report``, ``detailed``, ``warnings``,
        ``sensitivity``, ``diagnostics``, and ``metadata``. **Breaking change
        in 0.6.0**: previously returned ``(report, detailed)``; migrate by
        unpacking ``artifact.report`` and ``artifact.detailed``.

    Raises
    ------
    ValueError
        If ``models`` is empty or contains ``None`` values.
    DataValidationError
        If ``keep_contributions=True`` is set with more than
        ``max_kept_contributions`` evaluated decisions.
    """
    if not models:
        raise ValueError("models dict must not be empty")
    if any(v is None for v in models.values()):
        raise ValueError("models dict values must not be None")

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

    # Fit propensity model
    propensities, _ = fit_propensity_timecal(
        eval_design.X_phi,
        eval_design.A,
        eval_design.ts,
        n_splits=n_splits,
        random_state=random_state,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
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
            action_embedding=action_embedding,
            switch_tau=switch_tau,
            dros_lam=dros_lam,
            mips_bandwidth=mips_bandwidth,
            outcome_estimator=outcome_estimator,
            n_splits=n_splits,
            random_state=random_state,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )
        results.update(extras)

        detailed_results[model_name] = results

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
                    report_rows.append(row)
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

            report_rows.append(row)

    report = pd.DataFrame(report_rows)

    from .reporting import build_evaluation_artifact  # noqa: PLC0415

    return build_evaluation_artifact(
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
        },
    )


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
    action_embedding: np.ndarray | None = None,
    switch_tau: float = 5.0,
    dros_lam: float = 1.0,
    mips_bandwidth: float = 1.0,
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
        If ``keep_contributions=True`` is set with more than
        ``max_kept_contributions`` evaluated decisions.
    """

    logger.info("Starting pairwise evaluation")

    # Validate parameters
    if task_type not in ["regression", "binary"]:
        raise ValueError(
            f"Unknown task_type: {task_type}. Must be 'regression' or 'binary'"
        )
    if direction not in ["min", "max"]:
        raise ValueError(f"Unknown direction: {direction}. Must be 'min' or 'max'")

    # Resolve policy_train sentinel: None means "pre_split" + emit warning.
    if policy_train is None:
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
    if fit_models and policy_train == "all":
        feature_cols = design.cli_features + design.op_features
        X_fit = design.logs_df[feature_cols].values.astype(np.float32)
        y_fit = design.logs_df[metric_col].values
        for model in models.values():
            model.fit(X_fit, y_fit)
        logger.info(
            f"Fitted {len(models)} policy model(s) on {len(X_fit)} pairs (policy_train='all')"
        )

    # Induce policies (held-out for pre_split, full data for "all")
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
                action_embedding=action_embedding,
                switch_tau=switch_tau,
                dros_lam=dros_lam,
                mips_bandwidth=mips_bandwidth,
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

    return build_evaluation_artifact(
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
        },
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
