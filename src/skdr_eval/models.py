"""Extended model support for skdr-eval library."""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from .exceptions import ConfigurationError, DataValidationError, ModelValidationError

logger = logging.getLogger("skdr_eval")

# Optional imports for advanced models
try:
    import xgboost as xgb  # pragma: no cover

    XGBOOST_AVAILABLE = True  # pragma: no cover
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.debug("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb  # pragma: no cover

    LIGHTGBM_AVAILABLE = True  # pragma: no cover
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.debug("LightGBM not available. Install with: pip install lightgbm")

_BINARY_THRESHOLD = 2
_ADVANCED_MODEL_SAMPLE_THRESHOLD = 1000
_ADVANCED_MODEL_FEATURE_THRESHOLD = 10

# Default parameter lookup tables
_DEFAULT_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "classification": {
        "logistic": {"max_iter": 1000, "C": 1.0},
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
        },
        "hist_gradient": {"max_iter": 100, "learning_rate": 0.1},
        "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    },
    "regression": {
        "ridge": {"alpha": 1.0},
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
        },
        "hist_gradient": {"max_iter": 100, "learning_rate": 0.1},
        "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    },
}


class ModelFactory:
    """Factory class for creating model instances with consistent interfaces."""

    @staticmethod
    def create_classifier(
        model_type: str, random_state: Optional[int] = None, **kwargs: Any
    ) -> BaseEstimator:
        """Create a classifier instance.

        Parameters
        ----------
        model_type : str
            Type of classifier to create.
        random_state : int, optional
            Random seed.
        **kwargs
            Additional parameters for the model.

        Returns
        -------
        BaseEstimator
            Classifier instance.
        """
        if model_type == "logistic":
            return LogisticRegression(random_state=random_state, **kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(random_state=random_state, **kwargs)
        elif model_type == "hist_gradient":
            return HistGradientBoostingClassifier(random_state=random_state, **kwargs)
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "XGBoost is not installed. Install it with: pip install xgboost"
                )
            return xgb.XGBClassifier(
                random_state=random_state, **kwargs
            )  # pragma: no cover
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError(
                    "LightGBM is not installed. Install it with: pip install lightgbm"
                )
            return lgb.LGBMClassifier(
                random_state=random_state, **kwargs
            )  # pragma: no cover
        else:
            raise ModelValidationError(f"Unknown classifier type: {model_type}")

    @staticmethod
    def create_regressor(
        model_type: str, random_state: Optional[int] = None, **kwargs: Any
    ) -> BaseEstimator:
        """Create a regressor instance.

        Parameters
        ----------
        model_type : str
            Type of regressor to create.
        random_state : int, optional
            Random seed.
        **kwargs
            Additional parameters for the model.

        Returns
        -------
        BaseEstimator
            Regressor instance.
        """
        if model_type == "ridge":
            return Ridge(random_state=random_state, **kwargs)
        elif model_type == "random_forest":
            return RandomForestRegressor(random_state=random_state, **kwargs)
        elif model_type == "hist_gradient":
            return HistGradientBoostingRegressor(random_state=random_state, **kwargs)
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "XGBoost is not installed. Install it with: pip install xgboost"
                )
            return xgb.XGBRegressor(
                random_state=random_state, **kwargs
            )  # pragma: no cover
        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError(
                    "LightGBM is not installed. Install it with: pip install lightgbm"
                )
            return lgb.LGBMRegressor(
                random_state=random_state, **kwargs
            )  # pragma: no cover
        else:
            raise ModelValidationError(f"Unknown regressor type: {model_type}")

    @staticmethod
    def get_available_models() -> dict[str, list[str]]:
        """Get list of available model types.

        Returns
        -------
        dict[str, list[str]]
            Dictionary with 'classifiers' and 'regressors' keys.
        """
        classifiers = ["logistic", "random_forest", "hist_gradient"]
        regressors = ["ridge", "random_forest", "hist_gradient"]

        if XGBOOST_AVAILABLE:  # pragma: no cover
            classifiers.append("xgboost")  # pragma: no cover
            regressors.append("xgboost")  # pragma: no cover

        if LIGHTGBM_AVAILABLE:  # pragma: no cover
            classifiers.append("lightgbm")  # pragma: no cover
            regressors.append("lightgbm")  # pragma: no cover

        return {"classifiers": classifiers, "regressors": regressors}

    @staticmethod
    def get_default_params(model_type: str, task_type: str) -> dict[str, Any]:
        """Get default parameters for a model type.

        Parameters
        ----------
        model_type : str
            Type of model.
        task_type : str
            Task type ("classification" or "regression").

        Returns
        -------
        dict[str, Any]
            Default parameters.

        Raises
        ------
        ConfigurationError
            If model_type or task_type is not recognised.
        """
        if task_type not in _DEFAULT_PARAMS:
            raise ConfigurationError(
                f"Unknown task_type '{task_type}'. "
                "Must be 'classification' or 'regression'."
            )
        task_params = _DEFAULT_PARAMS[task_type]
        if model_type not in task_params:
            raise ConfigurationError(
                f"Unknown model_type '{model_type}' for task '{task_type}'."
            )
        return dict(task_params[model_type])


class ModelEvaluator:
    """Class for evaluating model performance."""

    @staticmethod
    def cross_validate_model(
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> dict[str, Any]:
        """Perform cross-validation on a model.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate.
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric. If None, uses default for task type.
        random_state : int, optional
            Random seed for reproducible fold splits.

        Returns
        -------
        dict[str, Any]
            Cross-validation results.
        """
        if len(X) != len(y):
            raise DataValidationError("X and y must have the same length")

        if len(X) < cv:
            raise DataValidationError(f"Need at least {cv} samples for {cv}-fold CV")

        is_classifier = hasattr(model, "predict_proba")
        if scoring is None:
            scoring = "accuracy" if is_classifier else "neg_mean_squared_error"

        # Use a seeded splitter so random_state is honoured
        splitter = (
            StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            if is_classifier
            else KFold(n_splits=cv, shuffle=True, random_state=random_state)
        )

        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "scoring": scoring,
        }

    @staticmethod
    def evaluate_model_performance(
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification",
    ) -> dict[str, float]:
        """Evaluate model performance on train and test sets.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate.
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test targets.
        task_type : str, default="classification"
            Task type ("classification" or "regression").

        Returns
        -------
        dict[str, float]
            Performance metrics.
        """
        if len(X_train) != len(y_train):
            raise DataValidationError("X_train and y_train must have the same length")
        if len(X_test) != len(y_test):
            raise DataValidationError("X_test and y_test must have the same length")
        if task_type not in ("classification", "regression"):
            raise ConfigurationError(
                f"Unknown task_type '{task_type}'. "
                "Must be 'classification' or 'regression'."
            )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results: dict[str, float] = {}

        if task_type == "classification":
            results["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
            results["test_accuracy"] = float(accuracy_score(y_test, y_test_pred))
            results["train_precision"] = float(
                precision_score(y_train, y_train_pred, average="weighted")
            )
            results["test_precision"] = float(
                precision_score(y_test, y_test_pred, average="weighted")
            )
            results["train_recall"] = float(
                recall_score(y_train, y_train_pred, average="weighted")
            )
            results["test_recall"] = float(
                recall_score(y_test, y_test_pred, average="weighted")
            )
            results["train_f1"] = float(
                f1_score(y_train, y_train_pred, average="weighted")
            )
            results["test_f1"] = float(
                f1_score(y_test, y_test_pred, average="weighted")
            )

            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)

                try:
                    results["train_log_loss"] = float(log_loss(y_train, y_train_proba))
                    results["test_log_loss"] = float(log_loss(y_test, y_test_proba))
                except ValueError:  # pragma: no cover
                    logger.debug(
                        "Could not compute log loss", exc_info=True
                    )  # pragma: no cover

                try:
                    if len(np.unique(y_test)) == _BINARY_THRESHOLD:
                        results["test_roc_auc"] = float(
                            roc_auc_score(y_test, y_test_proba[:, 1])
                        )
                except ValueError:  # pragma: no cover
                    logger.debug(
                        "Could not compute ROC AUC", exc_info=True
                    )  # pragma: no cover

        else:
            results["train_mse"] = float(mean_squared_error(y_train, y_train_pred))
            results["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
            results["train_mae"] = float(mean_absolute_error(y_train, y_train_pred))
            results["test_mae"] = float(mean_absolute_error(y_test, y_test_pred))
            results["train_r2"] = float(r2_score(y_train, y_train_pred))
            results["test_r2"] = float(r2_score(y_test, y_test_pred))

        return results


class ModelSelector:
    """Class for model selection and hyperparameter tuning."""

    @staticmethod
    def grid_search(
        model_type: str,
        param_grid: dict[str, list[Any]],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> dict[str, Any]:
        """Perform grid search for hyperparameter tuning.

        Parameters
        ----------
        model_type : str
            Type of model to tune.
        param_grid : dict[str, list[Any]]
            Parameter grid for search.
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        task_type : str, default="classification"
            Task type ("classification" or "regression").
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric.
        random_state : int, optional
            Random seed.

        Returns
        -------
        dict[str, Any]
            Grid search results.
        """
        if task_type == "classification":
            model = ModelFactory.create_classifier(
                model_type, random_state=random_state
            )
        else:
            model = ModelFactory.create_regressor(model_type, random_state=random_state)

        if scoring is None:
            scoring = (
                "accuracy"
                if task_type == "classification"
                else "neg_mean_squared_error"
            )

        gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        gs.fit(X, y)

        return {
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "best_estimator": gs.best_estimator_,
            "cv_results": gs.cv_results_,
        }

    @staticmethod
    def random_search(
        model_type: str,
        param_distributions: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
        n_iter: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> dict[str, Any]:
        """Perform random search for hyperparameter tuning.

        Parameters
        ----------
        model_type : str
            Type of model to tune.
        param_distributions : dict[str, Any]
            Parameter distributions for search.
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        task_type : str, default="classification"
            Task type ("classification" or "regression").
        n_iter : int, default=100
            Number of iterations.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric.
        random_state : int, optional
            Random seed.

        Returns
        -------
        dict[str, Any]
            Random search results.
        """
        if task_type == "classification":
            model = ModelFactory.create_classifier(
                model_type, random_state=random_state
            )
        else:
            model = ModelFactory.create_regressor(model_type, random_state=random_state)

        if scoring is None:
            scoring = (
                "accuracy"
                if task_type == "classification"
                else "neg_mean_squared_error"
            )

        rs = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
        )
        rs.fit(X, y)

        return {
            "best_params": rs.best_params_,
            "best_score": float(rs.best_score_),
            "best_estimator": rs.best_estimator_,
            "cv_results": rs.cv_results_,
        }


def get_model_recommendations(
    task_type: str,
    n_samples: int,
    n_features: int,
    problem_complexity: str = "medium",
) -> list[str]:
    """Get model recommendations based on problem characteristics.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    problem_complexity : str, default="medium"
        Problem complexity ("low", "medium", "high").

    Returns
    -------
    list[str]
        Recommended model types.
    """
    if task_type not in ("classification", "regression"):
        raise ConfigurationError("task_type must be 'classification' or 'regression'")

    if problem_complexity not in ("low", "medium", "high"):
        raise ConfigurationError(
            "problem_complexity must be 'low', 'medium', or 'high'"
        )

    if task_type == "classification":
        recommendations: list[str] = ["logistic", "random_forest"]
    else:
        recommendations = ["ridge", "random_forest"]

    # Add advanced models for large, high-dimensional datasets
    if (
        n_samples > _ADVANCED_MODEL_SAMPLE_THRESHOLD
        and n_features > _ADVANCED_MODEL_FEATURE_THRESHOLD
    ):
        recommendations.append("hist_gradient")
        if XGBOOST_AVAILABLE:  # pragma: no cover
            recommendations.append("xgboost")  # pragma: no cover
        if LIGHTGBM_AVAILABLE:  # pragma: no cover
            recommendations.append("lightgbm")  # pragma: no cover

    if problem_complexity == "high":
        if "random_forest" not in recommendations:
            recommendations.append("random_forest")
        if XGBOOST_AVAILABLE and "xgboost" not in recommendations:  # pragma: no cover
            recommendations.append("xgboost")  # pragma: no cover
    elif problem_complexity == "low":
        simple = "logistic" if task_type == "classification" else "ridge"
        recommendations = [simple] + [m for m in recommendations if m != simple]

    return recommendations


def create_model_ensemble(
    model_types: list[str],
    task_type: str,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Create an ensemble of models.

    Parameters
    ----------
    model_types : list[str]
        List of model types to include in ensemble.
    task_type : str
        Task type ("classification" or "regression").
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for ensemble.

    Returns
    -------
    BaseEstimator
        Ensemble model.
    """
    if task_type == "classification":
        estimators = [
            (f"{mt}_{i}", ModelFactory.create_classifier(mt, random_state=random_state))
            for i, mt in enumerate(model_types)
        ]
        kwargs.setdefault("voting", "soft")
        return VotingClassifier(estimators, **kwargs)
    else:
        estimators = [
            (f"{mt}_{i}", ModelFactory.create_regressor(mt, random_state=random_state))
            for i, mt in enumerate(model_types)
        ]
        return VotingRegressor(estimators, **kwargs)
