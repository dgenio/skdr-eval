"""Extended model support for skdr-eval library."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from .exceptions import DataValidationError, ModelValidationError

logger = logging.getLogger("skdr_eval")

# Optional imports for advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HIST_GRADIENT_AVAILABLE = True
except ImportError:
    HIST_GRADIENT_AVAILABLE = False
    logger.warning("HistGradientBoosting not available. Requires scikit-learn >= 0.21")


class ModelFactory:
    """Factory class for creating model instances with consistent interfaces."""

    @staticmethod
    def create_classifier(
        model_type: str,
        random_state: Optional[int] = None,
        **kwargs
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
        elif model_type == "hist_gradient" and HIST_GRADIENT_AVAILABLE:
            return HistGradientBoostingClassifier(random_state=random_state, **kwargs)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=random_state, **kwargs)
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {model_type}")

    @staticmethod
    def create_regressor(
        model_type: str,
        random_state: Optional[int] = None,
        **kwargs
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
        elif model_type == "hist_gradient" and HIST_GRADIENT_AVAILABLE:
            return HistGradientBoostingRegressor(random_state=random_state, **kwargs)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(random_state=random_state, **kwargs)
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown regressor type: {model_type}")

    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available model types.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with 'classifiers' and 'regressors' keys.
        """
        classifiers = ["logistic", "random_forest"]
        regressors = ["ridge", "random_forest"]

        if HIST_GRADIENT_AVAILABLE:
            classifiers.append("hist_gradient")
            regressors.append("hist_gradient")

        if XGBOOST_AVAILABLE:
            classifiers.append("xgboost")
            regressors.append("xgboost")

        if LIGHTGBM_AVAILABLE:
            classifiers.append("lightgbm")
            regressors.append("lightgbm")

        return {
            "classifiers": classifiers,
            "regressors": regressors
        }

    @staticmethod
    def get_default_params(model_type: str, task_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type.

        Parameters
        ----------
        model_type : str
            Type of model.
        task_type : str
            Task type ("classification" or "regression").

        Returns
        -------
        Dict[str, Any]
            Default parameters.
        """
        if task_type == "classification":
            if model_type == "logistic":
                return {"max_iter": 1000, "C": 1.0}
            elif model_type == "random_forest":
                return {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
            elif model_type == "hist_gradient":
                return {"max_iter": 100, "learning_rate": 0.1}
            elif model_type == "xgboost":
                return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
            elif model_type == "lightgbm":
                return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
        else:  # regression
            if model_type == "ridge":
                return {"alpha": 1.0}
            elif model_type == "random_forest":
                return {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
            elif model_type == "hist_gradient":
                return {"max_iter": 100, "learning_rate": 0.1}
            elif model_type == "xgboost":
                return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
            elif model_type == "lightgbm":
                return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}

        return {}


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
    ) -> Dict[str, float]:
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
            Random seed.

        Returns
        -------
        Dict[str, float]
            Cross-validation results.
        """
        if len(X) != len(y):
            raise DataValidationError("X and y must have the same length")

        if len(X) < cv:
            raise DataValidationError(f"Need at least {cv} samples for {cv}-fold CV")

        # Determine scoring metric if not provided
        if scoring is None:
            if hasattr(model, "predict_proba"):  # Classifier
                scoring = "accuracy"
            else:  # Regressor
                scoring = "neg_mean_squared_error"

        # Perform cross-validation
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scoring, random_state=random_state
        )

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "scoring": scoring
        }

    @staticmethod
    def evaluate_model_performance(
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification",
    ) -> Dict[str, float]:
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
        Dict[str, float]
            Performance metrics.
        """
        if len(X_train) != len(y_train):
            raise DataValidationError("X_train and y_train must have the same length")
        if len(X_test) != len(y_test):
            raise DataValidationError("X_test and y_test must have the same length")

        # Fit model
        model.fit(X_train, y_train)

        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results = {}

        if task_type == "classification":
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            results["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
            results["test_accuracy"] = float(accuracy_score(y_test, y_test_pred))
            results["train_precision"] = float(precision_score(y_train, y_train_pred, average="weighted"))
            results["test_precision"] = float(precision_score(y_test, y_test_pred, average="weighted"))
            results["train_recall"] = float(recall_score(y_train, y_train_pred, average="weighted"))
            results["test_recall"] = float(recall_score(y_test, y_test_pred, average="weighted"))
            results["train_f1"] = float(f1_score(y_train, y_train_pred, average="weighted"))
            results["test_f1"] = float(f1_score(y_test, y_test_pred, average="weighted"))

            # Add probability-based metrics if available
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)

                from sklearn.metrics import log_loss, roc_auc_score

                try:
                    results["train_log_loss"] = float(log_loss(y_train, y_train_proba))
                    results["test_log_loss"] = float(log_loss(y_test, y_test_proba))
                except ValueError:
                    pass  # Skip if log loss can't be computed

                try:
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        results["test_roc_auc"] = float(roc_auc_score(y_test, y_test_proba[:, 1]))
                except ValueError:
                    pass  # Skip if ROC AUC can't be computed

        else:  # regression
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning.

        Parameters
        ----------
        model_type : str
            Type of model to tune.
        param_grid : Dict[str, List[Any]]
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
        Dict[str, Any]
            Grid search results.
        """
        from sklearn.model_selection import GridSearchCV

        if task_type == "classification":
            model = ModelFactory.create_classifier(model_type, random_state=random_state)
        else:
            model = ModelFactory.create_regressor(model_type, random_state=random_state)

        # Determine scoring metric if not provided
        if scoring is None:
            if task_type == "classification":
                scoring = "accuracy"
            else:
                scoring = "neg_mean_squared_error"

        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, random_state=random_state
        )
        grid_search.fit(X, y)

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "best_estimator": grid_search.best_estimator_,
            "cv_results": grid_search.cv_results_
        }

    @staticmethod
    def random_search(
        model_type: str,
        param_distributions: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
        n_iter: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform random search for hyperparameter tuning.

        Parameters
        ----------
        model_type : str
            Type of model to tune.
        param_distributions : Dict[str, Any]
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
        Dict[str, Any]
            Random search results.
        """
        from sklearn.model_selection import RandomizedSearchCV

        if task_type == "classification":
            model = ModelFactory.create_classifier(model_type, random_state=random_state)
        else:
            model = ModelFactory.create_regressor(model_type, random_state=random_state)

        # Determine scoring metric if not provided
        if scoring is None:
            if task_type == "classification":
                scoring = "accuracy"
            else:
                scoring = "neg_mean_squared_error"

        # Perform random search
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=n_iter, cv=cv, 
            scoring=scoring, random_state=random_state
        )
        random_search.fit(X, y)

        return {
            "best_params": random_search.best_params_,
            "best_score": float(random_search.best_score_),
            "best_estimator": random_search.best_estimator_,
            "cv_results": random_search.cv_results_
        }


def get_model_recommendations(
    task_type: str,
    n_samples: int,
    n_features: int,
    problem_complexity: str = "medium",
) -> List[str]:
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
    List[str]
        Recommended model types.
    """
    if task_type not in ["classification", "regression"]:
        raise ValueError("task_type must be 'classification' or 'regression'")

    if problem_complexity not in ["low", "medium", "high"]:
        raise ValueError("problem_complexity must be 'low', 'medium', or 'high'")

    recommendations = []

    # Base recommendations
    if task_type == "classification":
        recommendations.extend(["logistic", "random_forest"])
    else:
        recommendations.extend(["ridge", "random_forest"])

    # Add advanced models based on problem characteristics
    if n_samples > 1000 and n_features > 10:
        if HIST_GRADIENT_AVAILABLE:
            recommendations.append("hist_gradient")
        if XGBOOST_AVAILABLE:
            recommendations.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            recommendations.append("lightgbm")

    # Adjust based on problem complexity
    if problem_complexity == "high":
        # For high complexity, prefer ensemble methods
        if "random_forest" not in recommendations:
            recommendations.append("random_forest")
        if XGBOOST_AVAILABLE and "xgboost" not in recommendations:
            recommendations.append("xgboost")
    elif problem_complexity == "low":
        # For low complexity, prefer simple models
        if task_type == "classification":
            recommendations = ["logistic"] + [m for m in recommendations if m != "logistic"]
        else:
            recommendations = ["ridge"] + [m for m in recommendations if m != "ridge"]

    return recommendations


def create_model_ensemble(
    model_types: List[str],
    task_type: str,
    random_state: Optional[int] = None,
    **kwargs
) -> BaseEstimator:
    """Create an ensemble of models.

    Parameters
    ----------
    model_types : List[str]
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
    from sklearn.ensemble import VotingClassifier, VotingRegressor

    if task_type == "classification":
        estimators = []
        for i, model_type in enumerate(model_types):
            model = ModelFactory.create_classifier(model_type, random_state=random_state)
            estimators.append((f"{model_type}_{i}", model))
        
        return VotingClassifier(estimators, **kwargs)
    else:
        estimators = []
        for i, model_type in enumerate(model_types):
            model = ModelFactory.create_regressor(model_type, random_state=random_state)
            estimators.append((f"{model_type}_{i}", model))
        
        return VotingRegressor(estimators, **kwargs)