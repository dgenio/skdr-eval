"""Tests for models module."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.svm import SVC

from skdr_eval.exceptions import (
    ConfigurationError,
    DataValidationError,
    ModelValidationError,
)
from skdr_eval.models import (
    _ADVANCED_MODEL_FEATURE_THRESHOLD,
    _ADVANCED_MODEL_SAMPLE_THRESHOLD,
    ModelEvaluator,
    ModelFactory,
    ModelSelector,
    create_model_ensemble,
    get_model_recommendations,
)


def test_model_factory_classifier():
    """Test ModelFactory for creating classifiers."""
    model = ModelFactory.create_classifier("logistic", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    model = ModelFactory.create_classifier("random_forest", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    model = ModelFactory.create_classifier("logistic", random_state=42, C=0.1)
    assert model.C == 0.1


def test_model_factory_regressor():
    """Test ModelFactory for creating regressors."""
    model = ModelFactory.create_regressor("ridge", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    model = ModelFactory.create_regressor("random_forest", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    model = ModelFactory.create_regressor("ridge", random_state=42, alpha=0.1)
    assert model.alpha == 0.1


def test_model_factory_available_models():
    """Test getting available model types."""
    available = ModelFactory.get_available_models()

    assert "classifiers" in available
    assert "regressors" in available
    assert "logistic" in available["classifiers"]
    assert "ridge" in available["regressors"]
    assert "random_forest" in available["classifiers"]
    assert "random_forest" in available["regressors"]
    assert "hist_gradient" in available["classifiers"]
    assert "hist_gradient" in available["regressors"]


def test_model_factory_default_params():
    """Test getting default parameters."""
    params = ModelFactory.get_default_params("logistic", "classification")
    assert "max_iter" in params
    assert "C" in params

    params = ModelFactory.get_default_params("ridge", "regression")
    assert "alpha" in params


def test_model_factory_default_params_unknown_raises():
    """get_default_params raises ConfigurationError for unknown inputs."""
    with pytest.raises(ConfigurationError):
        ModelFactory.get_default_params("logisitc", "classification")  # typo

    with pytest.raises(ConfigurationError):
        ModelFactory.get_default_params("logistic", "invalid_task")


def test_model_factory_optional_dep_importerror():
    """create_classifier/regressor raises ImportError when optional dep absent."""
    with patch("skdr_eval.models.XGBOOST_AVAILABLE", False):
        with pytest.raises(ImportError, match="pip install xgboost"):
            ModelFactory.create_classifier("xgboost")

        with pytest.raises(ImportError, match="pip install xgboost"):
            ModelFactory.create_regressor("xgboost")

    with patch("skdr_eval.models.LIGHTGBM_AVAILABLE", False):
        with pytest.raises(ImportError, match="pip install lightgbm"):
            ModelFactory.create_classifier("lightgbm")

        with pytest.raises(ImportError, match="pip install lightgbm"):
            ModelFactory.create_regressor("lightgbm")


def test_model_evaluator_cross_validate():
    """Test cross-validation functionality."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    model = ModelFactory.create_classifier("logistic", random_state=42)

    results = ModelEvaluator.cross_validate_model(model, X, y, cv=5, random_state=42)

    assert "mean_score" in results
    assert "std_score" in results
    assert "scores" in results
    assert "scoring" in results
    assert len(results["scores"]) == 5
    assert 0 <= results["mean_score"] <= 1
    assert results["scoring"] == "accuracy"


def test_model_evaluator_cross_validate_regressor_scoring():
    """cross_validate_model defaults to neg_mean_squared_error for regressors."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = ModelFactory.create_regressor("ridge", random_state=42)
    results = ModelEvaluator.cross_validate_model(model, X, y, cv=3, random_state=42)

    assert results["scoring"] == "neg_mean_squared_error"
    assert len(results["scores"]) == 3


def test_model_evaluator_cross_validate_random_state_reproducible():
    """random_state produces identical fold splits across calls."""
    np.random.seed(0)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    model = ModelFactory.create_classifier("logistic", random_state=42)
    r1 = ModelEvaluator.cross_validate_model(model, X, y, cv=5, random_state=7)
    r2 = ModelEvaluator.cross_validate_model(model, X, y, cv=5, random_state=7)
    assert r1["scores"] == r2["scores"]

    r3 = ModelEvaluator.cross_validate_model(model, X, y, cv=5, random_state=99)
    # Different seed should (almost always) produce different fold order
    assert r1["scores"] != r3["scores"]


def test_model_evaluator_performance():
    """Test model performance evaluation."""
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randint(0, 2, 20)

    model = ModelFactory.create_classifier("logistic", random_state=42)
    results = ModelEvaluator.evaluate_model_performance(
        model, X_train, y_train, X_test, y_test, task_type="classification"
    )

    for key in (
        "train_accuracy",
        "test_accuracy",
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
        "train_f1",
        "test_f1",
    ):
        assert key in results

    # Regression branch
    y_train_reg = np.random.randn(80)
    y_test_reg = np.random.randn(20)
    model_reg = ModelFactory.create_regressor("ridge", random_state=42)
    results_reg = ModelEvaluator.evaluate_model_performance(
        model_reg, X_train, y_train_reg, X_test, y_test_reg, task_type="regression"
    )

    for key in (
        "train_mse",
        "test_mse",
        "train_mae",
        "test_mae",
        "train_r2",
        "test_r2",
    ):
        assert key in results_reg


def test_model_evaluator_performance_invalid_task_type():
    """evaluate_model_performance raises ConfigurationError for invalid task_type."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    model = ModelFactory.create_classifier("logistic", random_state=42)
    model.fit(X, y)

    with pytest.raises(ConfigurationError):
        ModelEvaluator.evaluate_model_performance(
            model, X, y, X, y, task_type="invalid"
        )


def test_model_evaluator_performance_no_predict_proba():
    """evaluate_model_performance skips proba metrics when model lacks predict_proba."""
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randint(0, 2, 20)

    model = SVC()  # no predict_proba by default
    results = ModelEvaluator.evaluate_model_performance(
        model, X_train, y_train, X_test, y_test, task_type="classification"
    )

    assert "train_accuracy" in results
    assert "train_log_loss" not in results
    assert "test_roc_auc" not in results


def test_model_evaluator_performance_multiclass():
    """evaluate_model_performance skips roc_auc for multiclass (>2 classes)."""
    np.random.seed(42)
    X_train = np.random.randn(90, 5)
    y_train = np.tile([0, 1, 2], 30)
    X_test = np.random.randn(30, 5)
    y_test = np.tile([0, 1, 2], 10)

    model = ModelFactory.create_classifier("logistic", random_state=42, max_iter=500)
    results = ModelEvaluator.evaluate_model_performance(
        model, X_train, y_train, X_test, y_test, task_type="classification"
    )

    assert "train_accuracy" in results
    assert "train_log_loss" in results
    assert "test_roc_auc" not in results  # skipped for multiclass


def test_model_factory_hist_gradient():
    """hist_gradient classifier and regressor are created and can fit."""
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y_clf = np.random.randint(0, 2, 60)
    y_reg = np.random.randn(60)

    clf = ModelFactory.create_classifier("hist_gradient", random_state=42)
    clf.fit(X, y_clf)
    assert clf.predict(X).shape == (60,)

    reg = ModelFactory.create_regressor("hist_gradient", random_state=42)
    reg.fit(X, y_reg)
    assert reg.predict(X).shape == (60,)


def test_model_factory_default_params_hist_gradient():
    """get_default_params returns params for hist_gradient."""
    clf_params = ModelFactory.get_default_params("hist_gradient", "classification")
    assert "max_iter" in clf_params
    assert "learning_rate" in clf_params

    reg_params = ModelFactory.get_default_params("hist_gradient", "regression")
    assert "max_iter" in reg_params


def test_model_selector_grid_search_regression():
    """grid_search uses regression path when task_type='regression'."""
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = np.random.randn(80)

    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    results = ModelSelector.grid_search(
        "ridge", param_grid, X, y, task_type="regression", cv=3, random_state=42
    )

    assert "best_params" in results
    assert "alpha" in results["best_params"]
    assert results["best_score"] < 0  # neg_mean_squared_error is negative


def test_model_selector_random_search_regression():
    """random_search uses regression path when task_type='regression'."""
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = np.random.randn(80)

    param_distributions = {"alpha": [0.01, 0.1, 1.0, 10.0]}
    results = ModelSelector.random_search(
        "ridge",
        param_distributions,
        X,
        y,
        task_type="regression",
        n_iter=4,
        cv=3,
        random_state=42,
    )

    assert "best_params" in results
    assert results["best_score"] < 0  # neg_mean_squared_error is negative


def test_model_selector_grid_search():
    """Test grid search functionality."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    param_grid = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 1000]}
    results = ModelSelector.grid_search(
        "logistic", param_grid, X, y, task_type="classification", cv=3, random_state=42
    )

    assert "best_params" in results
    assert "best_score" in results
    assert "best_estimator" in results
    assert "cv_results" in results
    assert "C" in results["best_params"]
    assert "max_iter" in results["best_params"]


def test_model_selector_random_search():
    """Test random search functionality."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    param_distributions = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 1000]}
    results = ModelSelector.random_search(
        "logistic",
        param_distributions,
        X,
        y,
        task_type="classification",
        n_iter=10,
        cv=3,
        random_state=42,
    )

    assert "best_params" in results
    assert "best_score" in results
    assert "best_estimator" in results
    assert "cv_results" in results


def test_get_model_recommendations():
    """Test model recommendations for basic cases."""
    recs = get_model_recommendations("classification", 1000, 10, "medium")
    assert "logistic" in recs
    assert "random_forest" in recs

    recs = get_model_recommendations("regression", 1000, 10, "medium")
    assert "ridge" in recs
    assert "random_forest" in recs

    recs = get_model_recommendations("classification", 1000, 10, "high")
    assert "random_forest" in recs

    recs = get_model_recommendations("classification", 100, 5, "low")
    assert recs[0] == "logistic"

    recs = get_model_recommendations("regression", 100, 5, "low")
    assert recs[0] == "ridge"


def test_get_model_recommendations_advanced_models():
    """n_samples > 1000 and n_features > 10 triggers hist_gradient inclusion."""
    recs = get_model_recommendations("classification", 2000, 15, "medium")
    assert "hist_gradient" in recs

    # Boundary: n_samples exactly at threshold should NOT trigger advanced models
    recs_boundary = get_model_recommendations(
        "classification",
        _ADVANCED_MODEL_SAMPLE_THRESHOLD,
        _ADVANCED_MODEL_FEATURE_THRESHOLD + 1,
        "medium",
    )
    assert "hist_gradient" not in recs_boundary


def test_create_model_ensemble():
    """Test model ensemble creation."""
    ensemble = create_model_ensemble(
        ["logistic", "random_forest"], "classification", random_state=42
    )
    assert hasattr(ensemble, "fit")
    assert hasattr(ensemble, "predict")
    assert hasattr(ensemble, "predict_proba")

    ensemble = create_model_ensemble(
        ["ridge", "random_forest"], "regression", random_state=42
    )
    assert hasattr(ensemble, "fit")
    assert hasattr(ensemble, "predict")


def test_error_handling():
    """Test error handling in model functions."""
    with pytest.raises(ModelValidationError):
        ModelFactory.create_classifier("invalid_model")

    with pytest.raises(ModelValidationError):
        ModelFactory.create_regressor("invalid_model")

    # Mismatched data lengths
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 50)
    model = ModelFactory.create_classifier("logistic")

    with pytest.raises(DataValidationError):
        ModelEvaluator.cross_validate_model(model, X, y)

    # Insufficient data for CV
    X_small = np.random.randn(3, 5)
    y_small = np.random.randint(0, 2, 3)

    with pytest.raises(DataValidationError):
        ModelEvaluator.cross_validate_model(model, X_small, y_small, cv=5)

    # Invalid task type / complexity
    with pytest.raises(ConfigurationError):
        get_model_recommendations("invalid_task", 100, 10, "medium")

    with pytest.raises(ConfigurationError):
        get_model_recommendations("classification", 100, 10, "invalid_complexity")


def test_integration():
    """Test integration of model components."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 2, 200)

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    recommendations = get_model_recommendations("classification", 200, 5, "medium")

    for model_type in recommendations[:2]:
        model = ModelFactory.create_classifier(model_type, random_state=42)

        cv_results = ModelEvaluator.cross_validate_model(
            model, X_train, y_train, cv=3, random_state=42
        )
        perf_results = ModelEvaluator.evaluate_model_performance(
            model, X_train, y_train, X_test, y_test, task_type="classification"
        )

        assert cv_results["mean_score"] > 0
        assert perf_results["test_accuracy"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
