"""Tests for models module."""

import numpy as np
import pytest

from skdr_eval.models import (
    ModelEvaluator,
    ModelFactory,
    ModelSelector,
    create_model_ensemble,
    get_model_recommendations,
)


def test_model_factory_classifier():
    """Test ModelFactory for creating classifiers."""
    # Test logistic regression
    model = ModelFactory.create_classifier("logistic", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    
    # Test random forest
    model = ModelFactory.create_classifier("random_forest", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    
    # Test with custom parameters
    model = ModelFactory.create_classifier("logistic", random_state=42, C=0.1)
    assert model.C == 0.1


def test_model_factory_regressor():
    """Test ModelFactory for creating regressors."""
    # Test ridge regression
    model = ModelFactory.create_regressor("ridge", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    
    # Test random forest
    model = ModelFactory.create_regressor("random_forest", random_state=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    
    # Test with custom parameters
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


def test_model_factory_default_params():
    """Test getting default parameters."""
    # Test classification
    params = ModelFactory.get_default_params("logistic", "classification")
    assert "max_iter" in params
    assert "C" in params
    
    # Test regression
    params = ModelFactory.get_default_params("ridge", "regression")
    assert "alpha" in params


def test_model_evaluator_cross_validate():
    """Test cross-validation functionality."""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Create model
    model = ModelFactory.create_classifier("logistic", random_state=42)
    
    # Test cross-validation
    results = ModelEvaluator.cross_validate_model(
        model, X, y, cv=5, random_state=42
    )
    
    assert "mean_score" in results
    assert "std_score" in results
    assert "scores" in results
    assert "scoring" in results
    assert len(results["scores"]) == 5
    assert 0 <= results["mean_score"] <= 1


def test_model_evaluator_performance():
    """Test model performance evaluation."""
    # Create test data
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randint(0, 2, 20)
    
    # Create model
    model = ModelFactory.create_classifier("logistic", random_state=42)
    
    # Test performance evaluation
    results = ModelEvaluator.evaluate_model_performance(
        model, X_train, y_train, X_test, y_test, task_type="classification"
    )
    
    assert "train_accuracy" in results
    assert "test_accuracy" in results
    assert "train_precision" in results
    assert "test_precision" in results
    assert "train_recall" in results
    assert "test_recall" in results
    assert "train_f1" in results
    assert "test_f1" in results
    
    # Test regression
    y_train_reg = np.random.randn(80)
    y_test_reg = np.random.randn(20)
    
    model_reg = ModelFactory.create_regressor("ridge", random_state=42)
    results_reg = ModelEvaluator.evaluate_model_performance(
        model_reg, X_train, y_train_reg, X_test, y_test_reg, task_type="regression"
    )
    
    assert "train_mse" in results_reg
    assert "test_mse" in results_reg
    assert "train_mae" in results_reg
    assert "test_mae" in results_reg
    assert "train_r2" in results_reg
    assert "test_r2" in results_reg


def test_model_selector_grid_search():
    """Test grid search functionality."""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Define parameter grid
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "max_iter": [100, 1000]
    }
    
    # Test grid search
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
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Define parameter distributions
    param_distributions = {
        "C": [0.1, 1.0, 10.0],
        "max_iter": [100, 1000]
    }
    
    # Test random search
    results = ModelSelector.random_search(
        "logistic", param_distributions, X, y, task_type="classification", 
        n_iter=10, cv=3, random_state=42
    )
    
    assert "best_params" in results
    assert "best_score" in results
    assert "best_estimator" in results
    assert "cv_results" in results


def test_get_model_recommendations():
    """Test model recommendations."""
    # Test classification recommendations
    recs = get_model_recommendations("classification", 1000, 10, "medium")
    assert "logistic" in recs
    assert "random_forest" in recs
    
    # Test regression recommendations
    recs = get_model_recommendations("regression", 1000, 10, "medium")
    assert "ridge" in recs
    assert "random_forest" in recs
    
    # Test high complexity
    recs = get_model_recommendations("classification", 1000, 10, "high")
    assert "random_forest" in recs
    
    # Test low complexity
    recs = get_model_recommendations("classification", 100, 5, "low")
    assert "logistic" in recs


def test_create_model_ensemble():
    """Test model ensemble creation."""
    # Test classification ensemble
    ensemble = create_model_ensemble(
        ["logistic", "random_forest"], "classification", random_state=42
    )
    assert hasattr(ensemble, "fit")
    assert hasattr(ensemble, "predict")
    assert hasattr(ensemble, "predict_proba")
    
    # Test regression ensemble
    ensemble = create_model_ensemble(
        ["ridge", "random_forest"], "regression", random_state=42
    )
    assert hasattr(ensemble, "fit")
    assert hasattr(ensemble, "predict")


def test_error_handling():
    """Test error handling in model functions."""
    # Test with invalid model type
    with pytest.raises(ValueError):
        ModelFactory.create_classifier("invalid_model")
    
    with pytest.raises(ValueError):
        ModelFactory.create_regressor("invalid_model")
    
    # Test with mismatched data lengths
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 50)  # Wrong length
    
    model = ModelFactory.create_classifier("logistic")
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        ModelEvaluator.cross_validate_model(model, X, y)
    
    # Test with insufficient data for CV
    X_small = np.random.randn(3, 5)
    y_small = np.random.randint(0, 2, 3)
    
    with pytest.raises(Exception):  # Should raise DataValidationError
        ModelEvaluator.cross_validate_model(model, X_small, y_small, cv=5)
    
    # Test with invalid task type
    with pytest.raises(ValueError):
        get_model_recommendations("invalid_task", 100, 10, "medium")
    
    with pytest.raises(ValueError):
        get_model_recommendations("classification", 100, 10, "invalid_complexity")


def test_integration():
    """Test integration of model components."""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 2, 200)
    
    # Split data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Get model recommendations
    recommendations = get_model_recommendations("classification", 200, 5, "medium")
    
    # Create and evaluate models
    for model_type in recommendations[:2]:  # Test first 2 recommendations
        model = ModelFactory.create_classifier(model_type, random_state=42)
        
        # Cross-validate
        cv_results = ModelEvaluator.cross_validate_model(
            model, X_train, y_train, cv=3, random_state=42
        )
        
        # Evaluate performance
        perf_results = ModelEvaluator.evaluate_model_performance(
            model, X_train, y_train, X_test, y_test, task_type="classification"
        )
        
        assert cv_results["mean_score"] > 0
        assert perf_results["test_accuracy"] > 0


if __name__ == "__main__":
    pytest.main([__file__])