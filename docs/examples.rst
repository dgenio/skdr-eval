Examples
========

This page contains examples and tutorials for using skdr-eval.

Jupyter Notebooks
-----------------

Interactive examples are available as Jupyter notebooks:

.. toctree::
   :maxdepth: 2
   
   ../examples/advanced_evaluation

Python Scripts
--------------

Basic Usage
~~~~~~~~~~~

The quickstart example demonstrates basic usage:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :linenos:

Advanced Patterns
~~~~~~~~~~~~~~~~~

For more advanced usage patterns, see the Jupyter notebooks above or explore the following scenarios:

Custom Model Integration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.base import BaseEstimator, RegressorMixin
   import skdr_eval
   
   class CustomModel(BaseEstimator, RegressorMixin):
       def __init__(self, param1=1.0):
           self.param1 = param1
           
       def fit(self, X, y):
           # Your custom fitting logic
           return self
           
       def predict(self, X):
           # Your custom prediction logic
           return predictions
   
   # Use with skdr-eval
   models = {
       "Custom": CustomModel(param1=2.0),
       "Baseline": RandomForestRegressor()
   }
   
   report, results = skdr_eval.evaluate_sklearn_models(
       logs=logs, models=models, fit_models=True
   )

Time-Series Specific Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For time-series data with specific temporal patterns
   report, results = skdr_eval.evaluate_sklearn_models(
       logs=logs,
       models=models,
       n_splits=5,  # More splits for robust temporal evaluation
       policy_train="pre_split",  # Train policy before temporal splits
       policy_train_frac=0.7,  # Use 70% for policy training
   )

Bootstrap Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get confidence intervals for individual results
   dr_result = results["RandomForest"][0]  # First split result
   
   ci_lower, ci_upper = skdr_eval.block_bootstrap_ci(
       dr_result.logs_eval,
       dr_result.target_policy,
       dr_result.logging_policy,
       dr_result.outcome_preds,
       dr_result.propensity_scores,
       n_bootstrap=2000,  # More samples for tighter CIs
       block_size=100,    # Larger blocks for stronger temporal dependence
   )
