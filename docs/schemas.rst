Data Schemas and Structures
===========================

This page documents the data structures and schemas used in skdr-eval.

Input Data Schema
-----------------

Service Logs DataFrame
~~~~~~~~~~~~~~~~~~~~~~

The main input to skdr-eval is a pandas DataFrame representing service logs with the following required columns:

.. code-block:: python

   # Required columns for service logs
   logs_schema = {
       'timestamp': 'datetime64[ns]',  # When the request was made
       'operator': 'object',           # Which operator/policy was used
       'service_time': 'float64',      # Observed service time (outcome)
       # Additional context features (optional)
       'feature_1': 'float64',         # Any number of numeric features
       'feature_2': 'float64',         # used for propensity/outcome modeling
       # ... more features as needed
   }

Example:

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Example service logs structure
   logs = pd.DataFrame({
       'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
       'operator': np.random.choice(['op_A', 'op_B', 'op_C'], 1000),
       'service_time': np.random.exponential(2.0, 1000),
       'request_size': np.random.normal(100, 20, 1000),
       'server_load': np.random.uniform(0, 1, 1000),
   })

Output Data Structures
----------------------

DRResult
~~~~~~~~

.. autoclass:: skdr_eval.core.DRResult
   :members:
   :noindex:

The `DRResult` class contains the results of doubly robust evaluation:

.. code-block:: python

   @dataclass
   class DRResult:
       """Results from doubly robust evaluation."""
       
       dr_value: float              # DR estimate
       sndr_value: float           # Stabilized DR estimate  
       ess: float                  # Effective sample size
       match_rate: float           # Propensity score match rate
       propensity_scores: np.ndarray  # Individual propensity scores
       outcome_preds: np.ndarray   # Outcome model predictions
       weights: np.ndarray         # Importance weights
       clip_threshold: float       # Applied clipping threshold

Design
~~~~~~

.. autoclass:: skdr_eval.core.Design
   :members:
   :noindex:

The `Design` class encapsulates the experimental design:

.. code-block:: python

   @dataclass  
   class Design:
       """Experimental design for offline evaluation."""
       
       logs_train: pd.DataFrame    # Training data for models
       logs_eval: pd.DataFrame     # Evaluation data  
       target_policy: dict         # Target policy mapping
       logging_policy: dict        # Logging policy mapping
       feature_cols: List[str]     # Feature column names
       outcome_col: str            # Outcome column name
       operator_col: str           # Operator/action column name
       timestamp_col: str          # Timestamp column name

Configuration Schema
--------------------

Model Configuration
~~~~~~~~~~~~~~~~~~~

When using `evaluate_sklearn_models`, the models dictionary should follow this structure:

.. code-block:: python

   models = {
       'model_name': sklearn_estimator,  # Any sklearn-compatible estimator
       # Examples:
       'RandomForest': RandomForestRegressor(n_estimators=100),
       'XGBoost': XGBRegressor(n_estimators=100),
       'LinearRegression': LinearRegression(),
   }

Evaluation Parameters
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   evaluation_params = {
       'n_splits': int,                    # Number of time-series splits (default: 3)
       'outcome_estimator': str,           # 'rf', 'hgb', or 'linear' (default: 'hgb')  
       'random_state': int,                # Random seed (default: None)
       'policy_train': str,                # 'pre_split' or 'post_split' (default: 'pre_split')
       'policy_train_frac': float,         # Fraction for policy training (default: 0.8)
       'clip_thresholds': List[float],     # Clipping thresholds to try (default: [2,5,10,20,50,∞])
       'bootstrap_samples': int,           # Bootstrap samples for CI (default: 1000)
       'block_size': int,                  # Block size for bootstrap (default: 50)
   }

Return Value Schema
~~~~~~~~~~~~~~~~~~~

The `evaluate_sklearn_models` function returns a tuple:

.. code-block:: python

   report, detailed_results = evaluate_sklearn_models(...)
   
   # report: pd.DataFrame with columns:
   # - model: str (model name)
   # - dr_value: float (DR estimate)  
   # - sndr_value: float (SNDR estimate)
   # - dr_ci_lower: float (DR confidence interval lower bound)
   # - dr_ci_upper: float (DR confidence interval upper bound)
   # - sndr_ci_lower: float (SNDR confidence interval lower bound)  
   # - sndr_ci_upper: float (SNDR confidence interval upper bound)
   # - ess: float (effective sample size)
   # - match_rate: float (propensity score match rate)
   # - clip_threshold: float (selected clipping threshold)
   
   # detailed_results: Dict[str, List[DRResult]]
   # Keys are model names, values are lists of DRResult objects (one per split)

Validation Rules
----------------

Input Validation
~~~~~~~~~~~~~~~~

The library performs the following validation checks:

1. **Required Columns**: Logs must contain timestamp, operator, and outcome columns
2. **Data Types**: Timestamps must be datetime, numeric columns must be numeric
3. **No Missing Values**: Critical columns cannot have NaN values
4. **Temporal Order**: Timestamps must be in ascending order
5. **Operator Coverage**: All operators must appear in both training and evaluation sets
6. **Minimum Sample Size**: Each operator must have minimum samples for reliable estimation

Error Handling
~~~~~~~~~~~~~~

Common validation errors and their meanings:

.. code-block:: python

   # ValueError: Missing required columns
   "Logs must contain columns: ['timestamp', 'operator', 'service_time']"
   
   # ValueError: Invalid data types  
   "Column 'timestamp' must be datetime type"
   
   # ValueError: Insufficient data
   "Operator 'op_A' has only 5 samples, minimum 10 required"
   
   # ValueError: Temporal issues
   "Timestamps must be in ascending order"
