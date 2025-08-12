skdr-eval Documentation
=======================

**Offline policy evaluation for service-time minimization using Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators with time-aware splits and calibration.**

.. image:: https://badge.fury.io/py/skdr-eval.svg
   :target: https://badge.fury.io/py/skdr-eval
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/skdr-eval.svg
   :target: https://pypi.org/project/skdr-eval/
   :alt: Python versions

.. image:: https://github.com/dandrsantos/skdr-eval/workflows/CI/badge.svg
   :target: https://github.com/dandrsantos/skdr-eval/actions
   :alt: CI

.. image:: https://codecov.io/gh/dandrsantos/skdr-eval/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/dandrsantos/skdr-eval
   :alt: Coverage

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Features
--------

- 🎯 **Doubly Robust Estimation**: Implements both DR and Stabilized DR (SNDR) estimators
- ⏰ **Time-Aware Evaluation**: Uses time-series splits and calibrated propensity scores
- 🔧 **Sklearn Integration**: Easy integration with scikit-learn models
- 📊 **Comprehensive Diagnostics**: ESS, match rates, propensity score analysis
- 🚀 **Production Ready**: Type-hinted, tested, and documented
- 📈 **Bootstrap Confidence Intervals**: Moving-block bootstrap for time-series data

Installation
------------

.. code-block:: bash

   pip install skdr-eval

For development:

.. code-block:: bash

   git clone https://github.com/dandrsantos/skdr-eval.git
   cd skdr-eval
   pip install -e .[dev]

Quick Start
-----------

.. code-block:: python

   import skdr_eval
   from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

   # 1. Generate synthetic service logs
   logs, ops_all, true_q = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)

   # 2. Define candidate models
   models = {
       "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
       "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
   }

   # 3. Evaluate models using DR and SNDR
   report, detailed_results = skdr_eval.evaluate_sklearn_models(
       logs=logs,
       models=models,
       fit_models=True,
       n_splits=3,
       outcome_estimator="hgb",
       random_state=42,
   )

   print(report)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples
   schemas
   development

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   skdr_eval

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
