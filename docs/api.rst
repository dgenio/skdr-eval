API Reference
=============

This page contains the complete API reference for skdr-eval.

Core Functions
--------------

.. automodule:: skdr_eval.core
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

.. autoclass:: skdr_eval.core.DRResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: skdr_eval.core.Design
   :members:
   :undoc-members:
   :show-inheritance:

Synthetic Data Generation
-------------------------

.. automodule:: skdr_eval.synth
   :members:
   :undoc-members:
   :show-inheritance:

Main Interface
--------------

.. autofunction:: skdr_eval.evaluate_sklearn_models

.. autofunction:: skdr_eval.make_synth_logs

Evaluation Functions
--------------------

.. autofunction:: skdr_eval.dr_value_with_clip

.. autofunction:: skdr_eval.build_design

.. autofunction:: skdr_eval.fit_outcome_crossfit

.. autofunction:: skdr_eval.fit_propensity_timecal

.. autofunction:: skdr_eval.induce_policy_from_sklearn

Statistical Functions
---------------------

.. autofunction:: skdr_eval.block_bootstrap_ci
