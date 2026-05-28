"""Simulation studies for the DR/SNDR/IPS estimator family (#129, #130).

These studies are statistical-evidence assets, not unit tests of code
correctness. They cover:

* ``test_dr_misspecification`` — double robustness: DR is consistent if
  *either* the propensity or the outcome model is correct. Both wrong → bias.
* ``test_overlap_failure`` — bias grows monotonically as the logging
  policy becomes more deterministic and overlap collapses.
* ``test_bootstrap_validity`` — moving-block bootstrap coverage under
  iid, AR(1), and small-sample regimes.

All studies are gated by ``SIM_REPS`` (default 30 for CI; bump to ≥200 for a
thorough local check) — same convention as
``tests/test_estimator_recovery_simulation.py``.
"""
