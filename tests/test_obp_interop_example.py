"""Verify the OBP-interop example stays runnable (#179).

``docs/recipes/obp-interop.md`` and ``examples/obp_interop.py`` promise that a
practitioner coming from Open Bandit Pipeline can map an ``obp``-style
``bandit_feedback`` dict onto skdr-eval logs and evaluate a candidate policy,
with **no runtime dependency on OBP**. This test executes that example so the
recipe cannot rot silently: if the public ingestion/evaluation surface
(``skdr_eval.adapters.from_records``, ``evaluate_sklearn_models``) changes
shape, this fails alongside the library tests.

Mirrors ``tests/test_extending_example.py`` (#188), which guards the sibling
``custom_estimator.py`` example the same way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "obp_interop.py"


def _load_example():
    """Import the OBP-interop example script as a module."""
    assert _EXAMPLE.exists(), (
        f"OBP-interop example not found at {_EXAMPLE}. docs/recipes/obp-interop.md "
        "points here; if the file moved, update this test and the recipe together."
    )
    spec = importlib.util.spec_from_file_location("obp_interop", _EXAMPLE)
    assert spec is not None and spec.loader is not None, (
        f"Could not build an import spec for {_EXAMPLE}; it must be an "
        "importable Python module."
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["obp_interop"] = module
    spec.loader.exec_module(module)
    return module


def test_obp_interop_imports_without_obp() -> None:
    """The example must import with no runtime dependency on ``obp`` (#179)."""
    assert "obp" not in sys.modules
    module = _load_example()
    assert "obp" not in sys.modules, (
        "examples/obp_interop.py imported `obp` at runtime; the recipe promises "
        "no runtime OBP dependency."
    )
    assert hasattr(module, "OBP_FIELD_MAP") and module.OBP_FIELD_MAP


def test_field_mapping_produces_generic_records() -> None:
    """The one field-mapping step yields the generic record schema."""
    module = _load_example()
    feedback = module.make_obp_bandit_feedback(n=200, n_actions=4, seed=0)
    records = module.bandit_feedback_to_records(feedback)
    assert len(records) == 200
    assert set(records[0]) == {"context", "action", "reward", "propensity"}


def test_obp_interop_evaluates_to_finite_headline() -> None:
    """Records map onto logs and evaluate to a finite DR headline.

    Exercises the full public ingestion + evaluation path the recipe relies on
    (``adapters.from_records`` -> ``evaluate_sklearn_models``) at a small scale.
    """
    module = _load_example()
    feedback = module.make_obp_bandit_feedback(n=900, n_actions=4, seed=0)
    artifact = module.evaluate_records(module.bandit_feedback_to_records(feedback))

    assert "V_hat" in artifact.report.columns
    assert np.isfinite(artifact.report["V_hat"]).all()
    assert "support_health" in artifact.warnings.columns
