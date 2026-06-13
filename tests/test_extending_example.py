"""Verify the "write your own estimator" tutorial stays runnable (#188).

``docs/extending/add-an-estimator.md`` promises that following the worked
example produces a working estimator. This test executes that example
(``examples/extending/custom_estimator.py``) so the guide cannot rot silently:
if the strategy seam changes shape, this fails alongside the library tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "extending"
    / "custom_estimator.py"
)


@pytest.fixture(scope="module")
def custom_estimator_module():
    """Import the tutorial example script as a module."""
    spec = importlib.util.spec_from_file_location("custom_estimator", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_estimator"] = module
    spec.loader.exec_module(module)
    return module


def test_custom_estimator_runs_and_is_finite(custom_estimator_module):
    """The SoftClipDR tutorial estimator returns a finite value."""
    out = custom_estimator_module.run(n=2000, n_ops=4, seed=0)
    assert np.isfinite(out["SoftClipDR"])
    assert np.isfinite(out["DR"])
    assert out["SoftClipDR_ESS"] > 0


def test_custom_estimator_close_to_builtin_dr(custom_estimator_module):
    """SoftClipDR tracks built-in DR on well-supported synthetic logs.

    The smooth clip only reshapes heavy-tail weights, so on synthetic logs
    with healthy overlap the two estimates must land in the same ballpark.
    """
    out = custom_estimator_module.run(n=4000, n_ops=4, seed=0)
    assert abs(out["SoftClipDR"] - out["DR"]) < 0.5 * abs(out["DR"])
