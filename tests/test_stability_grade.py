"""Tests for the three-band ``stability_grade`` on ``summarize_sensitivity`` (#133)."""

from __future__ import annotations

import pandas as pd

from skdr_eval.core import DRResult
from skdr_eval.reporting import (
    STABILITY_GRADES,
    _stability_grade,
    summarize_sensitivity,
)


def _result(*, v_grid: dict[float, float], chosen_clip: float = 5.0) -> DRResult:
    """Build a synthetic DRResult with a known V_DR / V_SNDR grid."""
    grid = pd.DataFrame(
        {
            "clip": list(v_grid.keys()),
            "V_DR": list(v_grid.values()),
            "V_SNDR": list(v_grid.values()),
            "MSE_DR": [1.0] * len(v_grid),
            "MSE_SNDR": [1.0] * len(v_grid),
            "ESS": [100.0] * len(v_grid),
            "tail_mass": [0.0] * len(v_grid),
            "match_rate": [1.0] * len(v_grid),
            "min_pscore": [0.1] * len(v_grid),
        }
    )
    chosen_v = v_grid[chosen_clip]
    return DRResult(
        V_hat=chosen_v,
        SE_if=0.01 * abs(chosen_v),
        clip=chosen_clip,
        ESS=100.0,
        match_rate=1.0,
        min_pscore=0.1,
        tail_mass=0.0,
        MSE_est=1.0,
        pscore_q10=0.1,
        pscore_q05=0.05,
        pscore_q01=0.01,
        grid=grid,
    )


def test_stability_grade_helper_boundaries() -> None:
    # Below the "sensitive" floor + DR/SNDR agree → stable.
    assert _stability_grade(0.05, True) == "stable"
    # In the "sensitive" band → sensitive even if DR/SNDR disagree.
    assert _stability_grade(0.15, True) == "sensitive"
    # Below the "sensitive" floor but DR/SNDR disagree → still sensitive.
    assert _stability_grade(0.05, False) == "sensitive"
    # Above the "unstable" floor → unstable.
    assert _stability_grade(0.50, True) == "unstable"
    # NaN range fraction (degenerate input) → unstable.
    assert _stability_grade(float("nan"), True) == "unstable"


def test_stability_grade_values_are_known_constants() -> None:
    assert STABILITY_GRADES == ("stable", "sensitive", "unstable")


def test_summarize_sensitivity_stable_grade() -> None:
    """A tight grid where V_range is < 10% of |chosen_V| and DR == SNDR."""
    detailed = {
        "HGB": {
            "DR": _result(v_grid={2.0: 10.0, 5.0: 10.05, 10.0: 10.02}, chosen_clip=5.0),
            "SNDR": _result(
                v_grid={2.0: 10.0, 5.0: 10.05, 10.0: 10.02}, chosen_clip=5.0
            ),
        }
    }
    df = summarize_sensitivity(detailed)
    assert (df["stability_grade"] == "stable").all()
    assert (df["v_range_frac"] < 0.10).all()


def test_summarize_sensitivity_unstable_grade() -> None:
    """A wild grid where V_range > 25% of |chosen_V|."""
    detailed = {
        "HGB": {
            "DR": _result(v_grid={2.0: 10.0, 5.0: 15.0, 10.0: 25.0}, chosen_clip=5.0),
            "SNDR": _result(v_grid={2.0: 10.0, 5.0: 15.0, 10.0: 25.0}, chosen_clip=5.0),
        }
    }
    df = summarize_sensitivity(detailed)
    assert (df["stability_grade"] == "unstable").all()
    assert (df["v_range_frac"] > 0.25).all()


def test_summarize_sensitivity_sensitive_grade() -> None:
    """A middle case where 10% <= V_range/|chosen_V| < 25%."""
    detailed = {
        "HGB": {
            "DR": _result(v_grid={2.0: 10.0, 5.0: 11.5, 10.0: 11.8}, chosen_clip=5.0),
            "SNDR": _result(v_grid={2.0: 10.0, 5.0: 11.5, 10.0: 11.8}, chosen_clip=5.0),
        }
    }
    df = summarize_sensitivity(detailed)
    assert (df["stability_grade"] == "sensitive").all()
