"""Invariant tests for the recommendation / gating decision layer (#211).

The verdict is the product's headline output, so its decision boundaries are
locked in here:

* **Monotonicity** — degrading a single diagnostic (adding a warning, or
  removing the CI) never yields a *more* favourable verdict.
* **Verdict-set completeness** — only the four documented verdicts appear.
* **Card ↔ CLI consistency** — the verdict produced by the recommendation
  engine maps to exactly the documented CLI exit code, and ``do_not_deploy``
  takes precedence over ``insufficient_evidence``.

These tests encode the *current intended* behaviour. Per the issue, a real
monotonicity violation should be filed as a separate bug rather than silently
relaxing an assertion here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from skdr_eval.cli import (
    EXIT_DO_NOT_DEPLOY,
    EXIT_INSUFFICIENT_EVIDENCE,
    EXIT_OK,
    _verdict_exit_code,
)
from skdr_eval.recommendation import (
    Recommendation,
    RecommendationPolicy,
    _build_recommendation,
)
from skdr_eval.reporting import (
    WARN_HIGH_PARETO_K,
    WARN_LOW_ESS,
    WARN_LOW_MATCH_RATE,
    WARN_POOR_OVERLAP,
)

# The four — and only — documented verdicts, ordered most → least favourable.
# ``insufficient_evidence`` and ``do_not_deploy`` share the bottom tier (issue
# #197): both mean "not safe to auto-deploy". ``do_not_deploy`` is strictly the
# worst because a high-risk diagnostic actively fired.
_VERDICT_RANK = {
    "deploy": 3,
    "ab_test": 2,
    "insufficient_evidence": 1,
    "do_not_deploy": 0,
}


def _fake_artifact(
    *,
    warning_codes: tuple[str, ...] = (),
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    model: str = "m",
    estimator: str = "DR",
) -> SimpleNamespace:
    """Minimal duck-typed artifact accepted by ``_build_recommendation``."""
    report = pd.DataFrame(
        [
            {
                "model": model,
                "estimator": estimator,
                "diagnostic_warnings": ",".join(warning_codes),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        ]
    )
    return SimpleNamespace(detailed={model: {estimator: None}}, report=report)


def _verdict(
    *,
    warning_codes: tuple[str, ...] = (),
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    baseline: float = 0.0,
) -> str:
    art = _fake_artifact(
        warning_codes=warning_codes, ci_lower=ci_lower, ci_upper=ci_upper
    )
    rec = _build_recommendation(art, "m", "DR", RecommendationPolicy(baseline=baseline))
    return rec.verdict


# --------------------------------------------------------------------------- #
# Verdict-set completeness                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("warning_codes", "ci_lower", "ci_upper"),
    [
        ((), 1.0, 2.0),  # clean + CI above baseline
        ((WARN_LOW_ESS,), 1.0, 2.0),  # caution
        ((WARN_POOR_OVERLAP,), 1.0, 2.0),  # high-risk
        ((), -1.0, 1.0),  # CI overlaps baseline
        ((), None, None),  # no CI
    ],
)
def test_only_documented_verdicts(
    warning_codes: tuple[str, ...], ci_lower: float | None, ci_upper: float | None
) -> None:
    verdict = _verdict(
        warning_codes=warning_codes, ci_lower=ci_lower, ci_upper=ci_upper
    )
    assert verdict in _VERDICT_RANK


# --------------------------------------------------------------------------- #
# Anchored verdicts (happy / edge / error paths)                              #
# --------------------------------------------------------------------------- #


def test_clean_ci_above_baseline_deploys() -> None:
    assert _verdict(ci_lower=1.0, ci_upper=2.0) == "deploy"


def test_caution_flag_demotes_deploy_to_ab_test() -> None:
    assert (
        _verdict(warning_codes=(WARN_LOW_ESS,), ci_lower=1.0, ci_upper=2.0) == "ab_test"
    )


def test_high_risk_flag_blocks() -> None:
    assert (
        _verdict(warning_codes=(WARN_POOR_OVERLAP,), ci_lower=1.0, ci_upper=2.0)
        == "do_not_deploy"
    )


def test_ci_overlapping_baseline_is_ab_test() -> None:
    assert _verdict(ci_lower=-1.0, ci_upper=1.0) == "ab_test"


def test_no_ci_is_insufficient_evidence() -> None:
    assert _verdict(ci_lower=None, ci_upper=None) == "insufficient_evidence"


# --------------------------------------------------------------------------- #
# Monotonicity                                                                #
# --------------------------------------------------------------------------- #


def test_adding_a_caution_flag_never_improves_verdict() -> None:
    base = _verdict(ci_lower=1.0, ci_upper=2.0)
    worse = _verdict(warning_codes=(WARN_LOW_MATCH_RATE,), ci_lower=1.0, ci_upper=2.0)
    assert _VERDICT_RANK[worse] <= _VERDICT_RANK[base]


def test_adding_a_high_risk_flag_never_improves_verdict() -> None:
    for codes in [(), (WARN_LOW_ESS,)]:
        base = _verdict(warning_codes=codes, ci_lower=1.0, ci_upper=2.0)
        worse = _verdict(
            warning_codes=(*codes, WARN_HIGH_PARETO_K), ci_lower=1.0, ci_upper=2.0
        )
        assert _VERDICT_RANK[worse] <= _VERDICT_RANK[base]


def test_removing_the_ci_never_improves_verdict() -> None:
    with_ci = _verdict(ci_lower=1.0, ci_upper=2.0)
    without_ci = _verdict(ci_lower=None, ci_upper=None)
    assert _VERDICT_RANK[without_ci] <= _VERDICT_RANK[with_ci]


def test_high_risk_dominates_a_winning_ci() -> None:
    # Even with a CI that clears the baseline, a high-risk flag must block.
    assert (
        _VERDICT_RANK[
            _verdict(warning_codes=(WARN_POOR_OVERLAP,), ci_lower=5.0, ci_upper=9.0)
        ]
        == 0
    )


# --------------------------------------------------------------------------- #
# Card ↔ CLI exit-code consistency                                            #
# --------------------------------------------------------------------------- #


class _VerdictArtifact:
    """Duck-typed artifact whose ``recommendation`` returns fixed verdicts."""

    def __init__(self, verdicts: dict[tuple[str, str], str]) -> None:
        self._verdicts = verdicts
        self.detailed: dict[str, dict[str, object]] = {}
        for model, estimator in verdicts:
            self.detailed.setdefault(model, {})[estimator] = object()

    def recommendation(
        self, model_name: str, *, estimator: str = "SNDR"
    ) -> Recommendation:
        verdict = self._verdicts[(model_name, estimator)]
        return Recommendation(
            verdict=verdict,
            confidence="low",
            primary_blocker=None,
            reasons=[],
            recommended_estimator=estimator,
            model_name=model_name,
        )


@pytest.mark.parametrize(
    ("verdict", "expected_code"),
    [
        ("deploy", EXIT_OK),
        ("ab_test", EXIT_OK),
        ("insufficient_evidence", EXIT_INSUFFICIENT_EVIDENCE),
        ("do_not_deploy", EXIT_DO_NOT_DEPLOY),
    ],
)
def test_single_verdict_maps_to_documented_exit_code(
    verdict: str, expected_code: int
) -> None:
    art = _VerdictArtifact({("m", "DR"): verdict})
    assert _verdict_exit_code(art) == expected_code


def test_do_not_deploy_takes_precedence_over_insufficient_evidence() -> None:
    art = _VerdictArtifact(
        {("m", "DR"): "insufficient_evidence", ("m", "SNDR"): "do_not_deploy"}
    )
    assert _verdict_exit_code(art) == EXIT_DO_NOT_DEPLOY


def test_gate_scans_non_dr_sndr_estimators() -> None:
    # A do_not_deploy from MIPS alone must still trip the gate (#196).
    art = _VerdictArtifact({("m", "MIPS"): "do_not_deploy"})
    assert _verdict_exit_code(art) == EXIT_DO_NOT_DEPLOY


def test_recommendation_errors_are_surfaced_not_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RaisingArtifact:
        def __init__(self) -> None:
            self.detailed = {"m": {"DR": object()}}

        def recommendation(
            self, model_name: str, *, estimator: str = "SNDR"
        ) -> Recommendation:
            raise RuntimeError("boom")

    with caplog.at_level("WARNING", logger="skdr_eval.cli"):
        code = _verdict_exit_code(_RaisingArtifact())
    assert code == EXIT_OK  # no verdict could be computed → no gate trip
    assert any("boom" in rec.message for rec in caplog.records)
