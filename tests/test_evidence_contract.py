"""Tests for implementation maturity, diagnostics and evidence states."""

from __future__ import annotations

import pytest

from skdr_eval.evidence import (
    DiagnosticState,
    ESTIMATOR_VALIDATION_REGISTRY,
    EstimatorMaturity,
    EstimatorValidationRecord,
    EvidenceStatus,
    can_support_estimate,
    get_estimator_validation,
    required_diagnostics_support_evidence,
)
from skdr_eval.exceptions import DataValidationError


def test_initial_registry_keeps_dr_sndr_validation_pending() -> None:
    assert get_estimator_validation("DR").maturity is EstimatorMaturity.VALIDATION_PENDING
    assert (
        get_estimator_validation("SNDR").maturity
        is EstimatorMaturity.VALIDATION_PENDING
    )


def test_non_core_variants_start_experimental() -> None:
    for name in ("MRDR", "SWITCH-DR", "DRos", "MIPS"):
        assert get_estimator_validation(name).maturity is EstimatorMaturity.EXPERIMENTAL


def test_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        ESTIMATOR_VALIDATION_REGISTRY["DR"] = get_estimator_validation("DR")  # type: ignore[index]


def test_unknown_estimator_record_fails_closed() -> None:
    with pytest.raises(DataValidationError, match="Unknown estimator"):
        get_estimator_validation("made-up")
    assert not can_support_estimate(
        "made-up",
        {"overlap": "pass"},
        inside_validation_envelope=True,
    )


def test_evidence_status_contains_no_deployment_authorization() -> None:
    values = {state.value for state in EvidenceStatus}
    assert values == {
        "estimate_supported",
        "inconclusive",
        "insufficient_evidence",
        "unsupported",
        "invalid_evaluation",
    }
    forbidden = ("deploy", "online_test", "eligible")
    for value in values:
        assert not any(token in value for token in forbidden)


def test_unknown_required_diagnostic_blocks_positive_evidence() -> None:
    assert not required_diagnostics_support_evidence(
        {"ess": DiagnosticState.PASS, "pareto_k": DiagnosticState.UNKNOWN}
    )


def test_failed_required_diagnostic_blocks_positive_evidence() -> None:
    assert not required_diagnostics_support_evidence(
        {"ess": "pass", "overlap": "fail"}
    )


def test_pass_and_not_applicable_diagnostics_can_satisfy_hard_gate() -> None:
    assert required_diagnostics_support_evidence(
        {"ess": DiagnosticState.PASS, "pareto_k": DiagnosticState.NOT_APPLICABLE}
    )


def test_missing_diagnostic_contract_does_not_pass() -> None:
    assert not required_diagnostics_support_evidence({})


def test_malformed_diagnostic_state_does_not_pass() -> None:
    assert not required_diagnostics_support_evidence({"ess": "probably-fine"})


def test_all_initial_registry_estimators_fail_positive_evidence_gate() -> None:
    # No implementation is validated at the start of the evidence-first reset.
    for name in ESTIMATOR_VALIDATION_REGISTRY:
        assert not can_support_estimate(
            name,
            {"overlap": "pass", "ess": "pass"},
            inside_validation_envelope=True,
        )


def test_caller_created_validated_record_cannot_bypass_registry() -> None:
    fake = EstimatorValidationRecord(
        name="DR",
        maturity=EstimatorMaturity.REFERENCE_VALIDATED,
        estimand="fake runtime promotion",
        validation_report_id="not-a-project-validation-report",
    )
    assert fake.validated_for_positive_evidence is True

    # can_support_estimate accepts a canonical registry name, not an arbitrary
    # record. Passing the record anyway at runtime must fail closed rather than
    # treating caller-supplied maturity as authoritative.
    assert not can_support_estimate(  # type: ignore[arg-type]
        fake,
        {"overlap": "pass", "ess": "pass"},
        inside_validation_envelope=True,
    )


def test_validation_envelope_and_diagnostics_remain_separate_gates() -> None:
    # This remains false today because DR is validation_pending; the assertions
    # still protect the other gates so a future registry promotion cannot make
    # UNKNOWN diagnostics or out-of-envelope runs positive by accident.
    assert not can_support_estimate(
        "DR",
        {"overlap": "unknown", "ess": "pass"},
        inside_validation_envelope=True,
    )
    assert not can_support_estimate(
        "DR",
        {"overlap": "pass", "ess": "pass"},
        inside_validation_envelope=False,
    )
