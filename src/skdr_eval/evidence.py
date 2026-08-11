"""Machine-readable evidence and implementation-maturity contracts.

This module separates three concepts that must not be conflated in validated
statistical software (#245 / #259 / #282):

1. **Estimator implementation maturity** — how strongly a concrete estimator
   implementation has itself been validated.
2. **Diagnostic state** — whether a required diagnostic passed, failed, is
   unknown, or is genuinely not applicable for a concrete run.
3. **Evidence status** — whether the *particular evaluation* has enough valid
   evidence to support its estimate under a declared validation envelope.

A validated estimator can therefore still produce unsupported evidence.  These
states intentionally contain no deployment or online-experiment authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .exceptions import DataValidationError


class EstimatorMaturity(str, Enum):
    """Validation maturity of an estimator implementation."""

    EXPERIMENTAL = "experimental"
    VALIDATION_PENDING = "validation_pending"
    REFERENCE_VALIDATED = "reference_validated"
    INDEPENDENTLY_VALIDATED = "independently_validated"
    DEPRECATED = "deprecated"


class DiagnosticState(str, Enum):
    """Normalized state of one required diagnostic for a concrete run."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class EvidenceStatus(str, Enum):
    """Statistical/evidence state of a concrete evaluation.

    These are deliberately evidence-only semantics.  In particular, none of
    these values means "deploy", "do not deploy", or "approved for an online
    experiment".
    """

    ESTIMATE_SUPPORTED = "estimate_supported"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"
    INVALID_EVALUATION = "invalid_evaluation"


@dataclass(frozen=True)
class EstimatorValidationRecord:
    """One estimator entry in the implementation-maturity registry."""

    name: str
    maturity: EstimatorMaturity
    estimand: str
    validation_report_id: str | None = None
    reference_backend: str | None = None
    known_limitations: tuple[str, ...] = ()

    @property
    def validated_for_positive_evidence(self) -> bool:
        """Whether maturity can participate in a positive evidence claim.

        This is necessary but never sufficient: a concrete evaluation still
        needs supported data/diagnostics/inference.
        """

        return self.maturity in {
            EstimatorMaturity.REFERENCE_VALIDATED,
            EstimatorMaturity.INDEPENDENTLY_VALIDATED,
        }


_INITIAL_REGISTRY: dict[str, EstimatorValidationRecord] = {
    "DR": EstimatorValidationRecord(
        name="DR",
        maturity=EstimatorMaturity.VALIDATION_PENDING,
        estimand="finite-discrete one-step target-policy value",
        known_limitations=(
            "validated-v1 promotion requires explicit target policy, logged behavior propensity, genuine OOF outcome predictions, corrected action-specific q(x,a), end-to-end validation, and estimator-specific reference evidence",
        ),
    ),
    "SNDR": EstimatorValidationRecord(
        name="SNDR",
        maturity=EstimatorMaturity.VALIDATION_PENDING,
        estimand="finite-discrete one-step self-normalized target-policy value",
        known_limitations=(
            "must not inherit DR reference validation; requires its own semantically equivalent independent reference and inferential validation",
        ),
    ),
    "MRDR": EstimatorValidationRecord(
        name="MRDR",
        maturity=EstimatorMaturity.EXPERIMENTAL,
        estimand="experimental one-step target-policy value",
    ),
    "SWITCH-DR": EstimatorValidationRecord(
        name="SWITCH-DR",
        maturity=EstimatorMaturity.EXPERIMENTAL,
        estimand="experimental one-step target-policy value",
    ),
    "DRos": EstimatorValidationRecord(
        name="DRos",
        maturity=EstimatorMaturity.EXPERIMENTAL,
        estimand="experimental one-step target-policy value",
    ),
    "MIPS": EstimatorValidationRecord(
        name="MIPS",
        maturity=EstimatorMaturity.EXPERIMENTAL,
        estimand="experimental marginalized one-step target-policy value",
    ),
}

# Immutable public initial registry.  Promotion/demotion should happen through a
# deliberate versioned code change plus validation evidence, never mutable
# process state at runtime.
ESTIMATOR_VALIDATION_REGISTRY: Mapping[str, EstimatorValidationRecord] = MappingProxyType(
    _INITIAL_REGISTRY
)


def get_estimator_validation(name: str) -> EstimatorValidationRecord:
    """Return the implementation-maturity record for a canonical estimator."""

    canonical = str(name).strip()
    try:
        return ESTIMATOR_VALIDATION_REGISTRY[canonical]
    except KeyError as exc:
        raise DataValidationError(
            f"Unknown estimator validation record: {canonical!r}"
        ) from exc


def required_diagnostics_support_evidence(
    states: Mapping[str, DiagnosticState | str],
) -> bool:
    """Return whether all required diagnostics are affirmative or N/A.

    ``UNKNOWN`` is deliberately fail-closed for positive evidence.  This helper
    does not decide the final :class:`EvidenceStatus`; it only captures the hard
    invariant that a required unknown/failed diagnostic cannot satisfy a
    positive evidence contract.
    """

    if not states:
        return False

    for raw_state in states.values():
        try:
            state = (
                raw_state
                if isinstance(raw_state, DiagnosticState)
                else DiagnosticState(str(raw_state))
            )
        except ValueError:
            return False
        if state in {DiagnosticState.FAIL, DiagnosticState.UNKNOWN}:
            return False
    return True


def can_support_estimate(
    estimator: EstimatorValidationRecord,
    required_diagnostics: Mapping[str, DiagnosticState | str],
    *,
    inside_validation_envelope: bool,
) -> bool:
    """Hard preconditions for the future ``estimate_supported`` state.

    This intentionally does not inspect effect direction, confidence intervals,
    business risk, or deployment policy.  It merely encodes three necessary
    conditions that future evidence-state logic cannot bypass.
    """

    return (
        estimator.validated_for_positive_evidence
        and inside_validation_envelope
        and required_diagnostics_support_evidence(required_diagnostics)
    )


__all__ = [
    "DiagnosticState",
    "ESTIMATOR_VALIDATION_REGISTRY",
    "EstimatorMaturity",
    "EstimatorValidationRecord",
    "EvidenceStatus",
    "can_support_estimate",
    "get_estimator_validation",
    "required_diagnostics_support_evidence",
]
