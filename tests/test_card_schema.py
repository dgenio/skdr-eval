"""Tests for the :class:`EvaluationCard` schema (#88)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval import reporting as _rep
from skdr_eval.exceptions import DataValidationError
from skdr_eval.reporting import (
    CARD_SCHEMA_VERSION,
    DiagnosticsBlock,
    EvaluationCard,
    HeadlineBlock,
    ProvenanceBlock,
    SensitivityBlock,
    TrustBlock,
    _coerce_optional_float,
    _read_path_or_string,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build_artifact(
    seed: int = 7, n: int = 400, ci: bool = False
) -> skdr_eval.EvaluationArtifact:
    logs, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=seed)
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=seed)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=seed,
        ci_bootstrap=ci,
        policy_train="pre_split",
    )


class TestEvaluationCardSchema:
    def test_card_schema_version_constant(self):
        assert CARD_SCHEMA_VERSION == "1.0.0"

    def test_card_schema_method_returns_card(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card, EvaluationCard)
        assert card.card_schema_version == CARD_SCHEMA_VERSION
        assert card.model_name == "HGB"

    def test_headline_block_populated(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="SNDR")
        assert isinstance(card.headline, HeadlineBlock)
        assert card.headline.estimator == "SNDR"
        assert card.headline.V_hat is None or isinstance(card.headline.V_hat, float)

    def test_diagnostics_block_populated(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card.diagnostics, DiagnosticsBlock)
        # ESS may be None for tiny datasets, but if present must be float.
        if card.diagnostics.ESS is not None:
            assert isinstance(card.diagnostics.ESS, float)
        if card.diagnostics.gate is not None:
            assert "overall" in card.diagnostics.gate

    def test_sensitivity_block_populated(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card.sensitivity, SensitivityBlock)

    def test_provenance_block_populated(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card.provenance, ProvenanceBlock)
        assert card.provenance.evaluator == "evaluate_sklearn_models"
        assert card.provenance.n_samples is not None
        assert card.provenance.n_splits == 3

    def test_trust_block_carries_recommendation(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card.trust, TrustBlock)
        # Recommendation should be a dict with the verdict key when included.
        if card.trust.recommendation is not None:
            assert "verdict" in card.trust.recommendation
            assert card.trust.recommendation["verdict"] in {
                "deploy",
                "ab_test",
                "do_not_deploy",
                "insufficient_evidence",
            }

    def test_unknown_model_raises(self):
        art = _build_artifact()
        with pytest.raises(DataValidationError, match="not in artifact"):
            art.card_schema("does_not_exist")

    def test_unknown_estimator_raises(self):
        art = _build_artifact()
        with pytest.raises(DataValidationError, match="estimator"):
            art.card_schema("HGB", estimator="MRDR")

    def test_baseline_propagates(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR", baseline=10.0)
        assert card.headline.baseline == 10.0
        if card.headline.V_hat is not None:
            assert card.headline.delta_vs_baseline == pytest.approx(
                card.headline.V_hat - 10.0
            )

    def test_include_gate_false_drops_gate(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR", include_gate=False)
        assert card.diagnostics.gate is None

    def test_include_recommendation_false_drops_recommendation(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR", include_recommendation=False)
        assert card.trust.recommendation is None


class TestCardRoundTrip:
    def test_yaml_round_trip_string(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        text = card.to_yaml()
        assert isinstance(text, str)
        loaded = EvaluationCard.from_yaml(text)
        assert loaded == card

    def test_json_round_trip_string(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        text = card.to_json()
        assert isinstance(text, str)
        loaded = EvaluationCard.from_json(text)
        assert loaded == card

    def test_yaml_round_trip_file(self, tmp_path: Path):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        path = tmp_path / "card.yaml"
        card.to_yaml(path)
        assert path.is_file()
        loaded = EvaluationCard.from_yaml(path)
        assert loaded == card

    def test_json_round_trip_file(self, tmp_path: Path):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        path = tmp_path / "card.json"
        card.to_json(path)
        assert path.is_file()
        loaded = EvaluationCard.from_json(path)
        assert loaded == card

    def test_from_yaml_rejects_non_mapping(self):
        with pytest.raises(DataValidationError, match="YAML mapping"):
            EvaluationCard.from_yaml("- 1\n- 2\n")


class TestCardJSONSchema:
    def test_json_schema_returns_dict(self):
        schema = EvaluationCard.json_schema()
        assert isinstance(schema, dict)
        assert schema["title"] == "EvaluationCard"
        assert "properties" in schema

    def test_json_schema_includes_blocks(self):
        schema = EvaluationCard.json_schema()
        properties = schema["properties"]
        for required_key in (
            "headline",
            "trust",
            "diagnostics",
            "sensitivity",
            "provenance",
        ):
            assert required_key in properties, (
                f"Missing block {required_key!r} in EvaluationCard JSON Schema."
            )

    def test_json_schema_is_json_serializable(self):
        schema = EvaluationCard.json_schema()
        # Round-trip ensures no non-JSON values leaked into the schema dict.
        text = json.dumps(schema)
        assert isinstance(text, str)


class TestCardForwardCompatibility:
    def test_extra_field_preserved_under_extra_allow(self):
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        d = card.to_dict()
        d["future_field"] = {"some": "value"}
        loaded = EvaluationCard.from_dict(d)
        # ``ConfigDict(extra="allow")`` preserves unknown fields.
        assert loaded.model_dump(mode="json").get("future_field") == {"some": "value"}

    def test_card_schema_version_stable(self):
        # Stability guard: any bump must update tests + CHANGELOG.
        assert CARD_SCHEMA_VERSION == "1.0.0"


class TestReadPathOrString:
    """Exercises the _read_path_or_string heuristic in reporting.py."""

    def test_from_yaml_with_inline_string(self):
        """from_yaml with a multi-line YAML string (contains newlines)."""
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        yaml_text = card.to_yaml()
        assert "\n" in yaml_text
        loaded = EvaluationCard.from_yaml(yaml_text)
        assert loaded == card

    def test_from_yaml_with_short_nonexistent_path(self):
        """from_yaml with a short string that's not a valid file falls through."""
        # This hits the try/Path(s)/is_file() → False → return s path.
        with pytest.raises((DataValidationError, Exception)):
            # "no_such_file.yaml" is short, no newlines, but doesn't exist
            # as a file. _read_path_or_string returns it as-is, then YAML
            # parsing of that string fails or produces a non-mapping.
            EvaluationCard.from_yaml("no_such_file.yaml")

    def test_from_json_with_inline_string(self):
        """from_json with an inline JSON string."""
        art = _build_artifact()
        card = art.card_schema("HGB", estimator="DR")
        json_text = card.to_json()
        loaded = EvaluationCard.from_json(json_text)
        assert loaded == card


class TestCoerceOptionalFloat:
    """Exercises edge cases in _coerce_optional_float."""

    def test_none_value(self):
        assert _coerce_optional_float(None) is None

    def test_valid_float(self):
        assert _coerce_optional_float(3.14) == pytest.approx(3.14)

    def test_non_convertible_string(self):
        assert _coerce_optional_float("not_a_number") is None

    def test_inf_returns_none(self):
        assert _coerce_optional_float(float("inf")) is None
        assert _coerce_optional_float(float("-inf")) is None

    def test_nan_returns_none(self):
        assert _coerce_optional_float(float("nan")) is None


class TestReadPathOrStringOSError:
    """Cover the OSError branch in _read_path_or_string (line 2073-2075)."""

    def test_oserror_falls_through_to_return_string(self):
        # A filename > 255 chars (ext4 NAME_MAX) triggers OSError from
        # is_file() on Linux. On Windows it just returns False — either way
        # the function returns the original string.
        bad = "a" * 300 + ".yaml"
        result = _read_path_or_string(bad)
        assert result == bad


class TestBuildCardFallbackPaths:
    """Cover fallback branches in _build_card_from_row."""

    def test_recommendation_fallback_on_error(self, monkeypatch):
        """When recommendation() raises, card_schema still returns a card with rec=None."""
        art = _build_artifact()

        def _raise(*args, **kwargs):
            raise ValueError("simulated failure")

        monkeypatch.setattr(art, "recommendation", _raise)
        card = art.card_schema("HGB", estimator="DR", include_recommendation=True)
        assert card.trust.recommendation is None

    def test_gate_fallback_on_error(self, monkeypatch):
        """When gate_diagnostics raises, card_schema returns card with gate=None."""
        art = _build_artifact()

        def _raise(*args, **kwargs):
            raise KeyError("simulated gate failure")

        monkeypatch.setattr(_rep, "gate_diagnostics", _raise)
        card = art.card_schema("HGB", estimator="DR", include_gate=True)
        assert card.diagnostics.gate is None

    def test_empty_sensitivity_returns_empty_block(self):
        """When sensitivity has no matching rows, SensitivityBlock defaults."""
        art = _build_artifact()
        # Wipe sensitivity data for the model
        art.sensitivity = pd.DataFrame(columns=art.sensitivity.columns)
        card = art.card_schema("HGB", estimator="DR")
        assert isinstance(card.sensitivity, SensitivityBlock)
        assert card.sensitivity.V_min is None
        assert card.sensitivity.V_max is None

    def test_report_row_missing_raises(self):
        """When estimator is in detailed but not in report, raises DataValidationError."""
        art = _build_artifact()
        # Remove the DR row from the report DataFrame
        art.report = art.report[art.report["estimator"] != "DR"].reset_index(drop=True)
        # Estimator still in detailed but not in report → line 2118
        with pytest.raises(DataValidationError, match="No report row"):
            art.card_schema("HGB", estimator="DR")
