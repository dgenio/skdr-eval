"""Tests for the verdict-explanation surface (#201).

Covers :meth:`EvaluationArtifact.explain`, the schema-based
:func:`explain_artifact_schema`, and the :class:`Explanation` rendering — all of
which are a pure presentation layer over the existing recommendation/gate logic
(no statistical recomputation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval import Explanation, explain_artifact_schema, load_artifact_json

if TYPE_CHECKING:
    from pathlib import Path


def _artifact(*, ci_bootstrap: bool = False):
    logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
    models = {"hgb": HistGradientBoostingRegressor(max_iter=20, random_state=0)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=0,
        ci_bootstrap=ci_bootstrap,
        policy_train="pre_split",
    )


class TestEvaluationArtifactExplain:
    def test_explain_matches_recommendation_verdict(self):
        artifact = _artifact()
        rec = artifact.recommendation("hgb", estimator="SNDR")
        expl = artifact.explain("hgb", estimator="SNDR")
        assert isinstance(expl, Explanation)
        # The narrative must not drift from the structured recommendation.
        assert expl.verdict == rec.verdict
        assert expl.confidence == rec.confidence
        assert expl.primary_blocker == rec.primary_blocker
        assert [r.code for r in expl.reasons] == [r.code for r in rec.reasons]

    def test_explain_renders_reasons_with_thresholds(self):
        expl = _artifact().explain("hgb", estimator="DR")
        text = expl.to_text()
        assert "hgb / DR" in text
        assert expl.verdict.upper() in text
        # The embedded gate names a measured value against its threshold.
        if expl.gate is not None:
            assert "overlap" in expl.gate.to_text()

    def test_explain_unknown_model_raises(self):
        with pytest.raises(skdr_eval.DataValidationError):
            _artifact().explain("does-not-exist", estimator="SNDR")

    def test_explain_to_dict_is_jsonable(self):
        d = _artifact().explain("hgb", estimator="SNDR").to_dict()
        assert set(d) >= {"verdict", "confidence", "reasons", "gate", "V_hat"}
        assert isinstance(d["reasons"], list)


class TestExplainArtifactSchema:
    def test_schema_explanation_matches_live(self, tmp_path: Path):
        artifact = _artifact()
        live = artifact.explain("hgb", estimator="SNDR")
        path = tmp_path / "artifact.json"
        artifact.to_json(path)
        schema = load_artifact_json(path)
        from_schema = explain_artifact_schema(schema, "hgb", estimator="SNDR")
        # Reconstructed-from-disk explanation must equal the live one.
        assert from_schema.verdict == live.verdict
        assert from_schema.primary_blocker == live.primary_blocker
        assert [r.code for r in from_schema.reasons] == [r.code for r in live.reasons]
        assert from_schema.to_dict()["gate"] == live.to_dict()["gate"]

    def test_schema_explanation_unknown_estimator_raises(self, tmp_path: Path):
        artifact = _artifact()
        path = tmp_path / "artifact.json"
        artifact.to_json(path)
        schema = load_artifact_json(path)
        with pytest.raises(skdr_eval.DataValidationError):
            explain_artifact_schema(schema, "hgb", estimator="NOPE")

    def test_no_ci_yields_insufficient_evidence_narrative(self, tmp_path: Path):
        artifact = _artifact(ci_bootstrap=False)
        expl = artifact.explain("hgb", estimator="SNDR")
        # Without a high-risk blocker, no CI ⇒ insufficient_evidence; with one,
        # do_not_deploy. Either way the CI line must say it is unavailable.
        assert expl.verdict in {"insufficient_evidence", "do_not_deploy"}
        assert "not available" in expl.to_text()
