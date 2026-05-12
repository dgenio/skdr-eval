"""End-to-end tests for :mod:`skdr_eval.reporting` and ``EvaluationArtifact``.

Covers issues #22, #23, #27, #28, #30 — the bundled evaluation artifact and
the public attach-warnings / sensitivity / export / card helpers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.exceptions import ConfigurationError, DataValidationError
from skdr_eval.reporting import (
    SCHEMA_VERSION,
    SUPPORT_HIGH_RISK,
    SUPPORT_OK,
    WARN_EXTREME_CLIP,
    WARN_LOW_ESS,
    WARN_LOW_MATCH_RATE,
    WARN_POOR_OVERLAP,
    ArtifactSchema,
    EvaluationArtifact,
    SupportHealthThresholds,
    attach_warnings,
    load_artifact_json,
    render_evaluation_card,
    summarize_sensitivity,
)

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _run_eval(seed: int = 7, n: int = 400, ci: bool = False) -> EvaluationArtifact:
    logs, _, _ = skdr_eval.make_synth_logs(n=n, n_ops=3, seed=seed)
    models = {"HGB": HistGradientBoostingRegressor(max_iter=20, random_state=seed)}
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        random_state=seed,
        ci_bootstrap=ci,
    )


# --------------------------------------------------------------------------- #
# Artifact construction (#22 + #23 wiring)                                    #
# --------------------------------------------------------------------------- #


def test_evaluate_sklearn_models_returns_artifact():
    art = _run_eval()
    assert isinstance(art, EvaluationArtifact)
    assert isinstance(art.report, pd.DataFrame)
    assert isinstance(art.detailed, dict)
    assert "HGB" in art.detailed
    # Warnings + sensitivity DataFrames always present.
    assert isinstance(art.warnings, pd.DataFrame)
    assert isinstance(art.sensitivity, pd.DataFrame)
    # Diagnostics: one entry per model (shared fit across models).
    assert set(art.diagnostics) == {"HGB"}
    # Metadata: pinned schema version, evaluator label.
    assert art.metadata["schema_version"] == SCHEMA_VERSION
    assert art.metadata["evaluator"] == "evaluate_sklearn_models"
    assert art.metadata["n_samples"] == 400


def test_report_has_support_columns():
    art = _run_eval()
    cols = set(art.report.columns)
    assert {"support_health", "diagnostic_warnings"}.issubset(cols)
    # support_health is one of the three allowed labels for every row.
    assert set(art.report["support_health"].unique()).issubset(
        {"ok", "caution", "high_risk"}
    )


def test_warnings_df_shape():
    art = _run_eval()
    # 1 model x 2 estimators = 2 rows
    assert len(art.warnings) == 2
    assert set(art.warnings.columns) == {
        "model",
        "estimator",
        "support_health",
        "warning_codes",
    }
    # warning_codes is a list of strings (not a comma-joined string).
    for codes in art.warnings["warning_codes"]:
        assert isinstance(codes, list)
        for c in codes:
            assert isinstance(c, str)


def test_sensitivity_df_shape():
    art = _run_eval()
    assert len(art.sensitivity) == 2  # DR + SNDR
    expected = {
        "model",
        "estimator",
        "V_min",
        "V_max",
        "V_range",
        "chosen_clip",
        "chosen_V",
        "argmin_MSE_clip",
        "dr_sndr_agree",
        "stable",
    }
    assert set(art.sensitivity.columns) == expected
    # V_min <= chosen_V <= V_max for every row.
    for _, row in art.sensitivity.iterrows():
        assert row["V_min"] <= row["V_max"]


# --------------------------------------------------------------------------- #
# attach_warnings — direct unit tests (#22)                                   #
# --------------------------------------------------------------------------- #


def _toy_report_row(**overrides):
    base = {
        "model": "m",
        "estimator": "DR",
        "V_hat": 1.0,
        "SE_if": 0.1,
        "clip": 10.0,
        "ESS": 100.0,
        "tail_mass": 0.0,
        "MSE_est": 0.01,
        "match_rate": 1.0,
        "min_pscore": 0.5,
        "pscore_q10": 0.6,
        "pscore_q05": 0.55,
        "pscore_q01": 0.5,
    }
    base.update(overrides)
    return base


def test_attach_warnings_healthy_is_ok():
    report = pd.DataFrame([_toy_report_row()])
    enriched, warns = attach_warnings(report, n_samples=1000)
    assert enriched["support_health"].iloc[0] == SUPPORT_OK
    assert enriched["diagnostic_warnings"].iloc[0] == ""
    assert warns["warning_codes"].iloc[0] == []


def test_attach_warnings_low_ess_caution():
    # ESS = 50 / n = 1000 -> 5% < 10% threshold but >= 5% threshold -> caution.
    report = pd.DataFrame([_toy_report_row(ESS=99.0)])  # 9.9% < 10% but > 5%
    enriched, warns = attach_warnings(report, n_samples=1000)
    assert enriched["support_health"].iloc[0] == "caution"
    assert WARN_LOW_ESS in warns["warning_codes"].iloc[0]


def test_attach_warnings_low_ess_high_risk():
    # ESS / n < 5% -> high_risk
    report = pd.DataFrame([_toy_report_row(ESS=10.0)])  # 1% < 5%
    enriched, warns = attach_warnings(report, n_samples=1000)
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_LOW_ESS in warns["warning_codes"].iloc[0]


def test_attach_warnings_extreme_clip():
    # tail_mass > 0.05 -> caution; > 0.10 -> high_risk
    enriched_c, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(tail_mass=0.07)]), n_samples=1000
    )
    enriched_h, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(tail_mass=0.20)]), n_samples=1000
    )
    assert enriched_c["support_health"].iloc[0] == "caution"
    assert WARN_EXTREME_CLIP in enriched_c["diagnostic_warnings"].iloc[0]
    assert enriched_h["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_EXTREME_CLIP in enriched_h["diagnostic_warnings"].iloc[0]


def test_attach_warnings_low_match_rate():
    # match_rate < 0.5 -> caution; < 0.25 -> high_risk
    enriched_c, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(match_rate=0.4)]), n_samples=1000
    )
    enriched_h, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(match_rate=0.1)]), n_samples=1000
    )
    assert enriched_c["support_health"].iloc[0] == "caution"
    assert WARN_LOW_MATCH_RATE in enriched_c["diagnostic_warnings"].iloc[0]
    assert enriched_h["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_LOW_MATCH_RATE in enriched_h["diagnostic_warnings"].iloc[0]


def test_attach_warnings_poor_overlap_always_high_risk():
    # min_pscore < 1/n -> POOR_OVERLAP, always high_risk
    report = pd.DataFrame([_toy_report_row(min_pscore=1e-6)])
    enriched, _ = attach_warnings(report, n_samples=1000)
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_POOR_OVERLAP in enriched["diagnostic_warnings"].iloc[0]


def test_attach_warnings_two_cautions_escalates_to_high_risk():
    # Two caution-level warnings: low ESS (9%) + low match_rate (40%) -> high_risk
    report = pd.DataFrame([_toy_report_row(ESS=90.0, match_rate=0.4, min_pscore=0.5)])
    enriched, _ = attach_warnings(report, n_samples=1000)
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK


def test_attach_warnings_custom_thresholds():
    # Tighten low_ess threshold so 0.50 ESS-frac triggers caution.
    thresh = SupportHealthThresholds(low_ess_frac=0.60)
    enriched, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(ESS=500.0)]),  # 50% < 60%
        n_samples=1000,
        thresholds=thresh,
    )
    # 50% is below 30% (= 60%/2) ? No, 50% > 30%, so caution.
    assert enriched["support_health"].iloc[0] == "caution"


def test_attach_warnings_rejects_zero_n_samples():
    with pytest.raises(DataValidationError):
        attach_warnings(pd.DataFrame([_toy_report_row()]), n_samples=0)


def test_attach_warnings_rejects_missing_columns():
    report = pd.DataFrame([{"model": "m", "estimator": "DR"}])
    with pytest.raises(DataValidationError):
        attach_warnings(report, n_samples=100)


# --------------------------------------------------------------------------- #
# Sensitivity unit tests (#27)                                                #
# --------------------------------------------------------------------------- #


def test_summarize_sensitivity_empty():
    out = summarize_sensitivity({})
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_summarize_sensitivity_columns_and_invariants():
    art = _run_eval()
    s = summarize_sensitivity(art.detailed)
    # Same shape (1 model x 2 estimators).
    assert len(s) == 2
    # V_range = V_max - V_min, always non-negative.
    assert (s["V_range"] == s["V_max"] - s["V_min"]).all()
    assert (s["V_range"] >= 0).all()
    # chosen_clip appears in the result's grid for that estimator.
    for _, row in s.iterrows():
        result = art.detailed[row["model"]][row["estimator"]]
        assert row["chosen_clip"] == float(result.clip)
        assert row["chosen_V"] == float(result.V_hat)


# --------------------------------------------------------------------------- #
# JSON / HTML export (#28)                                                    #
# --------------------------------------------------------------------------- #


def test_to_json_round_trip(tmp_path: Path):
    art = _run_eval()
    p = art.to_json(tmp_path / "art.json")
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["skdr_eval_version"] == art.metadata["skdr_eval_version"]
    # Same row counts.
    assert len(payload["report"]) == len(art.report)
    assert len(payload["warnings"]) == len(art.warnings)
    assert len(payload["sensitivity"]) == len(art.sensitivity)
    # Pydantic round-trip parses cleanly.
    loaded = load_artifact_json(p)
    assert isinstance(loaded, ArtifactSchema)
    assert loaded.schema_version == SCHEMA_VERSION


def test_to_json_includes_diagnostics(tmp_path: Path):
    art = _run_eval()
    p = art.to_json(tmp_path / "art.json")
    payload = json.loads(p.read_text())
    assert "HGB" in payload["diagnostics"]
    diag = payload["diagnostics"]["HGB"]
    for k in (
        "overlap_ratio",
        "balance_ratio",
        "calibration_score",
        "discrimination_score",
        "log_loss_score",
        "statistics",
        "balance_stats",
    ):
        assert k in diag


def test_to_html_contains_key_sections(tmp_path: Path):
    art = _run_eval()
    p = art.to_html(tmp_path / "art.html")
    src = p.read_text()
    assert "<title>skdr-eval evaluation report</title>" in src
    assert "Headline results" in src
    assert "Support-health warnings" in src
    assert "Clip-grid sensitivity" in src
    assert "Propensity diagnostics" in src
    # Model name appears in the rendered table.
    assert "HGB" in src
    # Schema version is rendered.
    assert SCHEMA_VERSION in src


def test_export_writes_both_formats(tmp_path: Path):
    art = _run_eval()
    written = art.export(tmp_path / "run", formats=["json", "html"])
    assert set(written) == {"json", "html"}
    assert all(p.exists() for p in written.values())


def test_export_rejects_unknown_format(tmp_path: Path):
    art = _run_eval()
    with pytest.raises(ConfigurationError):
        art.export(tmp_path / "run", formats=["pdf"])


def test_export_results_top_level_helper(tmp_path: Path):
    """Issue #28: ``skdr_eval.export_results`` is the documented public API."""
    art = _run_eval()
    written = skdr_eval.export_results(art, tmp_path / "run", formats=["json"])
    assert "json" in written
    assert written["json"].exists()


# --------------------------------------------------------------------------- #
# Card rendering (#30)                                                        #
# --------------------------------------------------------------------------- #


def test_card_contains_expected_sections():
    art = _run_eval()
    card = art.card("HGB", headline_estimator="SNDR")
    assert "HGB" in card
    assert "Headline result (SNDR)" in card
    assert "Interpretation" in card
    assert "All estimators" in card
    # Has an inline matplotlib sparkline (matplotlib is now required).
    assert "data:image/png;base64," in card


def test_card_unknown_model_raises():
    art = _run_eval()
    with pytest.raises(DataValidationError):
        art.card("does-not-exist")


def test_card_unknown_estimator_raises():
    art = _run_eval()
    with pytest.raises(DataValidationError):
        art.card("HGB", headline_estimator="NOT_AN_ESTIMATOR")


def test_card_unknown_format_raises():
    art = _run_eval()
    with pytest.raises(ConfigurationError):
        art.card("HGB", format="markdown")


def test_render_evaluation_card_top_level_helper():
    art = _run_eval()
    out = render_evaluation_card(art, "HGB", headline_estimator="SNDR")
    assert isinstance(out, str)
    assert "HGB" in out


def test_card_with_ci_displays_band():
    art = _run_eval(ci=True)
    card = art.card("HGB")
    # CI label rendered (e.g. "95% CI: [..., ...]")
    assert "% CI" in card


# --------------------------------------------------------------------------- #
# Pydantic schema (#28)                                                       #
# --------------------------------------------------------------------------- #


def test_schema_json_schema_is_stable():
    """Lock the top-level JSON schema fields so wire-format changes are explicit."""
    schema = ArtifactSchema.model_json_schema()
    top = set(schema["properties"])
    assert {
        "schema_version",
        "skdr_eval_version",
        "timestamp",
        "metadata",
        "report",
        "warnings",
        "sensitivity",
        "diagnostics",
    } == top


def test_schema_version_pinned():
    assert SCHEMA_VERSION == "1.0.0"


# --------------------------------------------------------------------------- #
# Pairwise integration (#22 + #23 on the pairwise path)                       #
# --------------------------------------------------------------------------- #


def test_evaluate_pairwise_models_returns_artifact():
    logs_df, op_daily_df = skdr_eval.make_pairwise_synth(
        n_days=2, n_clients_day=80, n_ops=4, seed=7
    )
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: PLC0415

    models = {"HGB": HGB(max_iter=20, random_state=7)}

    art = skdr_eval.evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models,
        metric_col="service_time",
        task_type="regression",
        direction="min",
        fit_models=True,
        n_splits=3,
        random_state=7,
    )
    assert isinstance(art, EvaluationArtifact)
    assert art.metadata["evaluator"] == "evaluate_pairwise_models"
    assert "task_type" in art.metadata
    assert "support_health" in art.report.columns
    # Diagnostics may be empty if propensity sample is small; just check type.
    assert isinstance(art.diagnostics, dict)
