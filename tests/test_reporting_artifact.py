"""End-to-end tests for :mod:`skdr_eval.reporting` and ``EvaluationArtifact``.

Covers issues #22, #23, #27, #28, #30 — the bundled evaluation artifact and
the public attach-warnings / sensitivity / export / card helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
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
    WARN_HIGH_PARETO_K,
    WARN_LOW_ESS,
    WARN_LOW_MATCH_RATE,
    WARN_MISCAL_PROP,
    WARN_POOR_OVERLAP,
    ArtifactSchema,
    EvaluationArtifact,
    SupportHealthThresholds,
    _jsonable,
    attach_warnings,
    build_evaluation_artifact,
    load_artifact_json,
    render_evaluation_card,
    summarize_sensitivity,
)

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
        policy_train="all",  # explicit: legacy tests use full-data semantics
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
        # #133 — three-band stability grade + normalized range.
        "v_range_frac",
        "stability_grade",
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
        # nan pareto_k → no HIGH_PARETO_K warning (no-signal); overridable.
        "pareto_k": float("nan"),
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


# --------------------------------------------------------------------------- #
# HIGH_PARETO_K (#80) and MISCAL_PROP (#84) — warning-code tests              #
# --------------------------------------------------------------------------- #


def test_attach_warnings_high_pareto_k_caution():
    """0.5 ≤ k < 0.7 emits HIGH_PARETO_K as caution."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(pareto_k=0.6)]), n_samples=1000
    )
    assert enriched["support_health"].iloc[0] == "caution"
    assert WARN_HIGH_PARETO_K in warns["warning_codes"].iloc[0]


def test_attach_warnings_high_pareto_k_high_risk():
    """k ≥ 0.7 escalates HIGH_PARETO_K to high_risk."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(pareto_k=0.85)]), n_samples=1000
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_HIGH_PARETO_K in warns["warning_codes"].iloc[0]


def test_attach_warnings_pareto_k_nan_is_no_signal():
    """nan pareto_k must not emit HIGH_PARETO_K."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(pareto_k=float("nan"))]), n_samples=1000
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_OK
    assert WARN_HIGH_PARETO_K not in warns["warning_codes"].iloc[0]


def test_attach_warnings_pareto_k_below_caution_threshold_is_ok():
    """k < 0.5 is healthy — no warning emitted."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(pareto_k=0.3)]), n_samples=1000
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_OK
    assert WARN_HIGH_PARETO_K not in warns["warning_codes"].iloc[0]


def test_attach_warnings_miscal_prop_caution():
    """ECE in (0.10, 0.20] emits MISCAL_PROP as caution."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(model="m")]),
        n_samples=1000,
        model_ece={"m": 0.15},
    )
    assert enriched["support_health"].iloc[0] == "caution"
    assert WARN_MISCAL_PROP in warns["warning_codes"].iloc[0]


def test_attach_warnings_miscal_prop_high_risk():
    """ECE > 0.20 escalates MISCAL_PROP to high_risk."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(model="m")]),
        n_samples=1000,
        model_ece={"m": 0.30},
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK
    assert WARN_MISCAL_PROP in warns["warning_codes"].iloc[0]


def test_attach_warnings_miscal_prop_nan_is_no_signal():
    """nan / missing ECE must not emit MISCAL_PROP."""
    enriched, warns = attach_warnings(
        pd.DataFrame([_toy_report_row(model="m")]),
        n_samples=1000,
        model_ece={"m": float("nan")},
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_OK
    assert WARN_MISCAL_PROP not in warns["warning_codes"].iloc[0]

    # Same when the model is simply missing from the ECE map.
    enriched2, warns2 = attach_warnings(
        pd.DataFrame([_toy_report_row(model="other")]),
        n_samples=1000,
        model_ece={"m": 0.50},  # huge ECE for a different model
    )
    assert enriched2["support_health"].iloc[0] == SUPPORT_OK
    assert WARN_MISCAL_PROP not in warns2["warning_codes"].iloc[0]


def test_attach_warnings_thresholds_configurable_pareto_k():
    """Tightening high_pareto_k thresholds fires the warning at lower k."""
    thresh = SupportHealthThresholds(high_pareto_k_caution=0.2, high_pareto_k=0.4)
    enriched, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(pareto_k=0.45)]),
        n_samples=1000,
        thresholds=thresh,
    )
    # 0.45 > 0.4 → high_risk under the tightened threshold.
    assert enriched["support_health"].iloc[0] == SUPPORT_HIGH_RISK


def test_attach_warnings_thresholds_configurable_miscal_ece():
    """Loosening miscal_ece silences MISCAL_PROP on borderline ECE values."""
    thresh = SupportHealthThresholds(miscal_ece=0.40)
    enriched, _ = attach_warnings(
        pd.DataFrame([_toy_report_row(model="m")]),
        n_samples=1000,
        thresholds=thresh,
        model_ece={"m": 0.30},
    )
    assert enriched["support_health"].iloc[0] == SUPPORT_OK


# --------------------------------------------------------------------------- #
# End-to-end: report + diagnostics carry the new fields                       #
# --------------------------------------------------------------------------- #


def test_report_includes_pareto_k_column():
    art = _run_eval()
    assert "pareto_k" in art.report.columns
    # With synth data we expect a real number, but the diagnostic may return
    # nan if the matched-set tail is too short — accept either, just verify
    # type discipline.
    for v in art.report["pareto_k"]:
        if v is not None:
            assert isinstance(float(v), float)


def test_diagnostics_include_ece_and_brier():
    art = _run_eval()
    for _model_name, diag in art.diagnostics.items():
        # Both metrics finite under healthy synth data with n=400.
        assert math.isfinite(diag.ece), f"ECE not finite: {diag.ece}"
        assert math.isfinite(diag.brier_score), f"Brier not finite: {diag.brier_score}"
        assert 0.0 <= diag.ece <= 1.0
        assert 0.0 <= diag.brier_score <= 2.0  # multiclass Brier upper bound
        assert len(diag.reliability_curve) == diag.ece_n_bins


def test_json_roundtrip_carries_new_fields(tmp_path: Path):
    art = _run_eval()
    p = tmp_path / "art.json"
    art.to_json(p)
    payload = json.loads(p.read_text())
    # Pareto-k present on each report row (None when nan via _jsonable).
    for row in payload["report"]:
        assert "pareto_k" in row
    # ECE/Brier present on each diagnostics entry.
    for _name, diag_payload in payload["diagnostics"].items():
        assert "ece" in diag_payload
        assert "brier_score" in diag_payload
        assert "reliability_curve" in diag_payload
        assert diag_payload["ece_n_bins"] == 15


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
    assert isinstance(p, Path)
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


def test_to_json_no_arg_returns_string(tmp_path: Path):
    """#108: to_json() with no path returns the JSON string; to_json(path) writes."""
    art = _run_eval()

    text = art.to_json()
    assert isinstance(text, str)
    payload = json.loads(text)
    assert payload["schema_version"] == SCHEMA_VERSION

    # The string return must match what gets written to disk.
    written = art.to_json(tmp_path / "art.json")
    assert isinstance(written, Path)
    assert written.read_text() == text


def test_to_html_no_arg_returns_string(tmp_path: Path):
    """#108: to_html() with no path returns the HTML string; to_html(path) writes."""
    art = _run_eval()

    html = art.to_html()
    assert isinstance(html, str)
    assert "<title>skdr-eval evaluation report</title>" in html

    written = art.to_html(tmp_path / "art.html")
    assert isinstance(written, Path)
    assert written.read_text() == html


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
    """Lock the top-level JSON schema fields so wire-format changes are explicit.

    The pre-1.1.0 baseline was eight fields; #128 adds the estimand contract
    (``estimand_tex`` / ``estimand_summary`` / ``assumptions``) and #132 adds
    the baseline configuration (``baseline_kind`` / ``baseline_value``) onto
    the wire so the artifact JSON can be round-tripped without losing the
    statistical contract that the card already carries.
    """
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
        # #128 — estimand contract on the wire.
        "estimand_tex",
        "estimand_summary",
        "assumptions",
        # #132 — baseline configuration on the wire.
        "baseline_kind",
        "baseline_value",
    } == top


def test_schema_version_pinned():
    # 1.1.0: additive trust diagnostics (#80 pareto_k, #84 ECE/Brier).
    assert SCHEMA_VERSION == "1.1.0"


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
        policy_train="all",  # explicit: suppress DeprecationWarning in legacy test
    )
    assert isinstance(art, EvaluationArtifact)
    assert art.metadata["evaluator"] == "evaluate_pairwise_models"
    assert "task_type" in art.metadata
    assert "support_health" in art.report.columns
    # Diagnostics may be empty if propensity sample is small; just check type.
    assert isinstance(art.diagnostics, dict)


# --------------------------------------------------------------------------- #
# Regression tests for review-fix commits                                     #
# --------------------------------------------------------------------------- #


def test_jsonable_serializes_infinity_as_null():
    """Non-finite floats (``inf``, ``-inf``, ``NaN``) must serialize as JSON null."""
    assert _jsonable(float("inf")) is None
    assert _jsonable(float("-inf")) is None
    assert _jsonable(math.nan) is None
    assert _jsonable(np.float64("inf")) is None
    assert _jsonable(np.float64("nan")) is None
    # Finite floats and ints pass through.
    assert _jsonable(1.5) == 1.5
    assert _jsonable(np.float64(2.5)) == 2.5
    assert _jsonable(np.int64(3)) == 3
    assert _jsonable(np.bool_(True)) is True


def test_to_json_clip_inf_serializes_as_null(tmp_path: Path):
    """``clip = inf`` (the no-clipping sentinel) must serialize as JSON null."""
    art = _run_eval()
    # Force the chosen clip to inf so the wire-format gets exercised.
    for row in art.detailed.values():
        for est_name in list(row):
            row[est_name].clip = float("inf")
    # Mirror into the report DataFrame.
    art.report["clip"] = float("inf")

    payload = json.loads(art.to_json_str())
    for row in payload["report"]:
        assert row["clip"] is None, "clip=inf must serialize as null in JSON"

    # The Pydantic schema must accept the null clip without error.
    loaded = ArtifactSchema.model_validate(payload)
    assert all(r.clip is None for r in loaded.report)


def _force_inf_clip(art: EvaluationArtifact) -> None:
    """Mutate an artifact so every chosen clip is ``inf`` for render coverage."""
    for row in art.detailed.values():
        for est_name in list(row):
            row[est_name].clip = float("inf")
    art.report["clip"] = float("inf")
    if "chosen_clip" in art.sensitivity.columns:
        art.sensitivity["chosen_clip"] = float("inf")
    if "argmin_MSE_clip" in art.sensitivity.columns:
        art.sensitivity["argmin_MSE_clip"] = float("inf")


def test_to_html_str_with_inf_clip_renders_sentinel(tmp_path: Path):
    """HTML render must not crash when ``clip`` columns are the inf sentinel.

    Before the ``fmt_num`` filter, ``"%.4g"|format(None)`` raised ``TypeError``
    once ``_jsonable`` coerced ``inf`` to ``None``. This pins that branch.
    """
    art = _run_eval()
    _force_inf_clip(art)
    html_str = art.to_html_str()
    # No crash, and the sentinel is rendered where clip would have been.
    assert "<title>skdr-eval evaluation report</title>" in html_str
    assert "∞" in html_str
    # And no leftover ``None`` literal in the rendered numeric cells.
    assert ">None<" not in html_str


def test_card_with_inf_clip_renders_sentinel():
    """Card render must not crash when ``clip`` columns are the inf sentinel."""
    art = _run_eval()
    _force_inf_clip(art)
    card = art.card("HGB", headline_estimator="SNDR")
    assert "HGB" in card
    assert "∞" in card
    assert ">None<" not in card


def test_diagnostics_are_independent_per_model():
    """Mutating one model's diagnostics must not leak to other models."""
    logs, _, _ = skdr_eval.make_synth_logs(n=400, n_ops=3, seed=7)
    models = {
        "A": HistGradientBoostingRegressor(max_iter=10, random_state=7),
        "B": HistGradientBoostingRegressor(max_iter=10, random_state=7),
    }
    art = skdr_eval.evaluate_sklearn_models(
        logs=logs, models=models, fit_models=True, n_splits=3, random_state=7
    )
    assert {"A", "B"}.issubset(art.diagnostics)
    diag_a = art.diagnostics["A"]
    diag_b = art.diagnostics["B"]
    assert diag_a is not diag_b, "diagnostics objects must be distinct per model"
    # Mutating one must not affect the other.
    diag_a.statistics["mutated"] = 999.0
    assert "mutated" not in diag_b.statistics


def test_to_html_metadata_section_not_double_escaped(tmp_path: Path):
    """``metadata_json`` must be auto-escaped by Jinja exactly once."""
    art = _run_eval()
    art.metadata["note"] = '"quoted" & <tag>'
    html_str = art.to_html_str()
    # Jinja autoescape converts " -> &#34; and & -> &amp; (one pass only).
    # Double-escaping would produce &amp;quot;, &amp;amp;, &amp;lt; etc.
    assert "&amp;quot;" not in html_str
    assert "&amp;amp;" not in html_str
    assert "&amp;lt;" not in html_str


def test_export_treats_suffixless_path_as_stem(tmp_path: Path):
    """``"run"`` (no suffix, no trailing sep) -> ``run.json`` / ``run.html``."""
    art = _run_eval()
    written = art.export(tmp_path / "run", formats=["json", "html"])
    assert written["json"] == tmp_path / "run.json"
    assert written["html"] == tmp_path / "run.html"
    assert written["json"].exists()
    assert written["html"].exists()


def test_export_writes_inside_existing_directory(tmp_path: Path):
    """When ``path`` is an existing directory, write ``artifact.<fmt>`` inside."""
    art = _run_eval()
    written = art.export(tmp_path, formats=["json"])
    assert written["json"] == tmp_path / "artifact.json"
    assert written["json"].exists()


def test_export_replaces_existing_suffix(tmp_path: Path):
    """When ``path`` already has a suffix, the suffix is replaced per format."""
    art = _run_eval()
    written = art.export(str(tmp_path / "run.json"), formats=["json", "html"])
    assert written["json"] == tmp_path / "run.json"
    assert written["html"] == tmp_path / "run.html"


def test_export_trailing_slash_means_directory(tmp_path: Path):
    """A trailing path separator forces directory semantics."""
    target = str(tmp_path / "outdir") + "/"
    art = _run_eval()
    written = art.export(target, formats=["json"])
    assert written["json"] == tmp_path / "outdir" / "artifact.json"
    assert written["json"].exists()


def test_to_json_str_round_trips_without_path():
    """``to_json_str`` returns serialized JSON without writing to disk."""
    art = _run_eval()
    s = art.to_json_str(indent=None)
    payload = json.loads(s)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["skdr_eval_version"] == art.metadata["skdr_eval_version"]


def test_to_html_str_returns_self_contained_html():
    art = _run_eval()
    src = art.to_html_str()
    assert src.startswith("<!DOCTYPE html>")
    assert "</html>" in src


def test_save_card_writes_to_disk(tmp_path: Path):
    art = _run_eval()
    out = art.save_card(tmp_path / "card.html", "HGB")
    assert out == tmp_path / "card.html"
    contents = out.read_text()
    assert "HGB" in contents
    assert "Headline result" in contents


def test_summarize_sensitivity_empty_detailed():
    out = summarize_sensitivity({})
    assert out.empty


def test_load_artifact_json_round_trip(tmp_path: Path):
    art = _run_eval()
    p = art.to_json(tmp_path / "run.json")
    schema = load_artifact_json(p)
    assert isinstance(schema, ArtifactSchema)
    assert len(schema.report) == len(art.report)


def test_build_evaluation_artifact_without_propensities():
    """``build_evaluation_artifact(propensities=None)`` returns no diagnostics."""
    art_full = _run_eval()
    art = build_evaluation_artifact(
        report=art_full.report.drop(
            columns=["support_health", "diagnostic_warnings"], errors="ignore"
        ),
        detailed=art_full.detailed,
        n_samples=400,
        propensities=None,
        actions=None,
    )
    assert art.diagnostics == {}
    # Without random_state / alpha args, those keys are absent from metadata.
    assert "random_state" not in art.metadata
    assert "alpha" not in art.metadata


def test_build_evaluation_artifact_handles_failing_diagnostics():
    """Diagnostics that raise should be logged and omitted, not propagated."""
    art_full = _run_eval()
    # Pass too few samples so comprehensive_propensity_diagnostics raises
    # InsufficientDataError; the factory should catch it and return {}.
    art = build_evaluation_artifact(
        report=art_full.report.drop(
            columns=["support_health", "diagnostic_warnings"], errors="ignore"
        ),
        detailed=art_full.detailed,
        n_samples=400,
        propensities=np.array([[0.5, 0.5]]),
        actions=np.array([0]),
    )
    assert art.diagnostics == {}


def test_card_handles_missing_diagnostics():
    """Card should render even when ``artifact.diagnostics`` is empty."""
    art_full = _run_eval()
    art = build_evaluation_artifact(
        report=art_full.report.drop(
            columns=["support_health", "diagnostic_warnings"], errors="ignore"
        ),
        detailed=art_full.detailed,
        n_samples=400,
        propensities=None,
        actions=None,
    )
    card = art.card("HGB")
    assert "HGB" in card
    # Without diagnostics, the "Propensity diagnostics" subsection is omitted.
    assert "Propensity diagnostics" not in card


# --------------------------------------------------------------------------- #
# Recommendation API (#83)                                                    #
# --------------------------------------------------------------------------- #


from skdr_eval.reporting import (  # noqa: E402
    DiagnosticGate,
    GateResult,
    Reason,
    Recommendation,
    RecommendationPolicy,
    gate_diagnostics,
)


def _make_report_row(
    *,
    v_hat: float = 0.5,
    se: float = 0.1,
    clip: float = 5.0,
    ess: float = 200.0,
    tail_mass: float = 0.05,
    match_rate: float = 0.9,
    min_pscore: float = 0.05,
    pscore_q10: float = 0.05,
    pscore_q05: float = 0.03,
    pscore_q01: float = 0.01,
    pareto_k: float = 0.5,
    support_health: str = "ok",
    diagnostic_warnings: str = "",
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    estimator: str = "SNDR",
    model: str = "HGB",
) -> pd.DataFrame:
    row: dict[str, object] = {
        "model": model,
        "estimator": estimator,
        "V_hat": v_hat,
        "SE_if": se,
        "clip": clip,
        "ESS": ess,
        "tail_mass": tail_mass,
        "MSE_est": 0.01,
        "match_rate": match_rate,
        "min_pscore": min_pscore,
        "pscore_q10": pscore_q10,
        "pscore_q05": pscore_q05,
        "pscore_q01": pscore_q01,
        "pareto_k": pareto_k,
        "support_health": support_health,
        "diagnostic_warnings": diagnostic_warnings,
    }
    if ci_lower is not None:
        row["ci_lower"] = ci_lower
    if ci_upper is not None:
        row["ci_upper"] = ci_upper
    return pd.DataFrame([row])


def _make_artifact_from_row(report: pd.DataFrame) -> EvaluationArtifact:
    """Minimal stub artifact for recommendation/gate tests."""
    # Build minimal DRResult stubs from the report row.
    detailed: dict[str, dict[str, object]] = {}
    for _, row in report.iterrows():
        m = str(row["model"])
        e = str(row["estimator"])
        if m not in detailed:
            detailed[m] = {}
        # Minimal DRResult stub
        from skdr_eval.core import DRResult as _DRResult  # noqa: PLC0415

        grid = pd.DataFrame(
            {
                "clip": [row["clip"]],
                "V_hat": [row["V_hat"]],
                "SE_if": [row["SE_if"]],
                "ESS": [row["ESS"]],
                "MSE_est": [row["MSE_est"]],
                "tail_mass": [row["tail_mass"]],
                "match_rate": [row["match_rate"]],
                "min_pscore": [row["min_pscore"]],
            }
        )
        detailed[m][e] = _DRResult(
            V_hat=float(row["V_hat"]),
            SE_if=float(row["SE_if"]),
            clip=float(row["clip"]),
            ESS=float(row["ESS"]),
            tail_mass=float(row["tail_mass"]),
            MSE_est=float(row["MSE_est"]),
            match_rate=float(row["match_rate"]),
            min_pscore=float(row["min_pscore"]),
            pscore_q10=float(row["pscore_q10"]),
            pscore_q05=float(row["pscore_q05"]),
            pscore_q01=float(row["pscore_q01"]),
            pareto_k=float(row["pareto_k"]),
            grid=grid,
        )
    return EvaluationArtifact(
        report=report,
        detailed=detailed,  # type: ignore[arg-type]
        warnings=pd.DataFrame(
            columns=["model", "estimator", "support_health", "warning_codes"]
        ),
        sensitivity=pd.DataFrame(),
        diagnostics={},
        metadata={"n_samples": 400},
    )


class TestRecommendation:
    """Tests for EvaluationArtifact.recommendation() and Recommendation data class."""

    def test_deploy_verdict_clean_ci(self) -> None:
        """CI above baseline with no caution flags → 'deploy'."""
        report = _make_report_row(
            ci_lower=0.2, ci_upper=0.8, support_health="ok", diagnostic_warnings=""
        )
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        assert rec.verdict == "deploy"
        assert rec.confidence == "high"
        assert rec.primary_blocker is None
        assert rec.recommended_estimator == "SNDR"

    def test_ab_test_verdict_ci_above_baseline_with_caution(self) -> None:
        """CI above baseline + caution flags → 'ab_test' with medium confidence."""
        report = _make_report_row(
            ci_lower=0.2,
            ci_upper=0.8,
            support_health="caution",
            diagnostic_warnings="LOW_ESS",
        )
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        assert rec.verdict == "ab_test"
        assert rec.confidence == "medium"

    def test_ab_test_verdict_ci_overlaps_baseline(self) -> None:
        """CI overlapping baseline → 'ab_test' with low confidence."""
        report = _make_report_row(
            ci_lower=-0.1,
            ci_upper=0.3,
            support_health="ok",
            diagnostic_warnings="",
        )
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB", baseline=0.0)
        assert rec.verdict == "ab_test"
        assert rec.confidence == "low"

    def test_do_not_deploy_high_risk(self) -> None:
        """High-risk warning → 'do_not_deploy'."""
        report = _make_report_row(
            support_health="high_risk",
            diagnostic_warnings="POOR_OVERLAP",
        )
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        assert rec.verdict == "do_not_deploy"
        assert rec.primary_blocker == "POOR_OVERLAP"

    def test_insufficient_evidence_no_ci(self) -> None:
        """No CI available → 'insufficient_evidence'."""
        report = _make_report_row(support_health="ok", diagnostic_warnings="")
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        assert rec.verdict == "insufficient_evidence"

    def test_reasons_list_populated(self) -> None:
        report = _make_report_row(
            support_health="high_risk",
            diagnostic_warnings="POOR_OVERLAP",
        )
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        assert len(rec.reasons) >= 1
        assert all(isinstance(r, Reason) for r in rec.reasons)

    def test_to_dict_serializable(self) -> None:
        report = _make_report_row(ci_lower=0.2, ci_upper=0.8)
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        d = rec.to_dict()
        assert "verdict" in d
        assert "reasons" in d
        # Each reason is a plain dict
        for r in d["reasons"]:
            assert "code" in r
            assert "message" in r
            assert "severity" in r

    def test_from_dict_round_trip(self) -> None:
        report = _make_report_row(ci_lower=0.2, ci_upper=0.8)
        art = _make_artifact_from_row(report)
        rec = art.recommendation("HGB")
        d = rec.to_dict()
        rec2 = Recommendation.from_dict(d)
        assert rec2.verdict == rec.verdict
        assert rec2.confidence == rec.confidence
        assert rec2.primary_blocker == rec.primary_blocker

    def test_unknown_model_raises(self) -> None:
        art = _run_eval()
        with pytest.raises(DataValidationError, match="not in artifact"):
            art.recommendation("no_such_model")

    def test_unknown_estimator_raises(self) -> None:
        art = _run_eval()
        with pytest.raises(DataValidationError, match="not in detailed"):
            art.recommendation("HGB", estimator="QUANTUM")

    def test_recommendation_with_policy_object(self) -> None:
        report = _make_report_row(ci_lower=0.2, ci_upper=0.8)
        art = _make_artifact_from_row(report)
        policy = RecommendationPolicy(baseline=0.5)
        # baseline=0.5: ci_lower=0.2 < 0.5, so CI overlaps → ab_test
        rec = art.recommendation("HGB", policy=policy)
        assert rec.verdict == "ab_test"

    def test_recommendation_on_real_artifact(self) -> None:
        """End-to-end: recommendation runs on a real artifact without error."""
        art = _run_eval(ci=True)
        rec = art.recommendation("HGB", estimator="DR")
        assert rec.verdict in (
            "deploy",
            "ab_test",
            "do_not_deploy",
            "insufficient_evidence",
        )
        assert rec.model_name == "HGB"


# --------------------------------------------------------------------------- #
# DiagnosticGate API (#99)                                                    #
# --------------------------------------------------------------------------- #


class TestDiagnosticGate:
    """Tests for gate_diagnostics() and DiagnosticGate data class."""

    def test_gate_pass_healthy_artifact(self) -> None:
        """Healthy report row should produce overall=pass."""
        report = _make_report_row(
            min_pscore=0.05,
            ess=200.0,
            pareto_k=0.4,
            support_health="ok",
            diagnostic_warnings="",
        )
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        assert isinstance(gate, DiagnosticGate)
        assert gate.overall == "pass"
        assert gate.overlap.state == "pass"
        assert gate.ess.state == "pass"
        assert gate.calibration.state == "pass"

    def test_gate_fail_poor_overlap(self) -> None:
        """min_pscore ≤ 1/n → overlap gate fails."""
        report = _make_report_row(
            min_pscore=0.001,  # 1/400 = 0.0025; 0.001 < 0.0025
            ess=200.0,
            pareto_k=0.4,
        )
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        assert gate.overlap.state == "fail"
        assert gate.overall == "fail"

    def test_gate_warn_low_match_rate(self) -> None:
        """match_rate below threshold → overlap gate warns."""
        report = _make_report_row(
            min_pscore=0.05,
            match_rate=0.3,  # below default LOW_MATCH_RATE=0.5
            ess=200.0,
            pareto_k=0.4,
        )
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        assert gate.overlap.state == "warn"
        assert gate.overall == "warn"

    def test_gate_warn_low_ess(self) -> None:
        """ESS below threshold → ess gate warns/fails."""
        report = _make_report_row(min_pscore=0.05, ess=5.0, pareto_k=0.4)
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        assert gate.ess.state in ("warn", "fail")
        assert gate.overall in ("warn", "fail")

    def test_gate_warn_high_pareto_k(self) -> None:
        """Pareto-k above threshold → calibration gate warns/fails."""
        report = _make_report_row(
            min_pscore=0.05,
            ess=200.0,
            pareto_k=0.85,  # above default 0.7
        )
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        assert gate.calibration.state in ("warn", "fail")
        assert gate.overall in ("warn", "fail")

    def test_gate_custom_thresholds(self) -> None:
        """Custom SupportHealthThresholds affect gate outcomes."""
        report = _make_report_row(min_pscore=0.05, ess=200.0, pareto_k=0.6)
        art = _make_artifact_from_row(report)
        strict = SupportHealthThresholds(high_pareto_k=0.5)
        gate = gate_diagnostics(art, "HGB", "SNDR", thresholds=strict)
        # pareto_k=0.6 > 0.5 → calibration should warn/fail
        assert gate.calibration.state in ("warn", "fail")

    def test_gate_to_dict(self) -> None:
        report = _make_report_row()
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        d = gate.to_dict()
        assert "overlap" in d
        assert "ess" in d
        assert "calibration" in d
        assert "overall" in d
        assert d["overlap"]["check"] == "overlap"

    def test_gate_to_text(self) -> None:
        report = _make_report_row()
        art = _make_artifact_from_row(report)
        gate = gate_diagnostics(art, "HGB", "SNDR")
        txt = gate.to_text()
        assert "DiagnosticGate overall" in txt
        assert "overlap" in txt

    def test_gate_unknown_model_raises(self) -> None:
        art = _run_eval()
        with pytest.raises(DataValidationError, match="not in artifact"):
            gate_diagnostics(art, "no_such_model")

    def test_gate_unknown_estimator_raises(self) -> None:
        art = _run_eval()
        with pytest.raises(DataValidationError, match="not in detailed"):
            gate_diagnostics(art, "HGB", "QUANTUM")

    def test_gate_on_real_artifact(self) -> None:
        """End-to-end: gate_diagnostics runs on a real artifact without error."""
        art = _run_eval()
        gate = gate_diagnostics(art, "HGB", "DR")
        assert gate.overall in ("pass", "warn", "fail")
        assert isinstance(gate.overlap, GateResult)
        assert isinstance(gate.ess, GateResult)
        assert isinstance(gate.calibration, GateResult)

    def test_gate_gateresult_to_dict_complete(self) -> None:
        r = GateResult(
            check="overlap",
            state="pass",
            code="OVERLAP_OK",
            message="ok",
            value=0.05,
            threshold=0.0025,
        )
        d = r.to_dict()
        assert d["check"] == "overlap"
        assert d["state"] == "pass"
        assert d["value"] == 0.05
        assert d["threshold"] == 0.0025


# --------------------------------------------------------------------------- #
# Export / consumption surface (#184 #234 #237 #238 #249 #251)                #
# --------------------------------------------------------------------------- #


class TestTypedRows:
    def test_rows_match_report(self) -> None:
        art = _run_eval(ci=True)
        rows = art.rows()
        assert len(rows) == len(art.report)
        # Field parity vs the DataFrame for the SNDR row.
        r = art.row("HGB", estimator="SNDR")
        df_row = art.report[
            (art.report["model"] == "HGB") & (art.report["estimator"] == "SNDR")
        ].iloc[0]
        assert r.model == "HGB"
        assert r.estimator == "SNDR"
        assert math.isclose(r.V_hat, float(df_row["V_hat"]), rel_tol=0, abs_tol=1e-9)
        assert r.verdict in {
            "deploy",
            "ab_test",
            "insufficient_evidence",
            "do_not_deploy",
            None,
        }

    def test_row_unknown_raises(self) -> None:
        art = _run_eval()
        with pytest.raises(DataValidationError, match="No report row"):
            art.row("HGB", estimator="NOPE")

    def test_rows_verdict_never_positive_without_ci(self) -> None:
        # No bootstrap CI → the gate cannot return a positive verdict; it is
        # either insufficient_evidence or do_not_deploy (a high-risk blocker),
        # never deploy/ab_test.
        art = _run_eval(ci=False)
        for r in art.rows():
            assert r.verdict in {"insufficient_evidence", "do_not_deploy", None}


class TestToMarkdown:
    def test_markdown_has_table_and_rows(self) -> None:
        art = _run_eval(ci=True)
        md = art.to_markdown()
        assert md.startswith("# skdr-eval evaluation summary")
        assert "| Model | Estimator |" in md
        # One data row per (model, estimator).
        assert md.count("| HGB |") == len(art.report)

    def test_markdown_filters_by_model_and_estimator(self) -> None:
        art = _run_eval(ci=True)
        md = art.to_markdown("HGB", estimator="SNDR")
        assert md.count("| HGB |") == 1

    def test_markdown_escapes_pipes_in_model_name(self) -> None:
        report = _make_report_row()
        report["model"] = "a|b"
        art = _make_artifact_from_row(report)
        md = art.to_markdown()
        assert "a\\|b" in md


class TestDecisionSummary:
    def test_high_risk_leads_with_risk(self) -> None:
        report = _make_report_row()
        report["support_health"] = SUPPORT_HIGH_RISK
        art = _make_artifact_from_row(report)
        summary = art.decision_summary("HGB", estimator="SNDR")
        assert summary["support_health"] == SUPPORT_HIGH_RISK
        assert summary["summary"].lower().startswith("high-risk support")
        assert "do not deploy" in summary["summary"].lower()

    def test_delta_reported_against_baseline(self) -> None:
        art = _run_eval(ci=True)
        r = art.row("HGB", estimator="SNDR")
        summary = art.decision_summary("HGB", estimator="SNDR", baseline=0.0)
        assert summary["baseline"] == 0.0
        assert math.isclose(
            summary["delta_vs_baseline"], r.V_hat - 0.0, rel_tol=0, abs_tol=1e-9
        )

    def test_ok_support_reads_healthy(self) -> None:
        report = _make_report_row()
        report["support_health"] = SUPPORT_OK
        art = _make_artifact_from_row(report)
        summary = art.decision_summary("HGB", estimator="SNDR")
        assert "healthy" in summary["summary"].lower()

    def test_unknown_support_does_not_read_healthy(self) -> None:
        # A missing/unknown support_health must not be narrated as an endorsement.
        report = _make_report_row()
        report["support_health"] = None
        art = _make_artifact_from_row(report)
        summary = art.decision_summary("HGB", estimator="SNDR")
        assert summary["support_health"] is None
        assert "healthy" not in summary["summary"].lower()
        assert "unavailable" in summary["summary"].lower()


class TestSummaryFacts:
    def test_facts_keys_and_grounding(self) -> None:
        art = _run_eval(ci=True)
        facts = art.to_summary_facts("HGB", estimator="SNDR")
        expected = {
            "model",
            "estimator",
            "V_hat",
            "ci_lower",
            "ci_upper",
            "baseline",
            "delta_vs_baseline",
            "verdict",
            "confidence",
            "support_health",
            "primary_blocker",
            "warning_codes",
            "reasons",
        }
        assert set(facts) == expected
        assert facts["model"] == "HGB"
        assert isinstance(facts["warning_codes"], list)
        assert isinstance(facts["reasons"], list)


class TestBadge:
    def test_badge_color_tracks_support_health(self) -> None:
        report = _make_report_row()
        report["support_health"] = SUPPORT_HIGH_RISK
        art = _make_artifact_from_row(report)
        badge = art.badge("HGB", estimator="SNDR")
        # high_risk must render red — never oversell a thin-support result.
        assert badge["color"] == "#e05d44"
        assert badge["message"].endswith("high_risk")
        assert badge["svg"].startswith("<svg")
        assert "</svg>" in badge["svg"]
        assert badge["markdown"].startswith("![skdr-eval")

    def test_badge_ok_is_green(self) -> None:
        report = _make_report_row()
        report["support_health"] = SUPPORT_OK
        art = _make_artifact_from_row(report)
        badge = art.badge("HGB", estimator="SNDR")
        assert badge["color"] == "#4c1"


class TestCompare:
    def test_identical_runs_have_no_regression(self) -> None:
        art = _run_eval(seed=7, ci=True)
        other = _run_eval(seed=7, ci=True)
        diff = art.compare(other)
        assert not diff.verdict_regressed
        assert all(not r.verdict_regressed for r in diff.rows)
        assert all(r.status in {"unchanged", "changed"} for r in diff.rows)

    def test_verdict_regression_detected(self) -> None:
        art = _run_eval(seed=7, ci=True)
        # Build a healthy baseline by hand so the candidate's verdict is worse.
        baseline = _run_eval(seed=7, ci=True)
        baseline.report["support_health"] = SUPPORT_OK
        baseline.report["diagnostic_warnings"] = ""
        baseline.report["pareto_k"] = 0.1
        baseline.report["ci_lower"] = 1_000.0
        baseline.report["ci_upper"] = 1_100.0
        diff = art.compare(baseline)
        assert diff.verdict_regressed
        regressed = [r for r in diff.rows if r.verdict_regressed]
        assert regressed
        assert regressed[0].verdict_before == "deploy"
        md = diff.to_markdown()
        assert "regressed" in md.lower()

    def test_epsilon_suppresses_float_jitter(self) -> None:
        art = _run_eval(seed=7, ci=True)
        other = _run_eval(seed=7, ci=True)
        # Nudge a V_hat by less than epsilon; the delta must read as unchanged.
        other.report.loc[other.report["estimator"] == "SNDR", "V_hat"] += 1e-12
        diff = art.compare(other, epsilon=1e-9)
        sndr = next(r for r in diff.rows if r.estimator == "SNDR")
        assert sndr.delta_V_hat == 0.0

    def test_added_and_removed_rows(self) -> None:
        art = _run_eval(seed=7, ci=True)
        baseline = _run_eval(seed=7, ci=True)
        # Drop the DR row from the baseline → it appears as "added" in the diff.
        baseline.report = baseline.report[
            baseline.report["estimator"] != "DR"
        ].reset_index(drop=True)
        del baseline.detailed["HGB"]["DR"]
        diff = art.compare(baseline)
        statuses = {(r.estimator, r.status) for r in diff.rows}
        assert ("DR", "added") in statuses


class TestExportMarkdown:
    def test_export_writes_markdown(self, tmp_path: Path) -> None:
        art = _run_eval(ci=True)
        paths = art.export(tmp_path / "run", formats=["json", "markdown"])
        assert paths["markdown"].suffix == ".md"
        assert paths["markdown"].is_file()
        assert (
            paths["markdown"]
            .read_text(encoding="utf-8")
            .startswith("# skdr-eval evaluation summary")
        )

    def test_export_rejects_unknown_format(self, tmp_path: Path) -> None:
        art = _run_eval()
        with pytest.raises(ConfigurationError, match="Unknown export format"):
            art.export(tmp_path / "run", formats=["json", "pdf"])
