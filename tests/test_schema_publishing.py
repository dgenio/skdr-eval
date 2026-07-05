"""Schema publishing (#205) and artifact round-trip / forward-compat (#212).

Two guarantees are locked in here:

* The committed JSON Schemas under ``docs/schemas/`` never silently drift from
  the live Pydantic schema (#205). ``scripts/generate_schemas.py`` regenerates
  them; this test fails if they are stale.
* Older serialized artifacts (a genuine ``1.0.0``-shaped payload) still load
  through the current models, and the fields added since default correctly
  (#212). Fixtures live under ``tests/fixtures/artifacts/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skdr_eval.reporting import (
    SCHEMA_VERSION,
    ArtifactSchema,
    EvaluationCard,
    load_artifact_json,
)

_REPO = Path(__file__).resolve().parents[1]
_SCHEMA_DIR = _REPO / "docs" / "schemas"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "artifact_samples"

# Keep in sync with scripts/generate_schemas.py::SCHEMAS.
SCHEMA_FILES = {
    "artifact.schema.json": ArtifactSchema.json_schema,
    "card.schema.json": EvaluationCard.json_schema,
}


# --------------------------------------------------------------------------- #
# #205 — published schema files do not drift                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("filename", sorted(SCHEMA_FILES))
def test_committed_schema_matches_live(filename: str) -> None:
    """The committed schema file equals ``json_schema()`` (regenerate if not)."""
    producer = SCHEMA_FILES[filename]
    live = json.dumps(producer(), indent=2, sort_keys=True) + "\n"
    committed = (_SCHEMA_DIR / filename).read_text(encoding="utf-8")
    assert committed == live, (
        f"docs/schemas/{filename} is stale — run "
        "`python scripts/generate_schemas.py` and commit the result."
    )


def test_artifact_schema_advertises_current_version() -> None:
    """The artifact schema pins the current ``SCHEMA_VERSION`` as its default."""
    schema = ArtifactSchema.json_schema()
    assert schema["properties"]["schema_version"]["default"] == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# #212 — round-trip and forward-compat for older artifact payloads            #
# --------------------------------------------------------------------------- #


def test_current_fixture_round_trips() -> None:
    """The current-version fixture loads and re-serializes without loss."""
    schema = load_artifact_json(_FIXTURES / "v_current_sample.json")
    assert schema.report, "fixture must carry at least one report row"
    # Re-dump and reload to prove the round trip is stable.
    reloaded = ArtifactSchema.model_validate_json(schema.model_dump_json())
    assert reloaded.schema_version == schema.schema_version
    assert len(reloaded.report) == len(schema.report)


def test_v1_0_0_fixture_loads_under_current_models() -> None:
    """A genuine 1.0.0-shaped payload (no ``pareto_k``) still loads (#212)."""
    path = _FIXTURES / "v1_0_0_sample.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Sanity-check the fixture really is the older shape.
    assert raw["schema_version"] == "1.0.0"
    assert "pareto_k" not in raw["report"][0]

    schema = load_artifact_json(path)
    assert schema.schema_version == "1.0.0"
    # The 1.1.0 field defaults to None rather than raising on the missing key.
    assert schema.report[0].pareto_k is None
    # Estimand/baseline blocks (added after 1.0.0) default cleanly too.
    assert schema.estimand_tex is None
    assert schema.baseline_kind is None


def test_v1_0_0_fixture_is_forward_compatible_with_extra_fields() -> None:
    """Unknown future report columns are preserved via ``extra='allow'`` (#212)."""
    raw = json.loads((_FIXTURES / "v1_0_0_sample.json").read_text(encoding="utf-8"))
    raw["report"][0]["some_future_metric"] = 1.23
    schema = ArtifactSchema.model_validate(raw)
    dumped = schema.model_dump()
    assert dumped["report"][0]["some_future_metric"] == 1.23
