#!/usr/bin/env python3
"""Generate the published JSON Schema files under ``docs/schemas/`` (#205).

The committed schemas are the downloadable, versioned contract downstream
tooling uses to validate ``skdr-eval`` outputs (artifact JSON and card JSON)
without importing the library. Run this after any change to
:class:`skdr_eval.reporting.ArtifactSchema` or
:class:`skdr_eval.reporting.EvaluationCard`; a test
(``tests/test_schema_publishing.py``) fails if the committed files drift from
``json_schema()``.

Usage::

    python scripts/generate_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

from skdr_eval.reporting import ArtifactSchema

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "docs" / "schemas"

# Map of committed filename -> schema producer. Keep in sync with
# tests/test_schema_publishing.py::SCHEMA_FILES.
#
# Only the artifact payload schema is published. The card schema (which encodes
# the deploy/recommendation verdict) is deferred until the verdict contract
# stabilizes after the July 2026 experiment-eligibility audit.
SCHEMAS = {
    "artifact.schema.json": ArtifactSchema.json_schema,
}


def render(producer: object) -> str:
    """Return the deterministic, newline-terminated JSON text for a schema."""
    schema = producer()  # type: ignore[operator]
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, producer in SCHEMAS.items():
        path = SCHEMA_DIR / filename
        path.write_text(render(producer), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
