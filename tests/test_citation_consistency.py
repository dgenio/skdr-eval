"""Citation/version consistency guard (#242).

Runs the same check as ``scripts/check_citation_consistency.py`` /
``make citation-check`` inside pytest, so the version, DOI and repository slug
are verified to agree across ``CITATION.cff``, ``CITATION.bib``, ``.zenodo.json``
and the README citation block on the full CI matrix — not only when the
standalone script happens to be invoked.
"""

import importlib.util
import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_citation_consistency.py"


def _load_checker():
    """Import the standalone script as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("check_citation_consistency", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_citation_metadata_is_consistent() -> None:
    """Version, DOI and repo slug agree across all citation sources (#242)."""
    checker = _load_checker()
    errors = checker.check(_REPO_ROOT)
    assert not errors, "Citation metadata is inconsistent:\n" + "\n".join(
        f"  - {e}" for e in errors
    )


def test_checker_detects_version_drift(tmp_path: Path) -> None:
    """The guard actually fails when a source drifts — not a no-op (#242)."""
    checker = _load_checker()
    for name in ("CITATION.cff", "CITATION.bib", ".zenodo.json", "README.md"):
        (tmp_path / name).write_text(
            (_REPO_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    current = str(
        yaml.safe_load((tmp_path / "CITATION.cff").read_text(encoding="utf-8"))[
            "version"
        ]
    )
    readme = tmp_path / "README.md"
    original = readme.read_text(encoding="utf-8")
    tampered = original.replace(f"version = {{{current}}}", "version = {9.9.9}", 1)
    # The replacement must have taken effect, or the negative test is vacuous.
    assert tampered != original, f"README has no 'version = {{{current}}}' to tamper"
    readme.write_text(tampered, encoding="utf-8")

    errors = checker.check(tmp_path)
    assert any("Version mismatch" in e for e in errors), errors


def _copy_sources(tmp_path: Path) -> None:
    for name in ("CITATION.cff", "CITATION.bib", ".zenodo.json", "README.md"):
        (tmp_path / name).write_text(
            (_REPO_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8"
        )


def test_checker_flags_missing_readme_version(tmp_path: Path) -> None:
    """A README citation block with no version is an error, not silently OK (#242)."""
    checker = _load_checker()
    _copy_sources(tmp_path)
    current = str(
        yaml.safe_load((tmp_path / "CITATION.cff").read_text(encoding="utf-8"))[
            "version"
        ]
    )
    readme = tmp_path / "README.md"
    original = readme.read_text(encoding="utf-8")
    stripped = original.replace(f"  version = {{{current}}},\n", "", 1)
    assert stripped != original, "expected a README @software version line to drop"
    readme.write_text(stripped, encoding="utf-8")

    errors = checker.check(tmp_path)
    assert any("README.md citation block has no 'version" in e for e in errors), errors


def test_checker_flags_missing_bib_doi(tmp_path: Path) -> None:
    """If CITATION.cff declares a DOI, a bib @software without one is an error (#242)."""
    checker = _load_checker()
    _copy_sources(tmp_path)
    bib = tmp_path / "CITATION.bib"
    original = bib.read_text(encoding="utf-8")
    # Drop the DOI line from the @software entry only (it is the first doi = ...).
    stripped = re.sub(r"\n\s*doi\s*=\s*\{[^}]*\},", "", original, count=1)
    assert stripped != original, "expected a bib doi line to drop"
    bib.write_text(stripped, encoding="utf-8")

    errors = checker.check(tmp_path)
    assert any("missing the DOI declared in" in e for e in errors), errors
