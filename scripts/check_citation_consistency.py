#!/usr/bin/env python3
"""Citation/version consistency check (#242).

Verify that the version, DOI and repository slug agree across every place the
project advertises its citation metadata:

* ``CITATION.cff``      — machine-readable citation (CFF)
* ``CITATION.bib``      — BibTeX companion
* ``.zenodo.json``      — Zenodo archive metadata
* the ``## Citation`` BibTeX block in ``README.md``

A release that bumps the version (or mints a DOI) in one file but not the others
produces the stale-citation drift seen in #110 (wrong repo owner) and #111
(stale README version). This check fails loudly on any such mismatch so it can
gate CI cheaply.

Run directly (``python scripts/check_citation_consistency.py`` /
``make citation-check``) or via the pytest guard in
``tests/test_citation_consistency.py`` so it also runs on the full CI matrix.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SLUG = "dgenio/skdr-eval"

_BIB_VERSION_RE = re.compile(r"version\s*=\s*\{([^}]*)\}")
_BIB_DOI_RE = re.compile(r"doi\s*=\s*\{([^}]*)\}")
_GITHUB_SLUG_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+?)(?:\.git)?(?:[)\s/\"}]|$)")
# Only the project's own @software{...} entry carries the version/DOI we sync;
# the foundational-method references below it have their own (unrelated) DOIs.
_SOFTWARE_ENTRY_RE = re.compile(r"@software\{.*?\n\}", re.DOTALL)
# The README citation lives in a fenced ```bibtex block under "## Citation".
_README_BIBTEX_RE = re.compile(
    r"##\s*Citation.*?```bibtex\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _software_entry(bibtex: str) -> str:
    """Return just the ``@software{...}`` entry from a BibTeX string.

    Falls back to the whole string if no such entry is found, so a malformed
    file surfaces as a version/DOI mismatch rather than silently passing.
    """
    match = _SOFTWARE_ENTRY_RE.search(bibtex)
    return match.group(0) if match else bibtex


def _slugs(text: str) -> set[str]:
    """All ``owner/repo`` GitHub slugs referenced in ``text``."""
    return set(_GITHUB_SLUG_RE.findall(text))


def check(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a list of human-readable consistency errors (empty == OK)."""
    errors: list[str] = []

    cff_path = repo_root / "CITATION.cff"
    bib_path = repo_root / "CITATION.bib"
    zenodo_path = repo_root / ".zenodo.json"
    readme_path = repo_root / "README.md"

    for p in (cff_path, bib_path, zenodo_path, readme_path):
        if not p.exists():
            errors.append(f"Missing citation source: {p.relative_to(repo_root)}")
    if errors:
        return errors

    cff = yaml.safe_load(cff_path.read_text(encoding="utf-8"))
    bib_text = bib_path.read_text(encoding="utf-8")
    zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
    readme_text = readme_path.read_text(encoding="utf-8")

    # --- Version agreement: CITATION.cff vs CITATION.bib vs README block ----- #
    bib_software = _software_entry(bib_text)
    cff_version = str(cff.get("version", "")).strip()
    bib_versions = {v.strip() for v in _BIB_VERSION_RE.findall(bib_software)}

    readme_match = _README_BIBTEX_RE.search(readme_text)
    if readme_match is None:
        errors.append(
            "Could not find a ```bibtex block under '## Citation' in README.md."
        )
        readme_versions: set[str] = set()
    else:
        readme_versions = {
            v.strip() for v in _BIB_VERSION_RE.findall(readme_match.group(1))
        }

    if not cff_version:
        errors.append("CITATION.cff is missing a top-level 'version'.")

    # The @software BibTeX entry and the README citation block must each carry a
    # version that matches the CFF version. A *missing* version there is an error
    # too (not silently OK), or the guard would pass while a source has drifted
    # out of having any version at all. Foundational-reference entries have no
    # version field — that is why we only look inside the @software entry above.
    if not bib_versions:
        errors.append("CITATION.bib @software entry has no 'version = {...}'.")
    elif cff_version and bib_versions != {cff_version}:
        errors.append(
            f"Version mismatch: CITATION.cff has {cff_version!r} but "
            f"CITATION.bib has {sorted(bib_versions)!r}."
        )

    # readme_match is None only when the citation block itself is missing, which
    # is already reported above; otherwise the block must carry a version.
    if readme_match is not None:
        if not readme_versions:
            errors.append(
                "README.md citation block has no 'version = {...}' to check "
                "against CITATION.cff."
            )
        elif cff_version and readme_versions != {cff_version}:
            errors.append(
                f"Version mismatch: CITATION.cff has {cff_version!r} but "
                f"README.md has {sorted(readme_versions)!r}."
            )

    # --- DOI agreement: CITATION.cff vs CITATION.bib ------------------------- #
    cff_dois = {
        str(idf.get("value", "")).strip()
        for idf in cff.get("identifiers", [])
        if idf.get("type") == "doi"
    }
    bib_dois = {d.strip() for d in _BIB_DOI_RE.findall(bib_software)}
    if cff_dois and not bib_dois:
        errors.append(
            "CITATION.bib @software entry is missing the DOI declared in "
            f"CITATION.cff ({sorted(cff_dois)!r})."
        )
    elif cff_dois and bib_dois and cff_dois != bib_dois:
        errors.append(
            f"DOI mismatch: CITATION.cff has {sorted(cff_dois)!r} but "
            f"CITATION.bib has {sorted(bib_dois)!r}."
        )

    # --- Repository slug agreement across all four sources (#110) ------------ #
    sources = {
        "CITATION.cff": " ".join(
            str(cff.get(k, "")) for k in ("url", "repository-code")
        ),
        "CITATION.bib": bib_text,
        ".zenodo.json": json.dumps(zenodo),
        "README.md (citation block)": readme_match.group(1) if readme_match else "",
    }
    for label, text in sources.items():
        slugs = _slugs(text)
        if slugs and REPO_SLUG not in slugs:
            errors.append(
                f"Repository slug mismatch in {label}: expected "
                f"{REPO_SLUG!r}, found {sorted(slugs)!r}."
            )

    return errors


def main() -> int:
    """CLI entry point: print findings and return a process exit code."""
    errors = check()
    if errors:
        print("Citation consistency check FAILED:")
        for err in errors:
            print(f"  - {err}")
        print(
            "\nKeep the version/DOI/URL in sync across CITATION.cff, "
            "CITATION.bib, .zenodo.json and the README citation block."
        )
        return 1
    print("Citation consistency check passed (version, DOI and repo slug agree).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
