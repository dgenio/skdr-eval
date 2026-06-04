"""Cache + download plumbing for the public-dataset loaders (#70).

Loaders download their source files once into a per-user cache and reuse them
on subsequent calls. Integrity is recorded in a small ``manifest.json`` next
to the files (sha256 + size + source + license), so a cached dataset is
reproducible and verifiable across runs.

No new top-level dependency: downloads use :mod:`urllib.request` from the
standard library, and a ``source`` may equally be a local filesystem path
(copied instead of downloaded), which is what the offline tests exercise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..exceptions import DatasetError

logger = logging.getLogger("skdr_eval")

__all__ = [
    "default_cache_dir",
    "fetch_file",
    "read_manifest",
    "sha256_of",
    "write_manifest",
]

# A 16 KiB read block keeps memory flat while hashing / streaming downloads.
_BLOCK = 1 << 14


def default_cache_dir() -> Path:
    """Return the root cache directory for downloaded datasets.

    Honors the ``SKDR_EVAL_CACHE_DIR`` environment variable; otherwise defaults
    to ``~/.skdr_eval/datasets``. The directory is *not* created here — callers
    create the dataset-specific subdirectory when they actually write to it.
    """
    env = os.environ.get("SKDR_EVAL_CACHE_DIR")
    root = Path(env) if env else Path.home() / ".skdr_eval"
    return root / "datasets"


def sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of ``path``, read in fixed-size blocks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def _is_url(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _check_disk_space(dest_dir: Path, min_free_bytes: int) -> None:
    """Raise :class:`DatasetError` when free space is below ``min_free_bytes``."""
    probe = dest_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:  # pragma: no cover - disk_usage rarely fails
        return
    if free < min_free_bytes:
        raise DatasetError(
            "Insufficient disk space to download dataset",
            details={
                "required_bytes": min_free_bytes,
                "free_bytes": free,
                "path": str(probe),
            },
        )


def fetch_file(
    source: str,
    dest: Path,
    *,
    min_free_bytes: int = 16 * 1024 * 1024,
    force: bool = False,
) -> Path:
    """Fetch ``source`` to ``dest`` (cached), returning ``dest``.

    ``source`` may be an ``http(s)`` URL (downloaded via urllib) or a local
    filesystem path (copied). When ``dest`` already exists and ``force`` is
    False, the cached copy is returned without re-fetching.

    Raises
    ------
    DatasetError
        On network failure, a missing local source, or insufficient disk
        space — each with an actionable message.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        logger.debug("Using cached dataset file: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_space(dest.parent, min_free_bytes)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if _is_url(source):
        try:
            with urllib.request.urlopen(source) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=_BLOCK)
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise DatasetError(
                "Failed to download dataset file; check your network "
                "connection, or pass a local 'base_url' pointing at an "
                "already-downloaded copy",
                details={"source": source, "error": str(exc)},
            ) from exc
    else:
        src_path = Path(source)
        if not src_path.exists():
            raise DatasetError(
                "Local dataset source does not exist",
                details={"source": str(src_path)},
            )
        shutil.copyfile(src_path, tmp)

    tmp.replace(dest)
    return dest


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a manifest JSON, returning ``{}`` when it is absent or unreadable."""
    if not manifest_path.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def write_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``manifest_path`` as stable, sorted JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
