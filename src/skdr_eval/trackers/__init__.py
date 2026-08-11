"""Tracker protocol and working built-in trackers (#93 / #217).

The :class:`Tracker` protocol defines the minimum surface evaluators use for
optional result logging. The core-supported implementations are deliberately
small:

- :class:`NullTracker` — no-op default; used when ``tracker=None``.
- :class:`FileTracker` — writes JSONL metrics and artifact files to a run
  directory.

Historical MLflow / W&B / Aim modules exposed importable classes that always
raised ``NotImplementedError``. They are not working integrations and are no
longer advertised as capabilities. Their compatibility modules are being retired
under #217; users can implement the protocol directly if an external tracker is
needed before real demand justifies an official adapter.

This module has zero new mandatory dependencies. ``FileTracker`` uses only the
standard library and the existing ``pyyaml`` dependency already shipped in core.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType

    from skdr_eval.reporting import EvaluationCard

logger = logging.getLogger("skdr_eval")


@runtime_checkable
class Tracker(Protocol):
    """Minimum result-logging surface used by ``evaluate_*_models``.

    Implementations must be safe to use as context managers. ``NullTracker``
    and ``FileTracker`` are the supported built-ins; callers may provide their
    own implementation for third-party systems.
    """

    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...

    def log_card(self, card: EvaluationCard) -> None: ...

    def set_tag(self, key: str, value: str) -> None: ...

    def __enter__(self) -> Tracker: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class NullTracker:
    """A no-op tracker. Default when ``tracker=None``.

    Every method is a true no-op — calling any method (in any order) is
    guaranteed to have zero observable side effects on the filesystem or
    process state. The class is included in the public API so callers can
    spell their intent explicitly.
    """

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        del name, value, step

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        del path, artifact_path

    def log_card(self, card: EvaluationCard) -> None:
        del card

    def set_tag(self, key: str, value: str) -> None:
        del key, value

    def __enter__(self) -> NullTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


class FileTracker:
    """Built-in disk-based tracker.

    Writes a run directory containing:

    - ``metrics.jsonl`` — one JSON object per ``log_metric`` call;
    - ``tags.json`` — flat dict of tags;
    - ``artifacts/`` — files copied from ``log_artifact``;
    - ``cards/<model_name>_<estimator>.card.yaml`` — YAML evaluation cards.

    Parameters
    ----------
    root : str or Path
        Output directory. Created if it does not exist.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        (self.root / "cards").mkdir(exist_ok=True)
        self._metrics_path = self.root / "metrics.jsonl"
        self._tags_path = self.root / "tags.json"
        self._tags: dict[str, str] = {}
        if self._tags_path.exists():
            try:
                self._tags = json.loads(self._tags_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._tags = {}

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": str(name),
            "value": float(value),
        }
        if step is not None:
            record["step"] = int(step)
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {src}")
        sub = artifact_path or src.name
        artifacts_dir = (self.root / "artifacts").resolve()
        dest = (artifacts_dir / sub).resolve()
        try:
            dest.relative_to(artifacts_dir)
        except ValueError:
            raise ValueError(
                f"artifact_path {artifact_path!r} resolves outside the "
                "artifacts directory. Only relative, non-traversing paths "
                "are allowed."
            ) from None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

    def log_card(self, card: EvaluationCard) -> None:
        estimator = card.headline.estimator or "card"
        dest = self.root / "cards" / f"{card.model_name}_{estimator}.card.yaml"
        card.to_yaml(dest)

    def set_tag(self, key: str, value: str) -> None:
        self._tags[str(key)] = str(value)
        self._tags_path.write_text(
            json.dumps(self._tags, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def __enter__(self) -> FileTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


__all__ = ["FileTracker", "NullTracker", "Tracker"]
