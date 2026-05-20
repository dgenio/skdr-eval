"""Tracker protocol and built-in trackers (#93).

The :class:`Tracker` protocol defines the minimum surface
(``log_metric`` / ``log_artifact`` / ``log_card`` / ``set_tag`` plus context
management) that evaluators use to push results to disk or an external
experiment tracker. The core ships:

- :class:`NullTracker` — no-op default; used when ``tracker=None``.
- :class:`FileTracker` — writes JSONL metrics and artifact files to a run
  directory.

External adapters (MLflow / W&B / Aim) live in this package as separate
modules gated behind their own optional extras (``[mlflow]``, ``[wandb]``,
``[aim]``). They currently raise :class:`NotImplementedError` on
construction — the umbrella issue #73 tracks each adapter's full
implementation as a follow-up PR.

This module has zero new mandatory dependencies. The ``FileTracker`` uses
only the standard library and the existing ``pyyaml`` dep already shipped
in core.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType

    from skdr_eval.reporting import EvaluationCard

logger = logging.getLogger("skdr_eval")


@runtime_checkable
class Tracker(Protocol):
    """Minimum surface for experiment trackers used by ``evaluate_*_models``.

    Implementations must be safe to use as context managers. The
    :class:`NullTracker` and :class:`FileTracker` ship in core; external
    adapters (MLflow / W&B / Aim) live in sibling modules behind optional
    extras.
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
    spell their intent explicitly:

    >>> from skdr_eval.trackers import NullTracker
    >>> tracker = NullTracker()
    >>> tracker.log_metric("V_hat", 1.23)  # no-op
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

    - ``metrics.jsonl`` — one JSON object per ``log_metric`` call,
      append-only, deterministic.
    - ``tags.json`` — flat dict of tags (written on every ``set_tag`` call).
    - ``artifacts/`` — files copied from ``log_artifact`` (or sub-paths
      under it when ``artifact_path`` is provided).
    - ``cards/<model_name>.card.yaml`` — YAML dump of each ``log_card``
      payload.

    Parameters
    ----------
    root : str or Path
        Output directory. Created if it does not exist. Each instance writes
        all its records into this single directory; reuse a path across runs
        only if you want the metrics to be concatenated.
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
            "ts": datetime.now(UTC).isoformat(),
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
        dest = self.root / "artifacts" / sub
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


__all__ = [
    "FileTracker",
    "NullTracker",
    "Tracker",
]
