"""Input adapters that map external data into the skdr-eval log schema (#149).

Today this hosts the generic decision-trace adapter, which maps
``(context, action, reward[, timestamp, propensity])`` records — including
JSONL agent traces — into the canonical single-action logs frame consumed by
:func:`skdr_eval.evaluate_sklearn_models`.

Public API:

* :func:`from_records` — adapt an in-memory iterable of decision records.
* :func:`from_jsonl_trace` — adapt a JSONL trace file (one record per line).
* :class:`TraceAdapterResult` — the adapted logs plus mapping metadata.
"""

from __future__ import annotations

from .trace import TraceAdapterResult, from_jsonl_trace, from_records

__all__ = ["TraceAdapterResult", "from_jsonl_trace", "from_records"]
