"""Input adapters that map external data into the skdr-eval log schema (#149).

Today this hosts the generic decision-trace adapter, which maps
``(context, action, reward[, timestamp, propensity])`` records — including
JSONL agent traces — into the canonical single-action logs frame consumed by
:func:`skdr_eval.evaluate_sklearn_models`.

It also hosts adapters that broaden the *model* side of the API (#71): thin
wrappers for non-sklearn outcome / surrogate models (XGBoost / LightGBM /
CatBoost) and a generic callable adapter.

Public API:

* :func:`from_records` — adapt an in-memory iterable of decision records.
* :func:`from_jsonl_trace` — adapt a JSONL trace file (one record per line).
* :class:`TraceAdapterResult` — the adapted logs plus mapping metadata.
* :class:`XGBRegressorAdapter`, :class:`LGBMRegressorAdapter`,
  :class:`CatBoostRegressorAdapter` — GBDT outcome/surrogate adapters
  (``pip install 'skdr-eval[boosting]'``).
* :class:`CallableModelAdapter` — wrap a plain ``predict`` function.
"""

from __future__ import annotations

from .boosting import (
    CallableModelAdapter,
    CatBoostRegressorAdapter,
    LGBMRegressorAdapter,
    XGBRegressorAdapter,
)
from .trace import TraceAdapterResult, from_jsonl_trace, from_records

__all__ = [
    "CallableModelAdapter",
    "CatBoostRegressorAdapter",
    "LGBMRegressorAdapter",
    "TraceAdapterResult",
    "XGBRegressorAdapter",
    "from_jsonl_trace",
    "from_records",
]
