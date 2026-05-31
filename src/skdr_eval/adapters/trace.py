"""Generic decision-trace → OPE-log adapter (#149).

Maps generic ``(context, action, reward[, timestamp, propensity])`` decision
traces — e.g. JSONL agent traces — into the canonical single-action logs schema
consumed by :func:`skdr_eval.evaluate_sklearn_models` and validated by
:func:`skdr_eval.validate_logs`.

This lowers the on-ramp for *every* user with decision logs in a non-skdr
format: instead of hand-shaping a DataFrame with ``cli_*`` feature columns,
``*_elig`` eligibility columns, an ``action`` column and a reward column, you
describe where those fields live in your records and the adapter builds a
schema-valid frame.

Propensity handling
-------------------
skdr-eval estimates (calibrated) logging propensities internally from the logged
actions and features, so a logged ``propensity`` value in the trace is
*informational*: it is surfaced via
:attr:`TraceAdapterResult.had_logged_propensities` and a one-time warning, not
consumed by the DR/SNDR estimators. The no-logged-propensity case is the default
and requires nothing extra.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError, InsufficientDataError
from ..validation import validate_logs

logger = logging.getLogger("skdr_eval")

__all__ = ["TraceAdapterResult", "from_jsonl_trace", "from_records"]

# Sentinel distinguishing "key absent" from "value is None".
_MISSING = object()


@dataclass(frozen=True)
class TraceAdapterResult:
    """Result of adapting a decision trace into OPE logs.

    Attributes
    ----------
    logs : pd.DataFrame
        Schema-valid logs frame (passes :func:`skdr_eval.validate_logs` with
        ``y_col=reward_col``). Feed this straight to
        :func:`skdr_eval.evaluate_sklearn_models`.
    actions : list[str]
        Sorted action vocabulary (the eligible-operator universe).
    feature_names : list[str]
        Context-feature column names emitted into ``logs`` (all carry
        ``feature_prefix``).
    reward_col : str
        Name of the reward/outcome column in ``logs``. Pass this as ``y_col``
        to :func:`skdr_eval.evaluate_sklearn_models`.
    n_records : int
        Number of trace records mapped.
    had_logged_propensities : bool
        True when at least one record carried a logged propensity. Logged
        propensities are *not* consumed by the estimators (skdr-eval calibrates
        its own); this flag lets callers note the gap.
    synthesized_timestamps : bool
        True when no usable timestamp was found and a monotonic integer order
        was synthesized into ``arrival_ts``.
    """

    logs: pd.DataFrame
    actions: list[str]
    feature_names: list[str]
    reward_col: str
    n_records: int
    had_logged_propensities: bool
    synthesized_timestamps: bool

    def summary(self) -> str:
        """One-line human-readable summary of the adaptation."""
        ts = "synthesized" if self.synthesized_timestamps else "from trace"
        prop = "present (re-estimated)" if self.had_logged_propensities else "absent"
        return (
            f"{self.n_records} records -> logs[{len(self.logs.columns)} cols]; "
            f"{len(self.actions)} actions; {len(self.feature_names)} features; "
            f"timestamps {ts}; logged propensities {prop}"
        )


def _context_to_features(
    context: Any,
    feature_keys: list[str] | None,
    feature_len: int | None,
    record_index: int,
) -> list[float]:
    """Extract a numeric feature vector from one record's context.

    ``feature_keys``/``feature_len`` encode the schema fixed by the first
    record; subsequent records must match it. Raises ``DataValidationError`` on
    any mismatch or non-numeric value, naming the offending record.
    """
    if isinstance(context, Mapping):
        if feature_keys is None:  # pragma: no cover - schema fixed by caller
            raise DataValidationError("context schema not initialised")
        if set(context.keys()) != set(feature_keys):
            raise DataValidationError(
                f"record {record_index}: context keys {sorted(context.keys())} "
                f"differ from the first record's keys {sorted(feature_keys)}"
            )
        values = [context[k] for k in feature_keys]
    elif isinstance(context, list | tuple | np.ndarray):
        seq = list(context)
        if feature_len is not None and len(seq) != feature_len:
            raise DataValidationError(
                f"record {record_index}: context length {len(seq)} differs from "
                f"the first record's length {feature_len}"
            )
        values = seq
    else:
        raise DataValidationError(
            f"record {record_index}: context must be a mapping or a sequence of "
            f"numeric features, got {type(context).__name__}"
        )

    out: list[float] = []
    for j, v in enumerate(values):
        try:
            out.append(float(v))
        except (TypeError, ValueError) as exc:
            label = feature_keys[j] if feature_keys is not None else f"index {j}"
            raise DataValidationError(
                f"record {record_index}: context feature {label!r} is not "
                f"numeric (got {v!r})"
            ) from exc
    return out


def from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    context_key: str = "context",
    action_key: str = "action",
    reward_key: str = "reward",
    timestamp_key: str | None = "timestamp",
    propensity_key: str | None = "propensity",
    eligible_actions_key: str | None = "eligible_actions",
    feature_prefix: str = "cli_",
    reward_col: str = "reward",
    validate: bool = True,
) -> TraceAdapterResult:
    """Map generic decision-trace records into a schema-valid logs frame.

    Each record is a mapping carrying at least a context, an action and a
    reward. Context may be a mapping of named numeric features
    (``{"tokens": 812, "priority": 1}``) or a flat numeric sequence; either way
    it becomes ``feature_prefix``-prefixed columns. The chosen ``action`` and
    the (optional) per-record ``eligible_actions`` define the eligible-operator
    universe and the ``*_elig`` columns.

    Parameters
    ----------
    records : iterable of mapping
        The decision trace. Materialised once; must be non-empty.
    context_key, action_key, reward_key : str
        Record keys for the context, chosen action and observed reward.
    timestamp_key : str or None, default ``"timestamp"``
        Record key for a timestamp (numeric or anything ``pd.to_datetime`` can
        parse). When ``None``, or when any record lacks a usable value, a
        monotonic integer order is synthesized into ``arrival_ts``.
    propensity_key : str or None, default ``"propensity"``
        Record key for a logged propensity. Logged propensities are *not* used
        by the estimators (skdr-eval calibrates its own); presence is reported
        via :attr:`TraceAdapterResult.had_logged_propensities`.
    eligible_actions_key : str or None, default ``"eligible_actions"``
        Record key for the per-row list of eligible actions. When absent for a
        row, every action in the global vocabulary is treated as eligible there.
    feature_prefix : str, default ``"cli_"``
        Prefix for emitted feature columns. Keep ``"cli_"`` or ``"st_"`` so the
        downstream design matrix (:func:`skdr_eval.build_design`) recognises
        them.
    reward_col : str, default ``"reward"``
        Name of the emitted reward column; pass as ``y_col`` downstream.
    validate : bool, default True
        Run :func:`skdr_eval.validate_logs` (with ``cli_pref=feature_prefix``,
        ``y_col=reward_col``) on the result.

    Returns
    -------
    TraceAdapterResult

    Raises
    ------
    InsufficientDataError
        If ``records`` is empty.
    DataValidationError
        On a missing key, a non-numeric/inconsistent context, or a chosen
        action that is not in its own row's eligible set.
    """
    rows = list(records)
    if not rows:
        raise InsufficientDataError("trace has no records")

    first_ctx = rows[0].get(context_key, _MISSING)
    if first_ctx is _MISSING:
        raise DataValidationError(f"record 0: missing context key {context_key!r}")
    feature_keys: list[str] | None = None
    feature_len: int | None = None
    if isinstance(first_ctx, Mapping):
        feature_keys = list(first_ctx.keys())
        feature_names = [f"{feature_prefix}{k}" for k in feature_keys]
    elif isinstance(first_ctx, list | tuple | np.ndarray):
        feature_len = len(list(first_ctx))
        feature_names = [f"{feature_prefix}{i}" for i in range(feature_len)]
    else:
        raise DataValidationError(
            f"record 0: context must be a mapping or a sequence of numeric "
            f"features, got {type(first_ctx).__name__}"
        )
    if not feature_names:
        raise DataValidationError("record 0: context has no features")

    feature_matrix: list[list[float]] = []
    actions: list[str] = []
    rewards: list[float] = []
    raw_timestamps: list[Any] = []
    eligible_per_row: list[list[str] | None] = []
    had_propensities = False
    have_all_timestamps = timestamp_key is not None

    for i, rec in enumerate(rows):
        ctx = rec.get(context_key, _MISSING)
        if ctx is _MISSING:
            raise DataValidationError(
                f"record {i}: missing context key {context_key!r}"
            )
        feature_matrix.append(_context_to_features(ctx, feature_keys, feature_len, i))

        if action_key not in rec:
            raise DataValidationError(f"record {i}: missing action key {action_key!r}")
        actions.append(str(rec[action_key]))

        if reward_key not in rec:
            raise DataValidationError(f"record {i}: missing reward key {reward_key!r}")
        try:
            rewards.append(float(rec[reward_key]))
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"record {i}: reward {rec[reward_key]!r} is not numeric"
            ) from exc

        if propensity_key is not None and rec.get(propensity_key) is not None:
            had_propensities = True

        if timestamp_key is not None:
            ts = rec.get(timestamp_key)
            raw_timestamps.append(ts)
            if ts is None:
                have_all_timestamps = False

        if (
            eligible_actions_key is not None
            and rec.get(eligible_actions_key) is not None
        ):
            eligible_per_row.append([str(a) for a in rec[eligible_actions_key]])
        else:
            eligible_per_row.append(None)

    n = len(rows)

    # Action vocabulary: observed actions plus any declared eligible actions.
    vocab: set[str] = set(actions)
    for elig in eligible_per_row:
        if elig is not None:
            vocab.update(elig)
    action_vocab = sorted(vocab)

    # Eligibility matrix: declared set per row, else all-eligible. The chosen
    # action must be eligible on its own row (validate_logs enforces this too,
    # but we raise here with the originating record index).
    elig_matrix = np.zeros((n, len(action_vocab)), dtype=np.int8)
    col_of = {a: j for j, a in enumerate(action_vocab)}
    for i, elig in enumerate(eligible_per_row):
        allowed = elig if elig is not None else action_vocab
        for a in allowed:
            elig_matrix[i, col_of[a]] = 1
        if elig is not None and actions[i] not in elig:
            raise DataValidationError(
                f"record {i}: chosen action {actions[i]!r} is not in its "
                f"eligible set {sorted(elig)}"
            )

    features = np.asarray(feature_matrix, dtype=float)
    if not np.all(np.isfinite(features)):
        bad_col = int(np.argmax(~np.isfinite(features).all(axis=0)))
        raise DataValidationError(
            f"context feature {feature_names[bad_col]!r} contains non-finite values"
        )

    # Timestamps: parse when present for every row, else synthesize order.
    synthesized = True
    arrival_ts: np.ndarray | pd.Series
    if have_all_timestamps and raw_timestamps:
        if all(
            isinstance(t, (int, float)) and not isinstance(t, bool)
            for t in raw_timestamps
        ):
            arrival_ts = np.asarray(raw_timestamps, dtype=float)
            synthesized = False
        else:
            parsed = pd.to_datetime(pd.Series(raw_timestamps), errors="coerce")
            if parsed.isna().any():
                raise DataValidationError(
                    f"trace {timestamp_key!r} values could not all be parsed as "
                    "timestamps; drop the key to synthesize order instead"
                )
            arrival_ts = parsed
            synthesized = False
    else:
        arrival_ts = np.arange(n, dtype=np.int64)

    data: dict[str, Any] = {"arrival_ts": arrival_ts}
    for j, name in enumerate(feature_names):
        data[name] = features[:, j]
    for a in action_vocab:
        data[f"{a}_elig"] = elig_matrix[:, col_of[a]]
    data["action"] = actions
    data[reward_col] = rewards
    logs = pd.DataFrame(data)

    if had_propensities:
        logger.warning(
            "trace carried logged propensities under %r; skdr-eval estimates "
            "calibrated propensities internally, so the logged values are "
            "informational and not consumed by the DR/SNDR estimators.",
            propensity_key,
        )

    if validate:
        validate_logs(logs, cli_pref=feature_prefix, y_col=reward_col)

    return TraceAdapterResult(
        logs=logs,
        actions=action_vocab,
        feature_names=feature_names,
        reward_col=reward_col,
        n_records=n,
        had_logged_propensities=had_propensities,
        synthesized_timestamps=synthesized,
    )


def from_jsonl_trace(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    context_key: str = "context",
    action_key: str = "action",
    reward_key: str = "reward",
    timestamp_key: str | None = "timestamp",
    propensity_key: str | None = "propensity",
    eligible_actions_key: str | None = "eligible_actions",
    feature_prefix: str = "cli_",
    reward_col: str = "reward",
    validate: bool = True,
) -> TraceAdapterResult:
    """Read a JSONL decision trace and adapt it via :func:`from_records`.

    Each non-blank line must be a single JSON object (one decision record).
    See :func:`from_records` for the mapping keyword arguments.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.jsonl`` file (one JSON object per line).
    encoding : str, default ``"utf-8"``
        File encoding.

    Returns
    -------
    TraceAdapterResult

    Raises
    ------
    InsufficientDataError
        If the file holds no records.
    DataValidationError
        If a line is not valid JSON, or on any :func:`from_records` error.
    """
    records: list[Mapping[str, Any]] = []
    with Path(path).open(encoding=encoding) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{path}:{lineno}: not valid JSON ({exc})"
                ) from exc
            if not isinstance(obj, Mapping):
                raise DataValidationError(
                    f"{path}:{lineno}: each line must be a JSON object, got "
                    f"{type(obj).__name__}"
                )
            records.append(obj)
    return from_records(
        records,
        context_key=context_key,
        action_key=action_key,
        reward_key=reward_key,
        timestamp_key=timestamp_key,
        propensity_key=propensity_key,
        eligible_actions_key=eligible_actions_key,
        feature_prefix=feature_prefix,
        reward_col=reward_col,
        validate=validate,
    )
