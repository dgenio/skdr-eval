"""Open Bandit Dataset (OBD) loader (#70).

Loads the ZOZOTOWN Open Bandit Dataset into the canonical single-action logs
schema consumed by :func:`skdr_eval.evaluate_sklearn_models`, so the rest of
the library works unchanged. The dataset is published by ZOZO, Inc. under a
CC BY 4.0 license; see https://research.zozo.com/data.html and the paper
Saito et al., *Open Bandit Dataset and Pipeline* (arXiv:2008.07146).

The loader is deliberately thin (cf. the generic format adapter tracked by
#35): it downloads the ``{behavior_policy}/{campaign}.csv`` log file plus
``item_context.csv`` from a configurable ``base_url`` (default: the small
sample bundled in the zr-obp GitHub repository), caches them with a sha256
manifest, and maps the documented OBD columns onto the canonical schema:

==================  ===================================================
OBD column          Canonical mapping
==================  ===================================================
``user_feature_*``  ``cli_user_feature_*`` (label-encoded if categorical)
``position``        ``cli_position``
``item_id``         ``action`` as ``"item_<id>"``; one ``<action>_elig``
                    column per catalog item, all eligible
``click``           reward column (``y_col="click"``)
``timestamp``       ``arrival_ts``
==================  ===================================================

``base_url`` may be an ``http(s)`` URL prefix *or* a local directory, which is
how the offline tests and air-gapped users point the loader at a copy they
already have.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..exceptions import DatasetError
from ._cache import default_cache_dir, fetch_file, sha256_of, write_manifest

__all__ = ["DatasetBundle", "load_obd"]

# Default mirror: the small OBD sample committed to the zr-obp repository.
# The full 26M-row dataset lives at https://research.zozo.com/data.html; pass
# its location as ``base_url`` to load it instead.
_DEFAULT_BASE_URL = "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"
_OBD_LICENSE = "CC BY 4.0 (ZOZO, Inc.) — https://research.zozo.com/data.html"

_BEHAVIOR_POLICIES = ("random", "bts")
_CAMPAIGNS = ("all", "men", "women")


class DatasetBundle(NamedTuple):
    """A loaded dataset in the canonical ``make_synth_logs`` return shape.

    Unpacks as ``logs, ops_all, ground_truth`` so callers can use it exactly
    like :func:`skdr_eval.make_synth_logs`. ``ground_truth`` is ``None`` for
    logged real-world data (no on-policy oracle value is available).

    Attributes
    ----------
    logs : pd.DataFrame
        Schema-valid logs (passes :func:`skdr_eval.validate_logs` with the
        loader's reward column).
    ops_all : pd.Index
        The action universe (eligible-operator catalog).
    ground_truth : np.ndarray or None
        True per-(context, action) values when known; ``None`` otherwise.
    """

    logs: pd.DataFrame
    ops_all: pd.Index
    ground_truth: np.ndarray | None


def _resolve_source(base_url: str, rel: str) -> str:
    """Join ``base_url`` and ``rel`` for either a URL prefix or a local dir."""
    if base_url.startswith(("http://", "https://")):
        return f"{base_url.rstrip('/')}/{rel}"
    return str(Path(base_url) / rel)


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Label-encode any non-numeric feature columns to integer codes.

    OBD user features are anonymized category indices; some samples ship them
    as strings. ``evaluate_sklearn_models`` needs finite numeric features, so
    non-numeric columns are factorized deterministically (sorted categories).
    """
    out = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = series.to_numpy()
        else:
            as_str = series.astype("string")
            # Explicit sorted categories so the integer encoding is stable
            # regardless of row order (cache reproducibility / cross-run
            # comparability). NaNs map to code -1.
            categories = sorted(as_str.dropna().unique())
            cats = pd.Categorical(as_str, categories=categories)
            out[col] = cats.codes.astype(np.int64)
    return pd.DataFrame(out, index=frame.index)


def load_obd(
    behavior_policy: str = "random",
    campaign: str = "all",
    cache_dir: str | Path | None = None,
    *,
    base_url: str = _DEFAULT_BASE_URL,
    max_rows: int | None = None,
    force_download: bool = False,
) -> DatasetBundle:
    """Load the Open Bandit Dataset as canonical OPE logs.

    Parameters
    ----------
    behavior_policy : {"random", "bts"}, default="random"
        Logging policy whose interaction log to load.
    campaign : {"all", "men", "women"}, default="all"
        ZOZOTOWN fashion campaign.
    cache_dir : str or Path, optional
        Override the cache root (defaults to ``SKDR_EVAL_CACHE_DIR`` or
        ``~/.skdr_eval/datasets``).
    base_url : str, default=zr-obp sample mirror
        ``http(s)`` URL prefix *or* local directory containing
        ``{behavior_policy}/{campaign}.csv`` and ``item_context.csv``.
    max_rows : int, optional
        Truncate to the first ``max_rows`` decisions (useful for smoke runs).
    force_download : bool, default=False
        Re-fetch even when a cached copy exists.

    Returns
    -------
    DatasetBundle
        ``(logs, ops_all, ground_truth=None)``. The reward column is
        ``"click"`` — pass ``y_col="click"`` to ``evaluate_sklearn_models``.

    Raises
    ------
    DatasetError
        On an invalid ``behavior_policy`` / ``campaign``, network/disk failure,
        or a downloaded file failing its integrity check.
    """
    if behavior_policy not in _BEHAVIOR_POLICIES:
        raise DatasetError(
            f"Unknown behavior_policy {behavior_policy!r}",
            details={"allowed": list(_BEHAVIOR_POLICIES)},
        )
    if campaign not in _CAMPAIGNS:
        raise DatasetError(
            f"Unknown campaign {campaign!r}",
            details={"allowed": list(_CAMPAIGNS)},
        )

    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    dest_dir = root / "obd" / behavior_policy / campaign
    log_csv = dest_dir / f"{campaign}.csv"
    item_csv = dest_dir / "item_context.csv"

    log_src = _resolve_source(base_url, f"{behavior_policy}/{campaign}.csv")
    item_src = _resolve_source(base_url, f"{behavior_policy}/item_context.csv")

    fetch_file(log_src, log_csv, force=force_download)
    fetch_file(item_src, item_csv, force=force_download)

    write_manifest(
        dest_dir / "manifest.json",
        {
            "dataset": "open_bandit_dataset",
            "behavior_policy": behavior_policy,
            "campaign": campaign,
            "license": _OBD_LICENSE,
            "files": {
                log_csv.name: {
                    "sha256": sha256_of(log_csv),
                    "size_bytes": log_csv.stat().st_size,
                    "source": log_src,
                },
                item_csv.name: {
                    "sha256": sha256_of(item_csv),
                    "size_bytes": item_csv.stat().st_size,
                    "source": item_src,
                },
            },
        },
    )

    raw = pd.read_csv(log_csv)
    if max_rows is not None:
        raw = raw.head(max_rows)
    if raw.empty:
        raise DatasetError(
            "OBD log file is empty",
            details={"path": str(log_csv)},
        )

    items = pd.read_csv(item_csv)
    # Catalog universe: prefer the full item_context list so eligibility covers
    # every action, falling back to observed items if item_context lacks ids.
    if "item_id" in items.columns:
        catalog = sorted(int(i) for i in items["item_id"].unique())
    else:
        catalog = sorted(int(i) for i in raw["item_id"].unique())

    # Fail loud if a logged action references an item missing from the catalog:
    # eligibility columns are emitted per catalog item, so an out-of-catalog
    # ``item_id`` would yield an ``action`` with no matching ``<action>_elig``
    # column and be silently unrepresented downstream.
    logged_items = {int(i) for i in raw["item_id"].unique()}
    missing = sorted(logged_items - set(catalog))
    if missing:
        raise DatasetError(
            "OBD log references item_ids absent from the item_context catalog",
            details={"missing_item_ids": missing[:20], "n_missing": len(missing)},
        )

    n = len(raw)
    out: dict[str, object] = {}

    # Timestamp → arrival_ts (sorted-stable; OBD ships time-ordered).
    if "timestamp" in raw.columns:
        out["arrival_ts"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    else:
        out["arrival_ts"] = pd.date_range("2020-01-01", periods=n, freq="s")

    # Context features: user_feature_* → cli_user_feature_*, plus position.
    user_cols = [c for c in raw.columns if "user_feature" in c]
    if not user_cols:
        raise DatasetError(
            "OBD log file has no 'user_feature' columns; is this an OBD CSV?",
            details={"columns": list(raw.columns)},
        )
    encoded = _encode_features(raw[user_cols])
    for col in user_cols:
        out[f"cli_{col}"] = encoded[col].to_numpy()
    if "position" in raw.columns:
        out["cli_position"] = pd.to_numeric(raw["position"], errors="coerce").to_numpy()

    # Action + per-catalog-item eligibility (full catalog eligible everywhere).
    action = "item_" + raw["item_id"].astype(int).astype(str)
    out["action"] = action.to_numpy()
    for item in catalog:
        out[f"item_{item}_elig"] = np.ones(n, dtype=bool)

    # Reward.
    if "click" not in raw.columns:
        raise DatasetError(
            "OBD log file has no 'click' reward column",
            details={"columns": list(raw.columns)},
        )
    out["click"] = pd.to_numeric(raw["click"], errors="coerce").astype(float).to_numpy()

    logs = pd.DataFrame(out)
    # Drop rows with unparseable timestamps/features rather than emit NaNs that
    # would trip the design builder downstream. This covers the timestamp, the
    # reward, and every numeric feature column (``cli_position`` and the
    # ``cli_user_feature_*`` set), any of which can pick up NaNs from a
    # ``to_numeric(..., errors="coerce")`` on a malformed source row.
    feature_cols = [c for c in logs.columns if c.startswith("cli_")]
    logs = logs.dropna(subset=["arrival_ts", "click", *feature_cols]).reset_index(
        drop=True
    )

    ops_all = pd.Index([f"item_{i}" for i in catalog])
    return DatasetBundle(logs=logs, ops_all=ops_all, ground_truth=None)
