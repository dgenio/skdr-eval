"""Public-dataset loaders (#70).

A small, opinionated set of loaders that return the same canonical schema as
:func:`skdr_eval.make_synth_logs` — ``(logs, ops_all, ground_truth)`` — so the
rest of the library works unchanged on real benchmark data. This is the
lighter layer that lands before the generic OBP-format adapter (#35).

Loaders
-------
* :func:`load_obd` — Open Bandit Dataset (ZOZOTOWN); lands end-to-end.
* :func:`load_criteo_counterfactual` — Criteo counterfactual logs *(stub;
  full implementation tracked by #70)*.
* :func:`load_movielens_ope` — MovieLens OPE recipe *(stub; full
  implementation tracked by #70)*.

Each loader caches downloads under ``~/.skdr_eval/datasets`` (override with
``SKDR_EVAL_CACHE_DIR``) with a sha256 ``manifest.json`` for reproducibility.
``load_obd`` accepts a local ``base_url`` for air-gapped use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import DatasetError
from ._cache import default_cache_dir
from .obd import DatasetBundle, load_obd

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DatasetBundle",
    "default_cache_dir",
    "load_criteo_counterfactual",
    "load_movielens_ope",
    "load_obd",
]


def load_criteo_counterfactual(
    sample: str = "small",
    cache_dir: str | Path | None = None,
) -> DatasetBundle:
    """Load Criteo's counterfactual-learning logs *(not yet implemented)*.

    The Criteo Counterfactual Learning dataset requires explicit license
    acceptance and a multi-GB download; the full loader is tracked as a
    follow-up under #70. Until it ships this raises :class:`DatasetError` with
    a pointer to the dataset and to :func:`load_obd` as the available loader.
    """
    del sample, cache_dir
    raise DatasetError(
        "load_criteo_counterfactual is not implemented yet (tracked by #70). "
        "The Criteo Counterfactual Learning dataset requires license "
        "acceptance — see https://www.cs.cornell.edu/~adith/Criteo/. Use "
        "load_obd for a ready-to-run public benchmark in the meantime.",
    )


def load_movielens_ope(
    version: str = "1m",
    behavior: str = "random_plus_popularity",
    cache_dir: str | Path | None = None,
) -> DatasetBundle:
    """Load a MovieLens OPE recipe *(not yet implemented)*.

    A deterministic logging-policy recipe on MovieLens is tracked as a
    follow-up under #70. Until it ships this raises :class:`DatasetError`. Use
    :func:`load_obd` for a ready-to-run public benchmark in the meantime.
    """
    del version, behavior, cache_dir
    raise DatasetError(
        "load_movielens_ope is not implemented yet (tracked by #70). Use "
        "load_obd for a ready-to-run public benchmark in the meantime.",
    )
