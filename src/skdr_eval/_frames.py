"""Input-frame coercion for the public evaluators (#72).

The public API is pandas-first, but the ``[speed]`` extra ships ``polars``
and ``pyarrow`` and many callers already hold their logs in one of those
formats. Rather than forcing every caller to spell ``.to_pandas()`` at the
boundary, :func:`coerce_to_pandas` accepts a Polars ``DataFrame`` or a PyArrow
``Table`` (in addition to a pandas ``DataFrame``) and converts it once at
ingestion, so every downstream consumer — which already speaks pandas — is
unchanged.

Detection is by class identity, *not* by importing ``polars`` / ``pyarrow``
at module load: the conversion only imports the relevant package when an
object of that kind is actually passed, keeping these optional extras truly
optional. The matching ``to_polars`` / ``to_arrow`` accessors live on
:class:`skdr_eval.reporting.EvaluationArtifact`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .exceptions import DataValidationError

__all__ = ["coerce_to_pandas"]


def _is_instance_of(obj: Any, module: str, qualname: str) -> bool:
    """Return True if ``obj`` is an instance of ``module.qualname``.

    Walks the MRO and matches on ``__module__`` / ``__qualname__`` so we never
    import ``polars`` / ``pyarrow`` merely to run an ``isinstance`` check. The
    top-level module name is matched on its first component (e.g. ``polars``)
    so subpackage-defined classes (``polars.dataframe.frame.DataFrame``) match.
    """
    for klass in type(obj).__mro__:
        mod = getattr(klass, "__module__", "") or ""
        if mod.split(".", 1)[0] == module and klass.__qualname__ == qualname:
            return True
    return False


def coerce_to_pandas(obj: Any, *, name: str = "input") -> pd.DataFrame:
    """Coerce a supported tabular input into a pandas ``DataFrame``.

    Accepts a pandas ``DataFrame`` (returned unchanged), a Polars ``DataFrame``,
    or a PyArrow ``Table``. Any other type raises :class:`DataValidationError`
    with an actionable message naming the offending parameter.

    Parameters
    ----------
    obj : Any
        The value passed for a frame-typed parameter.
    name : str, default="input"
        Parameter name, used in error messages.

    Returns
    -------
    pd.DataFrame
        A pandas frame. Polars frames are converted with
        ``use_pyarrow_extension_array=True`` (zero-copy where possible);
        PyArrow tables via ``Table.to_pandas()``.

    Raises
    ------
    DataValidationError
        If ``obj`` is not a pandas / Polars / PyArrow frame.
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if _is_instance_of(obj, "polars", "DataFrame"):
        # Plain (NumPy-backed) conversion. We deliberately do *not* request
        # ``use_pyarrow_extension_array=True``: Arrow-backed extension columns
        # break the downstream NumPy paths (e.g. ``np.isfinite`` in the design
        # builder). Statistical correctness over a marginal zero-copy win.
        return obj.to_pandas()

    if _is_instance_of(obj, "pyarrow", "Table"):
        # ``types_mapper`` left default → primitive columns land as NumPy
        # dtypes, matching what the pandas-native path expects.
        return obj.to_pandas()

    raise DataValidationError(
        f"{name} must be a pandas DataFrame, polars DataFrame, or pyarrow "
        f"Table; got {type(obj).__name__}",
    )
