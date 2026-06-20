"""Runtime capability detection for skdr-eval optional extras.

Provides a side-effect-free way to discover which optional dependency
groups (``[viz]``, ``[speed]``) are installed, so callers and CI smoke
checks can short-circuit gracefully when a feature requires an extra
that was not installed.

The set of detected capabilities tracks the *truly* optional extras in
``pyproject.toml``: ``viz`` (``matplotlib``) and ``speed`` (``pyarrow``
+ ``polars``). ``scipy`` is a mandatory dependency, so condlogit
propensity estimation is always available and is not listed here.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

# Map of capability name -> list of module names whose presence enables it.
# Capability is True iff *all* listed modules can be located.
_CAPABILITY_SPECS: dict[str, tuple[str, ...]] = {
    "viz": ("matplotlib",),
    "speed": ("pyarrow", "polars"),
}

# Reverse map for missing-extras reporting.
_EXTRA_BY_CAPABILITY = {
    "viz": "viz",
    "speed": "speed",
}

# Full capability matrix (#215): every optional ``[extra]`` declared in
# ``pyproject.toml``, the modules that enable it, and the user-facing feature it
# unlocks. ``scipy`` is a mandatory dependency (conditional-logit propensities
# are always available), so it is intentionally absent here.
_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "viz": ("matplotlib",),
    "speed": ("pyarrow", "polars"),
    "cli": ("typer", "joblib", "pyarrow"),
    "boosting": ("xgboost", "lightgbm", "catboost"),
    "mlflow": ("mlflow",),
    "wandb": ("wandb",),
    "aim": ("aim",),
}

_EXTRA_FEATURES: dict[str, str] = {
    "viz": "Plotting helpers (skdr_eval.visualization).",
    "speed": "Accelerated parquet/feather I/O via pyarrow + polars.",
    "cli": "The 'skdr-eval' command-line interface.",
    "boosting": "XGBoost / LightGBM / CatBoost model adapters.",
    "mlflow": "MLflow experiment-tracker integration.",
    "wandb": "Weights & Biases experiment-tracker integration.",
    "aim": "Aim experiment-tracker integration.",
}


@dataclass(frozen=True)
class Capability:
    """One row of the optional-dependency capability matrix (#215).

    Attributes
    ----------
    extra : str
        The pip extra name (e.g. ``"viz"``), installable via
        ``pip install 'skdr-eval[<extra>]'``.
    installed : bool
        ``True`` iff *all* modules backing the extra can be located.
    feature : str
        Human-readable description of what the extra unlocks.
    install_hint : str
        Copy-pasteable ``pip install`` command that enables the extra.
    modules : tuple[str, ...]
        The import names probed to decide ``installed``.
    """

    extra: str
    installed: bool
    feature: str
    install_hint: str
    modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def get_capability_matrix() -> list[Capability]:
    """Return the full optional-dependency capability matrix (#215).

    Unlike :func:`get_capabilities` (which reports only the *truly optional*
    feature toggles ``viz`` / ``speed``), this covers every pip extra declared
    in ``pyproject.toml`` — including ``cli``, ``boosting`` and the experiment
    trackers — so ``doctor`` and the ``skdr-eval capabilities`` command can show
    users which features are available and how to unlock the rest.

    Detection is import-light: it only probes :func:`importlib.util.find_spec`
    and never imports the heavy extras themselves.

    Returns
    -------
    list[Capability]
        One :class:`Capability` per extra, ordered as declared in
        :data:`_EXTRA_MODULES`.
    """
    matrix: list[Capability] = []
    for extra, modules in _EXTRA_MODULES.items():
        installed = all(_module_available(m) for m in modules)
        matrix.append(
            Capability(
                extra=extra,
                installed=installed,
                feature=_EXTRA_FEATURES[extra],
                install_hint=f"pip install 'skdr-eval[{extra}]'",
                modules=modules,
            )
        )
    return matrix


def get_capabilities() -> dict[str, bool | list[str]]:
    """Return the set of optional capabilities available in this environment.

    The returned dict is suitable for preflight checks: a missing capability
    means the matching ``pip install 'skdr-eval[<extra>]'`` invocation will
    enable it.

    Returns
    -------
    dict[str, bool | list[str]]
        Keys:

        - ``"viz"`` (bool): plotting helpers under ``skdr_eval.visualization``
          (requires ``matplotlib``; install via ``pip install 'skdr-eval[viz]'``).
        - ``"speed"`` (bool): accelerated I/O paths (requires ``pyarrow``
          and ``polars``; install via ``pip install 'skdr-eval[speed]'``).
        - ``"missing_extras"`` (list[str]): pip extras that, if installed,
          would enable currently-disabled capabilities. Stable, sorted.

    Examples
    --------
    >>> caps = get_capabilities()  # doctest: +SKIP
    >>> caps["viz"]                 # doctest: +SKIP
    True
    >>> caps["missing_extras"]      # doctest: +SKIP
    ['speed']
    """
    result: dict[str, bool | list[str]] = {}
    missing: list[str] = []

    for capability, modules in _CAPABILITY_SPECS.items():
        available = all(_module_available(m) for m in modules)
        result[capability] = available
        if not available:
            missing.append(_EXTRA_BY_CAPABILITY[capability])

    result["missing_extras"] = sorted(missing)
    return result
