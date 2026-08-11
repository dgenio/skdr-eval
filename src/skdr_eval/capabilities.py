"""Runtime capability detection for skdr-eval optional extras.

Provides a side-effect-free way to discover which optional dependency
groups are installed, so callers and CI smoke checks can short-circuit
gracefully when a feature requires an extra that was not installed.

Only **working, user-facing capabilities** are reported here. Historical
MLflow / W&B / Aim extras were published before real tracker adapters existed;
those placeholders are intentionally not advertised as capabilities. Their
importable compatibility shims are deprecated separately under issue #217.

The set of detected capabilities tracks the optional features that currently
have working public implementations: ``viz`` (``matplotlib``), ``speed``
(``pyarrow`` + ``polars``), ``cli`` and ``boosting``. ``scipy`` is a mandatory
dependency, so conditional-logit propensity estimation is always available and
is not listed here.
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

# Full capability matrix (#215): every currently supported optional user-facing
# feature, the modules that enable it, and the behavior it unlocks.
#
# Historical ``mlflow`` / ``wandb`` / ``aim`` package extras are deliberately
# absent. They never had working adapters and are being retired under #217; a
# package being importable must not make ``skdr-eval capabilities`` claim that
# an integration exists.
_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "viz": ("matplotlib",),
    "speed": ("pyarrow", "polars"),
    "cli": ("typer", "joblib", "pyarrow"),
    "boosting": ("xgboost", "lightgbm", "catboost"),
}

_EXTRA_FEATURES: dict[str, str] = {
    "viz": "Plotting helpers (skdr_eval.visualization).",
    "speed": "Accelerated parquet/feather I/O via pyarrow + polars.",
    "cli": "The 'skdr-eval' command-line interface.",
    "boosting": "XGBoost / LightGBM / CatBoost model adapters.",
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
    """Return the working optional-dependency capability matrix (#215).

    Unlike :func:`get_capabilities` (which reports only the lightweight
    ``viz`` / ``speed`` feature toggles), this also covers the working ``cli``
    and ``boosting`` extras so ``doctor`` and ``skdr-eval capabilities`` can
    show users which implemented features are available.

    Deprecated placeholder tracker extras are intentionally excluded: the
    capability command describes functionality that actually exists, not every
    historical extra name still accepted during a deprecation window.

    Detection is import-light: it only probes :func:`importlib.util.find_spec`
    and never imports the heavy extras themselves.

    Returns
    -------
    list[Capability]
        One :class:`Capability` per supported extra, ordered as declared in
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
    """Return the lightweight optional capabilities available in this environment.

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
