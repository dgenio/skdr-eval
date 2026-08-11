"""Runtime capability detection for skdr-eval optional extras.

Provides a side-effect-free way to discover which optional dependency groups
are installed, so callers and CI smoke checks can short-circuit gracefully
when a *working* feature requires an extra that was not installed.

Only implemented user-facing extras belong in the capability matrix. Historical
MLflow / W&B / Aim extras exposed importable stubs that always raised
``NotImplementedError``; they are deliberately no longer advertised as
capabilities (#217).
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

# Map of lightweight capability name -> modules whose presence enables it.
# ``get_capabilities`` intentionally remains the small preflight surface used by
# the core evaluator.
_CAPABILITY_SPECS: dict[str, tuple[str, ...]] = {
    "viz": ("matplotlib",),
    "speed": ("pyarrow", "polars"),
}

_EXTRA_BY_CAPABILITY = {
    "viz": "viz",
    "speed": "speed",
}

# Full matrix of *implemented* user-facing optional extras. Maintainer/dev/docs
# extras are not runtime capabilities, and retired tracker stubs are excluded.
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
    """One implemented optional-dependency capability.

    Attributes
    ----------
    extra : str
        The pip extra name, installable via
        ``pip install 'skdr-eval[<extra>]'``.
    installed : bool
        ``True`` iff all modules backing the implemented extra can be located.
    feature : str
        Human-readable description of what the extra unlocks.
    install_hint : str
        Copy-pasteable ``pip install`` command that enables the extra.
    modules : tuple[str, ...]
        Import names probed to decide ``installed``.
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
    """Return the implemented optional-dependency capability matrix.

    Unlike :func:`get_capabilities` (which reports only the lightweight
    ``viz`` / ``speed`` toggles used by evaluator preflight), this also covers
    implemented CLI and boosting extras. It intentionally does **not** advertise
    optional packages whose skdr-eval adapters are only non-working compatibility
    stubs.

    Detection is import-light: it probes :func:`importlib.util.find_spec` and
    never imports heavy extras themselves.
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
    """Return lightweight optional capabilities available in this environment."""

    result: dict[str, bool | list[str]] = {}
    missing: list[str] = []

    for capability, modules in _CAPABILITY_SPECS.items():
        available = all(_module_available(m) for m in modules)
        result[capability] = available
        if not available:
            missing.append(_EXTRA_BY_CAPABILITY[capability])

    result["missing_extras"] = sorted(missing)
    return result
