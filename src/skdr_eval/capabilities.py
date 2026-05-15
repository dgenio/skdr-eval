"""Runtime capability detection for skdr-eval optional extras.

Provides a side-effect-free way to discover which optional dependency
groups (``[choice]``, ``[viz]``, ``[speed]``) are installed, so callers
and CI smoke checks can short-circuit gracefully when a feature requires
an extra that was not installed.
"""

from __future__ import annotations

import importlib.util

# Map of capability name -> list of module names whose presence enables it.
# Capability is True iff *all* listed modules can be located.
_CAPABILITY_SPECS: dict[str, tuple[str, ...]] = {
    "choice": ("scipy",),
    "viz": ("matplotlib",),
    "speed": ("pyarrow", "polars"),
}

# Reverse map for missing-extras reporting.
_EXTRA_BY_CAPABILITY = {
    "choice": "choice",
    "viz": "viz",
    "speed": "speed",
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def get_capabilities() -> dict[str, bool | list[str]]:
    """Return the set of optional capabilities available in this environment.

    The returned dict is suitable for preflight checks: a missing capability
    means the matching ``pip install 'skdr-eval[<extra>]'`` invocation will
    enable it.

    Returns
    -------
    dict[str, bool | list[str]]
        Keys:

        - ``"choice"`` (bool): conditional-logit propensity estimation
          (requires ``scipy``).
        - ``"viz"`` (bool): plotting helpers under ``skdr_eval.visualization``
          (requires ``matplotlib``).
        - ``"speed"`` (bool): accelerated I/O paths (requires ``pyarrow``
          and ``polars``).
        - ``"missing_extras"`` (list[str]): pip extras that, if installed,
          would enable currently-disabled capabilities. Stable, sorted.

    Examples
    --------
    >>> caps = get_capabilities()  # doctest: +SKIP
    >>> caps["choice"]              # doctest: +SKIP
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
