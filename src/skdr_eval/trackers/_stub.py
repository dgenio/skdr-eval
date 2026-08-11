"""Retired compatibility stubs for historical external tracker adapters.

MLflow, W&B, and Aim classes were publicly importable but never implemented:
construction always raised ``NotImplementedError``.  They remain temporarily
importable only so old imports fail with an accurate migration message while
#217 removes the misleading optional-integration surface.
"""

from __future__ import annotations

from typing import Any


def build_stub(package: str) -> type:
    """Construct a retired compatibility stub for an unimplemented adapter."""

    class _Stub:
        __slots__ = ()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise NotImplementedError(
                f"The {package} tracker adapter is not implemented and is no longer "
                "advertised as a skdr-eval capability. The historical optional "
                f"extra '[{package}]' does not enable a working adapter. Use "
                "NullTracker/FileTracker or provide your own Tracker implementation. "
                "See issue #217 for the compatibility cleanup."
            )

    _Stub.__name__ = f"{package.capitalize()}Tracker"
    _Stub.__qualname__ = _Stub.__name__
    _Stub.__doc__ = (
        f"Retired compatibility stub for the unimplemented {package} adapter. "
        "Use NullTracker/FileTracker or a custom Tracker implementation."
    )
    return _Stub
