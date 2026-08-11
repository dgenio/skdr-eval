"""Deprecated compatibility shims for historical tracker placeholders.

Early releases exposed importable MLflow / W&B / Aim tracker classes before
working adapters existed. Those names are retained only for a bounded
deprecation window so old imports fail with an honest migration message instead
of pretending that installing an extra will unlock functionality.

Issue #217 tracks their retirement. New code should use the built-in
``NullTracker`` / ``FileTracker`` or implement the small ``Tracker`` protocol
locally until repeated user demand justifies a real maintained integration.
"""

from __future__ import annotations

import warnings
from typing import Any


def build_stub(package: str) -> type:
    """Construct a deprecated compatibility shim for an unimplemented adapter."""

    class _Stub:
        __slots__ = ()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            warnings.warn(
                f"skdr_eval.trackers.{package} is a deprecated compatibility "
                "placeholder and is scheduled for removal. skdr-eval does not "
                f"currently provide a working {package} tracker integration. ",
                DeprecationWarning,
                stacklevel=2,
            )
            raise NotImplementedError(
                f"skdr-eval does not provide a working {package} tracker "
                "integration. Use NullTracker/FileTracker or implement the "
                "Tracker protocol locally. The historical placeholder is being "
                "retired under issue #217."
            )

    _Stub.__name__ = f"{package.capitalize()}Tracker"
    _Stub.__qualname__ = _Stub.__name__
    _Stub.__doc__ = (
        f"Deprecated compatibility placeholder for the {package} tracker. "
        "No working integration is provided."
    )
    return _Stub
