"""Shared stub-class builder for external tracker adapters.

Used by :mod:`mlflow`, :mod:`wandb`, and :mod:`aim` adapter modules to expose
importable classes that raise :class:`NotImplementedError` on construction
until their full implementations land under umbrella issue #73.
"""

from __future__ import annotations

from typing import Any


def build_stub(package: str) -> type:
    """Construct a stub tracker class that errors on instantiation.

    The class name is ``<Package>Tracker`` (capitalized). The error message
    references the matching ``pip install 'skdr-eval[<package>]'`` extra and
    points at the umbrella issue.
    """
    install_hint = f"pip install 'skdr-eval[{package}]'"

    class _Stub:
        __slots__ = ()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise NotImplementedError(
                f"The {package} tracker adapter is a stub. Install the optional "
                f"extra ({install_hint}) and wait for the full adapter to ship "
                f"under issue #73."
            )

    _Stub.__name__ = f"{package.capitalize()}Tracker"
    _Stub.__qualname__ = _Stub.__name__
    _Stub.__doc__ = (
        f"Stub for the {package} tracker adapter. "
        f"Raises NotImplementedError on construction. "
        f"Full implementation tracked by issue #73."
    )
    return _Stub
