"""Arena evaluation contexts.

A *context* bundles everything an evaluation needs to know about a benchmark family — its method
collection, its task metadata, and its subset predicates. Concrete contexts subclass
:class:`AbstractArenaContext`: :class:`TabArenaContext` is the TabArena-v0.1 context and
:class:`BeyondArenaContext` the data-foundry (BeyondArena) one. New benchmark families add a
context module here.

Re-exports are resolved lazily (PEP 562) so importing ``tabarena.contexts`` does not eagerly pull
in the heavy context modules — preserving the lazy-import cycle avoidance the runners rely on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abstract_arena_context import AbstractArenaContext
    from .beyondarena_context import BeyondArenaContext
    from .tabarena_context import TabArenaContext

__all__ = ["AbstractArenaContext", "BeyondArenaContext", "TabArenaContext"]

_EXPORTS = {
    "AbstractArenaContext": "abstract_arena_context",
    "TabArenaContext": "tabarena_context",
    "BeyondArenaContext": "beyondarena_context",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
