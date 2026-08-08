from __future__ import annotations

import importlib
import logging
import pkgutil

from tabarena.systems._system_info import SystemInfo

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, SystemInfo] | None = None


def discover_systems() -> dict[str, SystemInfo]:
    """Walk `tabarena.systems.<key>` packages, import each `info` submodule,
    and collect `SystemInfo` instances declared in them.

    Returns a dict keyed by `method_metadata.method` (the canonical, unique
    method identifier — required to be unique by `MethodMetadata`). Cached
    on first call; re-import the module to refresh.

    Mirrors :func:`tabarena.models.discover_models`, including its skip-and-warn on
    import failure: a package whose `info.py` cannot be imported (usually a missing
    optional dependency) logs a warning and is left out, so one broken system does not
    take the rest of the registry down with it.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    registry: dict[str, SystemInfo] = {}
    import tabarena.systems as pkg

    for _finder, name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if not is_pkg or name.startswith("_"):
            continue
        try:
            info_module = importlib.import_module(f"tabarena.systems.{name}.info")
        except ImportError as exc:
            logger.warning(
                "Skipping tabarena.systems.%s in registry: failed to import its "
                "info module (%s: %s). The system will not be discoverable until "
                "the import is fixed.",
                name,
                type(exc).__name__,
                exc,
            )
            continue
        for attr_name in dir(info_module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(info_module, attr_name)
            if not isinstance(obj, SystemInfo):
                continue
            key = obj.method_metadata.method
            if key in registry:
                raise RuntimeError(
                    f"Duplicate SystemInfo key {key!r}: {registry[key]} vs {obj} "
                    f"(from tabarena.systems.{name}.info::{attr_name})",
                )
            registry[key] = obj

    _REGISTRY = registry
    return registry


def get_system_registry() -> dict[str, SystemInfo]:
    """Return the cached `SYSTEM_REGISTRY`, building it on first call."""
    return discover_systems()
