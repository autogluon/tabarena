from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from tabarena.models._model_info import ModelInfo

if TYPE_CHECKING:
    pass


_REGISTRY: dict[str, ModelInfo] | None = None


def discover_models() -> dict[str, ModelInfo]:
    """Walk `tabarena.models.<key>` packages, import each `info` submodule,
    and collect `ModelInfo` instances declared in them.

    Returns a dict keyed by `method_metadata.method` (the canonical, unique
    method identifier — required to be unique by `MethodMetadata`). Cached
    on first call; re-import the module to refresh.

    Models that don't yet have an `info.py` are silently skipped — Stage 1
    is migration-friendly and lets old and new layouts coexist.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    registry: dict[str, ModelInfo] = {}
    import tabarena.models as pkg

    for _finder, name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if not is_pkg or name.startswith("_"):
            continue
        try:
            info_module = importlib.import_module(f"tabarena.models.{name}.info")
        except ImportError:
            # Model hasn't migrated to the new layout yet; skip silently.
            continue
        for attr_name in dir(info_module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(info_module, attr_name)
            if not isinstance(obj, ModelInfo):
                continue
            key = obj.method_metadata.method
            if key in registry:
                raise RuntimeError(
                    f"Duplicate ModelInfo key {key!r}: {registry[key]} vs {obj} "
                    f"(from tabarena.models.{name}.info::{attr_name})"
                )
            registry[key] = obj

    _REGISTRY = registry
    return registry


def get_model_registry() -> dict[str, ModelInfo]:
    """Return the cached `MODEL_REGISTRY`, building it on first call."""
    return discover_models()
