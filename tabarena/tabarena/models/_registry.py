from __future__ import annotations

import importlib
import logging
import pkgutil

from tabarena.models._model_info import ModelInfo

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, ModelInfo] | None = None


def discover_models() -> dict[str, ModelInfo]:
    """Walk `tabarena.models.<key>` packages, import each `info` submodule,
    and collect `ModelInfo` instances declared in them.

    Returns a dict keyed by `method_metadata.method` (the canonical, unique
    method identifier — required to be unique by `MethodMetadata`). Cached
    on first call; re-import the module to refresh.

    A package whose `info.py` fails to import logs a warning and is skipped;
    the model is then absent from the registry. The skip-and-warn behaviour
    keeps the rest of the registry usable when one model's optional deps
    are broken, while making the failure visible (silent skipping previously
    masked a real CatBoost discovery regression for an extended period).
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
        except ImportError as exc:
            logger.warning(
                "Skipping tabarena.models.%s in registry: failed to import its "
                "info module (%s: %s). The model will not be discoverable until "
                "the import is fixed.",
                name, type(exc).__name__, exc,
            )
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


def register_model_info(info: ModelInfo) -> None:
    """Register an additional `ModelInfo` with the core `MODEL_REGISTRY`.

    Intended for use by extension packages (e.g. `tabarena_extensions`) whose
    `tabarena_extensions/<key>/info.py` modules aren't reachable by
    `discover_models()`'s walk over `tabarena.models`. The extension's
    `__init__.py` calls `register_model_info(...)` for each of its
    `ModelInfo` instances to make them discoverable via `MODEL_REGISTRY`.

    Extensions sometimes redeclare a method that's already in the core
    registry (e.g. a re-benchmarked LinearModel with a different
    `artifact_name`). When `info.method_metadata.method` is already
    registered with a different `ModelInfo`, this function keys the new
    entry as ``f"{method}@{artifact_name}"`` instead, preserving the core
    entry under the bare method name.
    """
    registry = discover_models()
    key = info.method_metadata.method
    existing = registry.get(key)
    if existing is None or existing is info:
        registry[key] = info
        return
    # Disambiguate by appending artifact_name to the key.
    artifact = info.method_metadata.artifact_name or "ext"
    composite_key = f"{key}@{artifact}"
    if composite_key in registry and registry[composite_key] is not info:
        raise RuntimeError(
            f"Duplicate ModelInfo composite key {composite_key!r}: "
            f"already registered as {registry[composite_key]}; "
            f"attempted to re-register with {info}."
        )
    registry[composite_key] = info
