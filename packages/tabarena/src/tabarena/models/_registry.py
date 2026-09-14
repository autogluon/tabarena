from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from importlib.util import find_spec  # bound directly: the registry tests replace ``importlib``

from tabarena.models._model_info import ModelInfo

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, ModelInfo] | None = None


def assert_autogluon_resolves() -> None:
    """Fail fast when ``autogluon.tabular`` is shadowed by a namespace package.

    Starting python with a directory named ``autogluon/`` on ``sys.path[0]`` (for example
    ``python -m ...`` from the workspace directory that holds the editable AutoGluon checkout)
    makes ``import autogluon.tabular`` succeed as an empty namespace package. The real
    ``autogluon.tabular.models`` still resolves through the editable finder, but its import runs
    ``from autogluon.tabular import __version__`` and fails; every ``tabarena.models.<key>.info``
    reaches that import through its hpo or model module, so the discovery walk skips all of them
    and the registry (and any test parametrized over it) comes out empty.

    Raises:
        RuntimeError: ``autogluon.tabular`` resolves to a namespace package (``spec.origin`` is
            ``None``). Returns silently when the package is not installed (``ModuleNotFoundError``,
            surfacing as the usual ``ImportError`` at first real use) and when an already imported
            module has no spec to inspect (``ValueError``).
    """
    try:
        spec = find_spec("autogluon.tabular")
    except (ModuleNotFoundError, ValueError):
        return
    if spec is None or spec.origin is not None:
        return
    locations = list(spec.submodule_search_locations or [])
    raise RuntimeError(
        "autogluon.tabular resolved to a namespace package (no __init__.py) with search locations "
        f"{locations}. The installed package is shadowed, most likely by an 'autogluon/' directory "
        f"under sys.path[0]={sys.path[0]!r}. Run python from the repository root instead of the "
        "workspace directory that contains the editable AutoGluon checkout, or start it with "
        "'python -P' (or PYTHONSAFEPATH=1)."
    )


def _raise_if_all_skipped(kind: str, skipped: list[str], registry: dict) -> None:
    """Raise when a discovery walk skipped every package it found (``kind`` names the walk)."""
    if skipped and not registry:
        raise RuntimeError(
            f"{kind} registry is empty: every discovered package failed to import its info module "
            f"({', '.join(skipped)}). See the 'Skipping {kind}.<key>' warnings above for the import "
            "errors; this means the environment is broken, not that optional dependencies are missing."
        )


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

    The walk raises ``RuntimeError`` instead when ``autogluon.tabular`` resolves to a
    namespace package (see :func:`assert_autogluon_resolves`) and when every discovered
    package was skipped: an empty registry means a broken environment, not missing
    optional extras.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    assert_autogluon_resolves()
    registry: dict[str, ModelInfo] = {}
    skipped: list[str] = []
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
                name,
                type(exc).__name__,
                exc,
            )
            skipped.append(name)
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
                    f"(from tabarena.models.{name}.info::{attr_name})",
                )
            registry[key] = obj

    _raise_if_all_skipped("tabarena.models", skipped, registry)
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
    `suite`). When `info.method_metadata.method` is already
    registered with a different `ModelInfo`, this function keys the new
    entry as ``f"{method}@{suite}"`` instead, preserving the core
    entry under the bare method name.
    """
    registry = discover_models()
    key = info.method_metadata.method
    existing = registry.get(key)
    if existing is None or existing is info:
        registry[key] = info
        return
    # Disambiguate by appending suite to the key.
    artifact = info.method_metadata.suite or "ext"
    composite_key = f"{key}@{artifact}"
    if composite_key in registry and registry[composite_key] is not info:
        raise RuntimeError(
            f"Duplicate ModelInfo composite key {composite_key!r}: "
            f"already registered as {registry[composite_key]}; "
            f"attempted to re-register with {info}.",
        )
    registry[composite_key] = info
