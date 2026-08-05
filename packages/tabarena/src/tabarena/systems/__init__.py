"""Systems benchmarked by TabArena: self-contained pipelines rather than single models.

A *system* manages its own budget, model selection and ensembling: AutoGluon, TabFM+ (TabFM's
heavier ``ensemble`` interface), an AutoML library, an LLM-driven agent, a hosted API. It is
wrapped as an :class:`~tabarena.benchmark.exec_models.ExternalSystemModel`, paired with a
:class:`~tabarena.utils.config_utils.SystemConfigGenerator`, and run through the experiment
bundle's ``system_experiments=True`` mode.

Layout mirrors ``tabarena.models``: one subpackage per system holding ``system.py`` (the
exec-model wrapper), ``hpo.py`` (the config generator) and ``info.py`` (the
:class:`SystemInfo` + its :class:`~tabarena.models._method_metadata.MethodMetadata`).
:func:`discover_systems` walks those ``info`` modules into ``SYSTEM_REGISTRY``.

Systems stay out of the AutoGluon model registry on purpose: they have no ``ag_key`` and no
HPO search space, and their results are recorded as ``method_type="baseline"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.systems._registry import discover_systems, get_system_registry
from tabarena.systems._system_info import SystemInfo

if TYPE_CHECKING:
    from tabarena.systems.autogluon.system import AutoGluonSystemModel
    from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel


# Maps top-level public name -> module to import it from. Resolved lazily by `__getattr__`
# for the same reason `tabarena.models` does it: a system wrapper pulls in the heavy library
# it drives, and `import tabarena.systems` should stay cheap.
_LAZY_CLASSES: dict[str, str] = {
    "AutoGluonSystemModel": "tabarena.systems.autogluon.system",
    "TabFMPlusSystemModel": "tabarena.systems.tabfm_plus.system",
}


def __getattr__(name: str):
    module_path = _LAZY_CLASSES.get(name)
    if module_path is None:
        raise AttributeError(f"module 'tabarena.systems' has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(module_path), name)
    globals()[name] = obj  # cache so subsequent lookups skip __getattr__
    return obj


_EAGER_EXPORTS = (
    "SystemInfo",
    "discover_systems",
    "get_system_registry",
)

# Derived, so a name added to either mapping is automatically public. The TYPE_CHECKING block
# above still needs static imports for IDEs to resolve the lazy names.
__all__ = sorted({*_LAZY_CLASSES, *_EAGER_EXPORTS})  # noqa: PLE0605  # sorted() returns a valid list for __all__
