from __future__ import annotations

import copy

from autogluon.tabular.registry import ModelRegistry, ag_model_registry

# `tabarena_model_registry` and `_models_to_add` are built lazily on first
# access so that this module finishes loading before `get_model_registry()`
# (which transitively triggers `experiment_constructor` → this `registry`)
# tries to import from us. See PEP 562. Note: they are NOT predefined as
# module-level globals — that would short-circuit `__getattr__` and yield
# `None` on first access. The cache lives in `_lazy_state` instead.
_lazy_state: dict[str, object] = {}


def _build_tabarena_model_registry() -> tuple[ModelRegistry, list[type]]:
    """Auto-derive `tabarena_model_registry` and `_models_to_add` from the
    per-model `MODEL_REGISTRY`: every model class declared via a `ModelInfo`
    in `tabarena/models/<key>/info.py`. Multi-compute variants (e.g. TabM CPU +
    TabM_GPU) share one `model_cls`, hence the set dedup.
    """
    from tabarena.models import get_model_registry

    registry: ModelRegistry = copy.deepcopy(ag_model_registry)
    # Skip AG-builtin classes (e.g. CatBoostModel, LGBModel) whose `ag_key`
    # is already present in `ag_model_registry`. Those entries exist in
    # MODEL_REGISTRY for their MethodMetadata, but the underlying class
    # is the AG one — re-adding it would only trigger a "duplicate key"
    # warning and reinsert the same class.
    models_to_add: list[type] = list(
        {
            info.model_cls
            for info in get_model_registry().values()
            if info.model_cls.ag_key not in ag_model_registry.keys
        }
    )

    for model_cls in models_to_add:
        new_key = model_cls.ag_key
        if new_key in registry.keys:
            existing_model_cls = registry.key_to_cls(key=new_key)
            print(
                f"WARNING: Multiple models exist with the ag_key '{new_key}'..."
                f"\n\tOnly keeping the TabArena version..."
                f"\n\tThis can cause subtle bugs and should be resolved ASAP.",
            )
            registry.remove(model_cls=existing_model_cls)
        registry.add(model_cls)

    return registry, models_to_add


def __getattr__(name: str):
    if name in ("tabarena_model_registry", "_models_to_add"):
        if "tabarena_model_registry" not in _lazy_state:
            registry, models_to_add = _build_tabarena_model_registry()
            _lazy_state["tabarena_model_registry"] = registry
            _lazy_state["_models_to_add"] = models_to_add
        return _lazy_state[name]
    raise AttributeError(name)


def infer_model_cls(model_cls: str, model_register: ModelRegistry = None):
    if model_register is None:
        model_register = tabarena_model_registry  # noqa: F821
    if isinstance(model_cls, str):
        if model_cls in model_register.key_to_cls_map():
            model_cls = model_register.key_to_cls(key=model_cls)
        elif model_cls in model_register.name_map().values():
            for real_model_cls in model_register.model_cls_list:
                if real_model_cls.ag_name == model_cls:
                    model_cls = real_model_cls
                    break
        elif model_cls in [str(real_model_cls.__name__) for real_model_cls in model_register.model_cls_list]:
            for real_model_cls in model_register.model_cls_list:
                if model_cls == str(real_model_cls.__name__):
                    model_cls = real_model_cls
                    break
        else:
            raise AssertionError(f"Unknown model_cls: {model_cls}")
    return model_cls
