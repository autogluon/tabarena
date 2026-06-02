from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


@dataclass(frozen=True)
class ModelInfo:
    """Unified per-model contribution: model class, search space, metadata.

    Each `tabarena.models.<key>` package exports a `<key>_info: ModelInfo`
    that bundles the AutoGluon wrapper, the HPO search-space generator, and
    the `MethodMetadata` artifact entry. The auto-discovery registry (see
    :func:`tabarena.models.discover_models`) collects these into a single
    `MODEL_REGISTRY` keyed by `ag_name`.

    Attributes:
    ----------
    model_cls
        The AutoGluon model class (subclass of `AbstractModel`).
    search_space
        Callable that returns a `ConfigGenerator` (or compatible object) for
        this model's HPO search space.
    method_metadata
        The `MethodMetadata` artifact entry describing the benchmarked
        results (compute, ag_key, default config, etc.).
    pip_extra
        Pip-install specs required to use this model (e.g. `("tabstar==1.1.15",)`).
        Empty tuple means no extra dependencies beyond the base install.
    prefetch_weights
        Optional zero-arg ``prefetch_weights`` callable that ensures this model's (foundation)
        weights are present locally — by convention a ``prefetch_weights`` function in the model's
        ``model.py`` (or its ``info.py`` for models supported out of the box via an external class).
        ``None`` (the default) means the model has nothing to prefetch (tree / linear baselines).
        Consumed by :func:`tabarena.models.prefetch.prefetch_weights`, the single standardized entry
        point for warming weights before a benchmark runs.
    """

    model_cls: type
    search_space: Callable
    method_metadata: "MethodMetadata"
    pip_extra: tuple[str, ...] = field(default_factory=tuple)
    prefetch_weights: Callable[[], None] | None = None
