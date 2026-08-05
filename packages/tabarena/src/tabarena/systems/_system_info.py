from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tabarena.models._method_metadata import MethodClass

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabarena.benchmark.exec_models import ExternalSystemModel
    from tabarena.models._method_metadata import MethodMetadata
    from tabarena.utils.config_utils import SystemConfigGenerator


@dataclass(frozen=True)
class SystemInfo:
    """Unified per-system contribution: system class, config generator, metadata.

    The system counterpart of :class:`tabarena.models._model_info.ModelInfo`. Each
    `tabarena.systems.<key>` package exports a `<key>_info: SystemInfo`, collected by
    :func:`tabarena.systems.discover_systems` into `SYSTEM_REGISTRY`, keyed by
    `method_metadata.method`.

    Systems are deliberately absent from the model registry: they carry no AutoGluon
    registry keys and no HPO search space, and they are run through the experiment
    bundle's `system_experiments=True` mode rather than as AutoGluon models.

    Attributes:
    ----------
    system_cls
        The `ExternalSystemModel` subclass that fits and predicts.
    config_generator
        The `SystemConfigGenerator` pairing `system_cls` with a display name and the
        configurations to benchmark. This is what a benchmark run passes as a job entry.
    method_metadata
        The `MethodMetadata` artifact entry. Must have `method_class="system"`, i.e. be
        built with `MethodMetadata.system(...)`.
    pip_extra
        Pip-install specs required to run this system (e.g. `("lightautoml==0.4.0",)`).
        Empty tuple means no extra dependencies beyond the base install.
    prefetch_weights
        Optional zero-arg callable that ensures any (foundation) weights the system needs
        are present locally. `None` (the default) means there is nothing to prefetch.
    """

    system_cls: type[ExternalSystemModel]
    config_generator: SystemConfigGenerator
    method_metadata: MethodMetadata
    pip_extra: tuple[str, ...] = field(default_factory=tuple)
    prefetch_weights: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.method_metadata.method_class != MethodClass.SYSTEM:
            raise AssertionError(
                f"SystemInfo requires method_class='system', got "
                f"{self.method_metadata.method_class!r} (method={self.method_metadata.method!r}). "
                f"Build the metadata with `MethodMetadata.system(...)`.",
            )
