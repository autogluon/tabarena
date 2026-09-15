"""tabpfnwide-side pieces of the TabPFN-Wide shared-weights path.

This module imports tabpfnwide (and through it tabpfn and torch) at the top and is imported lazily
from ``model.py``, so ``import tabarena.models.tabpfnwide.model`` stays cheap.

``TabPFNWideClassifier`` builds its network in its constructor: ``__init__`` calls the static
``_build_model_specs``, which loads the tabpfn v2 classifier architecture, copies the wide
checkpoint's state dict into it, patches the encoder for narrow feature groups and wraps the result
in a ``ClassifierModelSpecs`` that it hands to ``TabPFNClassifier`` as ``model_path``.
:class:`SharedTabPFNWideClassifier` keeps that constructor and replaces only the build with a specs
object taken from the process-wide registry, so every fold child and the refit child reference one
network. Written against tabpfnwide 0.3.0 (``tabpfnwide/classifier.py``): the build happens inside
``__init__``, which is why the payload is a constructor argument rather than a post-construction seam.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from tabpfn.base import ClassifierModelSpecs
from tabpfnwide.classifier import TabPFNWideClassifier

from tabarena.models.prefetch import WeightsUnavailableError


@dataclasses.dataclass(eq=False, repr=False)
class SharedWideModelSpecs(ClassifierModelSpecs):
    """Wide classifier specs shared by every estimator of the process; ``deepcopy`` returns the object itself.

    A dataclass so the registry helpers can walk it for its module; ``sklearn.base.clone``
    deep-copies constructor arguments that are not estimators, and returning the same object keeps
    a cloned estimator on the shared network instead of copying it.
    """

    model: Any
    architecture_config: Any
    inference_config: Any

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedWideModelSpecs:
        return self


class SharedTabPFNWideClassifier(TabPFNWideClassifier):
    """``TabPFNWideClassifier`` whose network comes from ``shared_model_specs`` instead of being built.

    The library constructor still validates ``model_name`` / ``model_path`` and resolves the
    checkpoint file (which must already be on disk, so no download happens here), then calls
    ``_build_model_specs``; the override returns the registry payload, so the network is neither
    read from disk nor built. Everything else (fit, predict, attention bookkeeping) is the library's.
    """

    def __init__(self, *, shared_model_specs: SharedWideModelSpecs | None = None, **kwargs: Any) -> None:
        self.shared_model_specs = shared_model_specs
        super().__init__(**kwargs)

    def _build_model_specs(self, model_name, model_path, features_per_group, device):
        """The shared specs; the arguments were consumed when the registry key was derived."""
        if self.shared_model_specs is None:
            raise ValueError(
                "SharedTabPFNWideClassifier needs `shared_model_specs`; use TabPFNWideClassifier otherwise."
            )
        return self.shared_model_specs


def download_wide_checkpoint(model_name: str) -> str:
    """Download the wide checkpoint for ``model_name`` through the library (a no-op when cached).

    ``_get_model_path`` is an instance method that never reads ``self``; it is invoked on an
    uninitialized instance so the download logic (release URL derived from the installed package
    version, partial-file cleanup) stays the library's.
    """
    return TabPFNWideClassifier._get_model_path(object.__new__(TabPFNWideClassifier), model_name)


def tabpfn_v2_base_checkpoint() -> Path:
    """The tabpfn v2 default classifier checkpoint the wide models are built on, in tabpfn's cache."""
    from tabpfn.model_loading import resolve_model_path

    paths, _, _, _ = resolve_model_path(model_path=None, which="classifier", version="v2")
    return paths[0]


def build_shared_model_specs(*, checkpoint: str, model_name: str, features_per_group: int, device: str) -> Any:
    """Build the wide specs for one checkpoint with the network resident on ``device``.

    Delegates to the library's own ``_build_model_specs`` (v2 architecture, wide state dict,
    narrow-feature-group patch), so the module equals a per-estimator build. Both files must
    already be on disk: the wrapper's checkpoint resolution downloads them when the fetch policy
    allows, before the build runs. The module is moved to ``device`` once, set to eval mode and frozen.

    Raises:
        WeightsUnavailableError: The wide checkpoint or the tabpfn v2 base checkpoint is missing.
    """
    if not Path(checkpoint).is_file():
        raise WeightsUnavailableError(
            f"TabPFN-Wide checkpoint {checkpoint} is not on disk; prefetch it "
            "(tabarena.models.prefetch.prefetch_weights) or allow the fit to download it."
        )
    base = tabpfn_v2_base_checkpoint()
    if not base.is_file():
        raise WeightsUnavailableError(
            f"The tabpfn v2 base checkpoint {base} is not on disk; prefetch it or allow the fit to download it."
        )
    specs = TabPFNWideClassifier._build_model_specs(
        model_name=model_name, model_path=checkpoint, features_per_group=features_per_group, device=device
    )
    model = specs.model
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return SharedWideModelSpecs(
        model=model, architecture_config=specs.architecture_config, inference_config=specs.inference_config
    )
