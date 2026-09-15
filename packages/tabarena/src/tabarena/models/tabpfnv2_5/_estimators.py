"""tabpfn-side pieces of the TabPFN shared-weights path (TabPFN-3 and TabPFN-2.5 / 2.6).

This module imports tabpfn at the top and is imported lazily from the wrappers (inside the build
hook and the fit), so importing a wrapper module stays free of torch and tabpfn.

The registry payload is a tabpfn ``ModelSpecs`` container: ``TabPFNClassifier`` and
``TabPFNRegressor`` accept it as ``model_path`` and ``tabpfn.base.initialize_tabpfn_model`` then
returns its module, architecture config, criterion and inference config directly instead of
reading the checkpoint. The two subclasses here are dataclasses (so the registry helpers can walk
the container for its modules) whose ``copy.deepcopy`` returns the same object: ``sklearn.base.clone``
deep-copies every constructor argument that is not an estimator, and the ``ManyClassClassifier``
wrapper clones its base estimator once per output-coding row at predict time, so a plain specs
object would be copied (network included) per row.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
from tabpfn.inference_config import cpu_sample_limit
from tabpfn.model_loading import load_model_criterion_config, resolve_model_version

from tabarena.models.prefetch import WeightsUnavailableError


@dataclasses.dataclass(eq=False, repr=False)
class SharedClassifierModelSpecs(ClassifierModelSpecs):
    """Classifier specs shared by every estimator of the process; ``deepcopy`` returns the object itself."""

    model: Any
    architecture_config: Any
    inference_config: Any

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedClassifierModelSpecs:
        return self


@dataclasses.dataclass(eq=False, repr=False)
class SharedRegressorModelSpecs(RegressorModelSpecs):
    """Regressor specs shared by every estimator of the process; ``deepcopy`` returns the object itself."""

    model: Any
    architecture_config: Any
    inference_config: Any
    norm_criterion: Any

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedRegressorModelSpecs:
        return self


def build_shared_model_specs(
    checkpoint: str,
    estimator_type: str,
    device: str,
) -> SharedClassifierModelSpecs | SharedRegressorModelSpecs:
    """Build the specs for one checkpoint with the network resident on ``device``.

    Reproduces the path branch of ``tabpfn.base.initialize_tabpfn_model`` argument for argument
    (``load_model_criterion_config`` with the bar-distribution check for regressors only,
    ``cache_trainset_representation=False`` because the wrappers fit with tabpfn's default
    ``fit_mode="fit_preprocessors"``, the checkpoint's own version, and the same
    ``MAX_CPU_SAMPLES`` override), so a fit on the specs equals a fit on the path. The module comes
    out of tabpfn's ``_build_model`` in eval mode; it is moved to ``device`` once here, so the
    per-device cache of every estimator that receives it finds it on its device and never moves it.

    The checkpoint must already be on disk: the wrapper's checkpoint resolution downloads a missing
    file when the fetch policy allows, before the build runs. tabpfn's raw-checkpoint LRU (a private
    ``functools.lru_cache`` of size one) is cleared afterwards so the process does not keep a
    second copy of the state dict; the guard turns the call into a no-op should that cache
    disappear.

    Raises:
        WeightsUnavailableError: ``checkpoint`` does not exist.
    """
    if not Path(checkpoint).is_file():
        raise WeightsUnavailableError(
            f"TabPFN checkpoint {checkpoint} is not on disk; prefetch it (tabarena.models.prefetch.prefetch_weights) "
            "or allow the fit to download it."
        )
    version = resolve_model_version(checkpoint)
    models, criterion, configs, inference_config = load_model_criterion_config(
        model_path=checkpoint,
        check_bar_distribution_criterion=estimator_type == "regressor",
        cache_trainset_representation=False,
        estimator_type=estimator_type,
        version=version.value,
        download_if_not_exists=False,
    )
    inference_config = dataclasses.replace(inference_config, MAX_CPU_SAMPLES=cpu_sample_limit(version))
    model = models[0]
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    _clear_raw_checkpoint_cache()
    if estimator_type == "regressor":
        criterion.to(device)
        return SharedRegressorModelSpecs(model, configs[0], inference_config, criterion)
    return SharedClassifierModelSpecs(model, configs[0], inference_config)


def _clear_raw_checkpoint_cache() -> None:
    from tabpfn import model_loading

    cache_clear = getattr(getattr(model_loading, "_load_checkpoint_cached", None), "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
