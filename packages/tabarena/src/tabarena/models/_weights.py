"""The process-wide registry of shared pretrained weights: AutoGluon's, under TabArena's name.

The registry and the sharing mechanism live in AutoGluon: a model class declares
:class:`autogluon.core.models.abstract.SharedWeights` (which library call builds its network and which
inputs decide which network that is) and ``AbstractTorchModel`` memoizes that call per process,
pickles the fitted model without the network and takes it back on load. TabArena's wrappers only
declare (see ``tabarena/models/tabicl/model.py`` for the plain case and
``exaone_tabular/_estimators.py`` for a library that loads inside a constructor).

This module re-exports the registry so the benchmark's own code (the warm-up report, the exec-model
metadata, the audit CLI and the tests) has one import path: :func:`report` and :func:`release` for
the metadata and the cleanup, :func:`rng_guard` around the warm-up's dummy fit, ``WeightsKey`` and
``peek`` for the tests. It imports torch only inside the functions that need it.
"""

from __future__ import annotations

from autogluon.core.models.abstract._shared_weights_registry import (
    CAPACITY_ENV_VAR,
    DEFAULT_CAPACITY,
    LEGACY_CAPACITY_ENV_VAR,
    LoadedBy,
    SharedWeightsClassSettings,
    WeightsEntry,
    WeightsKey,
    capacity,
    contains,
    get_or_load,
    loaded_by,
    make_key,
    normalize_device,
    peek,
    release,
    report,
    reset_stats,
    rng_guard,
    set_capacity,
    shallow_copy,
    tensor_bytes,
)

__all__ = [
    "CAPACITY_ENV_VAR",
    "DEFAULT_CAPACITY",
    "LEGACY_CAPACITY_ENV_VAR",
    "LoadedBy",
    "SharedWeightsClassSettings",
    "WeightsEntry",
    "WeightsKey",
    "capacity",
    "contains",
    "get_or_load",
    "loaded_by",
    "make_key",
    "normalize_device",
    "peek",
    "release",
    "report",
    "reset_stats",
    "rng_guard",
    "set_capacity",
    "shallow_copy",
    "tensor_bytes",
]
