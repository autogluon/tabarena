"""Execution wrappers for benchmarked methods.

An "exec model" wraps an underlying method (an AutoGluon predictor, a single AG
model, or any custom fit/predict implementation) behind the common
:class:`AbstractExecModel` interface so the experiment runner can fit it, collect
predictions, and record metadata uniformly.

Note: ``tabarena_model_registry`` is intentionally NOT re-exported here, as it is
built lazily on first attribute access (see :mod:`tabarena.benchmark.exec_models.registry`).
Import it from that submodule directly to preserve the lazy import behavior.
"""

from __future__ import annotations

from tabarena.benchmark.exec_models.autogluon import (
    AGModelWrapper,
    AGSingleBagWrapper,
    AGSingleWrapper,
    AGWrapper,
)
from tabarena.benchmark.exec_models.autogluon_v2 import (
    AGSingleBagWrapperV2,
    AGSingleWrapperV2,
    AGWrapperV2,
)
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.exec_models.registry import infer_model_cls

__all__ = [
    "AGModelWrapper",
    "AGSingleBagWrapper",
    "AGSingleBagWrapperV2",
    "AGSingleWrapper",
    "AGSingleWrapperV2",
    "AGWrapper",
    "AGWrapperV2",
    "AbstractExecModel",
    "ExternalSystemModel",
    "infer_model_cls",
]
