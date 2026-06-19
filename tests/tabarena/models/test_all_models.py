from __future__ import annotations

import importlib.util

import pytest

from tabarena.models import get_model_registry

from .smoke_configs import smoke_for

# Built once at collection time. ``discover_models()`` imports each lightweight
# ``info.py`` but not the heavy model libraries, so a model whose optional
# dependency is missing still appears here and is skipped at fit time below.
_REGISTRY = get_model_registry()


def _cuda_available() -> bool:
    """True only if torch is installed and reports a usable CUDA device."""
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


_CUDA_AVAILABLE = _cuda_available()


@pytest.mark.models
@pytest.mark.parametrize("method", sorted(_REGISTRY), ids=str)
def test_model_smoke(method: str) -> None:
    """Fit each registered model on AutoGluon's toy datasets.

    Replaces the old per-model ``test_<model>.py`` files: one parametrized case
    per registry entry, using the fast toy hyperparameters in ``smoke_configs``.
    Run a single model during development with ``-k`` (e.g. ``pytest -m models -k TabM``).

    Skips (rather than fails) when a model cannot run in the current environment:
    its optional dependency is not installed (``ImportError``), or it is a
    GPU-only model (``compute='gpu'``) and no CUDA device is available.
    """
    info = _REGISTRY[method]
    if info.method_metadata.compute == "gpu" and not _CUDA_AVAILABLE:
        pytest.skip(f"{method}: requires a GPU (compute='gpu') and no CUDA device is available")

    cfg = smoke_for(method)
    try:
        from autogluon.tabular.testing import FitHelper

        kwargs: dict = {
            "model_cls": info.model_cls,
            "model_hyperparameters": dict(cfg.hyperparameters),
        }
        if cfg.problem_types is not None:
            kwargs["problem_types"] = list(cfg.problem_types)
        FitHelper.verify_model(**kwargs)
    except ImportError as err:
        pytest.skip(f"{method}: optional dependency not installed ({err})")
