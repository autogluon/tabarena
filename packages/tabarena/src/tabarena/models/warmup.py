"""Untimed environment warm-up for benchmarked methods.

A fresh benchmark process pays one-time environment costs that a long-lived real-world
deployment would not pay per fit: importing heavy libraries, JIT/kernel compilation
(numba, torch), and CUDA context creation. TabArena excludes these from the measured
``time_train_s`` / ``time_infer_s`` (and from fit time limits) by warming the environment
*before* the timed fit — see :attr:`~tabarena.benchmark.exec_models.base.AbstractExecModel.warmup_fn`
and ``ExperimentRunner.run_warmup``.

Fairness contract: a warm-up may only do data-independent work that is a one-time,
per-environment cost in a real deployment (imports, kernel/JIT compilation, CUDA context
initialization, hardware handles). It must never touch the task's data or carry any task-
or data-specific state into the fit.

Scope: warm-up runs once, in the job's main process, before the timed fit. It warms that
process and anything disk-backed (e.g. numba's on-disk kernel cache, which parallel-fold
workers then load instead of recompiling). It does *not* warm the in-memory state of
processes spawned later: a bagged fit with parallel (Ray) fold fitting still pays each
worker's imports / CUDA context inside the measured fit time. Where a library offers it,
prefer a warm-up that persists to disk — it is the only kind that carries over to workers.

Model classes opt in by declaring a ``warmup`` classmethod (see :func:`warmup_model_cls`);
models without one fall back to a generic torch warm-up (``AbstractTorchModel`` subclasses)
or an import-only warm-up (:data:`WARMUP_IMPORTS_BY_AG_KEY`).
"""

from __future__ import annotations

import importlib

#: Import-only warm-up for models whose class declares no ``warmup`` (keyed by ``ag_key``).
#: Covers AutoGluon built-ins whose heavy library import would otherwise land in the timed fit.
#: ``"torch"`` entries additionally initialize the CUDA context (via :func:`warmup_torch`).
WARMUP_IMPORTS_BY_AG_KEY: dict[str, tuple[str, ...]] = {
    "GBM": ("lightgbm",),
    "CAT": ("catboost",),
    "XGB": ("xgboost",),
    "EBM": ("interpret.glassbox",),
    "FASTAI": ("torch", "fastai.tabular.all"),
    "NN_TORCH": ("torch",),
}


def warmup_imports(*module_names: str) -> None:
    """Import ``module_names`` so later (timed) imports are cache hits."""
    for name in module_names:
        importlib.import_module(name)


def warmup_torch(*, cuda: bool | None = None) -> None:
    """Import torch and initialize the CUDA context (both one-time costs per process).

    A tiny device matmul materializes the CUDA context and cuBLAS handle; the allocator
    cache is emptied afterwards so no GPU memory stays reserved. ``cuda=None`` auto-detects.
    """
    import torch

    if cuda is None:
        cuda = torch.cuda.is_available()
    if cuda and torch.cuda.is_available():
        x = torch.zeros((8, 8), device="cuda")
        (x @ x).sum().item()
        del x
        torch.cuda.empty_cache()


def warmup_model_cls(
    model_cls: type,
    *,
    problem_type: str | None = None,
    num_cpus: int | None = None,
    num_gpus: float | None = None,
    hyperparameters: dict | None = None,
) -> None:
    """Warm the environment for one AutoGluon model class.

    The keyword context is what a real deployment also knows *before* seeing any data;
    all fields are optional. Dispatch order:

    1. A ``warmup`` classmethod declared by the model class, called with the full context.
       Convention: ``warmup(cls, *, problem_type=None, num_cpus=None, num_gpus=None,
       hyperparameters=None, **kwargs) -> None`` — read what you need, accept ``**kwargs``.
    2. ``AbstractTorchModel`` subclasses: generic torch warm-up (import + CUDA context;
       CUDA only when ``num_gpus`` doesn't rule it out).
    3. :data:`WARMUP_IMPORTS_BY_AG_KEY`: import-only warm-up for known heavy libraries.

    Models matching none of these need no warm-up (e.g. sklearn baselines).
    """
    warmup = getattr(model_cls, "warmup", None)
    if warmup is not None:
        warmup(
            problem_type=problem_type,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            hyperparameters=hyperparameters,
        )
        return

    cuda = None if num_gpus is None else num_gpus > 0

    from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

    if issubclass(model_cls, AbstractTorchModel):
        warmup_torch(cuda=cuda)
        return

    for module_name in WARMUP_IMPORTS_BY_AG_KEY.get(getattr(model_cls, "ag_key", None), ()):
        if module_name == "torch":
            warmup_torch(cuda=cuda)
        else:
            warmup_imports(module_name)
