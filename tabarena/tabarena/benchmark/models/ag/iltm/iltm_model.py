from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


class ILTMModel(AbstractTorchModel):
    """iLTM: Integrated Large Tabular Model.

    Paper: iLTM: Integrated Large Tabular Model (arXiv:2511.15941)
    Authors: Bonet, Comajoan Cara, Calafell, Mas Montserrat, Ioannidis
    Codebase: https://github.com/AI-sandbox/iLTM
    License: Apache-2.0
    """

    ag_key = "TA-ILTM"
    ag_name = "TA-iLTM"
    ag_priority = 65
    seed_name = "seed"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda:0" if num_gpus != 0 else "cpu"
        if (device == "cuda:0") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        # iLTM leaks torch backend flags and root logger handlers on import / fit;
        # see _isolate_iltm_global_state docstring for details.
        with _isolate_iltm_global_state():
            from iltm import iLTMClassifier, iLTMRegressor

            _ensure_iltm_logger_patched()

            if self.problem_type in ["binary", "multiclass"]:
                model_cls = iLTMClassifier
            elif self.problem_type == "regression":
                model_cls = iLTMRegressor
            else:
                raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

            hps = self._get_model_params()

            self.model = model_cls(
                **hps,
                device=device,
            )

            X = self.preprocess(X, y=y)
            if X_val is not None:
                X_val = self.preprocess(X_val)
                eval_set = (X_val, y_val)
            else:
                eval_set = None

            self.model = self.model.fit(
                X=X,
                y=y,
                eval_set=eval_set,
                fit_max_time=time_limit,
            )

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _predict_proba(self, X, **kwargs):
        # See _ensure_iltm_logger_patched docstring: bagged child models are
        # unpickled in the parent process without running iLTM's __init__, so
        # the module-level `logger` referenced inside predict() is undefined
        # unless we patch it here.
        _ensure_iltm_logger_patched()
        return super()._predict_proba(X, **kwargs)

    def get_device(self) -> str:
        return str(self.model.device)

    def _set_device(self, device: str):
        self.model.device = device
        if getattr(self.model, "_model", None) is not None:
            self.model._model = self.model._model.to(device)

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": False}


def _ensure_iltm_logger_patched() -> None:
    """Workaround for upstream bug in iltm==0.1.0.

    `iltm/inference_interface.py` declares `logger` only inside the predictor's
    `__init__` (via `global logger; logger = logging.getLogger(__name__)`).
    Any call path that reaches `_preprocess_test_data` (and other methods that
    log via `logger.debug(...)`) without first running `__init__` in the same
    process crashes with `NameError: name 'logger' is not defined`.

    This is exactly what happens with AutoGluon bagging: child models are fit
    in a Ray worker and pickled back to the parent; predict in the parent
    unpickles them but never re-runs `__init__`, so the parent's
    `iltm.inference_interface` module has no `logger` attribute.

    Setting the attribute on the module is idempotent and safe to call from
    anywhere iLTM might be used. Remove when fixed upstream.
    """
    import iltm.inference_interface as _ifi

    if not hasattr(_ifi, "logger"):
        _ifi.logger = logging.getLogger(_ifi.__name__)


@contextmanager
def _isolate_iltm_global_state():
    """Save/restore process-wide globals that the iLTM library mutates during fit.

    Upstream bugs in iltm==0.1.0 (https://github.com/AI-sandbox/iLTM) that leak
    out of fit() and pollute the host process:

    1. `iltm/inference_interface.py` sets `torch.backends.cuda.matmul.allow_tf32 = True`
       at module import time (not under a context manager / not reset).
    2. `iltm/utils.py::set_seed` sets `torch.backends.cudnn.deterministic = True`
       (and `cudnn.benchmark = False`) every time a predictor is generated.
    3. `iltm/log_config.py::setup_logging` calls `logging.getLogger().handlers = []`
       and then `logging.basicConfig(...)`, wiping the root logger's handlers and
       replacing them with its own `StreamHandler`.

    A library should not silently mutate global torch backend flags or the host
    application's root logger. AutoGluon's `FitHelper.verify_model` snapshots
    these globals and asserts they're unchanged after fit, so without this guard
    the model fails the test suite. Remove this wrapper if iLTM upstream stops
    leaking these settings.
    """
    import torch

    saved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn_deterministic = torch.backends.cudnn.deterministic
    saved_cudnn_benchmark = torch.backends.cudnn.benchmark

    root_logger = logging.getLogger()
    saved_handlers = list(root_logger.handlers)
    saved_root_level = root_logger.level

    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_allow_tf32
        torch.backends.cudnn.deterministic = saved_cudnn_deterministic
        torch.backends.cudnn.benchmark = saved_cudnn_benchmark
        root_logger.handlers = saved_handlers
        root_logger.setLevel(saved_root_level)