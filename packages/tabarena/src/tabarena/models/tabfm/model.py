from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


def _resolve_device(device: str | None, num_gpus: int, *, cuda_available: bool) -> str:
    """Resolve the torch device a TabFM fit should run on.

    ``device`` is the wrapper-only hyperparameter (``None``, ``"cpu"``, ``"gpu"``
    or ``"cuda"``); ``num_gpus`` is what AutoGluon allocated for the fit. Returns
    ``"cuda"`` or ``"cpu"``.

    ``None`` derives the device from ``num_gpus`` (GPU when one was allocated). An
    explicit GPU request (``"gpu"``/``"cuda"``) -- or ``None`` with an allocated
    GPU -- raises ``AssertionError`` when ``cuda_available`` is False rather than
    silently falling back to CPU.
    """
    if device is not None:
        device = str(device).lower()
    if device == "cpu":
        return "cpu"
    want_gpu = device in ("gpu", "cuda") or (device is None and bool(num_gpus))
    if want_gpu and not cuda_available:
        raise AssertionError(
            "TabFM fit requested a GPU, but torch reports no CUDA device. Install a "
            "CUDA-enabled torch build and ensure a GPU is visible, or set device='cpu'.",
        )
    return "cuda" if want_gpu else "cpu"


class TabFMModel(AbstractTorchModel):
    """TabFM: a tabular foundation model that predicts via in-context learning.

    TabFM is a pre-trained PyTorch model: at inference time it is shown the
    training data as context and predicts on the test rows without any per-dataset
    gradient training. It handles mixed numerical/categorical columns and missing
    values natively (via its own internal preprocessing pipeline), so the
    AutoGluon-side preprocessing is left as a no-op and the typed DataFrame is
    passed straight through.

    Wraps ``AbstractTorchModel`` so AutoGluon manages device placement: the network
    is moved to CPU before being pickled and back onto the training device (when
    available) on load, via ``get_device`` / ``_set_device``.

    Accepts an optional ``device`` hyperparameter: ``None`` (default) selects a GPU
    when AutoGluon allocated one and CPU otherwise, ``"cpu"`` forces CPU execution,
    and ``"gpu"``/``"cuda"`` requires a GPU.

    Paper: TabFM (Tabular Foundation Model)
    Authors: Google Research
    Codebase: https://github.com/google-research/tabfm
    License: Apache-2.0

    Install (PyTorch backend):
        pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
    """

    ag_key = "TA-TABFM"
    ag_name = "TA-TabFM"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        # `random_state` is injected by AutoGluon via `seed_name`; both the
        # classifier and the regressor accept it (and TabFM's other knobs default
        # sensibly), so the remaining params are forwarded as-is. `device` is a
        # wrapper-only knob (the TabFM estimators take their device from the
        # network's parameters), so it is popped here.
        hps = self._get_model_params()
        device = _resolve_device(
            hps.pop("device", None),
            num_gpus,
            cuda_available=torch.cuda.is_available(),
        )

        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

        if self.problem_type in ["binary", "multiclass"]:
            model_type = "classification"
            model_cls = TabFMClassifier
        elif self.problem_type == "regression":
            model_type = "regression"
            model_cls = TabFMRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Downloads the pre-trained PyTorch checkpoint from Hugging Face on first
        # use (see `prefetch_weights`); a no-op once cached. Loading with `device`
        # places the network there, which is where the estimator runs.
        base_model = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)

        self.model = model_cls(model=base_model, **hps)

        # Does nothing (TabFM handles categoricals/missing natively); kept for
        # future preprocessing extensions and parity with the other wrappers.
        X = self.preprocess(X, y=y)
        self.model = self.model.fit(X=X, y=y)

    def get_device(self) -> str:
        """Return the torch device of the fitted TabFM network."""
        param = next(self.model.model.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        """Move the fitted TabFM network to ``device`` (the estimator follows it)."""
        if getattr(self.model, "model", None) is not None:
            self.model.model.to(device)

    def __getstate__(self) -> dict:
        """Return a picklable state for AutoGluon's pickle-based ``save``.

        The fitted TabFM network (``self.model.model``) carries lambda activations
        built by ``get_activation`` that stdlib pickle cannot serialise. The
        network is dropped from a copy of the estimator here and reloaded from the
        cached checkpoint in ``__setstate__``; the live estimator is left intact so
        in-memory prediction after a save still works.
        """
        state = self.__dict__.copy()
        inner = state.get("model")
        if inner is not None:
            stripped = inner.__class__.__new__(inner.__class__)
            stripped.__dict__.update({k: v for k, v in inner.__dict__.items() if k != "model"})
            state["model"] = stripped
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the estimator and reload the network dropped on save."""
        self.__dict__.update(state)
        inner = self.__dict__.get("model")
        if inner is not None and getattr(inner, "model", None) is None:
            from tabfm import tabfm_v1_0_0_pytorch

            model_type = "classification" if self.problem_type in ["binary", "multiclass"] else "regression"
            inner.model = tabfm_v1_0_0_pytorch.load(model_type=model_type)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks.
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(
        self,
        is_gpu_available: bool = False,
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        refit_folds avoids storing one in-context model per fold (each carries the
        full training context), refitting a single model on all data instead.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            },
        )
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: support memory estimate!
        tags = super()._class_tags()
        tags["can_estimate_memory_usage_static"] = False
        return tags

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def prefetch_weights() -> None:
    """Pre-download the TabFM v1.0.0 PyTorch checkpoint from Hugging Face.

    Warms the local cache (``google/tabfm-1.0.0-pytorch``) so parallel / offline
    fits do not race on the download.
    """
    from huggingface_hub import snapshot_download
    from tabfm.src.pytorch.tabfm_v1_0_0 import HF_REPO_ID

    snapshot_download(repo_id=HF_REPO_ID)
