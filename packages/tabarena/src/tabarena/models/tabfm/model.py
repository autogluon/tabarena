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


def _build_tabfm_estimator(*, problem_type: str, device: str, interface: str, **hps):
    """Construct (but do not fit) a TabFM sklearn-style estimator for ``problem_type``.

    The single place both the AutoGluon wrapper (:class:`TabFMModel`) and the system model
    (:class:`~tabarena.models.tabfm.system.TabFMPlusSystemModel`) build a TabFM estimator, so the
    two never drift. ``interface`` selects the estimator's construction preset:

    * ``"default"`` — the plain ``TabFMClassifier`` / ``TabFMRegressor`` constructor.
    * ``"ensemble"`` — the ``.ensemble(...)`` preset (square-root feature-cross / SVD schedules,
      NNLS-weighted blending, probability averaging, per-problem calibration).

    ``interface`` is validated before the (cached) checkpoint is loaded, so an invalid value fails
    fast without touching Hugging Face. ``device`` is the resolved torch device (``"cuda"`` /
    ``"cpu"``, see :func:`_resolve_device`); the loaded network is placed there and the estimator
    runs where its network lives. Remaining ``hps`` are forwarded to the estimator (both the plain
    constructor and ``.ensemble`` accept the same keywords, e.g. ``random_state``).
    """
    if interface not in ("default", "ensemble"):
        raise ValueError(f"Unknown TabFM interface {interface!r}; expected 'default' or 'ensemble'.")

    from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

    if problem_type in ["binary", "multiclass"]:
        model_type, model_cls = "classification", TabFMClassifier
    elif problem_type == "regression":
        model_type, model_cls = "regression", TabFMRegressor
    else:
        raise AssertionError(f"Unsupported problem_type: {problem_type}")

    # Downloads the pre-trained PyTorch checkpoint from Hugging Face on first use (see
    # `prefetch_weights`); a no-op once cached. Loading with `device` places the network there,
    # which is where the estimator runs. The network bounds its own peak activation memory via
    # always-on internal chunking, so large tasks need no wrapper-side handling.
    base_model = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)

    factory = model_cls.ensemble if interface == "ensemble" else model_cls
    return factory(model=base_model, **hps)


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

        self.model = _build_tabfm_estimator(problem_type=self.problem_type, device=device, interface="default", **hps)

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
