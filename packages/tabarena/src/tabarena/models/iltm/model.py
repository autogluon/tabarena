from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.prefetch import WeightsUnavailableError

if TYPE_CHECKING:
    import pandas as pd


#: Hugging Face repository the library downloads its checkpoints from (``iltm.model_checkpoints.HF_REPO_ID``).
HF_REPO_ID = "dbonet/iLTM"
#: Environment variable the library reads for its checkpoint directory (``iltm.model_checkpoints.CKPT_DIR_ENV``).
CKPT_DIR_ENV = "ILTM_CKPT_DIR"
#: The checkpoint names the library resolves (``iltm.AVAILABLE_CHECKPOINTS``); ``<name>.pth`` in the repository.
CHECKPOINT_NAMES: tuple[str, ...] = ("xgbrconcat", "cbrconcat", "r128bn", "rnobn", "xgb", "catb", "rtr", "rtrcb")


def checkpoint_dir() -> Path:
    """The directory the library reads its checkpoints from, created when missing.

    The same rule as ``iltm.model_checkpoints._get_default_ckpt_dir``: ``ILTM_CKPT_DIR`` when the
    variable is set, else the platform cache directory (``$XDG_CACHE_HOME/iltm`` or ``~/.cache/iltm``
    on Linux, ``~/Library/Caches/iltm`` on macOS, ``%LOCALAPPDATA%/iltm`` on Windows). Replicated here
    so the head node can prefetch without importing ``iltm`` (see :func:`_isolate_iltm_global_state`).
    """
    if CKPT_DIR_ENV in os.environ:
        path = Path(os.environ[CKPT_DIR_ENV])
    elif sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        path = Path(appdata) / "iltm" if appdata else Path.home() / ".iltm"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Caches" / "iltm"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        path = Path(xdg_cache) / "iltm" if xdg_cache else Path.home() / ".cache" / "iltm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_filename(name: str) -> str:
    """The repository file of checkpoint ``name`` (``xgbrconcat`` selects ``xgbrconcat.pth``)."""
    return f"{name}.pth"


def resolve_checkpoint(name: str, *, allow_download: bool = True) -> str:
    """Local path of checkpoint ``name`` in the library's checkpoint directory, downloading only when allowed.

    Mirrors ``iltm.model_checkpoints._ensure_checkpoint``: a file already in :func:`checkpoint_dir` is
    returned without any Hub request, otherwise ``hf_hub_download(local_dir=...)`` places it there,
    which is exactly where the estimator's own resolution of the checkpoint name looks at fit time.

    Raises:
        WeightsUnavailableError: The file is missing and ``allow_download`` is False.
    """
    directory = checkpoint_dir()
    filename = checkpoint_filename(name)
    local_path = directory / filename
    if local_path.is_file():
        return str(local_path)
    if not allow_download:
        raise WeightsUnavailableError(
            f"iLTM checkpoint {filename} is not in {directory} and downloading is not allowed. Prefetch it on a "
            "node with network access (tabarena.models.prefetch.prefetch_weights) or allow the download."
        )
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(directory)))


class ILTMModel(AbstractTorchModel):
    """iLTM: Integrated Large Tabular Model.

    The wrapper declares no ``shared_weights``: the library
    shares the network itself through the class-level ``_iLTMBase._model_cache`` (keyed by checkpoint
    path, device string and the architecture hyperparameters), so the fold and refit children of one
    process already build the network once; from iltm 0.1.4 the estimator also releases the network
    after every fit, moves the cached module to the CPU and back per child, and pickles without it.
    Injecting a registry module would mean overriding those version-dependent hooks, so the library's
    own cache is left to do the sharing; the warm-up's dummy fit fills it untimed. The checkpoint config (preprocessing, tree embedding, retrieval, bottleneck) is keyed off the
    checkpoint *name*, so the name is passed through unchanged and :meth:`prefetch_weights` fills the
    library's own checkpoint directory, which its resolution reads local-first at fit time.

    Paper: iLTM: Integrated Large Tabular Model (arXiv:2511.15941)
    Authors: Bonet, Comajoan Cara, Calafell, Mas Montserrat, Ioannidis
    Codebase: https://github.com/AI-sandbox/iLTM
    License: Apache-2.0
    """

    ag_key = "TA-ILTM"
    ag_name = "TA-iLTM"
    ag_priority = 65
    seed_name = "seed"

    _supported_problem_types = ["binary", "multiclass", "regression"]

    _categorical_indices: list[int] | None
    """The indices of the categorical features, detected during preprocessing."""
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    warmup_modules: ClassVar[tuple[str, ...]] = ("iltm",)
    """Importing ``iltm`` sets ``torch.backends.cuda.matmul.allow_tf32`` as a side effect (see
    :func:`_isolate_iltm_global_state`). ``_fit`` pins that flag explicitly, so where the import happens
    no longer changes a fit, and the warm-up may import the library."""

    #: Knobs that make the warm-up's dummy fit cheap (one ensemble member, one fine-tuning epoch, a
    #: tiny tree embedding); none of them selects or shapes the checkpoint. The dummy fit also fills
    #: the library's class-level ``_model_cache`` untimed, which is the sharing win for iLTM.
    cheap_hyperparameters: ClassVar[dict] = {
        "n_ensemble": 1,
        "finetuning_max_steps": 1,
        "tree_n_estimators": 2,
    }

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Detect indices of pandas `category`-dtype columns for iLTM's `cat_features`.

        iLTM only auto-detects object-dtype string columns as categorical
        (iltm/inference_interface.py::detect_object_string_columns). AutoGluon's
        CategoryFeatureGenerator converts categoricals to pandas 'category' dtype,
        which CatBoost (used in iLTM's tree embedding) then rejects unless the
        column index is listed in cat_features.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            if categorical_cols:
                self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            else:
                self._categorical_indices = None

        return X

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
        fit_start = time.monotonic()

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

        with _isolate_iltm_global_state():
            from iltm import iLTMClassifier, iLTMRegressor

            if self.problem_type in ["binary", "multiclass"]:
                model_cls = iLTMClassifier
            elif self.problem_type == "regression":
                model_cls = iLTMRegressor
            else:
                raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

            hps = self._get_model_params()

            X = self.preprocess(X, y=y, is_train=True)
            if X_val is not None:
                X_val = self.preprocess(X_val)
                eval_set = (X_val, y_val)
            else:
                eval_set = None

            self.model = model_cls(
                **hps,
                device=device,
                cat_features=self._categorical_indices,
            )

            remaining_time = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - fit_start))
            self.model = self.model.fit(
                X=X,
                y=y,
                eval_set=eval_set,
                fit_max_time=remaining_time,
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        with _isolate_iltm_global_state():
            return super()._predict_proba(X, **kwargs)

    def prepare_for_inference(self) -> None:
        """Untimed, idempotent: keep the network in eval mode and synchronize CUDA.

        Never touches data and runs no forward pass. The network attribute is ``None`` on an estimator
        that released it after the fit (iltm 0.1.4 and later); the fitted predictors then carry
        everything the predict needs.
        """
        if self.model is None:
            return
        network = getattr(self.model, "_model", None)
        if network is None:
            return
        network.eval()
        if next(network.parameters()).device.type == "cuda":
            import torch

            torch.cuda.synchronize()

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Make every checkpoint the search space can select present in the library's checkpoint directory.

        Returns the local paths. A file already present is kept without a Hub request; a missing one
        is downloaded the way the estimator would download it at fit time, so a compute node that
        shares the directory (or a bundle prefetched on the same node) never downloads inside the timer.
        """
        return [resolve_checkpoint(name) for name in CHECKPOINT_NAMES]

    def get_device(self) -> str:
        return str(self.model.device)

    def _set_device(self, device: str):
        self.model.device = device
        if getattr(self.model, "_model", None) is not None:
            self.model._model = self.model._model.to(device)


@contextmanager
def _isolate_iltm_global_state():
    import torch

    saved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn_deterministic = torch.backends.cudnn.deterministic
    saved_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_allow_tf32
        torch.backends.cudnn.deterministic = saved_cudnn_deterministic
        torch.backends.cudnn.benchmark = saved_cudnn_benchmark
