from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.prefetch import WeightsUnavailableError

if TYPE_CHECKING:
    import pandas as pd


#: Hugging Face repository the library downloads its checkpoints from (``iltm.model_checkpoints.HF_REPO_ID``).
HF_REPO_ID = "dbonet/iLTM"
#: Commit of :data:`HF_REPO_ID` the prefetch downloads from, so the checkpoints never change silently when the
#: repository's default branch moves (the ``.pth`` files are unchanged since b3dcd2bd, 2025-11-18). Bump
#: deliberately, with a note on what changed.
HF_REVISION = "ef783ec2a246686ad4bd43f35b590c196e70a1ba"
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

    return str(hf_hub_download(repo_id=HF_REPO_ID, filename=filename, revision=HF_REVISION, local_dir=str(directory)))


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

    Validation data: every selection step runs on the passed ``X_val`` / ``y_val``. The network's
    fine-tuning early-stops on it and the adaptive retrieval alpha is chosen on it, both by the
    library's default. The tree embedding needs ``tree_use_external_eval_set=True`` (a default
    parameter of this wrapper): with the library's default it early-stops each tree on a 20% split of
    the fold's training rows, which a model submission may not do (autogluon/tabarena#637).

    Known gaps (iltm 0.1.6):

    - ``num_cpus`` is not passed on. The tree embedding fits XGBoost with ``n_jobs=-1`` and CatBoost
      with ``thread_count=-1`` (``iltm/tree_embedding.py``), and the library offers no thread argument.
    - ``fit_max_time`` is checked before each ensemble member and each fine-tuning step, not during the
      tree fit, the first fine-tuning step, or the adaptive-alpha predict on the validation rows after
      the loop, so a fit can overrun ``time_limit`` by about one member's non-fine-tuning work. With
      less than 10 s left the library raises a plain ``TimeoutError``.
    - ``auto_stop_on_low_cpu_memory`` (on by default) stops adding ensemble members when host memory
      runs low, so the member count can depend on the node.

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

    #: The checkpoints encode the label as a 100-wide one-hot (``n_classes_limit``), so a task with more
    #: classes would fail only after its tree embedding was fit; AutoGluon skips it up front instead.
    _default_auxiliary_params_extra = {"max_classes": 100}

    warmup_modules: ClassVar[tuple[str, ...]] = ("iltm",)
    """Already imported by :meth:`warmup` inside :func:`_isolate_iltm_global_state` (the import changes the
    process's float32 matmul precision); listed so the Ray worker warm-up imports it too."""

    #: Knobs that make the warm-up's dummy fit cheap (a short fine-tuning, a tiny tree embedding). Three
    #: ensemble members, because a ``checkpoint_mix`` config cycles its members through cbrconcat,
    #: xgbrconcat and rtr: the dummy fit then loads every checkpoint the config can use into the library's
    #: class-level ``_model_cache`` untimed, which is the sharing win for iLTM.
    cheap_hyperparameters: ClassVar[dict] = {
        "n_ensemble": 3,
        "finetuning_max_steps": 1,
        "tree_n_estimators": 2,
    }

    @classmethod
    def warmup(cls, **kwargs) -> None:
        """Import ``iltm`` inside :func:`_isolate_iltm_global_state`, before the generic module layer does.

        The library's first import in a process changes the float32 matmul precision; importing it here
        restores the caller's precision afterwards, and every later ``import iltm`` is a no-op.
        """
        with _isolate_iltm_global_state():
            pass

    def _set_default_params(self):
        # Early-stop the tree embedding on the passed validation rows, not on a split of the training rows.
        default_params = {"tree_use_external_eval_set": True}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

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
        X = self.preprocess(X, **kwargs)
        with _isolate_iltm_global_state():
            if self.problem_type == "regression":
                return self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(self._align_proba(y_pred_proba))

    def _align_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Widen the probabilities to the task's full class set.

        The library encodes only the classes present in the fold's training rows (``np.unique(y)``), so a
        bagged fold that misses a class (grouped and temporal inner splits make that common) returns one
        column per seen class. Put each column back under its label and leave the missing class at zero.
        """
        num_classes = self.num_classes
        if num_classes is None or y_pred_proba.ndim != 2 or y_pred_proba.shape[1] == num_classes:
            return y_pred_proba
        full = np.zeros((y_pred_proba.shape[0], num_classes), dtype=y_pred_proba.dtype)
        full[:, np.asarray(self.model.classes_, dtype=int)] = y_pred_proba
        return full

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
    """Run iLTM at full fp32 and leave the process-global state as the caller had it.

    Developer fix. ``iltm/inference_interface.py`` (iltm 0.1.6) calls
    ``torch.set_float32_matmul_precision("high")`` and then the legacy ``allow_tf32 = True`` at import,
    which turns TF32 on for CUDA and oneDNN matmuls for the rest of the process and mixes torch's two
    precision APIs (reading ``allow_tf32`` afterwards raises). Every fit also reseeds through
    ``iltm.utils.seed_everything``, which sets ``PYTHONHASHSEED``, and changes the cuDNN determinism flags.

    The context imports the library, so its first import in a process always happens here, then pins
    float32 matmuls to ``"highest"`` (the maintainers' decision of 2026-09-16: iLTM is fit and scored at
    full fp32, whichever code imported the library first), and on exit restores the matmul precision,
    the cuDNN flags and ``PYTHONHASHSEED``. The library should leave the precision to the caller, or
    scope TF32 to its own fit and predict behind a parameter; asked upstream in autogluon/tabarena#687.
    """
    import torch

    cuda_matmul = torch.backends.cuda.matmul
    mkldnn_matmul = getattr(getattr(torch.backends, "mkldnn", None), "matmul", None)
    precision_attrs = [obj for obj in (cuda_matmul, mkldnn_matmul) if hasattr(obj, "fp32_precision")]
    try:
        saved_precision = torch.get_float32_matmul_precision()
    except RuntimeError:  # the two precision APIs were mixed before this context (e.g. an unguarded import)
        saved_precision = "highest"
    saved_fp32_precisions = [obj.fp32_precision for obj in precision_attrs]
    saved_cudnn_deterministic = torch.backends.cudnn.deterministic
    saved_cudnn_benchmark = torch.backends.cudnn.benchmark
    saved_hash_seed = os.environ.get("PYTHONHASHSEED")

    try:
        import iltm  # noqa: F401  # the first import changes the precision, so pin it afterwards

        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        # The legacy-free reset first, then the caller's per-backend values on top of it.
        torch.set_float32_matmul_precision(saved_precision)
        for obj, value in zip(precision_attrs, saved_fp32_precisions, strict=False):
            obj.fp32_precision = value
        torch.backends.cudnn.deterministic = saved_cudnn_deterministic
        torch.backends.cudnn.benchmark = saved_cudnn_benchmark
        if saved_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = saved_hash_seed
