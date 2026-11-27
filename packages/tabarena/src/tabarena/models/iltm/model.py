from __future__ import annotations

import os
import sys
import time
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
    so the head node can prefetch without importing ``iltm`` and torch.
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
    fine-tuning and the tree embedding early-stop on it and the adaptive retrieval alpha is chosen on
    it. ``tree_use_external_eval_set=True`` is the library's default from iltm 0.1.7 and a default
    parameter of this wrapper as well: with ``False`` the trees early-stop on a split of the fold's
    training rows, which a model submission may not do (autogluon/tabarena#637).

    Precision: ``allow_tf32=True`` (a default parameter) runs CUDA float32 matmuls in TF32 during the
    fit and the predict, which the authors consider part of the method. The library sets it per call
    and restores the caller's setting afterwards; nothing changes at import.

    Small tables: a fit on at most :attr:`small_data_max_rows` training rows uses half the configured
    ``n_ensemble`` (8 becomes 4, 12 becomes 6). The authors proposed this in autogluon/tabarena#687 to
    cut the fixed cost per member, which dominates on small tables, where the bagged split already
    averages 8 or 25 fold models.

    Known gaps (iltm 0.1.8):

    - ``fit_max_time`` is checked between steps (before and after each tree fit, each fine-tuning
      step, the adaptive-alpha predict), so a single step can still overrun ``time_limit``. A fit whose
      budget runs out before the first member is complete raises ``TimeoutError``, as does a budget
      of 10 s or less.
    - ``auto_stop_on_low_cpu_memory`` (on by default) stops adding ensemble members when host memory
      runs low, so the member count can depend on the node; the fitted estimator reports it as
      ``n_ensemble_`` and in its log.

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
    #: A bagged fold claims a share of the GPU, so several folds can share one card: AutoGluon gives each fold
    #: the larger of this and ``1 / folds in parallel``, and caps the folds in parallel with the VRAM estimate
    #: (:meth:`_estimate_gpu_memory_usage_static`), the host-RAM estimate and the CPUs.
    minimum_num_gpus = 1 / 64

    #: The checkpoints encode the label as a 100-wide one-hot (``n_classes_limit``), so a task with more
    #: classes would fail only after its tree embedding was fit; AutoGluon skips it up front instead.
    _default_auxiliary_params_extra = {"max_classes": 100}

    warmup_modules: ClassVar[tuple[str, ...]] = ("iltm",)

    #: Knobs that make the warm-up's dummy fit cheap (a short fine-tuning, a tiny tree embedding). Three
    #: ensemble members, because a ``checkpoint_mix`` config cycles its members through cbrconcat,
    #: xgbrconcat and rtr: the dummy fit then loads every checkpoint the config can use into the library's
    #: class-level ``_model_cache`` untimed, which is the sharing win for iLTM.
    cheap_hyperparameters: ClassVar[dict] = {
        "n_ensemble": 3,
        "finetuning_max_steps": 1,
        "tree_n_estimators": 2,
    }

    #: Fits on at most this many training rows use half the configured ``n_ensemble`` (the class docstring).
    small_data_max_rows: ClassVar[int] = 1000

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls, *, X: pd.DataFrame, hyperparameters: dict | None = None, **kwargs
    ) -> int:
        """Peak VRAM of one fit and its predict, CUDA context included.

        Measured as the whole-device NVML peak with iltm 0.1.8 on an RTX PRO 6000, one fit per fresh process
        (the members run one after another, so their count does not change the peak): about 4.3 GB up to
        2,048 training rows, where one fine-tuning batch is the whole retrieval context; 7.6-8.0 GB at 5k-20k
        rows; 8.4-11.5 GB at 100k; 18-19.4 GB at 400k-1M. Wide tables add the fine-tuned random-feature
        matrix (input width x ``rf_size`` in float32, with its gradient and optimizer state), which the library
        caps near 6,000 inputs: a bagged fit with 4 members on 300 rows x 15,000 features peaked at 16 GB, and
        8 of them in parallel at 83 GB, so the term is sized for that (+11 GB at the cap). The others are
        rounded up by about 10%.
        """
        n_rows, n_features = X.shape
        if n_rows <= 2_048:
            base = 4.8e9
        elif n_rows <= 20_000:
            base = 8.8e9
        elif n_rows <= 100_000:
            base = 12.6e9
        else:
            base = 21.5e9
        rf_size = (hyperparameters or {}).get("rf_size", 32_768)
        return int(base + min(n_features, 6_000) * rf_size * 56)

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """Peak host RAM (RSS) of one fit and its predict.

        About 5 GB for the libraries and the checkpoint, plus the dense one-hot tree-leaf matrices, which grow
        with the rows: measured 10 GB at 100k rows, 48 GB at 400k and 118 GB at 1M (iltm 0.1.8, default config).
        """
        return int(5.5e9 + 120_000 * len(X))

    def _set_default_params(self):
        default_params = {
            # Early-stop the tree embedding on the passed validation rows, not on a split of the training rows.
            "tree_use_external_eval_set": True,
            # TF32 for CUDA matmuls during fit and predict, the authors' recipe; scoped by the library.
            "allow_tf32": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Detect indices of pandas `category`-dtype columns for iLTM's `cat_features`.

        AutoGluon's CategoryFeatureGenerator converts categoricals to pandas 'category' dtype, which
        CatBoost (used in iLTM's tree embedding) rejects unless the column index is listed in
        cat_features. iltm 0.1.7 also detects such columns itself and merges them with the passed list.
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
            n_jobs=num_cpus,
        )
        if len(X) <= self.small_data_max_rows:
            self.model.set_params(n_ensemble=max(1, self.model.n_ensemble // 2))

        remaining_time = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - fit_start))
        self.model = self.model.fit(
            X=X,
            y=y,
            eval_set=eval_set,
            fit_max_time=remaining_time,
        )

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        X = self.preprocess(X, **kwargs)
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
