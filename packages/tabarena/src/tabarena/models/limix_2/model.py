from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)

_DEFAULT_HF_REPO = "stable-ai/LimiX-2"
_DEFAULT_HF_FILENAME = "LimiX-2.ckpt"
#: Commit pinned so the checkpoint fetched here never silently changes if the
#: repo's default branch moves. Bump deliberately when picking up a newer file.
_DEFAULT_HF_REVISION = "20c07a07801973a0aec57c31062bdb2ea1cda2b2"

_LIMIX_INSTALL_HINT = (
    "LimiX-2 inference code is not installed. Install it with:\n"
    '  pip install "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git"'
)


def _redirect_unwritable_inference_cache() -> None:
    """V2.0 CacheManager mkdir's its default cache root in ``__init__``.

    That default is a cluster path (not always writable). If mkdir fails, point
    subsequent constructions at ``~/.cache/limix/infe_cache``.
    """
    try:
        from inference.v2_0.predictor import LimiXPredictor as V2Predictor
    except ImportError:
        return

    cache_cls = getattr(V2Predictor, "CacheManager", None)
    if cache_cls is None or getattr(cache_cls, "_tabarena_writable_cache", False):
        return

    orig_init = cache_cls.__init__

    def _init(self, cache_dir=None, *args, **kwargs):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "limix" / "infe_cache"
        cache_dir = Path(cache_dir)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache_dir = Path.home() / ".cache" / "limix" / "infe_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("LimiX-2 inference cache is not writable; using %s", cache_dir)
        orig_init(self, str(cache_dir), *args, **kwargs)

    cache_cls.__init__ = _init
    cache_cls._tabarena_writable_cache = True


class LimiX2Model(AbstractTorchModel):
    """LimiX-2 tabular foundation model (in-context learning, no train loop).

    Paper: LimiX-2: A Large Foundation Model for Structured Data (LDM)
    Authors: LimiX Team (Stable AI)
    Codebase: https://github.com/limix-ldm-ai/LimiX
    Weights: https://huggingface.co/stable-ai/LimiX-2
    License: StableAI LimiX Non-Commercial License v1.0 (weights);
        Stable AI Technology Co., Ltd. License, Version 1.0 (code)

    Install the official inference package, then download ``LimiX-2.ckpt`` from Hugging Face
    (this wrapper calls ``hf_hub_download`` with a pinned revision)::

        pip install "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git"

    ``LimiXPredictor.predict`` is in-context: the train table is stored at fit time and
    passed again at predict time. V2.0 regression already returns the original target
    scale, so this wrapper does not standardize or invert ``y``.

    A worker uses one GPU unless ``allow_multi_gpu=True``. AutoGluon bagged refit and
    sequential fold workers can be granted every system GPU; that only becomes
    pipeline-parallel inference when this flag is on.
    """

    ag_key = "TA-LIMIX-2"
    ag_name = "TA-LimiX-2"
    ag_priority = 100

    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    # Keep constant features so train/test column shapes stay aligned for ICL predict.
    _default_auxiliary_params_extra = {
        "max_rows": 100_000,
        "max_classes": 10,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.X_train_processed_: pd.DataFrame | None = None
        self.y_train_processed_: np.ndarray | None = None
        self.device = None
        self.gpu_ids: list[int] | None = None
        self.autobatch_flag = False
        self.softmax_temperature = 0.9
        self.inference_seed = 0
        self.inference_config: dict | list | None = None

    @classmethod
    def _default_param_dict(cls) -> dict:
        return {
            "use_default_params": True,
            "autobatch": True,
            "inference_seed": 0,
            "softmax_temperature": 0.9,
            "allow_multi_gpu": False,
        }

    def _set_default_params(self):
        for param, val in self._default_param_dict().items():
            self._set_default_param_value(param, val)

    def _allow_multi_gpu(self) -> bool:
        return bool(self._get_model_params().get("allow_multi_gpu", False))

    def _get_maximum_resources(self) -> dict[str, int | float]:
        """Keep a worker on one GPU unless ``allow_multi_gpu`` is set."""
        if self._allow_multi_gpu():
            return {}
        return {"num_gpus": 1}

    @staticmethod
    def _to_official_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
        """Cast pandas ``category`` / ``string`` columns to ``object``.

        Official demos load CSV, so categoricals arrive as ``object``. TabArena often
        stores them as ``category`` or pandas ``string``. Official
        ``encode_categorical_features`` also accepts those dtypes; we still
        normalize to ``object`` so the frames match the official CSV-demo path
        before ``LimiXPredictor`` encodes / scales.
        """
        X = X.copy()
        for col in X.columns:
            dtype = X[col].dtype
            if isinstance(dtype, pd.CategoricalDtype) or (
                pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype)
            ):
                X[col] = X[col].astype(object)
        return X

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        return self._to_official_feature_frame(X)

    def _default_inference_config(self) -> dict | list:
        """Load the official LimiX-2 / V2.0 noretrieval JSON shipped with the ``LimiX`` package."""
        try:
            import config as config_pkg
        except ImportError as err:
            raise ImportError(_LIMIX_INSTALL_HINT) from err

        name = (
            "cls_default_noretrieval_v2.json"
            if self.problem_type in [BINARY, MULTICLASS]
            else "reg_default_noretrieval_v2.json"
        )
        path = Path(config_pkg.__file__).with_name(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Packaged LimiX-2 inference config {name!r} was not found at {path}. {_LIMIX_INSTALL_HINT}",
            )
        logger.log(20, "LimiX-2: using inference config %s", path)
        with path.open("r") as f:
            return json.load(f)

    @classmethod
    def prefetch_weights(cls) -> str:
        """Download ``LimiX-2.ckpt`` from Hugging Face and return the local path.

        Tries the local cache first so offline compute nodes skip the etag
        HEAD-request that ``hf_hub_download`` performs by default.
        """
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required to download LimiX-2 weights. "
                "Install huggingface_hub and retry LimiX2Model.prefetch_weights().",
            ) from err

        try:
            try:
                return hf_hub_download(
                    repo_id=_DEFAULT_HF_REPO,
                    filename=_DEFAULT_HF_FILENAME,
                    revision=_DEFAULT_HF_REVISION,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                return hf_hub_download(
                    repo_id=_DEFAULT_HF_REPO,
                    filename=_DEFAULT_HF_FILENAME,
                    revision=_DEFAULT_HF_REVISION,
                )
        except Exception as err:
            raise RuntimeError(
                "Failed to download LimiX-2.ckpt from Hugging Face (stable-ai/LimiX-2). "
                "Install huggingface_hub and ensure network access, or set HF_HOME to a "
                "cache that already contains the file, then call LimiX2Model.prefetch_weights().",
            ) from err

    @classmethod
    def warmup(cls, **kwargs) -> None:
        """Import the installed LimiX inference package (untimed, data-independent)."""
        from tabarena.models.warmup import warmup_imports

        try:
            warmup_imports("limix")
        except ImportError as err:
            raise ImportError(_LIMIX_INSTALL_HINT) from err

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > 0 and (available_num_gpus == 0 or not torch.cuda.is_available()):
            logger.warning(
                "LimiX-2 was asked for %s GPU(s) but CUDA is not available; running on CPU. "
                "Official inference is much slower on CPU.",
                num_gpus,
            )
            num_gpus = 0
        elif num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        self.device = torch.device("cuda" if num_gpus != 0 else "cpu")
        # gpu_ids are logical indices inside this process (0..n-1), not physical ids.
        n_gpus = int(num_gpus) if self.device.type == "cuda" else 0
        allow_multi_gpu = self._allow_multi_gpu()
        if n_gpus >= 2 and not allow_multi_gpu:
            logger.log(20, "LimiX-2: allow_multi_gpu=False, using 1 GPU instead of %s", n_gpus)
            n_gpus = 1
        self.gpu_ids = list(range(n_gpus)) if n_gpus >= 2 else None
        if self.gpu_ids:
            logger.log(20, f"LimiX-2: pipeline-parallel inference on {n_gpus} GPUs {self.gpu_ids}")

        self.X_train_processed_ = self.preprocess(X, y=y, is_train=True)
        self.y_train_processed_ = np.asarray(y)

        hps: dict = self._get_model_params().copy()
        hps.pop("allow_multi_gpu", False)
        self.autobatch_flag = hps.pop("autobatch", True)
        self.softmax_temperature = hps.pop("softmax_temperature", 0.9)
        model_path = hps.pop("model_path", None) or self.prefetch_weights()
        self.inference_seed = hps.pop("inference_seed", 0)
        self.inference_config = hps.pop("inference_config", None) or self._default_inference_config()

        try:
            from limix import LimiXPredictor
            from model.v2_0.autobatch import AutobatchConfig
        except ImportError as err:
            raise ImportError(_LIMIX_INSTALL_HINT) from err

        _redirect_unwritable_inference_cache()
        AutobatchConfig.ENABLE_AUTOBATCH = self.autobatch_flag
        predictor_kwargs = {
            "device": self.device,
            "model_path": model_path,
            "inference_config": self.inference_config,
            "softmax_temperature": self.softmax_temperature,
            "seed": self.inference_seed,
        }
        if self.gpu_ids:
            predictor_kwargs["gpu_ids"] = self.gpu_ids
        self.model = LimiXPredictor(**predictor_kwargs)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Forward the stored train table plus the query; LimiX has no sklearn fit API.

        AutoGluon's binary contract is a 1-d positive-class probability, so the
        (n, 2) output is converted here — bagged fold scoring reads this array
        directly.
        """
        import torch

        X_test = self.preprocess(X, **kwargs)
        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"
        preds = self.model.predict(
            self.X_train_processed_,
            self.y_train_processed_,
            X_test,
            task_type=task_type,
        )
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().to(torch.float32).cpu().numpy()
        else:
            preds = np.asarray(preds, dtype=np.float32)
        return self._convert_proba_to_unified_form(preds)

    def get_device(self) -> str:
        if self.device is None:
            return "cpu"
        if isinstance(self.device, str):
            return self.device
        return self.device.type

    def _set_device(self, device: str):
        import torch

        self.device = torch.device(device)
        if self.model is not None:
            self.model.device = self.device
            if getattr(self.model, "model", None) is not None:
                self.model.model.to(self.device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
