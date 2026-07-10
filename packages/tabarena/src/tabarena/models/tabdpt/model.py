from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class TabDPTModelBase(AbstractTorchModel):
    """Shared AutoGluon wrapper for the TabDPT tabular foundation model.

    TabDPT is a tabular foundation model that performs in-context learning: one pre-trained
    transformer conditions on (a subset of) the training rows at inference time, with no
    per-dataset gradient training.

    This base holds everything common across TabDPT versions (preprocessing, device /
    flash-attention handling, resources, prediction, memory estimate). A concrete subclass pins a
    version purely by declaring:

    * :attr:`_constructor_defaults` — the estimator constructor kwargs (and this version's default
      values) to forward, so a version never receives a kwarg its ``tabdpt`` release doesn't accept
      and each version recovers its own defaults;
    * :attr:`_predict_hp_names` — the predict-time hyperparameters accepted per task.

    TabDPT auto-selects the matching checkpoint from the installed ``tabdpt`` package, so there is
    no per-version checkpoint path to set. Not registered directly (no ``info.py`` entry); use the
    concrete :class:`TabDPTModel` (v1.1) / :class:`TabDPTTurboModel` (v1.2) subclasses.

    Paper: "TabDPT: Scaling Tabular Foundation Models on Real Data" (NeurIPS 2025).
    Authors: Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari,
        Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, Maksims Volkovs.
    Codebase: https://github.com/layer6ai-labs/TabDPT-inference
    License: Apache-2.0.
    """

    ag_key = "NOTSET"
    ag_name = "NOTSET"
    ag_priority = 65
    seed_name = "seed"
    default_random_seed = 0

    #: Hugging Face repo hosting every TabDPT checkpoint.
    _hf_repo_id: ClassVar[str] = "Layer6/TabDPT"
    #: This version's checkpoint filename in :attr:`_hf_repo_id`. The installed ``tabdpt`` package
    #: hardcodes a single version (``tabdpt<VER>.safetensors``), so we pin the correct weights per
    #: version explicitly via ``model_weight_path`` rather than relying on the package default —
    #: otherwise every version would load whatever weights the installed package points at.
    #: Set per concrete subclass.
    _checkpoint_filename: ClassVar[str | None] = None

    #: Estimator constructor kwargs forwarded for this version, mapped to the version's default
    #: value (resolved from the fit hyperparameters, falling back to the default). Overridden per
    #: concrete subclass; ``device`` / ``use_flash`` / ``model_weight_path`` are always added on
    #: top in :meth:`_init_tabdpt_model`.
    _constructor_defaults: ClassVar[dict[str, object]] = {}
    #: Predict-time hyperparameters accepted by this version, split by task. ``temperature`` /
    #: ``permute_classes`` are classification-only. Overridden per concrete subclass.
    _predict_hp_names: ClassVar[dict[str, tuple[str, ...]]] = {"classifier": (), "regressor": ()}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None
        self._use_flash_og = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        is_classifier = self.problem_type in [BINARY, MULTICLASS]
        model_cls = TabDPTClassifier if is_classifier else TabDPTRegressor
        supported_predict_hps = self._predict_hp_names["classifier" if is_classifier else "regressor"]

        hps = self._get_model_params()
        random_seed = hps.pop(self.seed_name, self.default_random_seed)
        self._predict_hps = {k: v for k, v in hps.items() if k in supported_predict_hps}
        self._predict_hps["seed"] = random_seed
        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        self.model = self._init_tabdpt_model(model_cls=model_cls, device=device, hps=hps)
        self.model.fit(X=X, y=y)

    def _init_tabdpt_model(self, *, model_cls, device: str, hps: dict):
        """Construct (but do not fit) the underlying TabDPT estimator.

        Shared across versions: forwards ``device`` / ``use_flash`` / this version's checkpoint
        (``model_weight_path``) plus this version's :attr:`_constructor_defaults` (each resolved
        from ``hps`` with the version's default).
        """
        kwargs = {
            "device": device,
            "use_flash": self._use_flash(),
            "model_weight_path": self._download_checkpoint(),
        }
        for param, default in self._constructor_defaults.items():
            kwargs[param] = hps.get(param, default)
        return model_cls(**kwargs)

    @classmethod
    def _download_checkpoint(cls) -> str:
        """Resolve this version's checkpoint to a local path (from cache, else download).

        Tries the local cache first so prefetched / offline compute nodes skip the etag
        HEAD-request that ``hf_hub_download`` makes by default.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        assert cls._checkpoint_filename is not None, (
            f"{cls.__name__} must set `_checkpoint_filename` to pin its TabDPT weights."
        )
        try:
            return hf_hub_download(
                repo_id=cls._hf_repo_id,
                filename=cls._checkpoint_filename,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(repo_id=cls._hf_repo_id, filename=cls._checkpoint_filename)

    @classmethod
    def prefetch_weights(cls) -> str:
        """Pre-download this version's TabDPT checkpoint (warms the cache for offline/parallel fits)."""
        return cls._download_checkpoint()

    @staticmethod
    def _use_flash() -> bool:
        """Detect if torch's native flash attention is available on the current machine."""
        import torch

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)

        return capability != (7, 5)

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        self._use_flash_og = self.model.use_flash
        return self

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.use_flash = False
            self.model.model.use_flash = False
        else:
            self.model.use_flash = self._use_flash_og
            self.model.model.use_flash = self._use_flash_og

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(
        self,
        is_gpu_available: bool = False,
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X, **self._predict_hps)

        y_pred_proba = self.model.ensemble_predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabDPT requires numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X,
            )
        return X.to_numpy()

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        # `sequential_local` fold fitting avoids contention on the shared HF checkpoint cache;
        # `refit_folds` refits a single model on all data for faster inference at similar quality.
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    # FIXME: This is copied from TabPFN, but TabDPT is not the same
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147.

        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        memory_estimate = model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est

        # TabDPT memory estimation is very inaccurate because it is using TabPFN memory estimate. Double it to be safe.
        memory_estimate = memory_estimate * 2

        # Note: This memory estimate is way off if `context_size` is not None
        return int(memory_estimate)


class TabDPTModel(TabDPTModelBase):
    """TabDPT v1.1 (the original TabArena-benchmarked release).

    Uses FAISS retrieval as its default context reduction (the v1.1 default) and the v1.1
    constructor defaults. See :class:`TabDPTModelBase` for the shared implementation and paper /
    codebase / license details.
    """

    ag_key = "TA-TABDPT"
    ag_name = "TA-TabDPT"

    _checkpoint_filename: ClassVar[str] = "tabdpt1_1.safetensors"
    _constructor_defaults: ClassVar[dict[str, object]] = {
        "normalizer": "standard",
        "missing_indicators": False,
        "clip_sigma": 4,
        "feature_reduction": "pca",
        "faiss_metric": "l2",
    }
    _predict_hp_names: ClassVar[dict[str, tuple[str, ...]]] = {
        "classifier": ("context_size", "permute_classes", "temperature"),
        "regressor": ("context_size",),
    }


class TabDPTTurboModel(TabDPTModelBase):
    """TabDPT-Turbo (TabDPT v1.2).

    Accelerates fitting and inference by ~120x on average on TabArena versus v1.1 while improving
    predictive performance, chiefly by defaulting to subsampled context reduction (instead of
    v1.1's FAISS retrieval) plus long-context support and updated weights. Exposes the v1.2 predict
    knobs (``n_ensembles`` / ``batch_size``) and constructor surface (``compile`` / ``verbose`` /
    ``context_reduction``); see :class:`TabDPTModelBase` for the shared implementation.

    Paper: "TabDPT-Turbo" — https://openreview.net/pdf?id=Y00pwFyrHR

    Both wrappers share the ``tabdpt`` pip package (extra pinned to ``tabdpt>=1.2.0``), so a shared
    install runs v1.2 for both; the v1.1 wrapper then uses v1.2 defaults.
    """

    ag_key = "TA-TABDPT-TURBO"
    ag_name = "TA-TabDPT-Turbo"

    _checkpoint_filename: ClassVar[str] = "tabdpt1_2.safetensors"
    _constructor_defaults: ClassVar[dict[str, object]] = {
        # `compile` is off by default: torch.compile adds per-fit compilation overhead (costly
        # across TabArena's many small bagged/refit fits) and a compiled module complicates
        # AutoGluon's CPU-save / GPU-reload pickling cycle. The core Turbo speedup comes from
        # context_reduction="subsample" + the v1.2 weights, both kept below.
        "compile": False,
        "verbose": False,
        "normalizer": "standard",
        "missing_indicators": False,
        "clip_sigma": 8,  # v1.2 default (v1.1 uses 4)
        "feature_reduction": "pca",
        "context_reduction": "subsample",
        "faiss_metric": "l2",
    }
    _predict_hp_names: ClassVar[dict[str, tuple[str, ...]]] = {
        # v1.2 adds `n_ensembles` / `batch_size` to both tasks; `temperature` / `permute_classes`
        # remain classification-only (the regressor's predict() rejects them).
        "classifier": ("n_ensembles", "context_size", "batch_size", "permute_classes", "temperature"),
        "regressor": ("n_ensembles", "context_size", "batch_size"),
    }
