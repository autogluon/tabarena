from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract import SharedWeights
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
    concrete :class:`TabDPTModel` (v1.1) / :class:`TabDPTTurboModel` (v1.2) / :class:`TabDPTv13Model`
    (v1.3) subclasses.

    Paper: "TabDPT: Scaling Tabular Foundation Models on Real Data" (NeurIPS 2025).
    Authors: Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari,
        Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, Maksims Volkovs.
    Codebase: https://github.com/layer6ai-labs/TabDPT-inference
    License: Apache-2.0.
    """

    ag_key = "NOTSET"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabdpt", "huggingface_hub")
    ag_name = "NOTSET"
    ag_priority = 65
    seed_name = "seed"
    default_random_seed = 0

    #: Hugging Face repo hosting every TabDPT checkpoint.
    _hf_repo_id: ClassVar[str] = "Layer6/TabDPT"
    #: Commit pinned so checkpoints fetched here never silently change if the
    #: repo's default branch moves. Bump deliberately (with a note on what
    #: changed) when picking up newer checkpoints. A version whose checkpoint
    #: was uploaded after this commit overrides the pin on its own class.
    _hf_revision: ClassVar[str] = "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"
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
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 0.5

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
                revision=cls._hf_revision,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(
                repo_id=cls._hf_repo_id,
                filename=cls._checkpoint_filename,
                revision=cls._hf_revision,
            )

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

    Needs ``tabdpt>=1.2.0,<1.3``: the 1.3 release renamed the network's label encoders
    (``y_encoders`` became ``cls_y_encoders`` and ``reg_y_encoders``), so neither package version
    loads the other's checkpoint. :class:`TabDPTv13Model` is the installable entry and this one is
    ``superseded`` in ``info.py``. The 1.2 release builds its network inside the estimator
    constructor with no separable call, so this version declares no ``shared_weights`` and reads
    its checkpoint per fit, like v1.1; the 1.3 line shares (see :class:`TabDPTv13Model`).
    """

    ag_key = "TA-TABDPT-TURBO"
    ag_name = "TA-TabDPT-Turbo"
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_ensembles": 1}

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


class TabDPTv13Model(TabDPTTurboModel):
    """TabDPT v1.3.

    The v1.3 release keeps the v1.2 estimator surface (constructor arguments, predict knobs and
    their defaults) and ships weights retrained after small architecture changes: separate label
    encoders for classification and regression, plus a probabilistic regression output that the
    wrapper leaves at the ``"mean"`` point prediction. Upstream reports better predictive
    performance than v1.2 on CC18 and CTR23. This class extends :class:`TabDPTTurboModel` and pins
    the v1.3 checkpoint; see :class:`TabDPTModelBase` for the shared implementation and paper /
    codebase / license details.

    Release notes: https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.0

    Needs the ``tabdpt`` commit pinned in ``info.py``, the merge of layer6ai-labs/TabDPT-inference#79:
    it gives ``TabDPTEstimator`` a separable ``_load_model`` that the constructor calls once, which is
    this version's shared-weights loader. The 1.3.0 release predates it. :class:`TabDPTTurboModel`
    explains why the 1.2 and 1.3 packages cannot load each other's checkpoint.
    """

    ag_key = "TA-TABDPT-1.3"
    ag_name = "TA-TabDPT-1.3"
    #: The library's own loader is memoized: one build per checkpoint, flash-attention setting,
    #: clipping value and device per process. The constructor resolves ``use_flash`` and stores
    #: ``model_weight_path`` and ``clip_sigma`` before it calls ``_load_model``, so they key the entry.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabdpt.estimator:TabDPTEstimator._load_model",
        key=("model_weight_path", "use_flash", "clip_sigma"),
    )
    #: A later TabDPT version registered as a subclass owns its ``share_weights`` class setting.
    class_settings_per_subclass = True

    #: The commit that uploaded ``tabdpt1_3.safetensors`` (2026-09-08); the base pin predates it.
    _hf_revision: ClassVar[str] = "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a"
    _checkpoint_filename: ClassVar[str] = "tabdpt1_3.safetensors"

    def _shares_weights(self) -> bool:
        """Whether this fit takes its network from the registry; a compiling configuration builds its own.

        ``TabDPTEstimator.fit`` runs ``self.model.compile()`` in place when ``compile`` is set, which
        would write into the module every other fit holds. ``disabled_by`` cannot carry the rule: the
        constructor resolves ``compile`` after ``_load_model`` ran, so the loader's inputs never
        include it.
        """
        compiles = self._get_model_params().get("compile", self._constructor_defaults["compile"])
        return super()._shares_weights() and not compiles
