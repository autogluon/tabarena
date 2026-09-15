from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.pretrained_weights import PretrainedWeightsUnavailableError, unavailable_message
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models import prefetch as _hub
from tabarena.models._shared_estimators import check_payload_device
from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models._weights import normalize_device

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabarena.models._weights import WeightsKey

logger = logging.getLogger(__name__)

#: Hugging Face repo hosting every TabDPT checkpoint.
HF_REPO_ID = "Layer6/TabDPT"
#: Commit pinned so checkpoints fetched here never silently change if the repo's default branch
#: moves. Bump deliberately (with a note on what changed) when picking up newer checkpoints.
HF_REVISION = "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"


def flash_attention_available(device_type: str) -> bool:
    """Whether torch's native flash attention runs on ``device_type`` on this machine (never on the CPU).

    The library resolves ``use_flash`` the same way: only on CUDA, and not on compute capability
    7.5, whose kernels lack the bfloat16 path ``tabdpt.utils.flash_context`` autocasts to.
    """
    if normalize_device(device_type) != "cuda":
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.device("cuda:0")) != (7, 5)


def _network_flags(hyperparameters: dict, device_type: str) -> dict[str, Any]:
    """The two constructor values the module stores and reads in its forward pass.

    ``use_flash`` picks the autocast dtype and the FLASH SDPA backend (``tabdpt.utils.flash_context``),
    ``clip_sigma`` the outlier clipping in ``TabDPTModel.forward``; the latter is normalized to
    ``float`` so a config's ``8`` and the library default ``8.0`` name the same network.
    """
    return {
        "use_flash": flash_attention_available(device_type),
        "clip_sigma": float(hyperparameters.get("clip_sigma", 8.0)),
    }


def _compile_not_pinned_off(hyperparameters: dict) -> bool:
    """Sharing needs ``compile`` pinned to False: ``TabDPTEstimator.fit`` compiles the module in place otherwise."""
    return hyperparameters.get("compile") is not False


class TabDPTModelBase(AbstractTorchModel):
    """Shared AutoGluon wrapper for the TabDPT tabular foundation model.

    TabDPT is a tabular foundation model that performs in-context learning: one pre-trained
    transformer conditions on (a subset of) the training rows at inference time, with no
    per-dataset gradient training.

    This base holds everything common across TabDPT versions (preprocessing, device /
    flash-attention handling of an estimator that owns its network, resources, prediction, memory
    estimate). A concrete subclass pins a version purely by declaring:

    * :attr:`_constructor_defaults`: the estimator constructor kwargs (and this version's default
      values) to forward, so a version never receives a kwarg its ``tabdpt`` release doesn't accept
      and each version recovers its own defaults;
    * :attr:`_predict_hp_names`: the predict-time hyperparameters accepted per task;
    * :attr:`_checkpoint_filename`: the pinned checkpoint in :data:`HF_REPO_ID`.

    TabDPT auto-selects the matching checkpoint from the installed ``tabdpt`` package, but every
    version here pins its own checkpoint file explicitly. Not registered directly (no ``info.py``
    entry); use the concrete :class:`TabDPTModel` (v1.1) / :class:`TabDPTTurboModel` (v1.2)
    subclasses.

    Paper: "TabDPT: Scaling Tabular Foundation Models on Real Data" (NeurIPS 2025).
    Authors: Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari,
        Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, Maksims Volkovs.
    Codebase: https://github.com/layer6ai-labs/TabDPT-inference
    License: Apache-2.0.
    """

    ag_key = "NOTSET"
    #: ``import tabdpt`` pulls faiss, omegaconf, safetensors and the library's submodules.
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabdpt", "huggingface_hub", "huggingface_hub.errors")
    ag_name = "NOTSET"
    ag_priority = 65
    seed_name = "seed"
    default_random_seed = 0

    #: This version's checkpoint filename in :data:`HF_REPO_ID`. The installed ``tabdpt`` package
    #: hardcodes a single version (``tabdpt<VER>.safetensors``), so the correct weights are pinned
    #: per version explicitly via ``model_weight_path`` rather than relying on the package default;
    #: otherwise every version would load whatever weights the installed package points at.
    #: Set per concrete subclass.
    _checkpoint_filename: ClassVar[str | None] = None

    #: Estimator constructor kwargs forwarded for this version, mapped to the version's default
    #: value (resolved from the fit hyperparameters, falling back to the default). Overridden per
    #: concrete subclass; ``device`` / ``use_flash`` / ``model_weight_path`` are always added on
    #: top in :meth:`_constructor_kwargs`.
    _constructor_defaults: ClassVar[dict[str, object]] = {}
    #: Predict-time hyperparameters accepted by this version, split by task. ``temperature`` /
    #: ``permute_classes`` are classification-only. Overridden per concrete subclass.
    _predict_hp_names: ClassVar[dict[str, tuple[str, ...]]] = {"classifier": (), "regressor": ()}
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 0.5
    # `sequential_local` fold fitting avoids contention on the shared HF checkpoint cache;
    # `refit_folds` refits a single model on all data for faster inference at similar quality.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None
        self._use_flash_og = None

    # --- fit --------------------------------------------------------------------------------------

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
        """Construct (but do not fit) the library estimator, which reads this version's checkpoint itself."""
        from autogluon.common.utils.pretrained_weights import fetch_allowed

        allow_fetch = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="fit")
        checkpoint = self._download_checkpoint(allow_fetch=allow_fetch)
        return model_cls(**self._constructor_kwargs(device=device, hps=hps, checkpoint=checkpoint))

    def _constructor_kwargs(self, *, device: str, hps: dict, checkpoint: str) -> dict:
        """``device`` / ``use_flash`` / ``model_weight_path`` plus this version's :attr:`_constructor_defaults` resolved from ``hps``."""
        kwargs = {"device": device, "use_flash": flash_attention_available("cuda"), "model_weight_path": checkpoint}
        for param, default in self._constructor_defaults.items():
            kwargs[param] = hps.get(param, default)
        return kwargs

    # --- checkpoint resolution --------------------------------------------------------------------

    @classmethod
    def _download_checkpoint(cls, allow_fetch: bool = True, *, stage: str = "fit") -> str:
        """Resolve this version's checkpoint to a local path (from the cache, else download).

        The Hugging Face cache is asked first so prefetched or offline compute nodes skip the etag
        request ``hf_hub_download`` otherwise makes; the network is used only when ``allow_fetch``.

        Raises:
            PretrainedWeightsUnavailableError: The file is not cached and ``allow_fetch`` is False.
        """
        assert cls._checkpoint_filename is not None, (
            f"{cls.__name__} must set `_checkpoint_filename` to pin its TabDPT weights."
        )
        try:
            return _hub.resolve_hf_file(
                HF_REPO_ID, cls._checkpoint_filename, revision=HF_REVISION, allow_download=allow_fetch
            )
        except _hub.WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=cls.__name__, stage=stage, location=str(exc))
            ) from exc

    @classmethod
    def prefetch_weights(cls) -> str:
        """Download this version's checkpoint if it is not cached yet; returns its local path."""
        return cls._download_checkpoint(allow_fetch=True)

    # --- device and inference (an estimator that owns its network) --------------------------------

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        self._use_flash_og = self.model.use_flash
        return self

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)
        self._apply_use_flash(device)

    def _apply_use_flash(self, device: str) -> None:
        """Flash attention follows the device: off on the CPU, the fit-time value back on CUDA."""
        use_flash = False if normalize_device(device) == "cpu" else self._use_flash_og
        self.model.use_flash = use_flash
        self.model.model.use_flash = use_flash

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X, **self._predict_hps)

        y_pred_proba = self.model.ensemble_predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    # --- preprocessing, tags, resources -----------------------------------------------------------

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
    constructor defaults. Its ``tabdpt<1.2`` release pins no ``compile`` flag, so every estimator
    loads its own network through the library. See :class:`TabDPTModelBase` for the shared
    implementation and paper / codebase / license details.
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


class TabDPTTurboModel(SharedWeightsModelMixin, TabDPTModelBase):
    """TabDPT-Turbo (TabDPT v1.2).

    Accelerates fitting and inference by ~120x on average on TabArena versus v1.1 while improving
    predictive performance, chiefly by defaulting to subsampled context reduction (instead of
    v1.1's FAISS retrieval) plus long-context support and updated weights. Exposes the v1.2 predict
    knobs (``n_ensembles`` / ``batch_size``) and constructor surface (``compile`` / ``verbose`` /
    ``context_reduction``); see :class:`TabDPTModelBase` for the shared implementation.

    With the default ``compile=False`` every bagged child takes the 63.5M-parameter network from
    the weights registry (see :mod:`tabarena.models._shared_weights_model`) through the constructor
    replica in :mod:`tabarena.models.tabdpt._estimators`; ``compile=True`` compiles the module in
    place inside ``TabDPTEstimator.fit`` and keeps the library's per-estimator load. The fitted PCA
    basis ``V`` lives on the fit device and travels through the pickle on the CPU.

    Paper: "TabDPT-Turbo", https://openreview.net/pdf?id=Y00pwFyrHR

    Both wrappers share the ``tabdpt`` pip package (extra pinned to ``tabdpt>=1.2.0``), so a shared
    install runs v1.2 for both; the v1.1 wrapper then uses v1.2 defaults.
    """

    ag_key = "TA-TABDPT-TURBO"
    ag_name = "TA-TabDPT-Turbo"
    #: The registry-backed estimator module, imported lazily by a sharing fit on top of the library.
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabarena.models.tabdpt._estimators",)
    #: One ensemble member keeps the warm-up's dummy fit cheap; it changes no checkpoint-relevant key.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_ensembles": 1}

    _checkpoint_filename: ClassVar[str] = "tabdpt1_2.safetensors"
    _constructor_defaults: ClassVar[dict[str, object]] = {
        # `compile` is off by default: torch.compile adds per-fit compilation overhead (costly
        # across TabArena's many small bagged/refit fits), a compiled module complicates
        # AutoGluon's pickling cycle, and compiling mutates the module in place, which rules out
        # sharing it. The core Turbo speedup comes from context_reduction="subsample" + the v1.2
        # weights, both kept below.
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

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabdpt",
        checkpoint=CheckpointSpec(repo_id=HF_REPO_ID, filename=_checkpoint_filename, revision=HF_REVISION),
        variant="network",  # the classifier and the regressor run the same checkpoint and module
        default_params=lambda cls: dict(cls._constructor_defaults),
        key_flags=_network_flags,
        disable_when=(_compile_not_pinned_off,),
        unshareable_examples=({"compile": True},),
        network_attr="model",
        device_attrs=(("device", "str"),),
        owned_tensor_attrs=("V",),
        flag_attrs={"use_flash": ("use_flash", "model.use_flash")},
        estimator_move="to",
    )

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.tabdpt._estimators import load_network

        return load_network(key)

    def _init_tabdpt_model(self, *, model_cls, device: str, hps: dict):
        """The registry-backed estimator around the shared network, or the library estimator when this fit does not share."""
        key, payload = self._acquire_shared_weights(device=device)
        if key is None:
            return super()._init_tabdpt_model(model_cls=model_cls, device=device, hps=hps)
        from tabarena.models.tabdpt._estimators import shared_estimator_cls

        check_payload_device(payload, device)
        kwargs = self._constructor_kwargs(device=device, hps=hps, checkpoint=key.checkpoint)
        return shared_estimator_cls(model_cls).from_shared(payload, **kwargs)

    def _after_device_change(self, device_type: str) -> None:
        """An estimator that owns its network keeps the library's flash-attention rule; a shared one gets it from the key."""
        if self._shared_key is None and self.model is not None:
            self._apply_use_flash(device_type)
