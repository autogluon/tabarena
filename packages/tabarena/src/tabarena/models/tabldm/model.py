from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.models._weights import WeightsKey


class TabLDMModel(SharedWeightsModelMixin, AbstractTorchModel):
    """TabLDM: a tabular foundation model with a dual-stream column embedder and MoE backbone.

    Uses the enhanced sklearn estimators (``TabLDMEnhancedClassifier``/``TabLDMEnhancedRegressor``),
    which wrap the base MoE1 model with an ensemble/calibration pipeline on top.

    Codebase: pip name ``Xiaomi-TabLDM``, import name ``tabldm``, installed from GitHub since
    it is not on PyPI.
    Checkpoints: https://huggingface.co/occams/Xiaomi-TabLDM
    License: Apache-2.0 (Copyright Xiaomi Corporation)
    """

    ag_key = "TA-XIAOMI-TABLDM"
    #: ``import tabldm`` eagerly imports all four estimators and runs the import-time numba probe of
    #: ``tabldm._model.quantile_dist``; the second entry is TabArena's registry-aware estimator
    #: module, imported lazily by the fit.
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabldm", "tabarena.models.tabldm._estimators")
    #: Cheapness knob for the warm-up dummy fit; the ensemble size never touches the network or the key.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    ag_name = "TA-Xiaomi-TabLDM"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabldm",
        checkpoint=CheckpointSpec(
            repo_id="occams/Xiaomi-TabLDM",
            filename_param="checkpoint_version",
            default_filename={
                "classifier": "checkpoints/clf_default.ckpt",
                "regressor": "checkpoints/reg_default.ckpt",
            },
        ),
        # A user ``model_path`` selects other weights; the KV-cache path writes ``model_._cache`` on
        # the module itself, which a shared module must never carry.
        disable_when=("model_path", "kv_cache"),
        download_param="allow_auto_download",
        # ``requires_grad_(False)`` selects other CPU kernels and moves float32 probabilities by
        # about 3e-7; the parameters keep the flags the library's own load leaves.
        freeze_parameters=False,
        network_attr="model_",
        detach_attr="library",
        seam="load_model",
        post_attach_calls=("_build_inference_config", "_move_cache_to_device"),
        device_attrs=(("device_", "torch"),),
    )
    # Refitting one model on all data (like the other TFM wrappers: TabICL, TabSwift, LimiX, ...)
    # gives faster inference at similar quality to the bagged ensemble for an in-context model.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.tabldm._estimators import build_network

        return build_network(key)

    def get_model_cls(self):
        from tabarena.models.tabldm._estimators import estimator_cls

        return estimator_cls(self.shared_weights_spec.variant_for(self.problem_type))

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabLDM.

        As an in-context-learning foundation model there is no training loop and no early
        stopping, so (like the other TFM wrappers) ``X_val``/``y_val`` and ``time_limit`` are
        intentionally ignored. Categorical columns and missing values need no upfront handling
        here: the estimator's own ``TransformToNumerical`` preprocessing ordinal-encodes
        categorical dtypes and mean-imputes numeric NaNs internally when given a DataFrame, and
        for regression it standardizes/inverse-transforms the target itself, so ``X``/``y`` are
        passed straight through.
        """
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        model_cls = self.get_model_cls()
        hps = self._get_model_params()

        X = self.preprocess(X, y=y, is_train=True)

        key, payload = self._acquire_shared_weights(device=device)
        self.model = model_cls(device=device, n_jobs=num_cpus, **hps)
        if key is not None:
            self.model.use_shared_weights(key, payload, type(self)._load_shared_weights)
        self.model.fit(X, y)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
