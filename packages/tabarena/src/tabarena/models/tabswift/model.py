from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from sklearn.impute import SimpleImputer

from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models.prefetch import WeightsUnavailableError

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.models._weights import WeightsKey


logger = logging.getLogger(__name__)

_DEFAULT_HF_REPO = "LAMDA-Tabular/TabSwift"
_DEFAULT_HF_FILENAME = "swift.ckpt"
"""A single ``swift.ckpt`` checkpoint serves both the classifier and the regressor."""
#: Commit pinned so the checkpoint fetched here never silently changes if the
#: repo's default branch moves. Bump deliberately (with a note on what changed)
#: when picking up a newer checkpoint.
_DEFAULT_HF_REVISION = "b829456edb7c41ad93a2851a8df245db362e1c83"


def _clear_transient_state(network: Any) -> None:
    """Drop the hierarchical class tree a many-class forward leaves on ``icl_predictor``.

    Above the checkpoint's ``max_classes`` the vendored ``ICLearning`` builds a ``ClassNode`` tree
    holding the support representations and labels on the module (``root``) and reads it in the
    same forward; it is rebuilt before every use, so clearing it after a predict changes nothing and
    keeps one child's data off a module other children share.
    """
    icl = getattr(network, "icl_predictor", None)
    if icl is not None and getattr(icl, "root", None) is not None:
        icl.root = None


class TabSwiftModel(SharedWeightsModelMixin, AbstractTorchModel):
    """TabSwift: an efficient tabular foundation model with row-wise attention.

    Paper: TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention (ICML 2026)
    Authors: Si-Yang Liu, Han-Jia Ye
    Codebase: https://github.com/LAMDA-Tabular/TabSwift
    License: MIT
    """

    ag_key = "TA-TABSWIFT"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabarena.models.tabswift._vendor.classifier",
        "tabarena.models.tabswift._vendor.regressor",
        "tabarena.models.tabswift._estimators",
        "huggingface_hub",
    )
    #: Cheapness knob for the warm-up dummy fit; ``model_path`` (the only checkpoint-relevant key)
    #: is never overridden here.
    warmup_dummy_fit_hyperparameters: ClassVar[dict[str, Any]] = {"n_estimators": 1}
    ag_name = "TA-TabSwift"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    # ``refit_folds=True`` matches the other TFM wrappers (TabICL, LimiX, TabPFN-3, ...):
    # for an in-context-learning model, refitting one model on all data gives faster
    # inference at similar quality to the bagged ensemble.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabswift",
        # A ``model_path`` naming an existing file is shared under that path; one that does not
        # exist yet disables sharing and the vendored estimator downloads it itself.
        checkpoint=CheckpointSpec(
            repo_id=_DEFAULT_HF_REPO,
            filename=_DEFAULT_HF_FILENAME,
            revision=_DEFAULT_HF_REVISION,
            user_path_param="model_path",
        ),
        # One checkpoint and one ``TabSwift(**config)`` module serve the classifier and the
        # regressor (the head is chosen per forward by ``if_regression``), so the variant is a
        # constant; the checkpoint is float32 and ``use_amp`` autocasts activations only.
        variant="network",
        network_attr="model_",
        seam="load_model",
        device_attrs=(("device_", "torch"),),
        # The PCA basis a fit with more than ``pca_dim`` features stores on the fit device.
        owned_tensor_attrs=("pca_input_mean_", "pca_v_", "final_mean_", "final_var_"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._imputer: SimpleImputer | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        """Build the network for ``key`` on ``key.device``; the registry's loader.

        The checkpoint is read with ``torch.load(map_location="cpu", weights_only=True)`` and the
        module is constructed under ``torch.device("meta")`` so the random initialization that a
        strict ``load_state_dict`` overwrites anyway is skipped and no generator advances (the
        registry additionally runs every loader under its random-state guard).
        ``load_state_dict(assign=True)`` then adopts the checkpoint tensors; the module has no
        buffers, so nothing stays on the meta device. The result matches the vendored
        ``_load_model`` (plain construction plus strict ``load_state_dict``, ``eval()``,
        ``to(device)``) tensor for tensor.
        """
        import torch

        from tabarena.models.tabswift._vendor.model.tabswift import TabSwift

        checkpoint = torch.load(key.checkpoint, map_location="cpu", weights_only=True)
        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."
        with torch.device("meta"):
            network = TabSwift(**checkpoint["config"])
        network.load_state_dict(checkpoint["state_dict"], assign=True)
        return network.to(key.device)

    def _default_checkpoint_path(self) -> str:
        """Local path of the pinned checkpoint for a fit that loads its own network, honoring ``ag.fetch_pretrained_weights``."""
        allow = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="fit")
        checkpoint = self.shared_weights_spec.checkpoint
        try:
            return checkpoint.resolve(hyperparameters={}, variant="network", allow_download=allow, cls=type(self)).path
        except WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=self.name, stage="fit", location=str(exc))
            ) from exc

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Produce a dense float32 matrix for TabSwift.

        TabSwift's forward pass is a linear projection over numeric row vectors with no native
        categorical or missing-value handling, so (matching TALENT's ``cat_policy='indices'`` plus
        NaN processing) categoricals are ordinal-encoded to integer codes and any remaining missing
        numeric cells are mean-imputed. Encoder and imputer are fit on train and reused at predict
        so the two stay aligned.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        X = np.asarray(X.to_numpy(), dtype=np.float32)

        if is_train:
            # keep_empty_features=True: an all-NaN column becomes 0 instead of being dropped,
            # so the feature count stays constant between fit and predict.
            self._imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            X = self._imputer.fit_transform(X)
        else:
            X = self._imputer.transform(X)

        return np.asarray(X, dtype=np.float32)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabSwift.

        As an in-context-learning foundation model there is no training loop and no early
        stopping, so (like the other TFM wrappers) ``X_val`` / ``y_val`` and ``time_limit``
        are intentionally ignored: fitting prepares the data transforms and attaches the
        pretrained network.
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

        from tabarena.models.tabswift._estimators import estimator_cls

        hps = self._get_model_params()
        key, payload = self._acquire_shared_weights(device=device)
        if key is not None:
            model_path = key.checkpoint
        else:
            model_path = hps.get("model_path") or self._default_checkpoint_path()
        hps.pop("model_path", None)

        X_np = self.preprocess(X, y=y, is_train=True)

        if self.problem_type == "regression":
            # TabSwift's regressor does not standardize the target internally (matching
            # TALENT, which standardizes y up front and inverse-transforms predictions).
            y_np = np.asarray(y.to_numpy(), dtype=np.float32)
            self._y_mean = float(y_np.mean())
            self._y_std = float(y_np.std()) or 1.0
            y_fit = (y_np - self._y_mean) / self._y_std
        else:
            y_fit = np.asarray(y.to_numpy())

        self.model = estimator_cls(self.problem_type)(
            model_path=str(model_path),
            device=device,
            **hps,
        )
        if key is not None:
            self.model.use_shared_weights(key, payload, type(self)._load_shared_weights)
        self.model.fit(X_np, y_fit)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        try:
            if self.problem_type == "regression":
                out = np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1)
                y_pred_proba = out * self._y_std + self._y_mean
            else:
                y_pred_proba = self.model.predict_proba(X)
        finally:
            if self._shared_key is not None and self._network_attached():
                _clear_transient_state(self.model.model_)

        return self._convert_proba_to_unified_form(y_pred_proba)

    def _set_default_params(self):
        # TALENT's default TabSwift configuration (TALENT/model/methods/tabswift.py). The
        # framework seed is injected via ``seed_name`` into ``random_state``.
        default_params = {
            "n_estimators": 16,
            "norm_methods": ["none", "power"],
            "feat_shuffle_method": "latin",
            "class_shift": True,
            "outlier_threshold": 4.0,
            "softmax_temperature": 0.9,
            "average_logits": True,
            "use_hierarchical": True,
            "use_amp": True,
            "batch_size": 16,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
