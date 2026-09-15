from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_estimators import check_payload_device
from tabarena.models._shared_weights_model import ResolvedCheckpoint, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models._weights import normalize_device

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tabarena.models._weights import WeightsKey

logger = logging.getLogger(__name__)


def _default_variant_params(cls: type) -> dict[str, Any]:
    """The ``model`` hyperparameter default (the library's variant name) when the class sets one."""
    variant = getattr(cls, "default_model_variant", None)
    return {} if variant is None else {"model": variant}


class NoriModel(SharedWeightsModelMixin, AbstractTorchModel):
    """Nori: a tabular foundation model for regression via in-context learning.

    Paper/citation: Synthefy Nori
    Authors: Synthefy (Li Po-han, Aditya Narayanan, Sai Shankar Narasimhan, et al.)
    Codebase: https://github.com/Synthefy/synthefy-nori
    License: Apache-2.0

    The network is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`): every child's ``NoriPredictor`` is built eagerly
    in ``_fit`` around the registry payload through the library's ``NoriPredictor(model=...)``
    injection (:mod:`tabarena.models.nori._estimators`), pickled away and rebuilt on load; a
    device change rebuilds it around the registry entry of the new device type. Sharing is safe
    because the module has no forward-time attribute writes and every forward-time random draw comes
    from the global torch generator the predictor reseeds before each forward.

    Notes:
        - ``NoriRegressor`` is a scikit-learn estimator (``fit``/``predict``) and
          normalizes the target internally, so we pass ``y`` through unchanged and
          rely on the default regression ``_predict_proba`` path.
        - ``NoriRegressor.fit`` coerces ``X`` to a float32 array, so categoricals are
          label-encoded here. NaN is forwarded as-is: Nori's inference pipeline
          handles missing values natively (``allow-nan``).
        - ``NoriRegressor`` exposes no random seed (inference is deterministic given
          the context), so ``seed_name`` is left unset.
    """

    ag_key = "TA-NORI"
    #: Modules the timed fit and predict would otherwise import for the first time: the library's
    #: public API, its predictor module (which pulls in ``torch._dynamo``), the checkpoint loader,
    #: the Hub helpers used by checkpoint resolution and ``scipy.stats``, imported lazily by the
    #: predictor's default ``"yj"`` augmentation gate.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "synthefy_nori",
        "synthefy_nori.hf",
        "synthefy_nori.inference.predictor",
        "synthefy_nori.utils.loading",
        "huggingface_hub",
        "huggingface_hub.errors",
        "scipy.stats",
    )
    ag_name = "TA-Nori"
    ag_priority = 65
    _supported_problem_types = ["regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    # Cap context size at 100k rows; no feature or class limits (regression-only).
    _default_auxiliary_params_extra = {
        "max_rows": 100_000,
    }
    # Refit by default: a single forward-pass model has no per-fold validation cost.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}
    #: Default of the ``model`` hyperparameter (the library's variant name); ``None`` keeps the
    #: library's base checkpoint. Read by ``_set_default_params``, the shared-weights key and
    #: :meth:`prefetch_weights`, so a size variant only has to set this attribute.
    default_model_variant: ClassVar[str | None] = None

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="nori",
        checkpoint=None,  # the coordinates come from the library's variant registry; see ``_resolve_shared_checkpoint``
        variant=lambda problem_type: "regressor" if problem_type == "regression" else None,
        default_params=_default_variant_params,
        network_attr="_predictor.model",
        detach_attr="_predictor",
        device_attrs=(("device", "torch"), ("_predictor.device", "torch")),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._cat_indices: list[int] | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Label-encode categoricals to numeric and return a float32 array.

        NaN is preserved (Nori handles missing values natively); only categorical
        columns are encoded, numeric columns pass through untouched.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            if is_train:
                self._cat_indices = [X.columns.get_loc(c) for c in self._feature_generator.features_in]

        return np.asarray(X.to_numpy(), dtype=np.float32)

    def _set_default_params(self):
        for param, val in _default_variant_params(type(self)).items():
            self._set_default_param_value(param, val)

    # --- shared weights ---------------------------------------------------------------------------

    @classmethod
    def _resolve_shared_checkpoint(
        cls,
        *,
        problem_type: str,
        variant: str,
        hyperparameters: Mapping[str, Any],
        allow_download: bool,
        stage: str = "fit",
    ) -> ResolvedCheckpoint | None:
        """The checkpoint the ``model``, ``model_path`` and ``token`` hyperparameters select, local-first."""
        from tabarena.models.nori._estimators import resolve_checkpoint

        source = resolve_checkpoint(
            model=hyperparameters.get("model"),
            model_path=hyperparameters.get("model_path"),
            token=hyperparameters.get("token"),
            allow_download=allow_download,
        )
        return ResolvedCheckpoint(path=source.path, source=source.to_metadata())

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.nori._estimators import load_network

        return load_network(key)

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """Rebuild the estimator's predictor around ``payload`` for ``device`` through the library's ``model=`` injection.

        The predictor is rebuilt rather than re-pointed so its device-dependent constructor
        decisions (mixed precision, retrieval validation) always match a fresh build.
        """
        from tabarena.models.nori._estimators import build_predictor

        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        self.model._predictor = build_predictor(
            self.model, device=device_type, checkpoint=self._shared_key.checkpoint, network=payload
        )
        self._apply_device_bookkeeping(device_type)

    def _network_attached(self) -> bool:
        """An estimator that owns its predictor builds it lazily at the first predict; only a shared one can be detached."""
        return self._shared_key is None or super()._network_attached()

    def get_device(self) -> str:
        """The predictor's network device, or the estimator's device field while no predictor is built."""
        if self.model is not None and getattr(self.model, "_predictor", None) is None:
            return normalize_device(self.model.device)
        return super().get_device()

    def _prepare_owned_for_inference(self) -> None:
        """An estimator that owns its network builds its predictor through the library (checkpoint read, module build)."""
        self.model._get_predictor()

    @classmethod
    def prefetch_weights(cls) -> str:
        """Download the class's checkpoint from the Hugging Face Hub and return its local path.

        Runs on the head node before the jobs are dispatched: it revalidates the repository's
        default branch online (and fetches ``config.json`` for the Hub's download statistics, as the
        library does), so compute nodes resolving the same file local-first read the snapshot
        validated here. The base ``Synthefy/Nori`` repository is gated and needs a Hugging Face token
        (``HF_TOKEN`` / ``hf auth login``); the size variants' repositories are public.
        """
        from synthefy_nori.hf import download_checkpoint

        return download_checkpoint(model=cls.default_model_variant)

    # --- fit ------------------------------------------------------------------------------------

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit ``NoriRegressor`` on the label-encoded context and attach the shared network's predictor.

        With sharing on, the predictor is built right after ``fit`` around the registry network, so
        the OOF predict of a fold child and the first predict of the refit child run forward passes
        only. With sharing off, ``NoriRegressor`` is constructed and fitted as the library ships it
        and builds its own predictor at the first predict. ``NoriRegressor`` normalizes ``y``
        internally and denormalizes its predictions, so ``y`` is passed through unchanged.
        """
        import torch
        from synthefy_nori import NoriRegressor

        if self.problem_type != "regression":
            raise AssertionError(f"{self.ag_name} only supports regression, got problem_type={self.problem_type!r}.")

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, y=y, is_train=True)

        hps = self._get_model_params()
        hps.pop("device", None)  # device is set explicitly from the allocated resources

        key, payload = self._acquire_shared_weights(device=device)
        self.model = NoriRegressor(device=device, **hps)
        self.model.fit(X, y)
        if key is not None:
            self._attach_shared_weights(payload, device)

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """Assume a small-model baseline (weights + activations) plus the dataset footprint."""
        baseline_mem_est = 3 * 1e9  # 3 GB for the model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


class Nori30MModel(NoriModel):
    """Nori-30M: the ~29M-parameter version of Nori (in-context tabular regression).

    A version of :class:`NoriModel` with the same fit, predict, preprocessing and resource logic;
    only the checkpoint differs. ``default_model_variant = "nori-30m"`` routes ``NoriRegressor``,
    the shared-weights key and the prefetch to the 30M weights via synthefy-nori's variant registry
    (``>=0.10.0``), which resolves to the public `Synthefy/Nori-30M <https://huggingface.co/Synthefy/Nori-30M>`_
    Hugging Face repository.
    """

    ag_key = "TA-NORI-30M"
    ag_name = "TA-Nori-30M"
    ag_priority = 64  # just below the base Nori (65)
    default_model_variant: ClassVar[str | None] = "nori-30m"
