from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_weights_model import ResolvedCheckpoint
from tabarena.models.prefetch import WeightsUnavailableError
from tabarena.models.tabpfnv2_5._shared import TABPFN_SPEC, TabPFNSharedWeightsMixin, ensure_tabpfn_checkpoint

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from tabarena.models._shared_weights_model import SharedWeightsSpec
    from tabarena.models._weights import WeightsKey

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "wide-v2-8k"
#: The library's default number of features per token group; it patches the encoder, so it is part of the key.
DEFAULT_FEATURES_PER_GROUP = 1
#: Where the library releases its checkpoints; recorded as ``checkpoint_source`` (the files are not on the Hub).
CHECKPOINT_REPO = "not-a-feature/TabPFN-Wide"


def wide_checkpoint_path(model_name: str) -> Path:
    """The file ``TabPFNWideClassifier._get_model_path`` caches ``model_name`` at (``~/.tabpfnwide/models``)."""
    return Path(os.path.expanduser("~")) / ".tabpfnwide" / "models" / f"tabpfn-{model_name}.pt"


def wide_checkpoint_source(model_name: str, checkpoint: str | Path) -> dict[str, Any]:
    """Provenance of a wide checkpoint for the metadata: GitHub release coordinates, never a host path."""
    revision = None
    try:
        from importlib.metadata import version

        revision = f"v{version('tabpfnwide')}"
    except Exception:
        logger.debug("Could not read the tabpfnwide package version", exc_info=True)
    return {
        "repo_id": CHECKPOINT_REPO,
        "filename": Path(checkpoint).name,
        "revision": revision,
        "model_name": model_name,
    }


def _plain_v2(hyperparameters: Mapping[str, Any]) -> bool:
    """``model_name="v2"`` (or no name without a user ``model_path``) runs tabpfn's plain v2 classifier through the library's own loader."""
    if hyperparameters.get("model_path"):
        return False
    return hyperparameters.get("model_name", DEFAULT_MODEL_NAME) in (None, "", "v2")


class TabPFNWideModel(TabPFNSharedWeightsMixin, AbstractTorchModel):
    """TabPFN-Wide: a TabPFN variant specialized for wide tabular datasets
    (many features, few samples).

    The current default (v0.3.0) of TabPFN-Wide is based on TabPFNv2.

    The network is shared through :class:`tabarena.models.tabpfnv2_5._shared.TabPFNSharedWeightsMixin`
    and handed to ``tabarena.models.tabpfnwide._estimators.SharedTabPFNWideClassifier``, which
    builds nothing in its constructor. ``save_attention_maps`` (recording hooks on the module) and
    the plain ``model_name="v2"`` base model keep the library's own build.

    Paper: TabPFN-Wide (arXiv:2510.06162)
    Authors: Christopher Kolberg, Jules Kreuer, Jonas Huurdeman, Sofiane Ouaari,
        Katharina Eggensperger, Nico Pfeifer
    Codebase: https://github.com/not-a-feature/TabPFN-Wide
    """

    ag_key = "TA-TABPFN-WIDE"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabpfn",
        "tabpfn.base",
        "tabpfn.model_loading",
        "tabpfnwide.classifier",
        "tabarena.models.tabpfnwide._estimators",
    )
    ag_name = "TA-TabPFN-Wide"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {
        "max_rows": 10_000,
        "max_classes": 10,
    }

    # ``features_per_group`` patches the encoder, so two values are two networks; a ``device``
    # hyperparameter wins over the device derived from the allocated GPUs, exactly as in ``_fit``.
    shared_weights_spec: ClassVar[SharedWeightsSpec] = replace(
        TABPFN_SPEC,
        library="tabpfnwide",
        variant="classifier",
        default_params={"model_name": DEFAULT_MODEL_NAME, "features_per_group": DEFAULT_FEATURES_PER_GROUP},
        flag_params=("model_name", "features_per_group"),
        disable_when=(*TABPFN_SPEC.disable_when, "save_attention_maps", _plain_v2),
        unshareable_examples=(*TABPFN_SPEC.unshareable_examples, {"model_name": "v2"}),
        device_param="device",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        return X

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
        """A user ``model_path`` (shared when it exists), else the wide checkpoint the library caches for ``model_name``.

        The tabpfn v2 base checkpoint the build patches is made present alongside; both downloads
        happen only when ``allow_download`` is set (the wide file from the library's GitHub release,
        the base from tabpfn's own sources).
        """
        del problem_type, variant, stage
        from tabarena.models.tabpfnwide._estimators import download_wide_checkpoint, tabpfn_v2_base_checkpoint

        model_name = hyperparameters.get("model_name") or ""
        user_path = hyperparameters.get("model_path")
        if user_path:
            path = Path(user_path)
            if not path.is_file():
                return None
            source: dict[str, Any] = {"user_path": path.name}
        else:
            path = wide_checkpoint_path(model_name)
            if not path.is_file():
                if not allow_download:
                    raise WeightsUnavailableError(f"TabPFN-Wide checkpoint {path} is not cached locally")
                download_wide_checkpoint(model_name)
            source = wide_checkpoint_source(model_name, path)
        ensure_tabpfn_checkpoint(tabpfn_v2_base_checkpoint(), "classifier", allow_download=allow_download)
        return ResolvedCheckpoint(path=str(path.resolve()), source=source)

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey) -> Any:
        from tabarena.models.tabpfnwide._estimators import build_shared_model_specs

        flags = dict(key.flags)
        return build_shared_model_specs(
            checkpoint=key.checkpoint,
            model_name=flags.get("model_name", ""),
            features_per_group=int(flags.get("features_per_group", DEFAULT_FEATURES_PER_GROUP)),
            device=key.device,
        )

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Download the default wide checkpoint and the tabpfn v2 base classifier it is built on; returns both paths."""
        from tabarena.models.tabpfnwide._estimators import download_wide_checkpoint, tabpfn_v2_base_checkpoint

        wide = download_wide_checkpoint(DEFAULT_MODEL_NAME)
        base = ensure_tabpfn_checkpoint(tabpfn_v2_base_checkpoint(), "classifier", allow_download=True)
        return [str(wide), str(base)]

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """The preset's attach plus the two references the library keeps on the estimator itself."""
        super()._attach_shared_weights(payload, device)
        self.model.shared_model_specs = payload
        self.model.model_path = payload
        self.model._wide_model = payload.model

    def _detach_for_pickle(self, estimator: Any) -> Any:
        estimator = super()._detach_for_pickle(estimator)
        estimator._wide_model = None
        estimator.shared_model_specs = None
        return estimator

    # --- fit --------------------------------------------------------------------------------------

    @staticmethod
    def _get_estimator_classes() -> tuple[type, type]:
        """The library estimator and its registry-backed subclass (imported lazily)."""
        from tabpfnwide.classifier import TabPFNWideClassifier

        from tabarena.models.tabpfnwide._estimators import SharedTabPFNWideClassifier

        return TabPFNWideClassifier, SharedTabPFNWideClassifier

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

        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError(
                f"Unsupported problem_type: {self.problem_type}. TabPFN-Wide supports only classification.",
            )

        # The device derived from the allocated GPUs is the default; a ``device`` hyperparameter wins.
        hps = {"model_name": DEFAULT_MODEL_NAME, "device": device, **self._get_model_params()}

        X = self.preprocess(X, y=y, is_train=True)

        key, payload = self._acquire_shared_weights(device=device)
        plain_cls, shared_cls = self._get_estimator_classes()
        self.model = shared_cls(shared_model_specs=payload, **hps) if payload is not None else plain_cls(**hps)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            "model_name": DEFAULT_MODEL_NAME,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
