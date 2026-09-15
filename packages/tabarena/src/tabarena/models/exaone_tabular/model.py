from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
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

#: The released checkpoints' runtime defaults, identical for both tasks (``exaonetabular.presets``).
DEFAULT_PARAMS: dict[str, Any] = {"ensemble_count": 8, "compute_dtype": "float16"}

#: ``from_pretrained`` keyword arguments that never change which weights are loaded. ``ensemble_count``,
#: ``seed`` and ``max_vram_bytes`` live on the estimator's manifest, ``compute_dtype`` is part of the
#: registry key. Any other key (``weights``, ``revision``, ``cache_dir``, ``filename``, ``manifest``)
#: selects custom weights, and such a configuration falls back to the library loader unshared.
SHAREABLE_PARAMS: frozenset[str] = frozenset({"ensemble_count", "compute_dtype", "seed", "max_vram_bytes"})

#: The released checkpoint each problem type uses: binary and multiclass share the classifier.
CHECKPOINT_TASKS: dict[str, str] = {
    "binary": "classification",
    "multiclass": "classification",
    "regression": "regression",
}


def resolve_compute_dtype(hyperparameters: Mapping[str, Any], device: str) -> str:
    """The network dtype a fit on ``device`` runs with.

    Half precision is a GPU choice; several torch CPU kernels have no half implementation, so a CPU
    fit configured with ``float16`` runs in ``float32`` instead. Any other value is kept as is.
    """
    compute_dtype = hyperparameters.get("compute_dtype", DEFAULT_PARAMS["compute_dtype"])
    if normalize_device(device) == "cpu" and compute_dtype == "float16":
        return "float32"
    return compute_dtype


def _selects_custom_weights(hyperparameters: dict) -> bool:
    """Whether a configuration loads anything but the released checkpoint (a key outside ``SHAREABLE_PARAMS``)."""
    return not set(hyperparameters) <= SHAREABLE_PARAMS


class EXAONETabularModel(SharedWeightsModelMixin, AbstractTorchModel):
    """EXAONE Tabular: an in-context-learning tabular foundation model from LG AI Research.

    A Cross-axis Summary Transformer (CAST) that conditions on the training rows at inference time,
    with no per-dataset gradient training. Classification and regression are two separate released
    checkpoints of the same architecture, each loaded by its own estimator: a 20.8M-parameter
    10-class head (ECOC above that) and a 21.1M-parameter 999-quantile head read out as a trimmed
    mean over the quantile function.

    The released network of each task is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`) and handed to the estimator constructor's
    ``model=`` argument; the checkpoint coordinates come from the library's presets
    (:mod:`tabarena.models.exaone_tabular._estimators`). Configurations that select custom weights
    (see ``SHAREABLE_PARAMS``) keep the library's ``from_pretrained``.

    Paper: technical report not yet released (the repository's citation block is a placeholder).
    Authors: LG AI Research
    Codebase: https://github.com/LGAI-Research/EXAONE-Tabular
    License: code under the BSD-3-Clause-LG AI Research License; the released weights under the
        EXAONE AI Model License 1.1-NC, which permits non-commercial use only.
    """

    ag_key = "TA-EXAONE-TABULAR"
    ag_name = "TA-EXAONE-Tabular"
    ag_priority = 65
    seed_name = "seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    # ``refit_folds=True`` matches the other TFM wrappers (TabICL, TabSwift, TabPFN-3, ...): for
    # an in-context-learning model, refitting one model on all data gives faster inference at
    # similar quality to the bagged ensemble.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    #: Modules the timed fit would otherwise import for the first time: the two estimators (which
    #: pull in the checkpoint, presets and weights modules), safetensors (imported lazily inside the
    #: library's checkpoint reader), the Hub client used by the resolver and the registry-side
    #: plumbing module.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "exaonetabular.classifier",
        "exaonetabular.regressor",
        "safetensors.torch",
        "huggingface_hub",
        "huggingface_hub.errors",
        "tabarena.models.exaone_tabular._estimators",
    )
    #: Cheapness knob for the warm-up dummy fit; ``ensemble_count`` lives on the estimator's
    #: manifest and never touches the network, so the primed key is unaffected.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"ensemble_count": 1}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="exaone_tabular",
        checkpoint=None,  # the coordinates live in ``exaonetabular.presets``; see ``_resolve_shared_checkpoint``
        variant=CHECKPOINT_TASKS,
        default_params=DEFAULT_PARAMS,
        dtype=resolve_compute_dtype,
        disable_when=(_selects_custom_weights,),
        unshareable_examples=({"weights": "x"},),
        network_attr="model",
        device_attrs=(("device", "torch"),),
        # The library leaves the parameters trainable; freezing them selects other CPU kernels and
        # moves the float32 predictions (see ``_estimators.build_network``).
        freeze_parameters=False,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Produce the dense real-valued matrix EXAONE Tabular's estimators require.

        The estimators accept only a 2-D NumPy array of a real numeric dtype, so categoricals are
        ordinal-encoded (the encoding the upstream README asks callers to apply). Missing cells stay
        as NaN: the library's own preprocessor mean-imputes them and keeps a missing mask, which is
        strictly more informative than imputing here. Infinities are folded into NaN because the
        library rejects them outright.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        X = np.asarray(X.to_numpy(), dtype=np.float32)
        X[~np.isfinite(X)] = np.nan
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
        """The released checkpoint of ``variant`` (a task), local-first through the library's presets."""
        from tabarena.models.exaone_tabular._estimators import resolve_checkpoint

        source = resolve_checkpoint(variant, prefer_local=True, allow_download=allow_download)
        return ResolvedCheckpoint(path=source.path, source=source.to_metadata())

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.exaone_tabular._estimators import build_network

        return build_network(key)

    @classmethod
    def prefetch_weights(cls) -> dict[str, str]:
        """Validate and download both released checkpoints online; return ``{task: local path}``.

        Classification and regression are separate files, so a run covering both problem types
        needs both warmed before the jobs are dispatched. The mutable ``main`` revision is
        revalidated here; compute nodes then resolve the same snapshots local-first.
        """
        from tabarena.models.exaone_tabular._estimators import resolve_checkpoint

        return {
            task: resolve_checkpoint(task, prefer_local=False).path for task in sorted(set(CHECKPOINT_TASKS.values()))
        }

    # --- fit ------------------------------------------------------------------------------------

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit EXAONE Tabular, attaching the checkpoint network that matches the problem type.

        As an in-context-learning foundation model there is no training loop and no early stopping,
        so (like the other TFM wrappers) ``X_val`` / ``y_val`` and ``time_limit`` are intentionally
        ignored: fitting stores the support set. ``num_cpus`` is likewise unused: the library exposes
        no thread-count knob.

        With sharing on, the network comes from the registry and is injected through the estimator
        constructor; the per-child manifest (``ensemble_count``, the per-fold ``seed``,
        ``compute_dtype``) and ``max_vram_bytes`` stay on the estimator. With sharing off, the
        library's ``from_pretrained`` builds and loads the network for this child.

        Regression does one extra thing inside ``fit``: from ~10k support rows up, it holds out a
        fifth of them to solve for non-negative ensemble-member weights, which costs an additional
        forward pass. Below that threshold the members stay uniformly weighted.
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

        from tabarena.models.exaone_tabular._estimators import estimator_cls, released_manifest

        hps = self._get_model_params()
        if device == "cpu" and hps.get("compute_dtype") == "float16":
            # Half precision is a GPU choice; several torch CPU kernels have no half
            # implementation, so the CPU fallback path runs in float32 instead.
            logger.log(15, "Running on CPU: overriding compute_dtype 'float16' with 'float32'.")
            hps["compute_dtype"] = "float32"

        X_np = self.preprocess(X, y=y, is_train=True)
        # Passed through unscaled for both tasks: the regressor standardizes the target against its
        # own support set and maps its predictions back, and the classifier encodes the labels.
        y_np = np.asarray(y.to_numpy())

        task = CHECKPOINT_TASKS[self.problem_type]
        key, payload = self._acquire_shared_weights(device=device)
        if key is None:
            self.model = estimator_cls(task).from_pretrained(device=device, **hps)
        else:
            # The constructor runs ``model.to(self.device)`` in place, so the payload must already live there.
            check_payload_device(payload, device)
            manifest = released_manifest(
                task,
                ensemble_count=hps.get("ensemble_count"),
                compute_dtype=hps.get("compute_dtype"),
                seed=hps.get("seed"),
            )
            self.model = estimator_cls(task)(
                manifest, device=device, model=payload, max_vram_bytes=hps.get("max_vram_bytes")
            )
        self.model.fit(X_np, y_np)

    def _set_default_params(self):
        for param, val in DEFAULT_PARAMS.items():
            self._set_default_param_value(param, val)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
