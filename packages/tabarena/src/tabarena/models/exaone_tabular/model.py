from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)

#: Every released regression checkpoint, selectable by a config's ``regression_weight``. Declared
#: here rather than in ``hpo.py`` so the search space and :meth:`prefetch_weights` read the same list
#: and cannot drift — every file a sampled config can ask for is a file prefetch warms. The default
#: leads so it is the value a non-searching config falls back to. Classification has no counterpart:
#: only one classifier checkpoint is released, so its diversity is inference-side only.
REGRESSION_CHECKPOINTS: tuple[str, ...] = (
    "exaone-tabular-regressor-v1_default.safetensors",
    "exaone-tabular-regressor-v1_3eedbae97394.safetensors",
    "exaone-tabular-regressor-v1_45f9e04ea0fc.safetensors",
    "exaone-tabular-regressor-v1_511b873f561e.safetensors",
    "exaone-tabular-regressor-v1_8bd9bf482585.safetensors",
    "exaone-tabular-regressor-v1_aad9e74d4f2e.safetensors",
    "exaone-tabular-regressor-v1_ae491112a58e.safetensors",
    "exaone-tabular-regressor-v1_d5c5c527ac37.safetensors",
)


class EXAONETabularModel(AbstractTorchModel):
    """EXAONE Tabular: an in-context-learning tabular foundation model from LG AI Research.

    A Cross-axis Summary Transformer (CAST) that conditions on the training rows at inference time,
    with no per-dataset gradient training. Classification and regression are two separate released
    checkpoints of the same architecture, each loaded by its own estimator: a 20.8M-parameter
    10-class head (ECOC above that) and a 21.1M-parameter 999-quantile head read out as a trimmed
    mean over the quantile function.

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

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit EXAONE Tabular, loading the checkpoint that matches the problem type.

        As an in-context-learning foundation model there is no training loop and no early stopping,
        so (like the other TFM wrappers) ``X_val`` / ``y_val`` and ``time_limit`` are intentionally
        ignored — fitting loads the pre-trained checkpoint and stores the support set. ``num_cpus``
        is likewise unused: the library exposes no thread-count knob.

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

        from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor

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

        kwargs, manifest = self._resolve_config(hps)
        estimator_cls = EXAONETabularRegressor if self.problem_type == "regression" else EXAONETabularClassifier
        self.model = estimator_cls.from_pretrained(device=device, manifest=manifest, **kwargs)
        self.model.fit(X_np, y_np)

    #: Arm suffix naming a checkpoint file inside the released Hub repo.
    _WEIGHT_KEY = "weight"

    def _resolve_config(self, hps: dict) -> tuple[dict, object]:
        """Split a config into ``from_pretrained`` kwargs and the inference manifest to load with.

        Keys are unprefixed where both estimators accept the knob, and carry a ``classification_`` /
        ``regression_`` prefix where only one does. A prefixed key wins over the shared key of the
        same name, and the other task's keys are dropped — so one flat config describes both halves
        of the benchmark, which is what lets a single search space span a classification-only and a
        regression-only checkpoint while keeping every key a scalar the sampler can draw.

        ``weight`` names a checkpoint file in the released repo (routed to ``filename``, the
        parameter that keeps the repo/revision coordinates intact). Every other key is routed by
        which manifest section declares a field of that name, so adding a knob upstream needs no
        change here; anything no section claims is passed to ``from_pretrained`` and rejected there
        if it is not a real parameter.
        """
        from dataclasses import fields, replace

        from exaonetabular import released_manifest

        task = "regression" if self.problem_type == "regression" else "classification"
        shared = {k: v for k, v in hps.items() if not k.startswith(("classification_", "regression_"))}
        specific = {k[len(task) + 1 :]: v for k, v in hps.items() if k.startswith(f"{task}_")}
        arm = {**shared, **specific}

        kwargs = {}
        weight = arm.pop(self._WEIGHT_KEY, None)
        if weight is not None:
            kwargs["filename"] = weight

        manifest = released_manifest(task)
        # The task's own section first, so a knob it declares wins over a same-named field on the
        # shared sections rather than silently landing on the wrong one.
        sections = {name: getattr(manifest, name) for name in (task, "preprocessing", "runtime")}
        updates: dict[str, dict] = {name: {} for name in sections}
        for key, value in arm.items():
            for name, section in sections.items():
                if any(f.name == key for f in fields(section)):
                    updates[name][key] = value
                    break
            else:
                kwargs[key] = value

        changed = {name: replace(sections[name], **values) for name, values in updates.items() if values}
        return kwargs, replace(manifest, **changed) if changed else manifest

    def _set_default_params(self):
        # The released checkpoints' runtime defaults, identical for both (exaonetabular.presets).
        default_params = {
            "ensemble_count": 8,
            "compute_dtype": "float16",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        return self.model.device.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device = device
        self.model.model = self.model.model.to(device)

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Sequential fold fitting avoids contention on the shared Hugging Face checkpoint cache.

        ``refit_folds=True`` matches the other TFM wrappers (TabICL, TabSwift, TabPFN-3, ...): for
        an in-context-learning model, refitting one model on all data gives faster inference at
        similar quality to the bagged ensemble.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            },
        )
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: implement memory estimation and set to True
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
        """Warm torch (+ the CUDA context) and the ``exaonetabular`` import (untimed, data-independent).

        Declaring this overrides the generic ``AbstractTorchModel`` torch warm-up, so the torch part
        is re-done explicitly here; the extra piece is the library's own import chain (safetensors,
        scikit-learn, the model modules), which would otherwise land in the timed fit. Both
        estimator modules are imported regardless of the problem type: they share almost every
        dependency, so the second one costs nothing measurable.
        """
        from tabarena.models.warmup import warmup_imports, warmup_torch

        warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)
        warmup_imports("exaonetabular.classifier", "exaonetabular.regressor")

    @classmethod
    def download_checkpoint(cls, task: str, filename: str | None = None) -> str:
        """Download one released checkpoint, returning its local path.

        ``task`` (``"classification"`` / ``"regression"``) supplies the Hub coordinates from
        ``exaonetabular.presets`` rather than hardcoding them here, so a repo or revision bump in
        the library is picked up automatically. ``filename`` overrides which file inside that repo
        to fetch, for the alternative checkpoints the curated configs select.

        Deliberately no ``local_files_only`` fast path: the released files are served from a mutable
        ``main`` revision and have been republished in place at least once, so trusting a cache hit
        would pin a superseded checkpoint forever and silently benchmark the wrong weights. The
        normal call revalidates the etag and re-downloads when the bytes changed.
        """
        from exaonetabular import released_checkpoint
        from huggingface_hub import hf_hub_download

        checkpoint = released_checkpoint(task)
        return hf_hub_download(
            repo_id=checkpoint.repo_id,
            filename=filename or checkpoint.filename,
            revision=checkpoint.revision,
        )

    @classmethod
    def prefetch_weights(cls) -> dict[str, str]:
        """Pre-download every checkpoint the search space can select; return ``{name: path}``.

        That is the released classifier plus every entry of :data:`REGRESSION_CHECKPOINTS`, so no
        sampled config can reach a compute node needing a file the head node never warmed.
        """
        paths = {"classification": cls.download_checkpoint("classification")}
        paths.update({name: cls.download_checkpoint("regression", filename=name) for name in REGRESSION_CHECKPOINTS})
        return paths
