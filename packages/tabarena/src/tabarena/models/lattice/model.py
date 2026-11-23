from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_REVISIONS = {"default": "38d704ff3495ed36d97a473f9f321a791f7d9eea", "tiny": "c0ea3161d222fca03ef5474e365083995aeeceaa"}
_REPO_IDS = {"default": "hachure-ai/Lattice", "tiny": "hachure-ai/Lattice-tiny"}
_CHECKPOINT_FILES = ("cls.ckpt", "reg.ckpt")
# The two released classifiers both declare max_classes=10. Above this limit
# Lattice uses hierarchical prediction without KV.
_NATIVE_CLASSES = 10


@cache
def _release_snapshot(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    # The Hub records the pinned commit when downloading the files and
    # records that mapping so staged workers can find the snapshot offline.
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=["cls.ckpt", "reg.ckpt"]))


@cache
def _checkpoint(size: str, filename: str, weights_dir: Path | None) -> Path:
    """Resolve the released checkpoint path for this variant."""
    directory = _release_snapshot(_REPO_IDS[size], _REVISIONS[size]) if weights_dir is None else weights_dir / size
    return directory / filename


class LatticeModel(AbstractTorchModel):
    """Lattice, a frozen in-context model for classification and regression.

    Fit prepares the training context and can build its KV cache. There is no
    task-specific training loop to stop at ``time_limit``. The runner handles
    timeouts. The inner folds predict once, so only the refit configuration asks
    for KV. The released classifiers' hierarchical path does not support KV.

    Authors: Nanjing University
    Codebase and usage: https://github.com/hachure-ai/Lattice
    License: Apache-2.0 project code with third-party notices. CC-BY-NC-4.0 weights.

    ``LATTICE_WEIGHTS_DIR`` can point to an offline copy with ``default/`` and
    ``tiny/`` directories. The Hub route is pinned to the released checkpoint commits.
    """

    ag_key = "TA-LATTICE"
    ag_name = "TA-Lattice"
    ag_priority = 65
    seed_name = "random_state"
    model_size: ClassVar[str] = "default"
    default_n_estimators: ClassVar[int] = 8
    warmup_modules: ClassVar[tuple[str, ...]] = ("lattice", "huggingface_hub")
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1, "kv_cache": False}
    # The released checkpoints are frozen: every fit of the same file on the same
    # device builds an identical network, so the bagged children, the inner folds
    # and the refit share one. A cached fit keeps the checkpoint bytes off the
    # critical path, which is most of a fit's time on small tables.
    # ``kv_cache`` writes task-specific state into the network, so those fits keep
    # their own copy.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=("lattice:LatticeClassifier._load_model", "lattice:LatticeRegressor._load_model"),
        key=("model_path", "model_size"),
        disabled_by=("kv_cache",),
    )
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}

    @classmethod
    def _checkpoint_path(cls, filename: str) -> Path:
        weights_dir = os.environ.get("LATTICE_WEIGHTS_DIR")
        root = Path(weights_dir).expanduser().resolve() if weights_dir else None
        return _checkpoint(cls.model_size, filename, root)

    @classmethod
    def prefetch_weights(cls) -> list[Path]:
        """Stage the two released checkpoints for this variant."""
        return [cls._checkpoint_path(filename) for filename in _CHECKPOINT_FILES]

    _categorical_indices: list[int] | None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Record categorical columns so the encoder can use its type-aware recipe.

        The estimator detects categories from dtypes when no indices are given, but
        AutoGluon's feature pipeline can return integer-coded categorical columns;
        passing the positions found here keeps one-hot / count / missing-indicator
        branches available on categorical-heavy tables.
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns
            self._categorical_indices = [X.columns.get_loc(column) for column in categorical_cols] or None
        return X

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", self.default_n_estimators)
        self._set_default_param_value("kv_cache", False)
        self._set_default_param_value("preprocessing", "lattice")
        self._set_default_param_value("batch_size", "auto")
        self._set_default_param_value("use_amp", True)
        self._set_default_param_value("use_fa3", False)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ) -> None:
        from lattice import LatticeClassifier, LatticeRegressor

        if os.environ.get("LATTICE_KV_CACHE") == "1":
            raise ValueError("Unset LATTICE_KV_CACHE so the TabArena fold and refit configs control caching")

        import torch

        self.device = self._resolve_fit_device(num_gpus=num_gpus)
        regression = self.problem_type == "regression"
        estimator_cls = LatticeRegressor if regression else LatticeClassifier
        params = self._get_model_params()
        if not regression and y.nunique() > _NATIVE_CLASSES:
            params["kv_cache"] = False
        params.update(
            device=self.device,
            n_jobs=num_cpus,
            model_size=self.model_size,
            model_path=self._checkpoint_path("reg.ckpt" if regression else "cls.ckpt"),
        )
        # ``preprocess`` records the categorical columns; read them afterwards.
        train = self.preprocess(X, y=y, is_train=True)
        params["categorical_indices"] = getattr(self, "_categorical_indices", None)
        estimator = estimator_cls(**params)
        original_threads = torch.get_num_threads()
        try:
            if original_threads != num_cpus:
                torch.set_num_threads(num_cpus)
            self.model = estimator.fit(train, y)
        finally:
            if torch.get_num_threads() != original_threads:
                torch.set_num_threads(original_threads)

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        import numpy as np
        import torch

        X = self.preprocess(X, **kwargs)
        original_threads = torch.get_num_threads()
        try:
            if self.problem_type == "regression":
                return self.model.predict(X)
            proba = self.model.predict_proba(X)
        finally:
            # The library scopes n_jobs on a successful prediction. Restore it
            # here as well when preprocessing or inference raises.
            if torch.get_num_threads() != original_threads:
                torch.set_num_threads(original_threads)
        # Inner folds can be missing a class. Place the estimator's columns back
        # in AutoGluon's encoded class order, including zeroes for absent classes.
        ordered = np.zeros((len(X), self.num_classes), dtype=proba.dtype)
        ordered[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return self._convert_proba_to_unified_form(ordered)

    def get_device(self) -> str:
        return str(self.model.device_)

    def _set_device(self, device: str) -> None:
        target = self.to_torch_device(device)
        self.model.set_inference_device(target)
        self.model.ensemble_generator_.to(target)

    def prepare_for_inference(self) -> None:
        """Restore device-dependent inference state and place refit caches for serving.

        A shared fit is pickled as plain attributes, so the estimator's own
        ``__setstate__`` (which rebuilds ``inference_config_`` and reinstalls the
        fastpath on the reloaded network) does not run; the network comes back
        through ``ensure_network`` instead. Both are cheap and idempotent.
        """
        super().prepare_for_inference()
        model = getattr(self, "model", None)
        if model is None:
            return
        if self._shared_state is not None:
            model.set_inference_device(model.device)
        if model.device_.type == "cuda" and model.batch_size == "auto" and getattr(model, "model_kv_cache_", None):
            from lattice._sklearn.cache_residency import place_caches

            place_caches(model)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


class LatticeTinyModel(LatticeModel):
    """Lattice with the independently released tiny checkpoints."""

    ag_key = "TA-LATTICE-TINY"
    ag_name = "TA-Lattice-tiny"
    model_size = "tiny"
    default_n_estimators = 4
