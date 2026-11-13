from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "ai-for-data/inpriora"
#: Context budget. TabArena-v0.1 stays below both (150,000 rows; 12.9M cells); larger
#: BeyondArena tables are subsampled so their context cache fits the GPU.
MAX_CONTEXT_ROWS = 150_000
MAX_CONTEXT_CELLS = 15_000_000
#: Revision of the released classifier and regressor checkpoints.
DEFAULT_REVISION: str | None = "f8638e834d9cb26d77c007c4125744fc6c88381e"


def _weights_dir() -> Path | None:
    local = os.environ.get("INPRIORA_WEIGHTS_DIR")
    return Path(local).expanduser() if local else None


class InprioraModel(AbstractTorchModel):
    """Inpriora is a pretrained tabular foundation model that predicts by in-context learning.

    ``fit`` encodes the training table into a context the network attends over; ``predict`` runs a forward
    pass with no gradient updates. The released package ships one classifier and one regressor checkpoint,
    both selected from the task's problem type. Above the classifier's ten native classes the library
    builds a hierarchical label encoding itself, so the wrapper declares no class cap.

    The context cache is built for every fit and kept as torch tensors, which AutoGluon writes beside
    ``model.pkl`` (``model_tensors.pt``). Prediction from that cache reads only the fitted preprocessors,
    so the raw training matrix is dropped after the fit. A many-class fit disables the cache in the
    library; that path keeps the arrays prediction needs, and the wrapper moves them to torch tensors
    when the model is pickled, which keeps them out of ``model.pkl`` as well. The library halves its
    ensemble batch on an out-of-memory error during a fit or a prediction.

    Tables above 150,000 context rows or 15,000,000 cells are subsampled, stratified for
    classification; TabArena-v0.1 stays below both. The fit encodes every training row, and the
    library has no time-budget hook, so ``time_limit`` is not used; a hook in the library is needed
    to stop a fit early. Infinite feature values become NaN, which the library treats as missing.

    Paper: pending
    Codebase: https://github.com/ai-for-data/inpriora
    License: Apache-2.0
    """

    ag_key = "TA-INPRIORA"
    ag_name = "TA-Inpriora"
    ag_priority = 65
    seed_name = "random_seed"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "inpriora",
        "inpriora.core.model",
        "inpriora.estimators.classifier",
        "inpriora.estimators.regressor",
        "safetensors.torch",
    )
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0  # the library runs on the CPU as well
    gpu_strongly_recommended = True
    _default_auxiliary_params_extra = {
        "valid_raw_types": ["bool", "int", "float", "category"],
        "max_classes": None,
    }
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: A frozen network is built once per process and reused by the bagged children and the refit child.
    #: A user-supplied estimator (``model_path``) is private to its fit.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=(
            "inpriora.estimators.classifier:InprioraClassifier._load_model",
            "inpriora.estimators.regressor:InprioraRegressor._load_model",
        ),
        key=("model_path", "repo_id", "revision", "offline"),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def get_model_cls(self):
        if self.problem_type == "regression":
            from inpriora import InprioraRegressor

            return InprioraRegressor
        from inpriora import InprioraClassifier

        return InprioraClassifier

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            import inpriora  # noqa: F401
        except ImportError as err:
            logger.log(40, "\tFailed to import inpriora! Install the `inpriora` package or the tabarena extra.")
            raise err

        import torch
        from torch.cuda import is_available

        device = self._resolve_fit_device(num_gpus=num_gpus)
        if str(device).startswith("cuda") and not is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. Switch to CPU usage instead.",
            )
        self.device = device

        params = self._get_model_params()
        params["device"] = device
        params["n_jobs"] = num_cpus
        if params.get("model_path") is None:
            params.setdefault("repo_id", DEFAULT_REPO_ID)
            params.setdefault("revision", DEFAULT_REVISION)
        local = _weights_dir()
        if local is not None and params.get("model_path") is None:
            task = "regressor" if self.problem_type == "regression" else "classifier"
            params["model_path"] = str(local / task)

        model_cls = self.get_model_cls()
        self.model = model_cls(**params)
        # The library limits threads around predict only; the fit builds the context cache with torch's
        # ambient thread count, so the fit has to set the budget itself and restore it.
        threads_og = torch.get_num_threads()
        torch.set_num_threads(num_cpus)
        try:
            raw_rows = min(MAX_CONTEXT_ROWS, MAX_CONTEXT_CELLS // max(X.shape[1], 1))
            X, y = self._subsample_context(X, y, max(1, raw_rows))
            X = self.preprocess(X)
            max_rows = min(MAX_CONTEXT_ROWS, MAX_CONTEXT_CELLS // max(X.shape[1], 1))
            X, y = self._subsample_context(X, y, max(1, max_rows))
            self.model.fit(X=X, y=y)
        finally:
            torch.set_num_threads(threads_og)
        self._drop_training_tables()

    def _subsample_context(self, X, y, max_rows: int):
        """Cap the training context, stratified by class for classification."""
        if X.shape[0] <= max_rows:
            return X, y
        import numpy as np

        seed = self.params.get("random_seed", 0)
        rng = np.random.default_rng(0 if seed is None else int(seed))
        n_rows = X.shape[0]
        if self.problem_type == "regression":
            indices = rng.choice(n_rows, size=max_rows, replace=False)
        else:
            y_values = np.asarray(y)
            labels, counts = np.unique(y_values, return_counts=True)
            quota = counts.astype(float) * max_rows / n_rows
            take = np.minimum(np.floor(quota).astype(int), counts)
            if max_rows >= len(labels):
                take = np.maximum(take, 1)
            excess = int(take.sum()) - max_rows
            if excess > 0:
                for index in np.argsort(-take):
                    removable = min(excess, int(take[index]) - 1)
                    if removable > 0:
                        take[index] -= removable
                        excess -= removable
                    if excess == 0:
                        break
            remaining = max_rows - int(take.sum())
            order = np.argsort(-(quota - take))
            position = 0
            while remaining > 0:
                index = order[position % len(order)]
                if take[index] < counts[index]:
                    take[index] += 1
                    remaining -= 1
                position += 1
            parts = [
                rng.choice(np.flatnonzero(y_values == label), size=int(count), replace=False)
                for label, count in zip(labels, take, strict=False)
                if count > 0
            ]
            indices = np.concatenate(parts)
        logger.warning(
            "Inpriora: limiting the context from %d rows x %d features to %d rows.",
            n_rows,
            X.shape[1],
            max_rows,
        )
        y = y.iloc[indices] if hasattr(y, "iloc") else y[indices]
        return X.iloc[indices], y

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 8)
        self._set_default_param_value("cache", True)

    def get_device(self) -> str:
        return str(self.model.device_)

    def set_device(self, device: str):
        if self.model is None:
            self.device = device if isinstance(device, str) else device.type
            return
        super().set_device(device)
        # Shared weights replace the network without calling _set_device, so the cache and runner are
        # adjusted here for both the shared and the unshared path.
        self._sync_runtime_state(device)

    def _sync_runtime_state(self, device: str) -> None:
        import torch

        target = self.to_torch_device(device)
        if getattr(self.model, "model_cache_", None):
            # A cache built under CUDA autocast is float16; CPU attention needs both sides in float32.
            dtype = torch.float32 if target.type == "cpu" else None
            self.model.model_cache_ = {
                key: cache.to(target, dtype=dtype) for key, cache in self.model.model_cache_.items()
            }
        if hasattr(self.model, "runner_"):
            self.model._build_runner()

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.device = str(device)
        self.model.model_.to(device)

    # --- row-sized fitted state -------------------------------------------------------------------

    def _row_state_slots(self) -> list[tuple[object, object]]:
        """Slots holding arrays whose size grows with the training rows."""
        estimator = self.model
        slots: list[tuple[object, object]] = []
        ensemble = getattr(estimator, "ensemble_", None)
        if ensemble is not None:
            slots.extend((ensemble, name) for name in ("X_", "y_", "train_rank_"))
            rank = getattr(ensemble, "rank_reference_", None)
            columns = getattr(rank, "sorted_columns", None)
            if isinstance(columns, list):
                slots.extend((columns, index) for index in range(len(columns)))
            for pipeline in getattr(ensemble, "preprocessors_", {}).values():
                slots.append((pipeline, "X_transformed_"))
        for method, values in getattr(estimator, "target_values_", {}).items():
            slots.append((estimator.target_values_, method))
        return slots

    @staticmethod
    def _get_slot(owner: object, key: object):
        if isinstance(owner, (list, dict)):
            return owner.get(key) if isinstance(owner, dict) else owner[key]
        return getattr(owner, key, None)

    @staticmethod
    def _set_slot(owner: object, key: object, value) -> None:
        if isinstance(owner, (list, dict)):
            owner[key] = value
        else:
            setattr(owner, key, value)

    def _drop_training_tables(self) -> None:
        # The cached path predicts from ``model_cache_`` alone; the uncached path reads
        # ``X_transformed_``, ``y_``, ``train_rank_`` and, for regression, ``target_values_``.
        ensemble = getattr(self.model, "ensemble_", None)
        if ensemble is None:
            return
        if hasattr(ensemble, "X_"):
            ensemble.X_ = None
        if getattr(self.model, "cache_mode_", None) is not None:
            for name in ("y_", "train_rank_"):
                if hasattr(ensemble, name):
                    setattr(ensemble, name, None)
            for pipeline in getattr(ensemble, "preprocessors_", {}).values():
                pipeline.X_transformed_ = None
            if hasattr(self.model, "target_values_"):
                self.model.target_values_ = {}

    def _to_tensor_storage(self) -> None:
        import numpy as np
        import torch

        for owner, key in self._row_state_slots():
            value = self._get_slot(owner, key)
            if isinstance(value, np.ndarray):
                self._set_slot(owner, key, torch.as_tensor(value))

    def _to_numpy_storage(self) -> None:
        import torch

        for owner, key in self._row_state_slots():
            value = self._get_slot(owner, key)
            if isinstance(value, torch.Tensor):
                self._set_slot(owner, key, value.detach().cpu().numpy())

    def __getstate__(self):
        # AutoGluon bypasses the estimator's pickle protocol for shared weights; tensors moved here land in
        # ``model_tensors.pt`` instead of ``model.pkl``. The state holds this estimator itself, so the arrays
        # stay tensors until the next predict converts them back.
        self._to_tensor_storage()
        return super().__getstate__()

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        import numpy as np

        X = super()._preprocess(X, **kwargs)
        numeric = X.select_dtypes(include="number")
        if numeric.empty or not np.isinf(numeric.to_numpy()).any():
            return X
        # The library accepts NaN but rejects infinities, so an infinite cell becomes a missing one.
        X = X.copy()
        X[numeric.columns] = numeric.replace([np.inf, -np.inf], np.nan)
        return X

    def _predict_proba(self, X, **kwargs):
        self._to_numpy_storage()
        # A load restores the fit-time device; the library may have fallen back to the CPU when that
        # device is absent, and the runner and cache have to follow.
        self._sync_runtime_state(str(self.model.device_))
        if self.problem_type == "regression":
            return super()._predict_proba(X, **kwargs)
        import numpy as np

        proba = self.model.predict_proba(self.preprocess(X, **kwargs))
        # A fold can lack a class; the library then returns only the present classes' columns, so
        # place them back into AutoGluon's ``0..K-1`` order.
        ordered = np.zeros((len(X), self.num_classes), dtype=proba.dtype)
        ordered[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return self._convert_proba_to_unified_form(ordered)


def prefetch_weights() -> str:
    """Fetch both networks from the Hugging Face repository, or return the configured local directory."""
    local = _weights_dir()
    if local is not None:
        for task in ("classifier", "regressor"):
            for filename in ("config.json", "model.safetensors"):
                if not (local / task / filename).is_file():
                    raise FileNotFoundError(local / task / filename)
        return str(local)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        DEFAULT_REPO_ID,
        revision=DEFAULT_REVISION,
        allow_patterns=[
            f"{task}/{filename}"
            for task in ("classifier", "regressor")
            for filename in ("config.json", "model.safetensors")
        ],
    )
