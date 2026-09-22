from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
)
from autogluon.core.models.abstract import SharedWeights
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.utils.logging_utils import import_many_class_classifier

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
_CONFIG_DIR = _VENDOR_DIR / "config"

_DEFAULT_HF_REPO = "stableai-org/LimiX-16M"
_DEFAULT_HF_FILENAME = "LimiX-16M.ckpt"
#: Commit pinned so the checkpoint fetched here never silently changes if the
#: repo's default branch moves. Bump deliberately (with a note on what changed)
#: when picking up a newer checkpoint.
_DEFAULT_HF_REVISION = "da5f3072bf3633c70d957c02518c30d461007764"
_DEFAULT_CLS_CONFIG = "cls_default_16M_retrieval.json"
_DEFAULT_REG_CONFIG = "reg_default_16M_retrieval.json"
#: The retrieval-free classification pipeline: the cheap configuration of the warm-up's dummy fit.
_NORETRIEVAL_CLS_CONFIG = "cls_default_noretrieval.json"


def _load_bundled_config(filename: str) -> list:
    cfg_path = _CONFIG_DIR / filename
    with cfg_path.open("r") as f:
        return json.load(f)


def _build_and_fit_limix(
    *,
    X_np: np.ndarray,
    y_fit: np.ndarray,
    device_str: str,
    model_path: str,
    inference_config: list,
    cat_indices: list[int] | None,
    seed: int,
    hps: dict,
):
    """Construct, NaN-encoder-patch, and "fit" (store train data on) one
    ``LimiXPredictor`` instance. Factored out of ``LimiXModel._fit`` so the
    exact same construction logic can be reused per cloned sub-estimator when
    ``ManyClassClassifier`` (ECOC) wraps this model for many-class datasets --
    each ECOC sub-estimator needs its own independently NaN-patched instance,
    not a shared one.
    """
    import torch

    from tabarena.models.limix._vendor.inference.inference_method import InferenceAttentionMap
    from tabarena.models.limix._vendor.inference.predictor import LimiXPredictor

    model = LimiXPredictor(
        device=torch.device(device_str),
        model_path=str(model_path),
        inference_config=inference_config,
        categorical_features_indices=cat_indices,
        seed=int(seed),
        **hps,
    )
    # See `_NaNCleanEncoder` docstring for why this wrap is needed. We have to wrap
    # every loaded copy of the FeaturesTransformer, not just `LimiXPredictor.model`:
    # each `InferenceAttentionMap` step in `preprocess_pipelines` calls
    # `load_model(self.model_path)` in its own `__init__` and holds its own model
    # instance, used to compute sample-attention scores for retrieval. Without
    # wrapping those too, the very first attention-map pass at
    # `_vendor/inference/inference_method.py:309` still hits the NaN guard.
    nan_clean_encoder_cls = _nan_clean_encoder_cls()
    model.model.encoder_x = nan_clean_encoder_cls(model.model.encoder_x)
    for pipeline in model.preprocess_pipelines:
        for step in pipeline:
            if isinstance(step, InferenceAttentionMap):
                step.model.encoder_x = nan_clean_encoder_cls(step.model.encoder_x)
    # Save into model so pickling works better
    model._X_train = X_np
    model._y_train = y_fit
    return model


def _predict_proba_limix(model, X: np.ndarray, *, task_type: str, batch_test_n_rows: int) -> np.ndarray:
    """Chunked forward pass shared by the single-instance path and every
    ``ManyClassClassifier`` (ECOC) sub-estimator -- factored out of
    ``LimiXModel._predict_proba`` for the same reason as
    ``_build_and_fit_limix`` above. Classification only (ECOC never wraps
    regression), so there is no y-rescaling here -- that stays in
    ``LimiXModel._predict_proba`` for the non-ECOC path.
    """
    import torch

    chunk_size = batch_test_n_rows
    n_test = X.shape[0]
    chunks = []
    for start in range(0, n_test, chunk_size):
        chunk_out = model.predict(
            model._X_train,
            model._y_train,
            X[start : start + chunk_size],
            task_type=task_type,
        )
        # LimiX runs under autocast, so outputs can come back in fp16. Promote to
        # fp32 here so downstream math cannot overflow fp16's ~65504 max.
        if isinstance(chunk_out, torch.Tensor):
            chunk_out = chunk_out.detach().to(torch.float32).cpu().numpy()
        else:
            chunk_out = np.asarray(chunk_out, dtype=np.float32)
        chunks.append(chunk_out)
    return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


class _LimiXSklearnWrapper:
    """Thin sklearn-compatible wrapper around one LimiX predictor instance.

    ``ManyClassClassifier`` (tabpfn-extensions) clones this per ECOC
    sub-problem and requires each fitted sub-estimator to expose
    ``classes_`` and a standard ``fit(X, y)``/``predict_proba(X)`` interface
    -- LimiX's own ``LimiXPredictor`` does neither (it takes train data as
    arguments to ``predict()`` instead of storing it via ``fit()``). Mirrors
    ``autogluon.tabular.models.mitra.mitra_model._MitraSklearnWrapper``,
    which solves the identical problem for Mitra.
    """

    def __init__(self, *, device_str, model_path, inference_config, cat_indices, seed, hps, batch_test_n_rows):
        self._device_str = device_str
        self._model_path = model_path
        self._inference_config = inference_config
        self._cat_indices = cat_indices
        self._seed = seed
        self._hps = hps
        self._batch_test_n_rows = batch_test_n_rows

    def fit(self, X, y):
        self._model = _build_and_fit_limix(
            X_np=X,
            y_fit=y,
            device_str=self._device_str,
            model_path=self._model_path,
            inference_config=self._inference_config,
            cat_indices=self._cat_indices,
            seed=self._seed,
            hps=self._hps,
        )
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return _predict_proba_limix(
            self._model, X, task_type="Classification", batch_test_n_rows=self._batch_test_n_rows
        )

    def get_params(self, deep=True):  # sklearn clone() compatibility
        return {
            "device_str": self._device_str,
            "model_path": self._model_path,
            "inference_config": self._inference_config,
            "cat_indices": self._cat_indices,
            "seed": self._seed,
            "hps": self._hps,
            "batch_test_n_rows": self._batch_test_n_rows,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, f"_{k}", v)
        return self


class LimiXModel(AbstractTorchModel):
    """LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence.

    Paper: https://arxiv.org/abs/2509.03505
    Codebase: https://github.com/limix-ldm-ai/LimiX
    License: Apache-2.0

    Upstream is not pip-installable, so the inference-time sources are
    vendored under ``_vendor/`` next to this file.
    """

    ag_key = "TA-LIMIX"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabarena.models.limix._vendor.inference.predictor",
        "tabarena.models.limix._vendor.inference.inference_method",
        "huggingface_hub",
    )
    ag_name = "TA-LimiX"
    ag_priority = 100
    seed_name = "random_state"

    _supported_problem_types = ["binary", "multiclass", "regression"]

    subsample_train_n_rows: int = 75_000
    """Empirically, even with 140 GB of VRAM available we still hit OOM on LimiX's retrieval + clustering inference
    path on TabArena-scale datasets, so subsampling is the only reliable lever to keep it running.
    We-sub-sample datasets above 75k rows to 50k rows following the LimiX documentation examples."""
    batch_test_n_rows: int = 5_000
    """We batch forward passes with more than 10k test rows."""
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: The vendored predictor and its retrieval steps load the network through the vendored
    #: ``load_model``; one build per checkpoint file and device per process, shared by the predictor
    #: and every ``InferenceAttentionMap`` step. The loader has no device input, so ``_fit`` records
    #: the device on ``self.device`` before constructing the predictor.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=(
            "tabarena.models.limix._vendor.inference.predictor:load_model",
            "tabarena.models.limix._vendor.inference.inference_method:load_model",
        ),
        key=("model_path", "mask_prediction"),
    )
    #: A single retrieval-free pipeline keeps the warm-up's dummy fit cheap, and runs on a CPU too
    #: (the retrieval pipelines raise there); ``inference_config`` never touches the network.
    cheap_hyperparameters: ClassVar[dict] = {"inference_config": _load_bundled_config(_NORETRIEVAL_CLS_CONFIG)[:1]}
    # Sequential fold fitting avoids contention on the shared HF checkpoint cache.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    # We set the default to 100k to try to run on all of TabArena.
    # Note, all examples of LimiX code itself says one should skip above 50k.
    _default_auxiliary_params_extra = {
        # "max_rows": 50_000, # Technically from LimiX
        "max_classes": None,
        "many_class_threshold": 10,  # Use ManyClassClassifier (ECOC) above this class count
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._cat_indices: list[int] | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None
        self._use_many_class = False  # True when ManyClassClassifier ECOC wrapper is active

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """We preprocess for LimiX to ensure categorical features are passed as correct dtypes to LimiX."""
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

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
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
        device_str = "cuda" if num_gpus != 0 else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hps = self._get_model_params()
        many_class_threshold = self.params_aux.get("many_class_threshold", 10)
        self._use_many_class = (
            self.problem_type in [BINARY, MULTICLASS]
            and self.num_classes is not None
            and self.num_classes > many_class_threshold
        )
        random_state = hps.pop(self.seed_name, 0)
        model_path = hps.pop("model_path", None) or _download_default_checkpoint()
        inference_config = hps.pop("inference_config", None)
        if inference_config is None:
            cfg_filename = _DEFAULT_CLS_CONFIG if self.problem_type in ["binary", "multiclass"] else _DEFAULT_REG_CONFIG
            inference_config = _load_bundled_config(cfg_filename)

        X_np = self.preprocess(X, y=y, is_train=True)
        y_np = np.asarray(y.to_numpy(), dtype=np.float32 if self.problem_type == "regression" else None)

        if self.problem_type == "regression":
            # Following all documentation and examples, we scale at this level and inverse scale later.
            # Stats are computed on the full y before any subsampling so inverse scaling matches the
            # original target distribution.
            self._y_mean = float(y_np.mean())
            self._y_std = float(y_np.std()) or 1.0
            y_fit = (y_np - self._y_mean) / self._y_std
        else:
            y_fit = y_np

        # Cap n_train to keep the LimiX inference pipeline within VRAM.
        # Empirically, even with 140 GB of VRAM available we still hit this on
        # datasets at the TabArena scale, so subsampling is the only reliable lever.
        # LimiX's own documentation / examples flag >50k rows as out-of-distribution
        # for this model (see https://github.com/limix-ldm-ai/LimiX/blob/main/inference_classifier.py#L108-L110).
        if X_np.shape[0] >= self.subsample_train_n_rows:
            n_full = X_np.shape[0]
            target = 50_000
            if self.problem_type in [BINARY, MULTICLASS]:
                from sklearn.model_selection import train_test_split

                try:
                    X_np, _, y_fit, _ = train_test_split(
                        X_np,
                        y_fit,
                        train_size=target,
                        stratify=y_fit,
                        random_state=int(random_state),
                    )
                except ValueError:
                    # Stratification fails on classes with too few samples; fall back to random.
                    rng = np.random.default_rng(int(random_state))
                    idx = rng.choice(n_full, size=target, replace=False)
                    X_np, y_fit = X_np[idx], y_fit[idx]
            else:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(n_full, size=target, replace=False)
                X_np, y_fit = X_np[idx], y_fit[idx]
            logger.log(
                20,
                f"LimiX: subsampling train from {n_full} to {target} rows to bound VRAM at predict time",
            )

        self.device = device_str  # keys the shared network; the vendored loader names no device
        if self._use_many_class:
            try:
                ManyClassClassifier = import_many_class_classifier()
            except ImportError:
                logger.log(
                    40,
                    "\tLimiX: tabpfn-extensions not installed; cannot use ManyClassClassifier "
                    f"for {self.num_classes} classes (limit: {many_class_threshold}). "
                    "Install with: pip install tabpfn-extensions",
                )
                raise
            logger.log(
                20,
                f"\tLimiX: {self.num_classes} classes exceeds native limit ({many_class_threshold}). "
                "Using ManyClassClassifier (ECOC wrapper).",
            )
            base_model = _LimiXSklearnWrapper(
                device_str=device_str,
                model_path=model_path,
                inference_config=inference_config,
                cat_indices=self._cat_indices or None,
                seed=int(random_state),
                hps=hps,
                batch_test_n_rows=self.batch_test_n_rows,
            )
            self.model = ManyClassClassifier(estimator=base_model, alphabet_size=many_class_threshold).fit(X_np, y_fit)
        else:
            self.model = _build_and_fit_limix(
                X_np=X_np,
                y_fit=y_fit,
                device_str=device_str,
                model_path=model_path,
                inference_config=inference_config,
                cat_indices=self._cat_indices or None,
                seed=int(random_state),
                hps=hps,
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """LimiX does not support a sklearn API, thus, we have to call the forward pass this way."""
        X = self.preprocess(X, **kwargs)

        if self._use_many_class:
            # ManyClassClassifier (ECOC) exposes its own predict_proba, which
            # internally dispatches to each sub-estimator's predict_proba
            # (i.e. _LimiXSklearnWrapper.predict_proba, the same chunked
            # forward pass as the non-ECOC branch below).
            return self._convert_proba_to_unified_form(self.model.predict_proba(X))

        # Forward pass call via LimiX code
        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"
        out = _predict_proba_limix(self.model, X, task_type=task_type, batch_test_n_rows=self.batch_test_n_rows)

        if task_type == "Regression":
            out = out * self._y_std + self._y_mean
        y_pred_proba = out

        return self._convert_proba_to_unified_form(y_pred_proba)

    def _ag_params(self) -> set[str]:
        return super()._ag_params() | {"many_class_threshold"}

    def get_device(self) -> str:
        # `self.device` (set in `_fit`) is authoritative regardless of which branch ran --
        # unlike `self.model`, which is a `ManyClassClassifier` (no `.device` attribute of
        # its own) rather than a raw `LimiXPredictor` when `self._use_many_class`.
        return self.device

    def _set_device(self, device: str):
        import torch

        device = torch.device(device)

        def _move(predictor) -> None:
            predictor.device = device
            if predictor.model is not None:
                predictor.model.to(device)

        if self._use_many_class:
            # Output coding keeps no fitted rows: `ManyClassClassifier` refits every code row at predict
            # time from its base wrapper (`estimators_` is None), so the device to record is the base
            # wrapper's. The single-fit shortcut, taken when no codebook was needed, holds one fitted row.
            self.model.estimator._device_str = str(device)
            for estimator in self.model.estimators_ or ():
                _move(estimator._model)
        else:
            _move(self.model)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> str:
        """Pre-download the default LimiX checkpoint from Hugging Face.

        Returns the local cache path. Used by the foundation-model pre-download
        scripts to warm the cache before parallel fit runs. We try the local
        cache first so offline compute nodes (no internet / proxy timeouts)
        skip the HEAD-request-for-etag that ``hf_hub_download`` performs by
        default.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return hf_hub_download(
                repo_id=_DEFAULT_HF_REPO,
                filename=_DEFAULT_HF_FILENAME,
                revision=_DEFAULT_HF_REVISION,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(
                repo_id=_DEFAULT_HF_REPO,
                filename=_DEFAULT_HF_FILENAME,
                revision=_DEFAULT_HF_REVISION,
            )


def _download_default_checkpoint() -> str:
    return LimiXModel.prefetch_weights()


@functools.cache
def _nan_clean_encoder_cls() -> type:
    """Build (once) the ``nn.Module`` that sanitizes LimiX's ``encoder_x`` output.

    Defined inside a cached factory rather than at module scope so importing this
    module does not import ``torch``. That keeps the LimiX model off the import path
    of light-weight consumers (e.g. ``TabArenaContext`` pulls in every model's
    ``info.py``, which would otherwise transitively import ``torch``). ``functools.cache``
    gives a stable class identity across calls, which the idempotency check relies on.

    A class built inside a function is normally unpicklable: pickle stores a class by
    ``__module__`` + ``__qualname__`` and re-looks it up on load, and the default qualname
    here would be ``_nan_clean_encoder_cls.<locals>._NaNCleanEncoder``, which pickle rejects
    outright. Since AutoGluon pickles every fitted model (bagging alone pickles each fold
    child back to the parent), the qualname is rewritten to a plain module-level name and the
    module ``__getattr__`` below resolves it, rebuilding the class on demand in a process that
    has not called this factory yet. ``functools.cache`` is what makes that lookup return the
    *same* object, which is the identity check pickle performs.

    The wrapper itself: LimiX's bundled 16M checkpoint starts its preprocess pipeline
    with a ``NanEncoder`` (`_vendor/model/encoders.py:361`) that replaces NaN cells in
    ``x`` with the per-column mean computed over the *train portion only*
    (``calc_mean(x[:, :eval_pos, :], dim=1)``). LimiX's retrieval + clustering path
    (`_vendor/inference/inference_method.py`) shards inference into small train clusters
    per test group, and on datasets with heavy missingness the selected train rows for a
    given cluster can end up entirely NaN on some column. The per-column "mean" is then
    itself NaN, the imputation step substitutes NaN for NaN, and the NaN propagates
    through ``process_4_x`` and ``encoder_x`` until ``transformer.py:194`` raises:

        ValueError: embedded_all contains NaN values; please add a NanEncoder
        in the encoder

    Sanitizing here is the most surgical place — it catches NaN regardless of which
    upstream stage produced it, without modifying vendor code and without blanket-imputing
    the raw input (the model handles NaN correctly on most datasets and we don't want to
    overwrite that behavior).
    """
    import torch
    from torch import nn

    class _NaNCleanEncoder(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            # Idempotent: collapse nested wraps so re-applying is safe.
            if isinstance(inner, _NaNCleanEncoder):
                inner = inner.inner
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, dict) and isinstance(out.get("data"), torch.Tensor):
                out["data"] = torch.nan_to_num(out["data"], nan=0.0, posinf=0.0, neginf=0.0)
            return out

    # Make the class reachable as `<this module>._NaNCleanEncoder` so pickle can find it.
    _NaNCleanEncoder.__qualname__ = _NaNCleanEncoder.__name__
    return _NaNCleanEncoder


def __getattr__(name: str) -> type:
    """Resolve the lazily-built ``_NaNCleanEncoder`` for pickle (PEP 562).

    Only consulted for names missing from the module namespace, so it costs nothing on a
    normal attribute access and never imports ``torch`` on its own.
    """
    if name == "_NaNCleanEncoder":
        return _nan_clean_encoder_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
