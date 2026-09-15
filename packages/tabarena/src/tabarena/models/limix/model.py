from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
)
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models import prefetch as _hub
from tabarena.models._shared_estimators import check_payload_device, detach_by_path
from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models._weights import normalize_device

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from tabarena.models._weights import WeightsKey


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
#: Retrieval-free pipelines; the warm-up dummy fit runs the first one (see ``warmup_dummy_fit_hyperparameters``).
_NORETRIEVAL_CLS_CONFIG = "cls_default_noretrieval.json"

#: The checkpoint-relevant defaults of ``LimiXPredictor``: ``mask_prediction`` is applied to the network's
#: config when the checkpoint is loaded, so it is part of the registry key.
DEFAULT_PARAMS: dict[str, Any] = {"mask_prediction": False}


def _load_bundled_config(filename: str) -> list:
    cfg_path = _CONFIG_DIR / filename
    with cfg_path.open("r") as f:
        return json.load(f)


def _mask_prediction_flag(key: WeightsKey) -> bool:
    return str(dict(key.flags).get("mask_prediction", DEFAULT_PARAMS["mask_prediction"])).lower() == "true"


def _attention_steps(predictor: Any) -> Iterator[Any]:
    """The ``InferenceAttentionMap`` retrieval steps of a predictor's pipelines, which each hold the network.

    Duck-typed on the attribute the step's constructor sets, so the vendored inference modules need
    not be imported on load.
    """
    for pipeline in getattr(predictor, "preprocess_pipelines", None) or ():
        for step in pipeline:
            if hasattr(step, "calculate_sample_attention"):
                yield step


class LimiXModel(SharedWeightsModelMixin, AbstractTorchModel):
    """LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence.

    Paper: https://arxiv.org/abs/2509.03505
    Codebase: https://github.com/limix-ldm-ai/LimiX
    License: Apache-2.0

    Upstream is not pip-installable, so the inference-time sources are
    vendored under ``_vendor/`` next to this file.

    The network is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`) and handed to the vendored predictor's ``model=``
    argument, which also routes every ``InferenceAttentionMap`` retrieval step through it instead of
    the per-step ``load_model`` the upstream code performs. A custom ``model_path`` keeps the
    upstream load path.
    """

    ag_key = "TA-LIMIX"
    #: Modules the timed fit would otherwise import for the first time: the vendored predictor and
    #: inference code (which pull in torch, einops, sklearn and kditransform) and the Hub client the
    #: checkpoint resolver uses.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabarena.models.limix._vendor.inference.predictor",
        "tabarena.models.limix._vendor.inference.inference_method",
        "huggingface_hub",
        "huggingface_hub.errors",
    )
    #: Cheapness knob for the warm-up dummy fit: a single retrieval-free pipeline, which also runs on
    #: a CPU (the retrieval pipelines raise there). ``inference_config`` never touches the network,
    #: so the primed key is unaffected.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {
        "inference_config": _load_bundled_config(_NORETRIEVAL_CLS_CONFIG)[:1],
    }
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
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}
    # We set the default to 100k to try to run on all of TabArena.
    # Note, all examples of LimiX code itself says one should skip above 50k.
    _default_auxiliary_params_extra = {
        # "max_rows": 50_000, # Technically from LimiX
        "max_classes": 10,
    }

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="limix",
        checkpoint=CheckpointSpec(
            repo_id=_DEFAULT_HF_REPO, filename=_DEFAULT_HF_FILENAME, revision=_DEFAULT_HF_REVISION
        ),
        variant="network",  # one ``FeaturesTransformer`` checkpoint serves classification and regression alike
        default_params=DEFAULT_PARAMS,
        flag_params=("mask_prediction",),
        disable_when=("model_path",),
        network_attr="model",
        device_attrs=(("device", "torch"),),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._cat_indices: list[int] | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

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

    # --- shared weights ---------------------------------------------------------------------------

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        """Build the network for ``key`` exactly as the vendored ``load_model`` does.

        ``load_model`` reads the checkpoint on the CPU, builds the ``FeaturesTransformer`` from the
        stored config with ``mask_prediction`` applied and loads the state dict; the module is then
        moved to ``key.device`` and its ``encoder_x`` wrapped in the NaN-sanitizing encoder every
        per-child load also receives (see :func:`_nan_clean_encoder_cls`).
        """
        import torch

        from tabarena.models.limix._vendor.utils.loading import load_model

        network = load_model(model_path=key.checkpoint, mask_prediction=_mask_prediction_flag(key))
        network.encoder_x = _nan_clean_encoder_cls()(network.encoder_x)
        return network.to(torch.device(key.device))

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """Point the predictor and every ``InferenceAttentionMap`` step at ``payload``."""
        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        self.model.model = payload
        for step in _attention_steps(self.model):
            step.model = payload
        self._apply_device_bookkeeping(device_type)

    def _detach_for_pickle(self, estimator: Any) -> Any:
        """A shallow copy of the predictor with the network cleared from it and from every retrieval step."""
        copy = detach_by_path(estimator, "model")
        pipelines = getattr(estimator, "preprocess_pipelines", None)
        if pipelines is not None:
            copy.preprocess_pipelines = [
                [detach_by_path(step, "model") if hasattr(step, "calculate_sample_attention") else step for step in p]
                for p in pipelines
            ]
        return copy

    def _resolve_default_checkpoint(self, *, stage: str) -> str:
        """Local path of the released checkpoint for an unshared fit, honoring ``ag.fetch_pretrained_weights`` at ``stage``."""
        allow = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage=stage)
        try:
            resolved = type(self)._resolve_shared_checkpoint(
                problem_type=self.problem_type, variant="network", hyperparameters={}, allow_download=allow, stage=stage
            )
        except _hub.WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=self.name, stage=stage, location=str(exc))
            ) from exc
        return resolved.path

    # --- fit ------------------------------------------------------------------------------------

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit LimiX: store the (possibly subsampled) support set behind a predictor holding the network.

        With sharing on, the network comes from the registry and is injected through the vendored
        predictor's ``model=`` argument; the predictor's own state (``inference_config``, ``seed``,
        the preprocessing pipelines) stays per child. With sharing off, the predictor loads the
        checkpoint for this child exactly as upstream does, and the NaN-sanitizing encoder wrap is
        applied to every loaded copy here.
        """
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

        from tabarena.models.limix._vendor.inference.predictor import LimiXPredictor

        key, network = self._acquire_shared_weights(device=device_str)

        hps = self._get_model_params()
        random_state = hps.pop(self.seed_name, 0)
        model_path = hps.pop("model_path", None)
        inference_config = hps.pop("inference_config", None)
        if inference_config is None:
            cfg_filename = _DEFAULT_CLS_CONFIG if self.problem_type in ["binary", "multiclass"] else _DEFAULT_REG_CONFIG
            inference_config = _load_bundled_config(cfg_filename)

        if key is None:
            model_path = model_path or self._resolve_default_checkpoint(stage="fit")
        else:
            check_payload_device(network, device_str)
            model_path = key.checkpoint

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

        self.model = LimiXPredictor(
            device=torch.device(device_str),
            model_path=str(model_path),
            inference_config=inference_config,
            categorical_features_indices=self._cat_indices or None,
            seed=int(random_state),
            model=network,
            **hps,
        )
        if network is None:
            # See `_NaNCleanEncoder` docstring for why this wrap is needed. We have to wrap
            # every loaded copy of the FeaturesTransformer, not just `LimiXPredictor.model`:
            # each `InferenceAttentionMap` step in `preprocess_pipelines` calls
            # `load_model(self.model_path)` in its own `__init__` and holds its own model
            # instance, used to compute sample-attention scores for retrieval. Without
            # wrapping those too, the very first attention-map pass at
            # `_vendor/inference/inference_method.py:309` still hits the NaN guard.
            # (A shared network is wrapped once by `_build_shared_weights`.)
            nan_clean_encoder_cls = _nan_clean_encoder_cls()
            self.model.model.encoder_x = nan_clean_encoder_cls(self.model.model.encoder_x)
            for step in _attention_steps(self.model):
                step.model.encoder_x = nan_clean_encoder_cls(step.model.encoder_x)
        # Save into model so pickling works better
        self.model._X_train = X_np
        self.model._y_train = y_fit

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """LimiX does not support a sklearn API, thus, we have to call the forward pass this way."""
        import torch

        X = self.preprocess(X, **kwargs)

        # Forward pass call via LimiX code
        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"

        # Chunk the test set: a single forward pass over (n_train + n_test) rows can blow
        # past available VRAM on large datasets and surface as cudaErrorInvalidConfiguration.
        chunk_size = self.batch_test_n_rows
        n_test = X.shape[0]
        chunks = []
        for start in range(0, n_test, chunk_size):
            chunk_out = self.model.predict(
                self.model._X_train,
                self.model._y_train,
                X[start : start + chunk_size],
                task_type=task_type,
            )
            # LimiX runs under autocast, so outputs can come back in fp16. Promote to
            # fp32 here so downstream math (e.g. regression `out * self._y_std`) cannot
            # overflow fp16's ~65504 max on large-target regression problems.
            if isinstance(chunk_out, torch.Tensor):
                chunk_out = chunk_out.detach().to(torch.float32).cpu().numpy()
            else:
                chunk_out = np.asarray(chunk_out, dtype=np.float32)
            chunks.append(chunk_out)
        out = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        if task_type == "Regression":
            out = out * self._y_std + self._y_mean
        y_pred_proba = out

        return self._convert_proba_to_unified_form(y_pred_proba)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


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

    Sanitizing here is the most surgical place: it catches NaN regardless of which
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
