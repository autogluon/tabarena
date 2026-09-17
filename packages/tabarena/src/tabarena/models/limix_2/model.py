from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

_HF_REPO = "stable-ai/LimiX-2"
_HF_FILENAME = "LimiX-2.ckpt"
#: Commit pinned so the checkpoint fetched here never silently changes if the repo's default branch
#: moves. Bump deliberately (with a note on what changed) when picking up a newer checkpoint.
_HF_REVISION = "20c07a07801973a0aec57c31062bdb2ea1cda2b2"
#: The release's no-retrieval inference configs, one per task type, packaged in the LimiX
#: distribution's top-level ``config`` package.
_DEFAULT_CONFIGS = {
    "classification": "cls_default_noretrieval_v2.json",
    "regression": "reg_default_noretrieval_v2.json",
}
_INSTALL_HINT = (
    "LimiX-2 needs the LimiX inference package, installed without its dependency tree "
    "(it pins torch==2.9.1): see the LimiX2Model docstring."
)


def _patch_predictor(predictor_cls: type) -> None:
    """Developer fix (LimiX ``774aa3e``): two habits of ``LimiXPredictor`` that do not fit a process
    with a network shared across fits.

    ``__init__`` builds a ``CacheManager`` on a hard-coded cluster path (``/mnt/public/...``) even
    with ``use_data_cache=False``, and its ``os.makedirs`` fails on any other machine: an unwritable
    cache root is redirected to ``~/.cache/limix/infe_cache``. ``close()`` (also run by ``__del__``)
    moves the network to the CPU; with the network shared by the bagged children, one collected child
    would move it away from the estimators still using it, and the next timed predict would pay the
    copy back: ``close`` only shuts the pipeline-parallel worker pool down. Upstream should build the
    cache manager lazily and leave a network it did not build alone. Applied once per process.
    """
    if getattr(predictor_cls, "_tabarena_patched", False):
        return
    cache_cls = predictor_cls.CacheManager
    cache_init = cache_cls.__init__

    def init_cache(self, cache_dir="/mnt/public/infe_cache", *args, **kwargs):
        try:
            cache_init(self, cache_dir, *args, **kwargs)
        except OSError:
            fallback = Path.home() / ".cache" / "limix" / "infe_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            cache_init(self, str(fallback), *args, **kwargs)

    def close(self):
        pool = getattr(self, "_pipeline_gpu_pool", None)
        if pool is not None:
            pool.close()
            self._pipeline_gpu_pool = None

    cache_cls.__init__ = init_cache
    predictor_cls.close = close
    predictor_cls._tabarena_patched = True


class LimiX2Model(AbstractTorchModel):
    """LimiX-2 TabArena integration.

    LimiX-2 is Stable AI's 400M-parameter tabular foundation model, a Contextual Mechanism Network
    pretrained with context-conditional masked modeling on synthetic data from structural causal
    models. Prediction is in context: the fit stores the training table, and every predict passes it
    together with the query rows through an ensemble of preprocessing pipelines (32 members for
    classification, 8 for regression) into one forward pass each. No parameter is updated, so the
    fit consumes no validation data and ignores the time limit. Regression predictions come back on
    the original target scale.

    Paper: LimiX-2: A Large Foundation Model for Structured Data (LimiX Team, Stable AI, 2026),
    https://arxiv.org/abs/2609.17488
    Codebase: https://github.com/limix-ldm-ai/LimiX (Stable AI Technology Co., Ltd. License, Version 1.0)
    Weights: https://huggingface.co/stable-ai/LimiX-2 (StableAI LimiX Non-Commercial License v1.0)

    The inference package is not on PyPI and pins ``torch==2.9.1``, so install it without its
    dependency tree next to the torch already present (it needs ``nvtx`` on top)::

        pip install --no-deps "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@774aa3e1a994cbe38f33758e3d663e9951855554"
        pip install nvtx

    Hyperparameters: ``n_estimators`` keeps the first members of the packaged config (``None``, the
    default, keeps all), ``inference_config`` replaces the packaged config with a dict of the same
    shape, ``model_path`` points at another checkpoint, and every other key is forwarded to
    ``LimiXPredictor`` (``seed``, ``softmax_temperature``, ``test_batch_size``, ``deterministic``, ...).
    """

    ag_key = "TA-LIMIX-2"
    ag_name = "TA-LimiX-2"
    ag_priority = 100
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "limix",
        "inference.v2_0.predictor",
        "model.v2_0.loading",
        "huggingface_hub",
    )
    gpu_strongly_recommended = True  # in-context inference over the training table is far slower on a CPU

    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {"max_classes": 10}  # the checkpoint's classification head
    #: One fold at a time, and a refit on the full data instead of the bag: same quality for an
    #: in-context model, one network at inference.
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    #: The predictor builds its network through ``load_model`` (the ``utils.loading`` dispatcher, bound
    #: in the v2 predictor module), which reads the checkpoint on the CPU; one build per checkpoint
    #: and device per process. The loader names no device, so ``_fit`` records it on ``self.device``
    #: before constructing the predictor.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="inference.v2_0.predictor:load_model",
        key=("model_path", "mask_prediction", "deterministic"),
    )
    #: One ensemble member keeps the warm-up's dummy fit cheap; the member count never touches the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._X_train: pd.DataFrame | None = None
        self._y_train: np.ndarray | None = None

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Hand LimiX the frame it encodes itself, with pandas ``category`` and ``string`` columns as
        ``object`` (the dtype its CSV-fed examples see).
        """
        X = super()._preprocess(X, **kwargs)
        to_object = [
            col
            for col, dtype in X.dtypes.items()
            if isinstance(dtype, pd.CategoricalDtype)
            or (pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype))
        ]
        return X.astype(dict.fromkeys(to_object, object)) if to_object else X

    def _default_inference_config(self) -> dict:
        """The packaged no-retrieval configuration of this task type."""
        task = "classification" if self.problem_type in [BINARY, MULTICLASS] else "regression"
        try:
            text = resources.files("config").joinpath(_DEFAULT_CONFIGS[task]).read_text()
        except ModuleNotFoundError as err:
            raise ImportError(_INSTALL_HINT) from err
        return json.loads(text)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        try:
            from inference.v2_0.predictor import LimiXPredictor
        except ImportError as err:
            raise ImportError(_INSTALL_HINT) from err

        _patch_predictor(LimiXPredictor)
        self.device = self._resolve_fit_device(num_gpus)  # keys the shared network; the loader names no device
        hps = self._get_model_params()
        model_path = hps.pop("model_path", None) or self.prefetch_weights()
        inference_config = hps.pop("inference_config", None) or self._default_inference_config()
        n_estimators = hps.pop("n_estimators", None)
        if n_estimators is not None:
            inference_config = {**inference_config, "pipelines": inference_config["pipelines"][:n_estimators]}
        # The ``limix.LimiXPredictor`` factory reads the checkpoint itself and hands the dict to this
        # constructor; constructed directly, the v2 class reads it through ``load_model``, the call the
        # shared-weights declaration names.
        self.model = LimiXPredictor(
            device=torch.device(self.device),
            model_path=str(model_path),
            inference_config=inference_config,
            preprocess_num_jobs=num_cpus,
            **hps,
        )
        self._X_train = self.preprocess(X)
        self._y_train = y.to_numpy()

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """One in-context pass over the stored training table and the query rows (LimiX has no
        sklearn fit API). The regression decoder can return a tensor; both come back as float32.
        """
        import torch

        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"
        preds = self.model.predict(self._X_train, self._y_train, self.preprocess(X, **kwargs), task_type=task_type)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().float().cpu().numpy()
        return self._convert_proba_to_unified_form(np.asarray(preds, dtype=np.float32))

    def get_device(self) -> str:
        return self.model.device.type

    def _set_device(self, device: str):
        self.model.device = self.to_torch_device(device)
        self.model.model.to(self.model.device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}  # no validation data is consumed by the fit

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """A 10 GB baseline (the 400M-parameter network, its activations and the per-member
        preprocessing copies) plus five times the frame.
        """
        return int(10 * 1e9 + 5 * get_approximate_df_mem_usage(X).sum())

    @classmethod
    def prefetch_weights(cls) -> str:
        """Resolve the pinned ``LimiX-2.ckpt`` to a local path: the Hugging Face cache first (no
        network round trip on an offline node), a download otherwise.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = {"repo_id": _HF_REPO, "filename": _HF_FILENAME, "revision": _HF_REVISION}
        try:
            return hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            return hf_hub_download(**kwargs)
