from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from scipy.sparse.linalg import ArpackError
from sklearn.decomposition import TruncatedSVD

from tabarena.models.cell_budget import gpu_cell_budget, rows_within_budget, stratified_row_subsample
from tabarena.utils.logging_utils import import_many_class_classifier

logger = logging.getLogger(__name__)

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
#: Peak GPU memory per training cell measured on BeyondArena (about 21M cells on 96 GB) and a safety margin
#: below TabFM's: LimiX degrades into hours-long batch fallbacks before it raises.
LIMIX2_BYTES_PER_CELL = 4700
LIMIX2_CELL_SAFETY = 0.6

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


class _TruncatedSVDWithArpackFallback(TruncatedSVD):
    """``TruncatedSVD`` that retries with the randomized solver when ARPACK fails.

    Module-level so the fitted preprocessing pipelines that hold it pickle with the model.
    """

    _tabarena_arpack_fallback = True

    def fit_transform(self, X, y=None):
        try:
            return super().fit_transform(X, y)
        except ArpackError as exc:
            logger.log(
                20,
                f"\tLimiX-2: ARPACK failed on a {X.shape[0]} x {X.shape[1]} fold ({exc}); "
                "retrying the SVD member with the randomized solver.",
            )
            self.algorithm = "randomized"
            return super().fit_transform(X, y)


def _patch_svd_fallback() -> None:
    """Let LimiX's SVD preprocessing survive an ARPACK failure on a tiny or degenerate training fold.

    ``inference.v2_0.preprocess`` builds ``TruncatedSVD(algorithm="arpack", ...)`` for its SVD-augmented
    members. On a fold with few rows and near-duplicate columns ARPACK can raise ``ArpackError`` ("No
    shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration"), which failed a
    134-row BeyondArena split. The name the module looks up at construction time is replaced by
    :class:`_TruncatedSVDWithArpackFallback`; the member is otherwise unchanged. Idempotent. The
    library-side fix is the same fallback (or ``algorithm="randomized"``) in LimiX itself.
    """
    from inference.v2_0 import preprocess

    if getattr(preprocess.TruncatedSVD, "_tabarena_arpack_fallback", False):
        return
    preprocess.TruncatedSVD = _TruncatedSVDWithArpackFallback


def _build_limix2_predictor(*, device_str, model_path, inference_config, preprocess_num_jobs, hps):
    """Construct one ``LimiXPredictor`` instance (LimiX-2 is stateless per ``predict()`` call, so
    this is the whole "fit"). Factored out of ``LimiX2Model._fit`` so the exact same construction
    logic is reused per cloned sub-estimator when ``ManyClassClassifier`` (ECOC) wraps this model
    for many-class datasets -- each sub-estimator gets its own independently-built predictor,
    matching the identical fix already applied to LimiX-16M (``tabarena.models.limix.model``).
    """
    import torch
    from inference.v2_0.predictor import LimiXPredictor

    _patch_predictor(LimiXPredictor)
    _patch_svd_fallback()
    return LimiXPredictor(
        device=torch.device(device_str),
        model_path=str(model_path),
        inference_config=inference_config,
        preprocess_num_jobs=preprocess_num_jobs,
        **hps,
    )


def _predict_proba_limix2(predictor, X_train, y_train, X_test, *, task_type: str) -> np.ndarray:
    """One in-context forward pass, factored out of ``LimiX2Model._predict_proba`` so both the
    plain and ECOC-wrapped paths share it.
    """
    import torch

    preds = predictor.predict(X_train, y_train, X_test, task_type=task_type)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().float().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


class _LimiX2SklearnWrapper:
    """Sklearn-compatible ``fit``/``predict_proba`` wrapper for one ECOC sub-task.

    LimiX-2 has no sklearn fit API of its own -- ``LimiXPredictor.predict()`` takes the training
    table and query rows together, in-context, same as LimiX-16M -- so ``ManyClassClassifier``
    (which clones a base estimator per one-vs-rest-style sub-task and expects standard
    ``fit(X, y)``/``predict_proba(X)``/``classes_``) needs this wrapper. Mirrors LimiX-16M's own
    ``_LimiXSklearnWrapper`` and AutoGluon core's ``_MitraSklearnWrapper`` for the identical
    problem.
    """

    def __init__(self, *, device_str, model_path, inference_config, preprocess_num_jobs, hps):
        self._device_str = device_str
        self._model_path = model_path
        self._inference_config = inference_config
        self._preprocess_num_jobs = preprocess_num_jobs
        self._hps = hps

    def fit(self, X, y):
        self._model = _build_limix2_predictor(
            device_str=self._device_str,
            model_path=self._model_path,
            inference_config=self._inference_config,
            preprocess_num_jobs=self._preprocess_num_jobs,
            hps=self._hps,
        )
        self._X_train = X
        self._y_train = np.asarray(y)
        self.classes_ = np.unique(self._y_train)
        return self

    def predict_proba(self, X):
        return _predict_proba_limix2(self._model, self._X_train, self._y_train, X, task_type="Classification")

    def get_params(self, deep=True):  # sklearn clone() compatibility
        return {
            "device_str": self._device_str,
            "model_path": self._model_path,
            "inference_config": self._inference_config,
            "preprocess_num_jobs": self._preprocess_num_jobs,
            "hps": self._hps,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, f"_{key}", value)
        return self


class LimiX2Model(AbstractTorchModel):
    """LimiX-2 TabArena integration.

    Above a cell budget derived from the GPU (``rows x columns`` of the training table, about 12M cells on a
    96 GB card; :mod:`tabarena.models.cell_budget`) the fit sub-samples the stored in-context training table,
    stratified by class: LimiX has no row cap of its own and, at about 21M cells, spent hours in its
    smaller-batch fallback before AutoGluon's fold scheduler gave up. The ``cell_budget`` hyperparameter
    overrides the derived budget.

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

    The checkpoint's classification head is fixed-width (``many_class_threshold``, 10 classes); above
    that, ``_fit`` wraps a :class:`_LimiX2SklearnWrapper` in
    ``tabpfn_extensions.many_class.ManyClassClassifier`` (ECOC), matching Mitra/RealTabPFN/LimiX-16M's
    own native pattern, instead of failing outright.
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
    # max_classes moves from a hard 10 to None + many_class_threshold: 10, matching Mitra/RealTabPFN/
    # LimiX-16M's own config exactly -- see the ManyClassClassifier (ECOC) branch in _fit below.
    _default_auxiliary_params_extra = {"max_classes": None, "many_class_threshold": 10}
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
        self._use_many_class = False  # True when ManyClassClassifier (ECOC) is active

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
        try:
            import inference.v2_0.predictor  # noqa: F401  -- fail fast with the friendly hint below
        except ImportError as err:
            raise ImportError(_INSTALL_HINT) from err

        self.device = self._resolve_fit_device(num_gpus)  # keys the shared network; the loader names no device
        hps = self._get_model_params()
        many_class_threshold = self.params_aux.get("many_class_threshold", 10)
        model_path = hps.pop("model_path", None) or self.prefetch_weights()
        inference_config = hps.pop("inference_config", None) or self._default_inference_config()
        n_estimators = hps.pop("n_estimators", None)
        if n_estimators is not None:
            inference_config = {**inference_config, "pipelines": inference_config["pipelines"][:n_estimators]}

        self._X_train = self.preprocess(X)
        self._y_train = y.to_numpy()
        # Cell budget: LimiX has no row cap of its own, so the context every predict sees is sub-sampled
        # (stratified for classification). `cell_budget` overrides the GPU-derived budget (tests).
        cell_budget = hps.pop("cell_budget", None)
        if cell_budget is None:
            cell_budget = gpu_cell_budget(bytes_per_cell=LIMIX2_BYTES_PER_CELL, safety=LIMIX2_CELL_SAFETY)
        if cell_budget is not None:
            n_rows, n_cols = self._X_train.shape
            n_keep = rows_within_budget(n_rows, n_cols, cell_budget)
            if n_keep < n_rows:
                keep = stratified_row_subsample(
                    self._y_train,
                    n_keep,
                    classification=self.problem_type in [BINARY, MULTICLASS],
                    seed=self.random_seed if isinstance(getattr(self, "random_seed", None), int) else 0,
                )
                self._X_train = self._X_train.iloc[keep]
                self._y_train = self._y_train[keep]
                logger.log(
                    20,
                    f"\tLimiX-2: {n_rows} x {n_cols} training cells exceed the budget of {cell_budget} cells on this "
                    f"GPU; the in-context training table is sub-sampled to {n_keep} rows.",
                )
        self._use_many_class = (
            self.problem_type in [BINARY, MULTICLASS]
            and self.num_classes is not None
            and self.num_classes > many_class_threshold
        )

        if self._use_many_class:
            try:
                ManyClassClassifier = import_many_class_classifier()
            except ImportError:
                logger.log(
                    40,
                    "LimiX-2: many-class (ECOC) support requires tabpfn_extensions (install with the `models` extra).",
                )
                raise
            logger.log(
                20,
                f"\tLimiX-2: {self.num_classes} classes exceeds native limit ({many_class_threshold}). "
                "Using ManyClassClassifier (ECOC wrapper).",
            )
            base_model = _LimiX2SklearnWrapper(
                device_str=self.device,
                model_path=model_path,
                inference_config=inference_config,
                preprocess_num_jobs=num_cpus,
                hps=hps,
            )
            self.model = ManyClassClassifier(estimator=base_model, alphabet_size=many_class_threshold).fit(
                self._X_train, self._y_train
            )
        else:
            # The ``limix.LimiXPredictor`` factory reads the checkpoint itself and hands the dict to
            # this constructor; constructed directly, the v2 class reads it through ``load_model``,
            # the call the shared-weights declaration names.
            self.model = _build_limix2_predictor(
                device_str=self.device,
                model_path=model_path,
                inference_config=inference_config,
                preprocess_num_jobs=num_cpus,
                hps=hps,
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """One in-context pass over the stored training table and the query rows (LimiX has no
        sklearn fit API). The regression decoder can return a tensor; both come back as float32.
        """
        X_test = self.preprocess(X, **kwargs)
        if self._use_many_class:
            # ManyClassClassifier (ECOC) exposes its own predict_proba, which internally
            # dispatches to each sub-estimator's predict_proba (i.e.
            # _LimiX2SklearnWrapper.predict_proba, the same in-context forward pass below).
            return self._convert_proba_to_unified_form(self.model.predict_proba(X_test))

        task_type = "Classification" if self.problem_type in [BINARY, MULTICLASS] else "Regression"
        preds = _predict_proba_limix2(self.model, self._X_train, self._y_train, X_test, task_type=task_type)
        return self._convert_proba_to_unified_form(preds)

    def _ag_params(self) -> set[str]:
        return super()._ag_params() | {"many_class_threshold"}

    def get_device(self) -> str:
        # `self.device` (set in `_fit`) is authoritative regardless of which branch ran -- unlike
        # `self.model`, which is a `ManyClassClassifier` (no `.device` attribute of its own) rather
        # than a raw `LimiXPredictor` when `self._use_many_class` (confirmed the hard way on
        # LimiX-16M's identical fix: AttributeError: 'ManyClassClassifier' object has no attribute
        # 'device').
        return self.device

    def _set_device(self, device: str):
        torch_device = self.to_torch_device(device)

        def _move(predictor) -> None:
            predictor.device = torch_device
            predictor.model.to(torch_device)

        if self._use_many_class:
            # Output coding keeps no fitted rows: `ManyClassClassifier` refits every code row at predict
            # time from its base wrapper (`estimators_` is None), so the device to record is the base
            # wrapper's. The single-fit shortcut, taken when no codebook was needed, holds one fitted row.
            self.model.estimator._device_str = device
            for estimator in self.model.estimators_ or ():
                _move(estimator._model)
        else:
            _move(self.model)

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
