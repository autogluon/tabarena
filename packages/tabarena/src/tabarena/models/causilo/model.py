from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


class CausiloModelBase(AbstractTorchModel):
    """Pretrained tabular classification and regression by Nums AI Inc.

    Paper: Causilo Technical Report (https://arxiv.org/abs/2609.22866)
    Authors: Minyong Cho, Minho Jeong, Dooho Lee, Jinmo Lee, Jaemin Yoo
    Code and documentation: https://github.com/nums-ai/causilo
    Code license: Apache-2.0. The weights have separate non-commercial licenses: the Causilo License v1.0
    for 1.0 (https://huggingface.co/nums-ai/causilo/blob/main/LICENSE) and the Nums AI Model License for
    1.5 (https://huggingface.co/nums-ai/causilo-1.5/blob/main/LICENSE).

    One causilo release serves both checkpoints; ``model_version`` picks the one a subclass fits. A
    label set wider than the checkpoint's ten-class head is fit through the library's own
    error-correcting output codes, so the wrapper declares no ``max_classes`` cap.

    ``_fit`` does not read ``time_limit``: fitting prepares the context without training, and the
    library has no time budget for prediction, whose cost grows with context rows times query rows
    (the out-of-fold predictions inside a bagged fit included).
    """

    ag_key = "NOTSET"
    ag_name = "NOTSET"
    #: The checkpoint release the library loads (its ``model_version`` argument).
    model_version: ClassVar[str] = "NOTSET"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, **kwargs):
        import torch
        from causilo import CausiloClassifier, CausiloRegressor

        # Context fitting has no iterative training, early stopping or validation split.
        params = self._get_model_params().copy()
        requested_device = params.pop("device", "auto")
        if requested_device == "cpu":
            device = "cpu"
        else:
            device = self._resolve_fit_device(
                num_gpus=num_gpus, gpu_device="cuda" if requested_device == "auto" else requested_device
            )
        cls = CausiloRegressor if self.problem_type == "regression" else CausiloClassifier
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            self.device = device  # keys the shared network by fit device
            # "auto" places the network on the first visible GPU like "cuda", and the estimator's own
            # pickle records it, so a saved GPU fit also restores on a host without one.
            library_device = "auto" if device == "cuda" else device
            self.model = cls(device=library_device, model_version=self.model_version, **params)
            self.model.fit(self.preprocess(X, is_train=True), y)
        finally:
            torch.set_num_threads(previous_threads)

    def _predict_proba(self, X, **kwargs):
        import numpy as np
        import torch

        X = self.preprocess(X, **kwargs)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._inference_threads)
            if self.problem_type == "regression":
                return self.model.predict(X)
            proba = self.model.predict_proba(X)
        finally:
            torch.set_num_threads(previous_threads)
        # The library returns one column per class of the fitted labels; a fold's training rows can miss
        # a class, so the columns are placed at their positions among AutoGluon's ``num_classes``.
        ordered = np.zeros((len(X), self.num_classes), dtype=proba.dtype)
        ordered[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return self._convert_proba_to_unified_form(ordered)

    def get_device(self):
        return str(self.model._engine.device)

    def _set_device(self, device):
        """Restore the fitted context on ``device`` through the estimator's own save and load.

        ``export_estimator`` keeps the prepared context without the network, and ``import_estimator``
        places it on the requested device and loads the network there, as ``load_context(device=...)``
        does from a file.
        """
        import torch
        from autogluon.core.models.abstract.shared_weights import scope
        from causilo.serialization import export_estimator, import_estimator

        current = self.model._engine.device
        target = torch.device(device)
        if target.type == current.type and target.index in (None, current.index):
            return
        estimator = type(self.model).__new__(type(self.model))
        # The model owns its network (only such a model gets here), so the load neither takes from nor
        # adds to the shared registry. "auto" keeps the pickle loadable on a host without a GPU, as in _fit.
        with scope(self, enabled=False):
            import_estimator(
                estimator, export_estimator(self.model), device="auto" if str(device) == "cuda" else str(device)
            )
        self.model = estimator

    @classmethod
    def prefetch_weights(cls) -> str:
        """Download the pinned checkpoints of ``model_version`` into the Hugging Face cache; return the snapshot folder.

        The returned path lets the node staging and the SkyPilot seeding copy the repository directory,
        including the hub's ``trees`` listing that ``snapshot_download`` of a pinned commit needs offline.
        """
        from causilo.model_selection import MODEL_SOURCES
        from huggingface_hub import snapshot_download

        sources = [MODEL_SOURCES[cls.model_version, task] for task in ("classification", "regression")]
        (repository,) = {source.repository for source in sources}
        (revision,) = {source.revision for source in sources}
        return snapshot_download(
            repository,
            revision=revision,
            allow_patterns=[
                f"{source.folder}/{name}" for source in sources for name in ("config.json", "model.safetensors")
            ],
        )

    @classmethod
    def _class_tags(cls):
        # The estimator pickles through causilo's own ``export_estimator``, which leaves the network out.
        return {"pickles_pretrained_weights": False}

    def _more_tags(self):
        return {"can_refit_full": True}


class CausiloModel(CausiloModelBase):
    """Causilo 1.0 as used on TabArena."""

    ag_key = "TA-CAUSILO"
    ag_name = "Causilo"
    model_version: ClassVar[str] = "1.0"
    _default_ag_args_ensemble_extra = {
        **CausiloModelBase._default_ag_args_ensemble_extra,
        "vary_seed_across_folds": False,
    }
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "causilo",
        "causilo.serialization",
        "causilo.v1_0.engine",
        "causilo.v1_0.runtime",
    )
    #: The 1.0 runtime builds the network in ``causilo.v1_0.checkpoints.load_hosted_model(task)`` and the
    #: engine moves it to its device; one build per task and device per process. The loader has no
    #: device input, so ``_fit`` records the device on ``self.device`` before the fit.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="causilo.v1_0.checkpoints:load_hosted_model", key=("task",)
    )
    #: Causilo 1.0 and 1.5 are registered separately; each owns its ``share_weights`` class setting.
    class_settings_per_subclass = True

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 8)


class Causilo15Model(CausiloModelBase):
    """Causilo 1.5 with the library's task-specific ensemble defaults."""

    ag_key = "TA-CAUSILO-1.5"
    ag_name = "TA-Causilo-1.5"
    model_version: ClassVar[str] = "1.5"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "causilo",
        "causilo.serialization",
        "causilo.v1_5.engine",
        "causilo.v1_5.runtime",
        # Triton kernels of the CUDA path, imported at their first call.
        "causilo.v1_5.kernels.triton.cls_attention",
        "causilo.v1_5.kernels.triton.cls_readout",
        "causilo.v1_5.kernels.triton.head_normalization",
        "causilo.v1_5.kernels.triton.query_gain",
        "causilo.v1_5.kernels.triton.regression_mean",
        "causilo.v1_5.kernels.triton.residual_normalization",
        "causilo.v1_5.kernels.triton.swiglu",
    )
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="causilo:load_pretrained_model",
        key=("task", "model_version"),
        disabled_by=("checkpoint_path", "device_memory_budget_bytes"),
    )
    class_settings_per_subclass = True

    def _set_default_params(self):
        super()._set_default_params()
        # The authors' setting for large tables: each member samples its own 200k-row context. Tables up to
        # 200k training rows (every TabArena-v0.1 split) take the same full-context path as max_context=None.
        self._set_default_param_value("max_context", 200_000)

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Give integer category labels string labels.

        The 1.5 runtime reads numeric category labels as ordered levels and interpolates a category
        missing from the context between its neighbours. AutoGluon codes every categorical column as
        integers (TabArena) or keeps a dataset's integer IDs (BeyondArena), so a fold-unseen category of a
        nominal column would get an interpolated code instead of the library's encoding for an unseen text
        category. The zero-padded labels keep the codes' order, so seen categories encode as before;
        AutoGluon drops categories seen fewer than twice, so every category a test row carries is in the
        refit model's context and only out-of-fold rows change.

        The fit decides which columns get labels and stores them: the same column can reach predict with
        another category dtype (BeyondArena's preprocessing returns an ``object`` index of Python ints
        there), and a column labelled at fit but not at predict matches no category of the context.
        """
        import pandas as pd

        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._labelled_categories = [
                column
                for column in X.columns
                if isinstance(X[column].dtype, pd.CategoricalDtype)
                and len(X[column].cat.categories)
                and all(_integer_label(value) is not None for value in X[column].cat.categories)
            ]
        columns = [
            column
            for column in getattr(self, "_labelled_categories", ())
            if column in X.columns and isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        if not columns:
            return X
        X = X.copy(deep=False)
        for column in columns:
            X[column] = X[column].cat.rename_categories(_category_label)
        return X


def _integer_label(value) -> int | None:
    """``value`` as a Python int when it is an integer (also an integral float or an ``object`` int), else None."""
    import numbers

    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return int(value)
    return None


def _category_label(value) -> str:
    """The string label of a category: ``c`` plus the integer zero-padded to 19 digits, else ``s`` plus its text.

    A fixed width keeps a code's label the same whatever other categories the column's dtype holds
    (BeyondArena's preprocessing keeps unseen categories in the dtype at predict time).
    """
    code = _integer_label(value)
    return f"c{code:019d}" if code is not None else f"s{value}"
