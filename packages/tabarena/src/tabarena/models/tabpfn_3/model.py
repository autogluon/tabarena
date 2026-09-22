from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.benchmark.preprocessing.group_feature_generators import S_GROUP_ID

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


def _mutates_network(params: Mapping[str, Any]) -> bool:
    """Whether tabpfn writes into or casts the network under these estimator parameters.

    ``fit_mode="fit_with_cache"`` writes the train-set representation into the module, and a
    ``torch.dtype`` ``inference_precision`` makes the per-device model cache cast the module in
    place; such a fit builds its own network.
    """
    if params.get("fit_mode", "fit_preprocessors") != "fit_preprocessors":
        return True
    return not isinstance(params.get("inference_precision", "auto"), str)


class TabPFN3Model(AbstractTorchModel):
    """TabPFN-3 TabArena Integration."""

    ag_key = "TA-TABPFN-3"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabpfn", "tabpfn.model_loading")
    ag_name = "TA-TabPFN-3"
    ag_priority = 105
    seed_name = "random_state"

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    _supported_problem_types = ["binary", "multiclass", "regression"]

    checkpoint_param_name: str = "checkpoint_per_problem_type"
    """Name of the optional config hyperparameter that overrides the checkpoint per problem type.

    Its value is a dict mapping a problem type to a checkpoint. Keys may be ``"binary"`` /
    ``"multiclass"`` / ``"regression"``, or the ``"classification"`` umbrella (used for both
    ``"binary"`` and ``"multiclass"`` unless a more specific key is given). Each value is a bare
    filename (resolved in the tabpfn cache dir) or an absolute path to a ``.ckpt``. Problem types
    not listed fall back to ``default_classification_model`` / ``default_regression_model``. It is
    popped from the hyperparameters in :meth:`_fit` (it is not a tabpfn estimator argument).
    """

    _categorical_indices: list[int] | None
    """The indices of the categorical features, detected during preprocessing."""
    fixed_random_state: int = 0
    """Using a fixed random seed, as in TabPFN-2.6."""
    _default_auxiliary_params_extra = {
        "max_classes": 160,
        # Batch inference once we exceed 150_000 samples (batching starts at 150_001).
        "max_batch_size": 150_000,
    }
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: tabpfn builds its network inside ``_initialize_model_variables``, which ``fit`` calls; one
    #: build per checkpoint and device per process. A fit whose configuration writes into the
    #: network (a differentiable input, another fit mode, a forced dtype) builds its own.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=(
            "tabpfn.classifier:TabPFNClassifier._initialize_model_variables",
            "tabpfn.regressor:TabPFNRegressor._initialize_model_variables",
        ),
        key=("model_path",),
        disabled_by=("differentiable_input", _mutates_network),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def _preprocess(self, X: pd.DataFrame, *, is_train=False, **kwargs) -> pd.DataFrame:
        """Minimal model-specific preprocessing to detect the indices of categorical features.

        A group key the model-agnostic pipeline exposed (float codes tagged ``S_GROUP_ID`` in the
        feature metadata) is declared categorical here, after the model-specific category filter
        has run, so tabpfn treats the codes as levels rather than as a number.
        """
        group_id_features = self._group_id_features(X)
        if group_id_features:
            X = X.copy()
            X[group_id_features] = X[group_id_features].astype("category")
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            if categorical_cols:
                self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            else:
                self._categorical_indices = None

        return X

    def _group_id_features(self, X: pd.DataFrame) -> list[str]:
        """The columns of ``X`` the feature metadata tags as an exposed group key."""
        metadata = self._feature_metadata
        if metadata is None:
            return []
        return [f for f in metadata.get_features(required_special_types=[S_GROUP_ID]) if f in X.columns]

    def _get_model_class(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]

        return TabPFNClassifier if is_classification else TabPFNRegressor

    def _resolve_checkpoint_for_problem_type(self, checkpoint_per_problem_type: dict[str, str] | None) -> str | None:
        """Select this task's checkpoint name from an optional per-problem-type override.

        Resolution order: the exact ``problem_type`` key (``"binary"`` / ``"multiclass"`` /
        ``"regression"``) -> the ``"classification"`` umbrella key (for binary/multiclass only) ->
        the ``default_classification_model`` / ``default_regression_model`` class attribute.
        """
        is_classification = self.problem_type in ["binary", "multiclass"]
        overrides = checkpoint_per_problem_type or {}
        model = overrides.get(self.problem_type)
        if model is None and is_classification:
            model = overrides.get("classification")
        if model is None:
            model = self.default_classification_model if is_classification else self.default_regression_model
        return model

    def _get_model_checkpoint(self, checkpoint_per_problem_type: dict[str, str] | None = None):
        """Resolve the checkpoint to a full path: pick the name (see
        :meth:`_resolve_checkpoint_for_problem_type`) then prepend the tabpfn cache dir (a no-op
        for an absolute path).
        """
        from tabpfn.model_loading import prepend_cache_path

        return prepend_cache_path(self._resolve_checkpoint_for_problem_type(checkpoint_per_problem_type))

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        X = self.preprocess(X, y=y, is_train=True)

        # Set hyperparameters
        hps = dict(self._get_model_params())
        checkpoint_per_problem_type = hps.pop(self.checkpoint_param_name, None)
        default_hps = dict(
            model_path=self._get_model_checkpoint(checkpoint_per_problem_type),
            device=self._resolve_tabpfn_device(num_gpus=num_gpus),
            n_jobs=num_cpus,
            categorical_features_indices=self._categorical_indices,
        )
        default_hps[self.seed_name] = self.fixed_random_state
        hps = {**default_hps, **hps}  # hps later to override any conflicting keys default keys.

        # Initialize and fit the model
        model_class = self._get_model_class()
        self.model = model_class(**hps)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    # --- Model Behavior Management ---
    def _set_default_params(self):
        default_params = {
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO:
    #  - add support for many-class wrapper to remove the limit fully
    #  - add row/col limit?
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Ensure one fold is fit at a time and refits is enabled by default."""
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": default_ag_args_ensemble.pop("refit_folds", True),
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # --- Resource and GPU Management ---
    @staticmethod
    def _resolve_tabpfn_device(num_gpus: int) -> str | list[str]:
        """Return device type based on number of GPUs, ensuring that if
        GPUs are requested, they are available.
        """
        if num_gpus <= 0:
            return "cpu"

        import torch

        if not torch.cuda.is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if num_gpus == 1:
            return "cuda"

        return [f"cuda:{i}" for i in range(num_gpus)]

    def get_device(self) -> str:
        base = self.model
        if hasattr(base, "devices_"):
            return base.devices_[0].type

        from collections.abc import Sequence

        device = base.device
        if isinstance(device, Sequence):
            return device[0]

        return device

    def _set_device(self, device: str):
        self.model.to(device)

    # TODO: obtain memory estimation with/without chunking
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        """Assume a 10 GB baseline (model + activations) plus the dataset memory footprint."""
        baseline_mem_est = 10 * 1e9  # 10 GB minimum for TabPFN-3 model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)


def prefetch_weights() -> list[Path]:
    """Download the TabPFN-3 checkpoints missing from the tabpfn cache; return the paths of all of them.

    Only the v3 classifier and regressor checkpoints the wrapper can load: the ``default_*_model``
    files and the named variants a ``checkpoint_per_problem_type`` config may pick. Each file is
    fetched on its own, so a missing license token or a failed download raises instead of being
    logged and skipped, and the returned paths let the node staging and the SkyPilot seeding copy the
    files without enumerating the cache.
    """
    from tabpfn.model_loading import ModelSource, ModelVersion, download_model, resolve_model_path

    _, model_dir, _, _ = resolve_model_path(model_path=None, which="classifier")
    cache_dir = Path(model_dir[0])
    paths: list[Path] = []
    for which, source in (
        ("classifier", ModelSource.get_classifier_v3()),
        ("regressor", ModelSource.get_regressor_v3()),
    ):
        for name in source.filenames:
            path = cache_dir / name
            if not path.exists():
                result = download_model(to=path, version=ModelVersion.V3, which=which, model_name=name)
                if result != "ok":
                    raise RuntimeError(f"Could not download the TabPFN-3 checkpoint {name}: {result}")
            paths.append(path)
    return paths
