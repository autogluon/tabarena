from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

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


class TabPFN35Model(AbstractTorchModel):
    """TabPFN-3.5 TabArena integration.

    TabPFN-3.5 is the September 2026 release of Prior Labs' tabular foundation model, an in-context
    learner that predicts in a forward pass. From this release on, one multitask checkpoint carries
    both the classification and the regression head, so the same file backs ``TabPFNClassifier``
    and ``TabPFNRegressor``. The checkpoint accepts up to 1,000,000 rows, 20,000 features and 160
    classes natively.

    Codebase: https://github.com/PriorLabs/TabPFN (Apache 2.0)
    Weights: https://huggingface.co/Prior-Labs/tabpfn_3_5 (TabPFN-3.5 License v1.0, non-commercial)
    Paper: TabPFN-3.5: Technical Report (Jäger, Erickson, Grinsztajn, Birkel, Flöge et al., 2026),
    https://arxiv.org/abs/2609.17895
    Model page: https://docs.priorlabs.ai/models
    """

    ag_key = "TA-TABPFN-3.5"
    ag_name = "TA-TabPFN-3.5"
    ag_priority = 105
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabpfn", "tabpfn.model_loading")
    gpu_strongly_recommended = True  # in-context inference is an order of magnitude slower on CPU

    model_version: ClassVar[str] = "v3.5"
    """The tabpfn ``ModelVersion`` value this class runs; it selects the checkpoint's download source."""
    default_checkpoint: str = "tabpfn-v3.5-20260909.safetensors"
    """The multitask checkpoint used for every problem type unless ``checkpoint_per_problem_type``
    names another one. A bare filename, resolved in the tabpfn cache dir."""

    checkpoint_param_name: str = "checkpoint_per_problem_type"
    """Name of the optional config hyperparameter that overrides the checkpoint per problem type.

    Its value is a dict mapping a problem type to a checkpoint. Keys may be ``"binary"`` /
    ``"multiclass"`` / ``"regression"``, or the ``"classification"`` umbrella (used for both
    ``"binary"`` and ``"multiclass"`` unless a more specific key is given). Each value is a bare
    filename (resolved in the tabpfn cache dir) or an absolute path to a checkpoint, for example
    ``{"multiclass": "tabpfn-v3.5-20260909_multiclass.safetensors"}`` for the experimental
    multiclass variant. Problem types not listed fall back to ``default_checkpoint``. It is popped
    from the hyperparameters in :meth:`_fit` (it is not a tabpfn estimator argument).
    """

    fixed_random_state: int = 0
    """Using a fixed random seed, as in TabPFN-2.6 and TabPFN-3: tabpfn's random state decides its
    preprocessing, so a refit with another seed would make the validation score misleading."""

    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {
        "max_classes": 160,
        # Batch inference once we exceed 150_000 samples (batching starts at 150_001).
        "max_batch_size": 150_000,
    }
    #: One fold at a time (parallel folds race on the checkpoint download), and a refit on the full
    #: data instead of the bag: same quality for an in-context model, one network at inference.
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    default_resources_physical_cores_only = True
    default_num_gpus = 1
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
    #: TabPFN-3.5 and TabPFN-3.5-Fast are registered separately; each owns its ``share_weights``
    #: class setting.
    class_settings_per_subclass = True
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    _categorical_indices: list[int] | None = None
    """The indices of the categorical features, detected during training preprocessing."""

    def _preprocess(self, X: pd.DataFrame, *, is_train=False, **kwargs) -> pd.DataFrame:
        """Minimal model-specific preprocessing to detect the indices of categorical features."""
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols] or None

        return X

    def _get_model_class(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        return TabPFNClassifier if self.problem_type in ["binary", "multiclass"] else TabPFNRegressor

    def _resolve_model_path(self, checkpoint_per_problem_type: dict[str, str] | None) -> str:
        """The checkpoint path for this task: the ``problem_type`` key of the override, then its
        ``"classification"`` umbrella key (binary/multiclass only), then ``default_checkpoint``,
        with the tabpfn cache dir prepended (a no-op for an absolute path).
        """
        from tabpfn.model_loading import prepend_cache_path

        overrides = checkpoint_per_problem_type or {}
        checkpoint = overrides.get(self.problem_type)
        if checkpoint is None and self.problem_type in ["binary", "multiclass"]:
            checkpoint = overrides.get("classification")
        return prepend_cache_path(checkpoint or self.default_checkpoint)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        X = self.preprocess(X, y=y, is_train=True)

        device: str | list[str] = self._resolve_fit_device(num_gpus=num_gpus)
        if num_gpus > 1:
            device = [f"cuda:{i}" for i in range(num_gpus)]

        hps = dict(self._get_model_params())
        checkpoint_per_problem_type = hps.pop(self.checkpoint_param_name, None)
        # ``n_preprocessing_jobs`` stays at tabpfn's default of 1: a worker pool costs a fixed 15 to 20 s
        # per fit to start, which dwarfs the preprocessing it parallelizes and was charged to the
        # timed fit of every split in suite tabarena-2026-09-17.
        default_hps = {
            "model_path": self._resolve_model_path(checkpoint_per_problem_type),
            "device": device,
            "categorical_features_indices": self._categorical_indices,
            self.seed_name: self.fixed_random_state,
        }
        hps = {**default_hps, **hps}  # the config wins over the defaults

        self.model = self._get_model_class()(**hps).fit(X=X, y=y)

    def _set_default_params(self):
        self._set_default_param_value("ignore_pretraining_limits", True)  # no warnings or size limits

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}  # no validation data is consumed by the fit

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
        baseline_mem_est = 10 * 1e9  # 10 GB minimum for TabPFN-3.5 model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)

    @classmethod
    def prefetch_weights(cls) -> Path:
        """Download this version's default checkpoint into the tabpfn cache and return its path.

        tabpfn resolves the checkpoint by name through its own download function, which reads the
        Hugging Face repo and file that ``model_version`` selects (see ``tabpfn.model_loading``).
        """
        from pathlib import Path

        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import download_model, prepend_cache_path

        target = Path(prepend_cache_path(cls.default_checkpoint))
        # The multitask checkpoint serves both estimator types, so one download covers both.
        result = download_model(
            to=target,
            version=ModelVersion(cls.model_version),
            which="classifier",
            model_name=cls.default_checkpoint,
        )
        if result != "ok":
            raise RuntimeError(f"Downloading {cls.default_checkpoint} failed: {result}") from result[0]
        return target


class TabPFN35FastModel(TabPFN35Model):
    """TabPFN-3.5-Fast TabArena integration.

    The smaller and faster sibling of TabPFN-3.5, released alongside it from the same Hugging Face
    repo (Prior Labs reports up to 6x faster inference; the model is marked alpha). Same limits,
    license and estimator surface as TabPFN-3.5; only the checkpoint differs.
    """

    ag_key = "TA-TABPFN-3.5-FAST"
    ag_name = "TA-TabPFN-3.5-Fast"

    model_version: ClassVar[str] = "v3.5-fast"
    default_checkpoint: str = "tabpfn-v3.5-fast-20260909.safetensors"
