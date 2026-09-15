from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.tabpfnv2_5._shared import TABPFN_SPEC, TabPFNSharedWeightsMixin

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.models._shared_weights_model import SharedWeightsSpec


def _device_from_hyperparameters(hyperparameters: dict, allocated: str) -> str | None:
    """A ``device`` hyperparameter string names the shared network's device; a list or ``"auto"`` disables sharing.

    tabpfn deep-copies the module per device of a device list, and ``"auto"`` is resolved by the
    library at fit time, so neither can be keyed before the fit.
    """
    requested = hyperparameters.get("device")
    if requested is None:
        return allocated
    if not isinstance(requested, str) or requested == "auto":
        return None
    return requested


class TabPFN3Model(TabPFNSharedWeightsMixin, AbstractTorchModel):
    """TabPFN-3 TabArena Integration.

    The network is shared through :class:`tabarena.models.tabpfnv2_5._shared.TabPFNSharedWeightsMixin`;
    a user ``model_path`` keeps the library's own load.
    """

    ag_key = "TA-TABPFN-3"
    #: Every module the loader and the fit import lazily (the tabpfn import chain costs about 0.5 s warm).
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabpfn",
        "tabpfn.base",
        "tabpfn.classifier",
        "tabpfn.regressor",
        "tabpfn.model_loading",
        "tabpfn.inference_config",
        "tabarena.models.tabpfnv2_5._estimators",
    )
    ag_name = "TA-TabPFN-3"
    ag_priority = 105
    seed_name = "random_state"

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    _supported_problem_types = ["binary", "multiclass", "regression"]

    checkpoint_param: ClassVar[str] = "checkpoint_per_problem_type"
    """Name of the optional config hyperparameter that overrides the checkpoint per problem type.

    Its value is a dict mapping a problem type to a checkpoint. Keys may be ``"binary"`` /
    ``"multiclass"`` / ``"regression"``, or the ``"classification"`` umbrella (used for both
    ``"binary"`` and ``"multiclass"`` unless a more specific key is given). Each value is a bare
    filename (resolved in the tabpfn cache dir) or an absolute path to a ``.ckpt``. Problem types
    not listed fall back to ``default_classification_model`` / ``default_regression_model``. It is
    popped from the hyperparameters in :meth:`_fit` (it is not a tabpfn estimator argument).
    """

    shared_weights_spec: ClassVar[SharedWeightsSpec] = replace(
        TABPFN_SPEC,
        disable_when=(*TABPFN_SPEC.disable_when, "model_path"),
        device_from_params=_device_from_hyperparameters,
    )

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

    def _preprocess(self, X: pd.DataFrame, *, is_train=False, **kwargs) -> pd.DataFrame:
        """Minimal model-specific preprocessing to detect the indices of categorical features."""
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            if categorical_cols:
                self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            else:
                self._categorical_indices = None

        return X

    def _get_model_class(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]

        return TabPFNClassifier if is_classification else TabPFNRegressor

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit TabPFN-3 on the checkpoint that matches the problem type.

        ``num_cpus`` is unused: tabpfn's ``n_jobs`` is deprecated and ignored.
        """
        X = self.preprocess(X, y=y, is_train=True)

        # Set hyperparameters
        user_hps = dict(self._get_model_params())
        user_hps.pop(self.checkpoint_param, None)
        device = self._resolve_tabpfn_device(num_gpus=num_gpus)
        default_hps = dict(
            model_path=self._checkpoint_path(problem_type=self.problem_type, hyperparameters=self._get_model_params()),
            device=device,
            categorical_features_indices=self._categorical_indices,
        )
        default_hps[self.seed_name] = self.fixed_random_state
        hps = {**default_hps, **user_hps}  # hps later to override any conflicting keys default keys.

        # A device list (several GPUs) keeps the library's own per-device load.
        key, payload = self._acquire_shared_weights(device=device) if isinstance(device, str) else (None, None)
        hps = self._swap_in_shared_specs(hps, payload)

        # Initialize and fit the model
        model_class = self._get_model_class()
        self.model = model_class(**hps)
        self.model = self.model.fit(
            X=X,
            y=y,
        )
        self._finish_shared_fit(payload)

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
