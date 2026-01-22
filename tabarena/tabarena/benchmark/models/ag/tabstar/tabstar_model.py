from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


# TODO:
#   - make sure AutoGluon gives class labels not ordinal encoded! (local branch of AG + PR https://github.com/autogluon/autogluon/pull/5482)
#   - support for metric_name was rolled back, so maybe in the future add support for AG metrics again.
class TabSTARModel(AbstractModel):
    """TabSTAR Model: https://arxiv.org/abs/2505.18125."""

    ag_key = "TABSTAR"
    ag_name = "TabSTAR"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
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
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
        from tabstar.training.hyperparams import LORA_BATCH, VAL_BATCH

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = TabSTARClassifier
        elif self.problem_type in ["regression"]:
            model_cls = TabSTARRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Simple heuristic for batch size
        train_batch_size = LORA_BATCH
        predict_batch_size = VAL_BATCH
        if X.shape[1] > 200:
            train_batch_size = 16
            predict_batch_size = 16

        hps = self._get_model_params()
        self.model = model_cls(
            **hps,
            lora_batch=train_batch_size,
            val_batch_size=predict_batch_size,
            time_limit=time_limit,
            device=device,
            output_dir=self.path + "/model_checkpoints",
        )

        if X_val is None:
            # FIXME: make this a general utility function in autogluon that also handles
            #  ratio better! Or handle it before _fit based on `can_refit_full`
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.33,
                random_state=0,
            )

        # Does nothing but might be used for future extensions
        X = self.preprocess(X, y=y)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # Inverse label transformation to retain original semantics for classification
        #   - hasattr for backward compatibility
        if self.problem_type in ["binary", "multiclass"]:
            if (not hasattr(self, "label_cleaner")) or (self.label_cleaner is None):
                raise ValueError("Label cleaner missing from AbstractModel!")

            y = self.label_cleaner.inverse_transform(y)
            if y_val is not None:
                y_val = self.label_cleaner.inverse_transform(y_val)

        # FIXME: .fit does not return self as expected from sklearn API
        self.model.fit(
            X=X,
            y=y,
            x_val=X_val,
            y_val=y_val,
        )

    def _set_default_params(self):
        # Default values from the current version of the code base
        default_params = {
            # Large max epochs, we want to stop based on time limit or early stopping
            "max_epochs": 10_000,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(
        self, is_gpu_available: bool = False
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        # TODO: switch to parallel fitting on one GPU once VRAM memory estimation is supported
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: support memory estimate!
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
