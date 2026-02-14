from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.tabular import __version__

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class TabICLModelBase(AbstractModel):
    """TabICL is a foundation model for tabular data using in-context learning
    that is scalable to larger datasets than TabPFNv2. It is pretrained purely on synthetic data.
    TabICL currently only supports classification tasks.

    TabICL is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai

    Paper: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
    Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan
    Codebase: https://github.com/soda-inria/tabicl
    License: BSD-3-Clause
    """

    ag_key = "NOTSET"
    ag_name = "NOTSET"
    ag_priority = 65
    seed_name = "random_state"

    default_classification_model: str | None = None
    default_regression_model: str | None = None

    def get_model_cls(self):
        if self.problem_type in ["binary", "multiclass"]:
            from tabicl import TabICLClassifier

            model_cls = TabICLClassifier
        else:
            from tabicl import TabICLRegressor

            model_cls = TabICLRegressor
        return model_cls

    def get_checkpoint_version(self, hyperparameter: dict) -> str:
        clf_checkpoint = self.default_classification_model
        reg_checkpoint = self.default_regression_model

        # Resolve HPO
        if "checkpoint_version" in hyperparameter:
            if isinstance(hyperparameter["checkpoint_version"], str):
                clf_checkpoint = hyperparameter["checkpoint_version"]
                reg_checkpoint = hyperparameter["checkpoint_version"]
            elif isinstance(hyperparameter["checkpoint_version"], tuple):
                clf_checkpoint = hyperparameter["checkpoint_version"][0]
                reg_checkpoint = hyperparameter["checkpoint_version"][1]
            else:
                raise ValueError(
                    "checkpoint_version hyperparameter must be either "
                    "a string or a tuple of two strings (clf, reg)."
                )

        if self.problem_type in ["binary", "multiclass"]:
            return clf_checkpoint

        return reg_checkpoint

    # TODO: is this still correct for TabICLv2?
    @staticmethod
    def _get_batch_size(n_cells: int):
        if n_cells <= 4_000_000:
            return 8
        if n_cells <= 6_000_000:
            return 4
        return 2

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            import tabicl
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabicl! To use the TabICL model, "
                f"do: `pip install autogluon.tabular[tabicl]=={__version__}`.",
            )
            raise err

        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        hyp["batch_size"] = hyp.get(
            "batch_size", self._get_batch_size(X.shape[0] * X.shape[1])
        )
        hyp["checkpoint_version"] = self.get_checkpoint_version(hyperparameter=hyp)

        self.model = model_cls(
            **hyp,
            device=device,
            n_jobs=num_cpus,
        )
        X = self.preprocess(X)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

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

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    # TODO: move memory estimate to specific models below.
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate that is very primitive.
        Can be vastly improved.
        """
        if hyperparameters is None:
            hyperparameters = {}

        dataset_size_mem_est = (
            3 * get_approximate_df_mem_usage(X).sum()
        )  # roughly 3x DataFrame memory size
        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        n_rows = X.shape[0]
        n_features = X.shape[1]
        batch_size = hyperparameters.get(
            "batch_size", cls._get_batch_size(X.shape[0] * X.shape[1])
        )
        embedding_dim = 128
        bytes_per_float = 4
        model_mem_estimate = (
            2 * batch_size * embedding_dim * bytes_per_float * (4 + n_rows) * n_features
        )

        model_mem_estimate *= 1.3  # add 30% buffer

        # TODO: Observed memory spikes above expected values on large datasets, increasing mem estimate to compensate
        model_mem_estimate *= 2.0  # Note: 1.5 is not large enough, still gets OOM

        return model_mem_estimate + dataset_size_mem_est + baseline_overhead_mem_est

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @staticmethod
    def checkpoint_search_space() -> list[str | tuple[str, str]]:
        raise NotImplementedError("This method must be implemented in the subclass.")


class TabICLModel(TabICLModelBase):
    """TabICLv1.1 model as used on TabArena."""

    ag_key = "TA-TABICL"
    ag_name = "TA-TabICL"

    default_classification_model: str | None = "tabicl-classifier-v1.1-20250506.ckpt"

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    @staticmethod
    def checkpoint_search_space() -> list[str]:
        return [
            "tabicl-classifier-v1.1-20250506.ckpt",
            "tabicl-classifier-v1-20250208.ckpt",
        ]


class TabICLv2Model(TabICLModelBase):
    """TabICLv1.1 model as used on TabArena."""

    ag_key = "TA-TABICLv2"
    ag_name = "TA-TabICLv2"

    default_classification_model: str | None = "tabicl-classifier-v2-20260212.ckpt"
    default_regression_model: str | None = "tabicl-regressor-v2-20260212.ckpt"

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    # TODO: search over v1 checkpoints too?
    @staticmethod
    def checkpoint_search_space() -> list[tuple[str, str]]:
        return [
            (
                "tabicl-classifier-v2-20260212.ckpt",
                "tabicl-regressor-v2-20260212.ckpt",
            )
        ]
