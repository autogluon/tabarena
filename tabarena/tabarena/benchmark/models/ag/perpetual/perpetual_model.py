from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# TODO:
#   - handle sequential_local switch to parllel and support memory estimation
#   - add support for setting a random seed (in rust?)
#   - time limit is not strict
#   - memory limit is not strict and not well supported across threads/ray
class PerpetualBoostingModel(AbstractModel):
    ag_key = "PB"
    ag_name = "PerpetualBoosting"

    # FIXME: random seed not supported
    # seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features: list[str] = None

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if self._category_features is None:
            self._category_features = X.select_dtypes(
                include=["category"]
            ).columns.tolist()

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: float | None = None,
        num_cpus: int | str = "auto",
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        **kwargs,
    ):
        # Preprocess data.
        X = self.preprocess(X, is_train=True)
        paras = self._get_model_params()

        from perpetual import PerpetualBooster

        memory_limit = ResourceManager().get_memory_size(format="GB")

        # Safety factors to account for the non-strict time-limit of Perpetual
        memory_limit = int(memory_limit * 0.95)
        time_limit = time_limit * 0.95

        if self.problem_type in [BINARY, MULTICLASS]:
            objective = "LogLoss"
        elif self.problem_type == REGRESSION:
            metric_map = {
                "mean_squared_error": "HuberLoss",
                "root_mean_squared_error": "SquaredLoss",
            }
            objective = metric_map.get(self.eval_metric.name, "SquaredLoss")
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

        self.model = PerpetualBooster(
            objective=objective,
            num_threads=num_cpus,
            memory_limit=memory_limit,
            categorical_features=self._category_features,
            timeout=time_limit,
            **paras,
        )

        self.model.fit(X=X, y=y, sample_weight=sample_weight)

    def _set_default_params(self):
        # Default from repo and told to use by authors
        default_params = {
            "iteration_limit": 1_000,
            "budget": 0.5,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            # FIXME: Default to sequential_local due to problems with memory limit and
            #   how it is handled by PerpetualBooster. Change back if needed.
            "fold_fitting_strategy": "sequential_local",
            # Following https://github.com/perpetual-ml/perpetual/issues/66#issuecomment-3073175292
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble
