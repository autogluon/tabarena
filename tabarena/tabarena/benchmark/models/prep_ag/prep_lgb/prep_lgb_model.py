from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series

from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

from scipy.special import softmax
from .linear_init import GroupedLinearInitScore, LinearInitScore, OOFLinearInitScore

from autogluon.tabular.models.lgb.lgb_model import LGBModel


class PrepLGBModel(ModelAgnosticPrepMixin, LGBModel):
    ag_key = "prep_GBM"
    ag_name = "prep_LightGBM"

    @classmethod
    def _estimate_memory_usage_static(cls, **kwargs) -> int:
        memory_usage = super()._estimate_memory_usage_static(**kwargs)
        # FIXME: 1.5 runs OOM on kddcup09_appetency fold 2 repeat 0 prep_LightGBM_r49_BAG_L1
        return memory_usage * 2.0  # FIXME: For some reason this underestimates mem usage without this

    def _predict_proba(self, X, num_cpus=0, **kwargs) -> np.ndarray:
        if not self.use_residuals:
            return super()._predict_proba(X=X, num_cpus=num_cpus, **kwargs)
        y_pred_linear = self.lin_init.init_score(X, is_train=False)
        X = self.preprocess(X, **kwargs)
        y_pred_lgb = self.model.predict(X, num_threads=num_cpus, raw_score=True)
        y_pred_proba = y_pred_lgb + y_pred_linear
        if self.problem_type == 'binary':
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        elif self.problem_type in ['multiclass', 'softclass']:
            y_pred_proba = softmax(y_pred_proba, axis=1)
        return self._post_process_predictions(y_pred_proba=y_pred_proba)

    def generate_datasets(
        self,
        X: DataFrame,
        y: Series,
        X_val=None,
        X_test=None,
        init_train=None,
        init_val=None,
        init_test=None,
        **kwargs,
    ):
        ag_params = self._get_ag_params()

        max_dataset_size_for_residuals = ag_params.get("max_dataset_size_for_residuals", 1000)
        use_residuals = ag_params.get("use_residuals", False)
        if max_dataset_size_for_residuals is not None and (X.shape[0] > max_dataset_size_for_residuals):
            use_residuals = False
        self.use_residuals = use_residuals
        residual_init_kwargs = ag_params.get("residual_init_kwargs", {"scaler": "squashing"})

        if use_residuals:
            residual_type = ag_params.get("residual_type", "oof")
            if residual_type == 'grouped':
                self.lin_init = GroupedLinearInitScore(target_type=self.problem_type, init_kwargs=residual_init_kwargs, random_state=self.random_seed)
            elif residual_type == 'oof':
                self.lin_init = OOFLinearInitScore(target_type=self.problem_type, init_kwargs=residual_init_kwargs, random_state=self.random_seed)
            elif residual_type == 'knn':
                raise ValueError(f"Invalid residual_type: {residual_type!r}")
            else:
                self.lin_init = LinearInitScore(target_type=self.problem_type, init_kwargs=residual_init_kwargs, random_state=self.random_seed)
            self.lin_init.fit(X, y)
            init_train = self.lin_init.init_score(X, is_train=True)
            init_val = self.lin_init.init_score(X_val, is_train=False) if X_val is not None else None
            init_test = self.lin_init.init_score(X_test, is_train=False) if X_test is not None else None

        return super().generate_datasets(
            X=X,
            y=y,
            X_val=X_val,
            X_test=X_test,
            init_train=init_train,
            init_val=init_val,
            init_test=init_test,
            **kwargs,
        )

    def _ag_params(self) -> set:
        ag_params = super()._ag_params()
        new_ag_params = {
            "use_residuals",
            "residual_type",
            "residual_init_kwargs",
            "max_dataset_size_for_residuals",
        }
        ag_params = ag_params.union(new_ag_params)
        return ag_params
