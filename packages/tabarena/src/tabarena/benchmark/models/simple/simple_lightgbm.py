from __future__ import annotations

import pandas as pd

from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel


class SimpleLightGBM(AbstractExecModel):
    def __init__(self, hyperparameters: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters

    def get_model_cls(self):
        from lightgbm import LGBMClassifier, LGBMRegressor
        is_classification = self.problem_type in ['binary', 'multiclass']
        if is_classification:
            model_cls = LGBMClassifier
        elif self.problem_type == 'regression':
            model_cls = LGBMRegressor
        else:
            raise AssertionError(f"LightGBM does not recognize the problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        model_cls = self.get_model_cls()
        self.model = model_cls(**self.hyperparameters)
        self.model.fit(
            X=X,
            y=y
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        return pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)
