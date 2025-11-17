from __future__ import annotations

from typing import Type

import pandas as pd

from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.models import AbstractModel


class AGModelWrapper(AbstractExecModel):
    def __init__(self, model_cls: Type[AbstractModel], hyperparameters: dict = None, **kwargs):
        super().__init__(**kwargs)
        assert issubclass(model_cls, AbstractModel)
        self.model_cls = model_cls
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.model = self.model_cls(
            path="",
            name=self.model_cls.__name__,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=self.hyperparameters,
        )
        self.model.fit(
            X=X,
            y=y,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == "binary":
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)
        y_pred_proba = pd.DataFrame(y_pred_proba, index=X.index)
        return y_pred_proba
