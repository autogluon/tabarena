from __future__ import annotations

import numpy as np
from autogluon.tabular.models.lr.lr_model import LinearModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

class AutoFeatLinearModel(ModelAgnosticPrepMixin, LinearModel):
    ag_key = "AutoFeatLR"
    ag_name = "AutoFeatLinearModel"

    def __init__(self,
                #  feateng_steps: int = 2,
                #  featsel_runs: int = 5,
                #  max_gb: int | None = None,
                #  transformations: list | tuple = ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # self.feateng_steps = feateng_steps
        # self.featsel_runs = featsel_runs
        # self.max_gb = max_gb
        # self.transformations = transformations

    def _fit(self, **kwargs):
        self.autofeat_feateng_steps = self.params.pop("feateng_steps", 2)
        self.autofeat_featsel_runs = self.params.pop("featsel_runs", 3)
        self.autofeat_max_gb = self.params.pop("max_gb", None)
        self.autofeat_transformations = self.params.pop("transformations", ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"))
        super()._fit(**kwargs)

    def _preprocess(self, X, y=None, is_train=False, **kwargs):
        if is_train:
            if self.problem_type == "regression":
                from autofeat import AutoFeatRegressor
                self.autofeat_model_cls = AutoFeatRegressor
            else:
                from autofeat import AutoFeatClassifier
                self.autofeat_model_cls = AutoFeatClassifier
            # self.autofeat_categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
            self.autofeat_model = self.autofeat_model_cls(
                # random_state=self.random_state,
                # categorical_cols=self.autofeat_categorical_cols,
                feateng_steps=self.autofeat_feateng_steps,
                featsel_runs=self.autofeat_featsel_runs,
                max_gb=self.autofeat_max_gb,
                transformations=self.autofeat_transformations,
            )
            X_autofeat_in = X.select_dtypes(np.number).copy()
            if X_autofeat_in.shape[1]>0:
                self.autofeat_colna_maps = {} 
                for col in X_autofeat_in.select_dtypes(np.number):
                    self.autofeat_colna_maps[col] = X[col].mean() 
                    if X_autofeat_in[col].isna().any():
                        X_autofeat_in[col] = X_autofeat_in[col].fillna(self.autofeat_colna_maps[col])
                
                for col in X_autofeat_in.select_dtypes(include=["category", "object"]):
                    self.autofeat_colna_maps[col] = X[col].mode().iloc[0] 
                    if X_autofeat_in[col].isna().any():
                        X_autofeat_in[col] = X_autofeat_in[col].fillna(self.autofeat_colna_maps[col])

                self.autofeat_model.fit(X_autofeat_in, y)
                X_new = self.autofeat_model.transform(X_autofeat_in)
                X_new = X_new.drop([i for i in X_new.columns if i in X.columns],axis=1)
                feature_types = self._get_types_of_features(X)
                X = self._preprocess_train(X, feature_types, self.params["vectorizer_dict_size"])

                X = np.concatenate([X, X_new.values],axis=1)
            else:
                feature_types = self._get_types_of_features(X)
                X = self._preprocess_train(X, feature_types, self.params["vectorizer_dict_size"])

        else:
            X_autofeat_in = X.select_dtypes(np.number).copy()
            if X_autofeat_in.shape[1]>0:
                for col in X_autofeat_in:
                    if X[col].isna().any():
                        X_autofeat_in[col] = X[col].fillna(self.autofeat_colna_maps[col])
                X_new = self.autofeat_model.transform(X_autofeat_in)
                X_new = X_new.drop([i for i in X_new.columns if i in X.columns],axis=1)
                X = self._pipeline.transform(X)
                X = np.concatenate([X, X_new.values],axis=1)
            else:
                X = self._pipeline.transform(X)
        return X

