from __future__ import annotations

import pandas as pd
import numpy as np
import os
from autogluon.tabular.models.lgb.lgb_model import LGBModel

class OpenFELGBModel(LGBModel):
    ag_key = "OpenFELGBModel"
    ag_name = "OpenFELGBModel"

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

    # def _fit(self, **kwargs):
    #     self.autofeat_feateng_steps = self.params.pop("feateng_steps", 2)
    #     self.autofeat_featsel_runs = self.params.pop("featsel_runs", 3)
    #     self.autofeat_max_gb = self.params.pop("max_gb", None)
    #     self.autofeat_transformations = self.params.pop("transformations", ("1/", "exp", "log", "abs", "sqrt", "^2", "^3"))
    #     super()._fit(**kwargs)

    def _preprocess(self, X, y=None, is_train=False, **kwargs):
        if is_train:
            from leakage_free_openfe import OpenFE
            self.openfe_categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

            if self.problem_type == "regression":
                task = "regression"
                metric ="rmse"
            elif self.problem_type == "binary":
                task = "classification"
                metric = "auc"
            elif self.problem_type == "multiclass":
                task = "classification"
                metric = "multi_logloss"

            self.openfe_model = OpenFE()

            X_openfe = X.copy()
            if not os.path.exists('./openfe'):
                os.makedirs('./openfe')
            self.openfe_model.fit(data=X_openfe, label=y, 
                                  categorical_features=self.openfe_categorical_features,
                                  task=task,
                                  metric=metric,
                                  tmp_save_path=f'./openfe/openfe_tmp_{np.random.randint(1e10)}.feather',
                                  n_jobs=1
                                  )
            # NOTE: OpenFE uses is_train the opposite way: True means that it is already trained
            X_openfe = self.openfe_model.transform(X_openfe, is_train=True).drop(X.columns,axis=1)
            X_openfe.index = X.index

            X_openfe = X_openfe.replace(np.inf, np.nan)
            X_openfe = X_openfe.replace(-np.inf, np.nan)
    
            X = pd.concat([X, X_openfe],axis=1)
        else:
            X_openfe = X.copy()
            X_openfe = self.openfe_model.transform(X_openfe, is_train=False, n_jobs=1).drop(X.columns,axis=1)
            X_openfe.index = X_openfe.index
            X_openfe = X_openfe.replace(np.inf, np.nan)
            X_openfe = X_openfe.replace(-np.inf, np.nan)
            X = pd.concat([X, X_openfe],axis=1)
        return X

