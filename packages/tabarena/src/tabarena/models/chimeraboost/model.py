"""ChimeraBoost TabArena model: a pure-Python, CatBoost-inspired, numba-backed
oblivious gradient-boosting library (https://github.com/bbstats/chimeraboost).

Drop this package at `tabarena/tabarena/models/chimeraboost/` in a fork of
autogluon/tabarena (see ../REGISTER.md). The model class MUST live in its own
file (not the run script) because TabArena pickles it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import pandas as pd


class ChimeraBoostModel(AbstractModel):
    """ChimeraBoost as an AutoGluon/TabArena model (scikit-learn style API)."""

    ag_key = "CHIMERA"
    ag_name = "ChimeraBoost"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """Label-encode categoricals and hand the model an object matrix while
        recording which columns are categorical, so ChimeraBoost applies its
        NATIVE ordered-target-statistics encoding (and auto cat_combinations on
        all-categorical data) rather than seeing cats as plain numbers."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
            # Positions of the categorical columns in the final column order,
            # passed verbatim as cat_features to ChimeraBoost. Computed once on
            # train and reused at predict (column order is identical).
            self._cat_indices = [X.columns.get_loc(c)
                                 for c in self._feature_generator.features_in]
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        # object dtype: cat columns stay integer-coded categories (ChimeraBoost
        # factorizes them), numeric columns stay float. fillna(0) for numerics;
        # the label encoder already maps unseen/NaN cats to a reserved code.
        return X.fillna(0).to_numpy(dtype=object)

    def _fit(self, X, y, num_cpus: int = 1, **kwargs):
        if self.problem_type in ["regression"]:
            from chimeraboost import ChimeraBoostRegressor

            model_cls = ChimeraBoostRegressor
        else:  # 'binary' and 'multiclass'
            from chimeraboost import ChimeraBoostClassifier

            model_cls = ChimeraBoostClassifier

        X = self.preprocess(X, y=y, is_train=True)
        params = self._get_model_params()
        # Tuned configs may sample linear_leaves=True, which raises on multiclass;
        # None = auto (binary on, multiclass off) keeps the config valid everywhere.
        if self.problem_type == "multiclass" and params.get("linear_leaves") is True:
            params["linear_leaves"] = None
        self.model = model_cls(**params)
        # early_stopping=True with no eval_set => ChimeraBoost auto-splits a
        # validation fraction internally and stops on it. cat_features triggers
        # native ordered-TS encoding + (on all-cat data) auto cat_combinations.
        cat = self._cat_indices or None
        self.model.fit(X, y, cat_features=cat)

    def _set_default_params(self):
        default_params = {
            "n_estimators": 500,
            "early_stopping": True,
            "thread_count": -1,
            "random_state": 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params
