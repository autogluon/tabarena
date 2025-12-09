from __future__ import annotations

import logging

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
warnings.filterwarnings("ignore", category=FutureWarning, message="Dask dataframe query")  # lightGBM dask-expr warning
logger = logging.getLogger(__name__)


from autogluon.features import ArithmeticFeatureGenerator
from autogluon.features import CategoricalInteractionFeatureGenerator
from autogluon.features import OOFTargetEncodingFeatureGenerator

class ModelAgnosticPrepMixin:
    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        prep_params = self._get_ag_params().get("prep_params", None)
        if prep_params is None:
            prep_params = {}

        if prep_params:
            X = X.copy()
        for preprocessor_cls_name, init_params in prep_params.items():
            if preprocessor_cls_name == 'ArithmeticFeatureGenerator':
                prep_cls = ArithmeticFeatureGenerator(target_type=self.problem_type, **init_params)
                num_new_feats, affected_features = prep_cls.estimate_no_of_new_features(X)
                X_new = pd.DataFrame(np.random.random(size=[X.shape[0], num_new_feats]), index=X.index, columns=[f'arithmetic_{i}' for i in range(num_new_feats)]).astype(prep_cls.out_dtype)
                prep_cls
                X = pd.concat([X, X_new], axis=1)
            elif preprocessor_cls_name == 'CategoricalInteractionFeatureGenerator':
                # TODO: Test whether it is also fine to just do the actual preprocessing and use the X resulting from that
                prep_cls = CategoricalInteractionFeatureGenerator(target_type=self.problem_type, **init_params)
                num_new_feats, affected_features = prep_cls.estimate_no_of_new_features(X)
                if prep_cls.only_freq:
                    X = pd.concat([X, pd.DataFrame(np.random.random(size=[X.shape[0], num_new_feats]), index=X.index, columns=[f'cat_int_freq_{i}' for i in range(num_new_feats)])], axis=1)
                elif prep_cls.add_freq:
                    shape = X.shape[0]
                    max_card = X.nunique().max()
                    X_cat_new = pd.DataFrame(np.random.randint(0, int(shape*(max_card/shape)), [shape, num_new_feats]), index=X.index, columns=[f'cat_int{i}' for i in range(num_new_feats)]).astype('category')
                    X = pd.concat([X, X_cat_new, pd.DataFrame(np.random.random(size=[X.shape[0], num_new_feats]), index=X.index, columns=[f'cat_int_freq_{i}' for i in range(num_new_feats)])], axis=1)
                else:
                    shape = X.shape[0]
                    max_card = X.nunique().max()
                    X_cat_new = pd.DataFrame(np.random.randint(0, int(shape*(max_card/shape)), [shape, num_new_feats]), index=X.index, columns=[f'cat_int_freq_{i}' for i in range(num_new_feats)]).astype('category')
                    X = pd.concat([X, X_cat_new], axis=1)
            elif preprocessor_cls_name == 'OOFTargetEncodingFeatureGenerator':
                prep_cls = OOFTargetEncodingFeatureGenerator(target_type=self.problem_type, **init_params)
                num_new_feats, affected_features = prep_cls.estimate_no_of_new_features(X, self.num_classes)
                if prep_cls.keep_original:
                    X_new = pd.DataFrame(np.random.random(size=[shape, num_new_feats]), index=X.index, columns=['oof_te_' + str(num) for num in range(num_new_feats)])
                    X = pd.concat([X, X_new], axis=1)
                else:
                    X = X.drop(columns=affected_features)
                    X_new = pd.DataFrame(np.random.random(size=[shape, num_new_feats]), index=X.index, columns=['oof_te_' + str(num) for num in range(num_new_feats)])
                    X = pd.concat([X, X_new], axis=1)

        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    def get_preprocessors(self, prep_params: dict = None) -> list:
        if prep_params is None:
            return []
        
        preprocessors = []
        for prep_name, init_params in prep_params.items():
            preprocessor_class = eval(prep_name)
            if preprocessor_class is not None:
                _init_params = dict(verbosity=0, random_state=self.random_seed)
                _init_params.update(**init_params)
                preprocessors.append(preprocessor_class(target_type=self.problem_type, **_init_params))
            else:
                raise ValueError(f"Preprocessor {prep_name} not recognized.")

        return preprocessors

    def _preprocess(self, X, y = None, is_train=False, prep_params: dict = None, **kwargs):
        X_out = X.copy()
        if is_train:
            self.preprocessors = self.get_preprocessors(prep_params=prep_params)
            for prep in self.preprocessors:
                X_out = prep.fit_transform(X_out, y)
            self.feature_metadata = self.feature_metadata.from_df(X_out) # TODO: Unsure whether that is the appropriate way to set the metadata 
            self._is_features_in_same_as_ex = True
        else:
            for prep in self.preprocessors:
                X_out = prep.transform(X_out)

        return super()._preprocess(X_out, is_train=is_train, **kwargs)
