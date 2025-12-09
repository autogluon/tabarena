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
from autogluon.features import BulkFeatureGenerator
from autogluon.features.generators.abstract import AbstractFeatureGenerator


# TODO: In future we can have a feature generator registry like what is done for models
_feature_generator_class_lst = [
    ArithmeticFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
]

_feature_generator_class_map = {
    feature_generator_cls.__name__: feature_generator_cls for feature_generator_cls in _feature_generator_class_lst
}


# TODO: Why is `prep_params` a dict instead of a list?
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

    def get_preprocessors(self) -> list[AbstractFeatureGenerator]:
        prep_params = self._get_ag_params().get("prep_params", None)
        if prep_params is None:
            return []
        
        preprocessors = []
        for prep_name, init_params in prep_params.items():
            preprocessor_class = _feature_generator_class_map[prep_name]
            if preprocessor_class is not None:
                _init_params = dict(verbosity=0, random_state=self.random_seed)
                _init_params.update(**init_params)
                preprocessors.append(preprocessor_class(target_type=self.problem_type, **_init_params))
            else:
                raise ValueError(f"Preprocessor {prep_name} not recognized.")

        return preprocessors

    def _preprocess(self, X: pd.DataFrame, y = None, is_train: bool = False, **kwargs):
        X_out = X.copy()
        if is_train:
            self.preprocessors = self.get_preprocessors()
            if self.preprocessors:
                assert y is not None, f"y must be specified to fit preprocessors... Likely the inheriting class isn't passing `y` in its `preprocess` call."
                self.preprocessors = [BulkFeatureGenerator(
                    generators=[[preprocessor] for preprocessor in self.preprocessors],
                    verbosity=0
                    # post_drop_duplicates=True,  # FIXME: add `post_drop_useless`, example: anneal has many useless features
                )]
                feature_metadata_in = self._feature_metadata
                for prep in self.preprocessors:
                    X_out = prep.fit_transform(X_out, y, feature_metadata_in=feature_metadata_in)
                    # FIXME: Nick: This is incorrect because it strips away special dtypes. Need to do this properly by fixing in the preprocessors
                    feature_metadata_in = prep.feature_metadata
                self._feature_metadata = feature_metadata_in
                self._features_internal = self._feature_metadata.get_features()
        else:
            for prep in self.preprocessors:
                X_out = prep.transform(X_out)

        return super()._preprocess(X_out, y=y, is_train=is_train, **kwargs)
