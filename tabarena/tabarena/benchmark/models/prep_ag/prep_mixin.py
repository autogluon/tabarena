from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from typing import Type

from autogluon.features import ArithmeticFeatureGenerator
from autogluon.features import CategoricalInteractionFeatureGenerator
from autogluon.features import OOFTargetEncodingFeatureGenerator
from autogluon.features import BulkFeatureGenerator
from autogluon.features.generators.abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)

# TODO: In future we can have a feature generator registry like what is done for models
_feature_generator_class_lst = [
    ArithmeticFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
]

_feature_generator_class_map = {
    feature_generator_cls.__name__: feature_generator_cls for feature_generator_cls in _feature_generator_class_lst
}


def _recursive_expand_prep_param(prep_param: tuple | list[list | tuple]) -> list[tuple]:
    if isinstance(prep_param, list):
        out = []
        for p in prep_param:
            out += _recursive_expand_prep_param(p)
        return out
    elif isinstance(prep_param, tuple):
        return [prep_param]
    else:
        raise ValueError(f"Invalid value for prep_param: {prep_param}")


# TODO: Why is `prep_params` a dict instead of a list?
class ModelAgnosticPrepMixin:
    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        prep_params = self._get_ag_params().get("prep_params", None)
        if prep_params is None:
            prep_params = []

        # FIXME: Temporarily simplify for memory calculation
        prep_params = _recursive_expand_prep_param(prep_params)
        
        shape = X.shape[0]

        for preprocessor_cls_name, init_params in prep_params:
            if preprocessor_cls_name == 'ArithmeticFeatureGenerator':
                prep_cls = ArithmeticFeatureGenerator(target_type=self.problem_type, **init_params)
                num_new_feats, affected_features = prep_cls.estimate_no_of_new_features(X)
                X_new = pd.DataFrame(np.random.random(size=[shape, num_new_feats]), index=X.index, columns=[f'arithmetic_{i}' for i in range(num_new_feats)]).astype(prep_cls.out_dtype)
                X = pd.concat([X, X_new], axis=1)
            elif preprocessor_cls_name == 'CategoricalInteractionFeatureGenerator':
                # TODO: Test whether it is also fine to just do the actual preprocessing and use the X resulting from that
                prep_cls = CategoricalInteractionFeatureGenerator(target_type=self.problem_type, **init_params)
                num_new_feats, affected_features = prep_cls.estimate_no_of_new_features(X)
                if not num_new_feats:
                    continue
                if prep_cls.only_freq:
                    X = pd.concat([X, pd.DataFrame(np.random.random(size=[shape, num_new_feats]), index=X.index, columns=[f'cat_int_freq_{i}' for i in range(num_new_feats)])], axis=1)
                elif prep_cls.add_freq:
                    max_card = X[affected_features].nunique().max()
                    X_cat_new = pd.DataFrame(np.random.randint(0, int(shape*(max_card/shape)), [shape, num_new_feats]), index=X.index, columns=[f'cat_int{i}' for i in range(num_new_feats)]).astype('category')
                    X = pd.concat([X, X_cat_new, pd.DataFrame(np.random.random(size=[shape, num_new_feats]), index=X.index, columns=[f'cat_int_freq_{i}' for i in range(num_new_feats)])], axis=1)
                else:
                    max_card = X[affected_features].nunique().max()
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

    def _init_preprocessor(
        self,
        preprocessor_cls: Type[AbstractFeatureGenerator] | str,
        init_params: dict | None,
    ) -> AbstractFeatureGenerator:
        if isinstance(preprocessor_cls, str):
            preprocessor_cls = _feature_generator_class_map[preprocessor_cls]
        if init_params is None:
            init_params = {}
        _init_params = dict(
            verbosity=0,
            random_state=self.random_seed,
            target_type=self.problem_type,
            passthrough=False,  # FIXME
        )
        _init_params.update(**init_params)
        return preprocessor_cls(
            **_init_params,
        )

    def _recursive_init_preprocessors(self, prep_param: tuple | list[list | tuple]):
        if isinstance(prep_param, list):
            out = []
            for i, p in enumerate(prep_param):
                out.append(self._recursive_init_preprocessors(p))
            return out
        elif isinstance(prep_param, tuple):
            preprocessor_cls = prep_param[0]
            init_params = prep_param[1]
            return self._init_preprocessor(preprocessor_cls=preprocessor_cls, init_params=init_params)
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

    def get_preprocessors(self) -> list[AbstractFeatureGenerator]:
        prep_params = self._get_ag_params().get("prep_params", None)
        if prep_params is None:
            return []

        preprocessors = self._recursive_init_preprocessors(prep_param=prep_params)
        if len(preprocessors) == 1 and isinstance(preprocessors[0], AbstractFeatureGenerator):
            return preprocessors
        else:
            preprocessors = [BulkFeatureGenerator(generators=preprocessors, verbosity=0)]
            return preprocessors

    def _preprocess(self, X: pd.DataFrame, y = None, is_train: bool = False, **kwargs):
        if is_train:
            self.preprocessors = self.get_preprocessors()
            if self.preprocessors:
                assert y is not None, f"y must be specified to fit preprocessors... Likely the inheriting class isn't passing `y` in its `preprocess` call."
                # FIXME: add `post_drop_useless`, example: anneal has many useless features
                feature_metadata_in = self._feature_metadata
                for prep in self.preprocessors:
                    X = prep.fit_transform(X, y, feature_metadata_in=feature_metadata_in)
                    # FIXME: Nick: This is incorrect because it strips away special dtypes. Need to do this properly by fixing in the preprocessors
                    feature_metadata_in = prep.feature_metadata
                self._feature_metadata = feature_metadata_in
                self._features_internal = self._feature_metadata.get_features()
        else:
            for prep in self.preprocessors:
                X = prep.transform(X)

        return super()._preprocess(X, y=y, is_train=is_train, **kwargs)
