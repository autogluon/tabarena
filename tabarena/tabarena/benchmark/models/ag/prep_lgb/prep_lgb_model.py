from __future__ import annotations

import gc
import logging
import os

import time
import warnings
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.common.utils.try_import import try_import_lightgbm
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

from autogluon.tabular.models.lgb import lgb_utils
from autogluon.tabular.models.lgb.hyperparameters.parameters import DEFAULT_NUM_BOOST_ROUND, get_lgb_objective, get_param_baseline
from autogluon.tabular.models.lgb.lgb_utils import construct_dataset, train_lgb_model

from autogluon.features import ArithmeticFeatureGenerator
from autogluon.features import CategoricalInteractionFeatureGenerator
from autogluon.features import OOFTargetEncodingFeatureGenerator

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
warnings.filterwarnings("ignore", category=FutureWarning, message="Dask dataframe query")  # lightGBM dask-expr warning
logger = logging.getLogger(__name__)

from scipy.special import softmax
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from .linear_init import GroupedLinearInitScore, LinearInitScore, OOFLinearInitScore

def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None, init_score=None):
    # FIXME: Copied function from AG and added init_score - should rather be added to AG
    try_import_lightgbm()
    import lightgbm as lgb

    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight, init_score=init_score)

    if save:
        assert location is not None
        saving_path = f"{location}.bin"
        if os.path.exists(saving_path):
            os.remove(saving_path)

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        dataset.save_binary(saving_path)
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()

    return dataset


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class PrepLGBModel(LGBModel):
    """
    LightGBM model: https://lightgbm.readthedocs.io/en/latest/

    Hyperparameter options: https://lightgbm.readthedocs.io/en/latest/Parameters.html

    Extra hyperparameter options:
        ag.early_stop : int, specifies the early stopping rounds. Defaults to an adaptive strategy. Recommended to keep default.
    """
    ag_key = "prep_GBM"
    ag_name = "prep_LightGBM"
    ag_priority = 90
    ag_priority_by_problem_type = MappingProxyType({
        SOFTCLASS: 100
    })
    seed_name = "seed"
    seed_name_alt = ["seed_value", "random_seed", "random_state"]

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        default_params['prep_params'] = {}
        default_params['use_residuals'] = False
        default_params['max_dataset_size_for_residuals'] = 1000
        default_params['residual_type'] = 'oof'
        default_params['residual_init_kwargs'] = {'scaler': 'squashing'}

        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()

        X = X.copy()
        for preprocessor_cls_name, init_params in hyperparameters['prep_params'].items():
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

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        **kwargs,
    ) -> int:
        memory_usage = super()._estimate_memory_usage_static(**kwargs)
        # FIXME: 1.5 runs OOM on kddcup09_appetency fold 2 repeat 0 prep_LightGBM_r49_BAG_L1
        return memory_usage * 2.0  # FIXME: For some reason this underestimates mem usage without this

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=0, sample_weight=None, sample_weight_val=None, verbosity=2, **kwargs):
        try_import_lightgbm()  # raise helpful error message if LightGBM isn't installed
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        generate_curves = ag_params.get("generate_curves", False)

        if generate_curves:
            X_test = kwargs.get("X_test", None)
            y_test = kwargs.get("y_test", None)
        else:
            X_test = None
            y_test = None

        if verbosity <= 1:
            log_period = False
        elif verbosity == 2:
            log_period = 1000
        elif verbosity == 3:
            log_period = 50
        else:
            log_period = 1

        stopping_metric, stopping_metric_name = self._get_stopping_metric_internal()

        num_boost_round = params.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND)
        dart_retrain = params.pop("dart_retrain", False)  # Whether to retrain the model to get optimal iteration if model is trained in 'dart' mode.
        if num_gpus != 0:
            if "device" not in params:
                # TODO: lightgbm must have a special install to support GPU: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version
                #  Before enabling GPU, we should add code to detect that GPU-enabled version is installed and that a valid GPU exists.
                #  GPU training heavily alters accuracy, often in a negative manner. We will have to be careful about when to use GPU.
                params["device"] = "gpu"
                logger.log(20, f"\tWarning: Training LightGBM with GPU. This may negatively impact model quality compared to CPU training.")
        logger.log(15, f"\tFitting {num_boost_round} rounds... Hyperparameters: {params}")

        if "num_threads" not in params:
            params["num_threads"] = num_cpus
        if "objective" not in params:
            params["objective"] = get_lgb_objective(problem_type=self.problem_type)
        if self.problem_type in [MULTICLASS, SOFTCLASS] and "num_classes" not in params:
            params["num_classes"] = self.num_classes
        if "verbose" not in params:
            params["verbose"] = -1

        if X.shape[0] > params.get("max_dataset_size_for_residuals", 1000):
            params["use_residuals"] = False
        self.use_residuals = params.get("use_residuals", False) # NOTE: Added to be used at inference time, since params seems not to be available then.
        self.residual_init_kwargs = params.pop("residual_init_kwargs", {})

        num_rows_train = len(X)
        dataset_train, dataset_val, dataset_test = self.generate_datasets(
            X=X, y=y, params=params, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, sample_weight=sample_weight, sample_weight_val=sample_weight_val
        )
        gc.collect()

        callbacks = []
        valid_names = []
        valid_sets = []
        if dataset_val is not None:
            from autogluon.tabular.models.lgb.callbacks import early_stopping_custom

            # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)
            if early_stopping_rounds is None:
                early_stopping_rounds = 999999
            reporter = kwargs.get("reporter", None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            if train_loss_name is not None:
                if "metric" not in params or params["metric"] == "":
                    params["metric"] = train_loss_name
                elif train_loss_name not in params["metric"]:
                    params["metric"] = f'{params["metric"]},{train_loss_name}'
            # early stopping callback will be added later by QuantileBooster if problem_type==QUANTILE
            early_stopping_callback_kwargs = dict(
                stopping_rounds=early_stopping_rounds,
                metrics_to_use=[("valid_set", stopping_metric_name)],
                max_diff=None,
                start_time=start_time,
                time_limit=time_limit,
                ignore_dart_warning=True,
                verbose=False,
                manual_stop_file=False,
                reporter=reporter,
                train_loss_name=train_loss_name,
            )
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(**early_stopping_callback_kwargs)
            ]
            valid_names = ["valid_set"] + valid_names
            valid_sets = [dataset_val] + valid_sets
        else:
            early_stopping_callback_kwargs = None

        from lightgbm.callback import log_evaluation, record_evaluation

        if log_period is not None:
            callbacks.append(log_evaluation(period=log_period))

        train_params = {
            "params": {key: value for key, value in params.items() if key not in ["prep_params"]},
            "train_set": dataset_train,
            "num_boost_round": num_boost_round,
            "valid_names": valid_names,
            "valid_sets": valid_sets,
            "callbacks": callbacks,
            "keep_training_booster": generate_curves,
        }

        if generate_curves:
            scorers = ag_params.get("curve_metrics", [self.eval_metric])
            use_curve_metric_error = ag_params.get("use_error_for_curve_metrics", False)
            metric_names = [scorer.name for scorer in scorers]

            if stopping_metric_name in metric_names:
                idx = metric_names.index(stopping_metric_name)
                scorers[idx].name = f"_{stopping_metric_name}"
                metric_names[idx] = scorers[idx].name

            custom_metrics = [
                lgb_utils.func_generator(
                    metric=scorer,
                    is_higher_better=scorer.greater_is_better_internal,
                    needs_pred_proba=not scorer.needs_pred,
                    problem_type=self.problem_type,
                    error=use_curve_metric_error,
                )
                for scorer in scorers
            ]

            eval_results = {}
            train_params["callbacks"].append(record_evaluation(eval_results))
            train_params["feval"] = custom_metrics

            if dataset_test is not None:
                train_params["valid_names"] = ["train_set", "test_set"] + train_params["valid_names"]
                train_params["valid_sets"] = [dataset_train, dataset_test] + train_params["valid_sets"]
            else:
                train_params["valid_names"] = ["train_set"] + train_params["valid_names"]
                train_params["valid_sets"] = [dataset_train] + train_params["valid_sets"]

        # NOTE: lgb stops based on first metric if more than one
        if not isinstance(stopping_metric, str):
            if generate_curves:
                train_params["feval"].insert(0, stopping_metric)
            else:
                train_params["feval"] = stopping_metric
        elif isinstance(stopping_metric, str):
            if "metric" not in train_params["params"] or train_params["params"]["metric"] == "":
                train_params["params"]["metric"] = stopping_metric
            elif stopping_metric not in train_params["params"]["metric"]:
                train_params["params"]["metric"] = f'{stopping_metric},{train_params["params"]["metric"]}'

        if self.problem_type == SOFTCLASS:
            train_params["params"]["objective"] = lgb_utils.softclass_lgbobj
            train_params["params"]["num_classes"] = self.num_classes
        elif self.problem_type == QUANTILE:
            train_params["params"]["quantile_levels"] = self.quantile_levels

        # Train LightGBM model:
        # Note that self.model contains a <class 'lightgbm.basic.Booster'> not a LightBGMClassifier or LightGBMRegressor object
        from lightgbm.basic import LightGBMError

        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0, future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset.")
            warnings.filterwarnings("ignore", message="categorical_column in param dict is overridden.")
            try:
                self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            except LightGBMError:
                if train_params["params"].get("device", "cpu") not in ["gpu", "cuda"]:
                    raise
                else:
                    if train_params["params"]["device"] == "gpu":
                        logger.warning(
                            "Warning: GPU mode might not be installed for LightGBM, "
                            "GPU training raised an exception. Falling back to CPU training..."
                            "Refer to LightGBM GPU documentation: "
                            "https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version"
                            "One possible method is:"
                            "\tpip uninstall lightgbm -y"
                            "\tpip install lightgbm --install-option=--gpu"
                        )
                    elif train_params["params"]["device"] == "cuda":
                        # Current blocker for using CUDA over GPU: https://github.com/microsoft/LightGBM/issues/6828
                        # Note that device="cuda" works if AutoGluon (and therefore LightGBM) is installed via conda.
                        logger.warning(
                            "Warning: CUDA mode might not be installed for LightGBM, "
                            "CUDA training raised an exception. Falling back to CPU training..."
                            "Refer to LightGBM CUDA documentation: "
                            "https://github.com/Microsoft/LightGBM/tree/master/python-package#build-cuda-version"
                        )
                    train_params["params"]["device"] = "cpu"
                    self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            retrain = False
            if train_params["params"].get("boosting_type", "") == "dart":
                if dataset_val is not None and dart_retrain and (self.model.best_iteration != num_boost_round):
                    retrain = True
                    if time_limit is not None:
                        time_left = time_limit + start_time - time.time()
                        if time_left < 0.5 * time_limit:
                            retrain = False
                    if retrain:
                        logger.log(15, f"Retraining LGB model to optimal iterations ('dart' mode).")
                        train_params.pop("callbacks", None)
                        train_params.pop("valid_sets", None)
                        train_params.pop("valid_names", None)
                        train_params["num_boost_round"] = self.model.best_iteration
                        self.model = train_lgb_model(**train_params)
                    else:
                        logger.log(15, f"Not enough time to retrain LGB model ('dart' mode)...")

        if generate_curves:

            def og_name(key):
                if key == f"_{stopping_metric_name}":
                    return stopping_metric_name
                return key

            def filter(d, keys):
                return {og_name(key): d[key] for key in keys if key in d}

            curves = {"train": filter(eval_results["train_set"], metric_names)}
            if X_val is not None:
                curves["val"] = filter(eval_results["valid_set"], metric_names)
            if X_test is not None:
                curves["test"] = filter(eval_results["test_set"], metric_names)

            if f"_{stopping_metric_name}" in metric_names:
                idx = metric_names.index(f"_{stopping_metric_name}")
                metric_names[idx] = stopping_metric_name

            self.save_learning_curves(metrics=metric_names, curves=curves)

        if dataset_val is not None and not retrain:
            self.params_trained["num_boost_round"] = self.model.best_iteration
        else:
            self.params_trained["num_boost_round"] = self.model.current_iteration()

    
    def _predict_proba(self, X, num_cpus=0, **kwargs) -> np.ndarray:
        if self.use_residuals:
            y_pred_linear = self.lin_init.init_score(X, is_train=False)
            X = self.preprocess(X, **kwargs)
            y_pred_lgb = self.model.predict(X, num_threads=num_cpus, raw_score=True)
            y_pred_proba = y_pred_lgb + y_pred_linear
            if self.problem_type == 'binary':
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            elif self.problem_type in ['multiclass', 'softclass']:
                y_pred_proba = softmax(y_pred_proba, axis=1)
        else:
            X = self.preprocess(X, **kwargs)
            y_pred_proba = self.model.predict(X, num_threads=num_cpus)
        if self.problem_type == QUANTILE:
            # y_pred_proba is a pd.DataFrame, need to convert
            y_pred_proba = y_pred_proba.to_numpy()
        if self.problem_type in [REGRESSION, QUANTILE, MULTICLASS]:
            return y_pred_proba
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == SOFTCLASS:  # apply softmax
            y_pred_proba = np.exp(y_pred_proba)
            y_pred_proba = np.multiply(y_pred_proba, 1 / np.sum(y_pred_proba, axis=1)[:, np.newaxis])
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def get_preprocessors(self, prep_params: dict = None) -> list:
        if prep_params is None:
            return []
        
        preprocessors = []
        for prep_name, init_params in prep_params.items():
            preprocessor_class = eval(prep_name)
            if preprocessor_class is not None:
                _init_params = dict(verbosity=0)
                _init_params.update(**init_params)
                preprocessors.append(preprocessor_class(target_type=self.problem_type, **_init_params, random_state=self.random_seed))
            else:
                raise ValueError(f"Preprocessor {prep_name} not recognized.")

        return preprocessors

    def _preprocess(self, X, y = None, is_train=False, prep_params: dict = None, **kwargs):
        X_out = X.copy()
        if is_train:
            self.preprocessors = self.get_preprocessors(prep_params=prep_params)
            for prep in self.preprocessors:
                X_out = prep.fit_transform(X_out, y)
        else:
            for prep in self.preprocessors:
                X_out = prep.transform(X_out)

        return X_out
    
    def generate_datasets(
        self,
        X: DataFrame,
        y: Series,
        params,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        sample_weight=None,
        sample_weight_val=None,
        sample_weight_test=None,
        save=False,
    ):
        lgb_dataset_params_keys = ["two_round"]  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike

        y_og = None
        y_val_og = None
        y_test_og = None
        if self.problem_type == SOFTCLASS:
            y_og = np.array(y)
            y = None
            if X_val is not None:
                y_val_og = np.array(y_val)
                y_val = None
            if X_test is not None:
                y_test_og = np.array(y_test)
                y_test = None

        if 'use_residuals' in params and params['use_residuals']:
            if params['residual_type'] == 'grouped':
                self.lin_init = GroupedLinearInitScore(target_type=self.problem_type, init_kwargs=self.residual_init_kwargs, random_state=self.random_seed)
            elif params['residual_type'] == 'oof':
                self.lin_init = OOFLinearInitScore(target_type=self.problem_type, init_kwargs=self.residual_init_kwargs, random_state=self.random_seed)
            elif params['residual_type'] == 'knn':
                self.lin_init = OOFKNNInitScore(target_type=self.problem_type, init_kwargs=self.residual_init_kwargs, random_state=self.random_seed)
            else:
                self.lin_init = LinearInitScore(target_type=self.problem_type, init_kwargs=self.residual_init_kwargs, random_state=self.random_seed)
            self.lin_init.fit(X, y)
            init_train = self.lin_init.init_score(X, is_train=True)
            init_valid = self.lin_init.init_score(X_val, is_train=False) if X_val is not None else None
            init_test = self.lin_init.init_score(X_test, is_train=False) if X_test is not None else None
        else:
            init_train = None
            init_valid = None
            init_test = None
        
        X = self.preprocess(X=X, y=y, is_train=True, prep_params=params['prep_params'])
        if X_val is not None:
            X_val = self.preprocess(X_val)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        
        # X, W_train = self.convert_to_weight(X=X)
        dataset_train = construct_dataset(
            x=X, y=y, location=os.path.join("self.path", "datasets", "train"), params=data_params, save=save, weight=sample_weight, init_score=init_train
        )
        # dataset_train = construct_dataset_lowest_memory(X=X, y=y, location=self.path + 'datasets/train', params=data_params)
        if X_val is not None:
            # X_val, W_val = self.convert_to_weight(X=X_val)
            dataset_val = construct_dataset(
                x=X_val,
                y=y_val,
                location=os.path.join(self.path, "datasets", "val"),
                reference=dataset_train,
                params=data_params,
                save=save,
                weight=sample_weight_val,
                init_score=init_valid,
            )
            # dataset_val = construct_dataset_lowest_memory(X=X_val, y=y_val, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        else:
            dataset_val = None

        if X_test is not None:
            dataset_test = construct_dataset(
                x=X_test,
                y=y_test,
                location=os.path.join(self.path, "datasets", "test"),
                reference=dataset_train,
                params=data_params,
                save=save,
                weight=sample_weight_test,
                init_score=init_test,
            )
        else:
            dataset_test = None

        if self.problem_type == SOFTCLASS:
            if y_og is not None:
                dataset_train.softlabels = y_og
            if y_val_og is not None:
                dataset_val.softlabels = y_val_og
            if y_test_og is not None:
                dataset_test.softlabels = y_test_og
        return dataset_train, dataset_val, dataset_test