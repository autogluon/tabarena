from __future__ import annotations

import copy
import gc
import inspect
import shutil
from typing import Type

import numpy as np
import pandas as pd

from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel, _apply_inv_perm, _make_perm

from loguru import logger

class AGWrapper(AbstractExecModel):
    can_get_error_val = True
    can_get_oof = True

    def __init__(
        self,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        preprocess_data: bool = False,
        preprocess_label: bool = False,
        target_name: str | None = None,
        **kwargs,
    ):
        super().__init__(preprocess_data=preprocess_data, preprocess_label=preprocess_label, target_name=target_name, **kwargs)
        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        if target_name is None:
            target_name = "__label__"
        self.label = target_name

    def _resolve_validation_protocol(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> tuple[pd.DataFrame, dict, dict]:
        """Update the AutoGluon validation protocol."""
        init_kwargs = copy.deepcopy(self.init_kwargs)
        fit_kwargs = copy.deepcopy(self.fit_kwargs)

        # TODO: think about if we can reset the index here without breaking simulation artifacts
        train_data = X

        num_folds = fit_kwargs.pop("num_bag_folds", None)
        num_repeats = fit_kwargs.pop("num_bag_folds", None)

        custom_splits, num_folds, num_repeats = self.resolve_validation_splits(
            X=train_data.reset_index(drop=True),
            y=y.reset_index(drop=True),
            num_folds=num_folds,
            num_repeats=num_repeats
        )

        if num_folds is not None:
            logger.info(f"Using num_folds: {num_folds}")
            fit_kwargs["num_bag_folds"] = num_folds
        if num_repeats is not None:
            logger.info(f"Using num_repeats: {num_repeats}")
            fit_kwargs["num_bag_sets"] = num_repeats
        if custom_splits is not None:
            logger.info("Using custom_splits for validation protocol.")
            if "ag_args_ensemble" not in fit_kwargs:
                fit_kwargs["ag_args_ensemble"] = {}
            fit_kwargs["ag_args_ensemble"]["custom_splits"] = custom_splits

        feature_generator_cls = fit_kwargs.pop("feature_generator_cls", None)
        if feature_generator_cls is not None:
            feature_generator_kwargs = fit_kwargs.pop("feature_generator_kwargs", {})
            sig = inspect.signature(feature_generator_cls.__init__)
            group_params = {
                "group_cols": self.group_on,
                "group_labels": self.group_labels,
                "group_time_on": self.group_time_on,
            }
            for param, value in group_params.items():
                if param in sig.parameters:
                    feature_generator_kwargs[param] = value
            fit_kwargs["feature_generator"] = feature_generator_cls(**feature_generator_kwargs)

        train_data[self.label] = y
        if X_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.label] = y_val
            fit_kwargs["tuning_data"] = tuning_data

        return train_data, init_kwargs, fit_kwargs


    def _fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            **kwargs
    ):
        from autogluon.tabular import TabularPredictor

        # FIXME: should we not reset the index of train data here?
        train_data, init_kwargs, fit_kwargs = self._resolve_validation_protocol(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            **init_kwargs
        )
        self.predictor.fit(
            train_data=train_data,
            **fit_kwargs
        )

        # FIXME: persist
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predictor.predict(X)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.predictor.predict_proba(X)
        return y_pred_proba

    def get_oof(self):
        # TODO: Rename method
        simulation_artifact = self.predictor.simulation_artifact()
        simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"][self.predictor.model_best]
        return simulation_artifact

    def get_metric_error_val(self) -> float:
        # FIXME: this shouldn't be calculating its own val score, that should be external. This should simply give val pred and val pred proba
        leaderboard = self.predictor.leaderboard(score_format="error", set_refit_score_to_parent=True)
        metric_error_val = leaderboard.set_index("model").loc[self.predictor.model_best]["metric_error_val"]
        if metric_error_val is not None and not np.isnan(metric_error_val):
            metric_error_val = float(metric_error_val)
        return metric_error_val

    def cleanup(self):
        shutil.rmtree(self.predictor.path, ignore_errors=True)
        gc.collect()
        try:
            import torch
        except ImportError:
            pass
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class AGSingleWrapper(AGWrapper):
    """
    Wrapper for a single model being fit in AutoGluon

    Parameters
    ----------
    model_cls: str | Type["AbstractModel"]
        The model_cls normally used for the model family in `predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})
    model_hyperparameters
        The model_hyperparameters normally used in `predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})
    calibrate : bool | str, default False
    init_kwargs
    fit_kwargs
    preprocess_data
    preprocess_label
    kwargs

    """
    def __init__(
        self,
        model_cls: str | Type["AbstractModel"],
        model_hyperparameters: dict,
        calibrate: bool | str = False,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        preprocess_data: bool = False,
        preprocess_label: bool = False,
        **kwargs,
    ):
        from autogluon.tabular.models import AbstractModel
        assert (isinstance(model_cls, str) or issubclass(model_cls, AbstractModel))
        assert isinstance(model_hyperparameters, dict)

        if fit_kwargs is None:
            fit_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        assert "hyperparameters" not in fit_kwargs, f"Must not specify `hyperparameters` in AGSingleWrapper."
        assert "num_stack_levels" not in fit_kwargs, f"num_stack_levels is not allowed for `AGSingleWrapper"
        assert "presets" not in fit_kwargs, f"AGSingleWrapper does not support `presets`"
        assert "fit_weighted_ensemble" not in fit_kwargs, f"Must not specify `fit_weighted_ensemble` in AGSingleWrapper... It is always set to False."
        assert "calibrate" not in fit_kwargs, f"Specify calibrate directly rather than in `fit_kwargs`"
        assert "ag_args_fit" not in fit_kwargs, f"ag_args_fit must be specified in `model_hyperparameters`, not in `fit_kwargs` for `AGSingleWrapper"
        assert "ag_args_ensemble" not in fit_kwargs, f"ag_args_ensemble must be specified in `model_hyperparameters`, not in `fit_kwargs` for `AGSingleWrapper"

        self.init_kwargs_extra = init_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["calibrate"] = calibrate

        self.fit_kwargs_extra = fit_kwargs
        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["fit_weighted_ensemble"] = False
        fit_kwargs["hyperparameters"] = {model_cls: model_hyperparameters}

        self._model_cls = model_cls
        self.model_hyperparameters = model_hyperparameters

        super().__init__(init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, preprocess_data=preprocess_data, preprocess_label=preprocess_label, **kwargs)

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        self.failure_artifact = self.get_metadata_failure()

    def get_hyperparameters(self):
        hyperparameters = self.predictor.model_hyperparameters(model=self.predictor.model_best, output_format="user")
        return hyperparameters

    @property
    def model_cls(self) -> Type["AbstractModel"]:
        if not isinstance(self._model_cls, str):
            model_cls = self._model_cls
        else:
            # TODO: Get it from predictor instead? What if we allow passing custom model registry?
            from autogluon.tabular.registry import ag_model_registry  # If this raises an exception, you need to update to latest mainline AutoGluon
            model_cls = ag_model_registry.key_to_cls(key=self._model_cls)
        return model_cls

    def _load_model(self, assert_single_model: bool = True):
        model_names = self.predictor.model_names(can_infer=True)
        if assert_single_model:
            assert len(model_names) == 1
            model_name = self.predictor.model_names()[0]
        else:
            model_name = self.predictor.model_best
        return self.predictor._trainer.load_model(model_name)

    def get_metadata_init(self) -> dict:
        metadata = {}
        metadata["hyperparameters"] = self.get_hyperparameters()
        metadata["model_cls"] = self.model_cls.__name__
        metadata["model_type"] = self.model_cls.ag_key  # TODO: rename to ag_key?
        metadata["name_prefix"] = self.model_cls.ag_name  # TODO: rename to ag_name?
        metadata["model_hyperparameters"] = self.model_hyperparameters
        metadata["init_kwargs_extra"] = self.init_kwargs_extra
        metadata["fit_kwargs_extra"] = self.fit_kwargs_extra
        return metadata

    def get_metadata_fit(self) -> dict:
        metadata = {}
        model = self._load_model(assert_single_model=False)
        metadata["info"] = model.get_info(include_feature_metadata=False)
        metadata["disk_usage"] = model.disk_usage()
        metadata["num_cpus"] = model.fit_num_cpus
        metadata["num_gpus"] = model.fit_num_gpus
        metadata["num_cpus_child"] = model.fit_num_cpus_child
        metadata["num_gpus_child"] = model.fit_num_gpus_child
        metadata["fit_metadata"] = model.get_fit_metadata()
        if hasattr(model, "_memory_usage_estimate"):
            metadata["memory_usage_estimate"] = model._memory_usage_estimate
        return metadata

    def get_metadata_failure(self) -> dict:
        metadata = {
            "model_failures": self.predictor.model_failures()
        }
        return metadata

    def get_metadata(self) -> dict:
        metadata = self.get_metadata_init()
        metadata_fit = self.get_metadata_fit()

        metadata.update(metadata_fit)
        return metadata


class AGSingleBagWrapper(AGSingleWrapper):
    can_get_per_child_oof = True
    can_get_per_child_val_idx = True

    def bag_artifact(self, X_test: pd.DataFrame):
        model = self._load_model()
        bag_info = {}
        bag_info["pred_proba_test_per_child"] = self.get_per_child_test(X_test=X_test, model=model)
        bag_info["val_idx_per_child"] = self.get_per_child_val_idx(model=model)
        return bag_info

    def get_per_child_val_idx(self, model=None) -> list[np.ndarray]:
        if model is None:
            model = self._load_model()
        X, y = self.predictor.load_data_internal()
        all_kfolds = []
        # TODO: Make this a bagged ensemble method
        if model._child_oof:
            all_kfolds = [(None, X.index.values)]
        else:
            for n_repeat, k in enumerate(model._k_per_n_repeat):
                kfolds = model._cv_splitters[n_repeat].split(X=X, y=y)
                cur_kfolds = kfolds[n_repeat * k: (n_repeat + 1) * k]
                all_kfolds += cur_kfolds

        val_idx_per_child = []
        for fold_idx, (train_idx, val_idx) in enumerate(all_kfolds):
            val_idx = pd.to_numeric(val_idx, downcast="integer")  # memory opt
            val_idx_per_child.append(val_idx)

        return val_idx_per_child

    # TODO: Can avoid predicting on test twice by doing it all in one go
    def get_per_child_test(self, X_test: pd.DataFrame, model=None) -> list[np.ndarray]:
        original_index = X_test.index
        inv_perm = None
        if self.shuffle_test:
            perm, inv_perm = _make_perm(len(X_test), seed=self.shuffle_seed)
            X_test = X_test.iloc[perm]
        if self.reset_index_test:
            X_test = X_test.reset_index(drop=True)

        X_test = self.transform_X(X=X_test)

        if model is None:
            model = self._load_model()
        X_test_inner = self.predictor.transform_features(data=X_test, model=model.name)

        if model.can_predict_proba():
            per_child_test_preds = model.predict_proba_children(X=X_test_inner)
        else:
            per_child_test_preds = model.predict_children(X=X_test_inner)

        if self.shuffle_test:
            # Inverse-permute outputs back to original X_test order
            per_child_test_preds = [_apply_inv_perm(y_pred, inv_perm, index=original_index) for y_pred in per_child_test_preds]

        per_child_test_preds = [preds_child.astype(np.float32) for preds_child in per_child_test_preds]  # memory opt
        return per_child_test_preds
