from __future__ import annotations

import copy
import logging
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@contextmanager
def set_logger_level(logger_name: str, level: int):
    _logger = logging.getLogger(logger_name)
    old_level = _logger.level
    _logger.setLevel(level)
    try:
        yield
    finally:
        _logger.setLevel(old_level)


class XRFMImplementation:
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.kwargs = kwargs

    def fit(self, X, y, X_val, y_val):
        import torch
        import xrfm

        # preprocessing
        self.cat_cols_ = X.select_dtypes(
            include=["category", "string", "object"]
        ).columns.tolist()
        self.num_cols_ = X.select_dtypes(
            exclude=["category", "string", "object"]
        ).columns.tolist()

        # Initialize encoders
        self.ohe_ = (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            if self.cat_cols_
            else None
        )
        self.scaler_ = StandardScaler()

        # Encode categorical variables
        if self.cat_cols_:
            x_cat_enc = self.ohe_.fit_transform(X.loc[:, self.cat_cols_])
        else:
            x_cat_enc = np.empty((len(X), 0))

        x_num_enc = X.loc[:, self.num_cols_].to_numpy().astype(np.float32)

        # Process validation data
        if self.cat_cols_:
            x_val_cat_enc = self.ohe_.transform(X_val.loc[:, self.cat_cols_])
        else:
            x_val_cat_enc = np.empty((len(X_val), 0))

        x_val_num_enc = X_val.loc[:, self.num_cols_].to_numpy().astype(np.float32)

        if self.kwargs.get("standardize_cats", False):
            X = np.concatenate([x_cat_enc, x_num_enc], axis=1)
            X_val = np.concatenate([x_val_cat_enc, x_val_num_enc], axis=1)
            X = self.scaler_.fit_transform(X)
            X_val = self.scaler_.transform(X_val)
        else:
            if len(self.num_cols_) > 0:
                x_num_enc = self.scaler_.fit_transform(x_num_enc)
                x_val_num_enc = self.scaler_.transform(x_val_num_enc)
            X = np.concatenate([x_cat_enc, x_num_enc], axis=1)
            X_val = np.concatenate([x_val_cat_enc, x_val_num_enc], axis=1)

        X = torch.from_numpy(X).float()
        X_val = torch.from_numpy(X_val).float()

        if self.problem_type == REGRESSION:
            # standardize regression target (may not be necessary)
            y = np.asarray(y)
            self.mean_ = np.mean(y, axis=0)
            self.std_ = np.std(y, axis=0) + 1e-30
            y = (y - self.mean_) / self.std_
            if y_val is not None:
                y_val = np.asarray(y_val)
                y_val = (y_val - self.mean_) / self.std_

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)

        y = torch.from_numpy(y)
        y_val = torch.from_numpy(y_val)

        if self.cat_cols_:
            idx = 0
            categorical_indices = []
            categorical_vectors = []
            numerical_indices_parts = []
            for cat in self.ohe_.categories_:
                cat_len = len(cat)
                cat_idxs = torch.tensor(range(idx, idx + cat_len))
                if cat_len > self.kwargs.get("max_cardinality_for_one_hot", 100):
                    categorical_indices.append(cat_idxs)
                    if self.kwargs.get("standardize_cats", False):
                        cat_vectors = np.asarray(self.scaler_.scale_)[
                            None, cat_idxs.numpy()
                        ] * (
                            np.eye(cat_len)
                            - np.asarray(self.scaler_.mean_)[None, cat_idxs.numpy()]
                        )
                        categorical_vectors.append(
                            torch.tensor(cat_vectors, dtype=torch.float32)
                        )
                    else:
                        categorical_vectors.append(torch.eye(cat_len))
                else:
                    numerical_indices_parts.append(cat_idxs)
                idx += cat_len

            # numerical indices include small-cardinality one-hot columns and then the true numerical block
            numerical_block = idx + torch.arange(x_num_enc.shape[1])
            if len(numerical_indices_parts) > 0:
                numerical_indices = torch.cat(
                    [*numerical_indices_parts, numerical_block]
                )
            else:
                numerical_indices = numerical_block
            self.categorical_info_ = {
                "categorical_indices": categorical_indices,
                "categorical_vectors": categorical_vectors,
                "numerical_indices": numerical_indices,
            }
        else:
            self.categorical_info_ = None

        self.model = xrfm.xRFM(
            categorical_info=self.categorical_info_,  # set in preprocess()
            **self.kwargs,
        )

        self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def preprocess_transform(self, X):
        # Encode categorical variables using the same strategy as in fit

        if self.cat_cols_:
            x_cat_enc = self.ohe_.transform(X.loc[:, self.cat_cols_])
        else:
            x_cat_enc = np.empty((len(X), 0))

        x_num_enc = X.loc[:, self.num_cols_].to_numpy().astype(np.float32)

        if self.kwargs.get("standardize_cats", False):
            X_processed = np.concatenate([x_cat_enc, x_num_enc], axis=1)
            X_processed = self.scaler_.transform(X_processed)
        else:
            if len(self.num_cols_) > 0:
                x_num_enc = self.scaler_.transform(x_num_enc)
            X_processed = np.concatenate([x_cat_enc, x_num_enc], axis=1)

        return X_processed

    def predict(self, X):
        y_pred = np.squeeze(self.model.predict(self.preprocess_transform(X)), axis=1)
        if self.problem_type == REGRESSION:
            y_pred = y_pred * self.std_ + self.mean_
        return y_pred

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocess_transform(X))


# pip install xrfm
class XRFMModel(AbstractModel):
    ag_key = "XRFM"
    ag_name = "xRFM"
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None
        self._device = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()

        import torch

        # FIXME: code assume we only see one GPU in the fit process.
        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        self._device = device
        hyp = self._get_model_params()

        metric_map = {
            "roc_auc": "auc",
            "accuracy": "accuracy",
            "balanced_accuracy": "brier",
            "log_loss": "logloss",
            "rmse": "mse",
            "root_mean_squared_error": "mse",
            "r2": "mse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        tuning_metric = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = copy.copy(hyp)

        if tuning_metric is not None:
            init_kwargs["tuning_metric"] = tuning_metric

        bool_to_cat = hyp.pop("bool_to_cat", False)
        impute_bool = hyp.pop("impute_bool", True)

        init_kwargs["standardize_cats"] = init_kwargs.get("standardize_cats", False)
        init_kwargs["classification_mode"] = init_kwargs.get(
            "classification_mode", "prevalence"
        )

        solver = init_kwargs.get("solver", "solve")
        if self.problem_type in (REGRESSION, MULTICLASS):
            solver = "solve"

        if solver == "log_reg":
            classification_mode = "zero_one"
        else:
            classification_mode = init_kwargs.get("classification_mode", "prevalence")
        init_kwargs["classification_mode"] = classification_mode

        exponent = init_kwargs.get("exponent", 1.0)

        # todo: adjust this dynamically and factor it into the VRAM estimate?
        batch_size = min(init_kwargs.get("M_batch_size", len(X)), 5_000)
        init_kwargs["rfm_params"] = {
            "model": {
                "kernel": init_kwargs.get("kernel", "l2"),
                "bandwidth": init_kwargs.get("bandwidth", 10.0),
                "exponent": exponent,
                "norm_p": exponent + (2 - exponent) * init_kwargs.get("p_interp", 1.0),
                "diag": init_kwargs.get("diag", True),
                "bandwidth_mode": init_kwargs.get("bandwidth_mode", "constant"),
                "fast_categorical": init_kwargs.get("fast_categorical", True),
            },
            "fit": {
                "reg": init_kwargs.get("reg", 1e-3),
                "iters": init_kwargs.get("iters", 5),
                "M_batch_size": batch_size,
                "verbose": True,
                "early_stop_rfm": init_kwargs.get("early_stop_rfm", True),
                "early_stop_multiplier": init_kwargs.get("early_stop_multiplier", 1.05),
                "solver": solver,
            },
        }

        # todo: should we set default_rfm_params as well? -> I guess not?
        # todo: set M_batch_size?
        # todo: do we already need to move stuff to the correct device?

        X = self.preprocess(
            X, is_train=True, bool_to_cat=bool_to_cat, impute_bool=impute_bool
        )

        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = XRFMImplementation(
            problem_type=self.problem_type,
            n_threads=num_cpus if num_gpus == 0 else 1,  # avoid VRAM leak
            device=device,
            time_limit_s=time_limit - (time.time() - start_time)
            if time_limit is not None
            else None,
            **init_kwargs,
        )

        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        return super()._predict_proba(X=X, kwargs=kwargs)

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        impute_bool: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        # FIXME: is copy needed?
        X = X.copy(deep=True)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(
                required_special_types=["bool"]
            )
            if impute_bool:  # Technically this should do nothing useful because bools will never have NaN
                self._features_to_impute = self._feature_metadata.get_features(
                    valid_raw_types=["int", "float"]
                )
                self._features_to_keep = self._feature_metadata.get_features(
                    invalid_raw_types=["int", "float"]
                )
            else:
                self._features_to_impute = self._feature_metadata.get_features(
                    valid_raw_types=["int", "float"], invalid_special_types=["bool"]
                )
                self._features_to_keep = [
                    f
                    for f in self._feature_metadata.get_features()
                    if f not in self._features_to_impute
                ]
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [
                    c
                    for c in self._imputer.get_feature_names_out()
                    if c not in self._features_to_impute
                ]
        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            X_impute = pd.DataFrame(
                X_impute, index=X.index, columns=self._imputer.get_feature_names_out()
            )
            if self._indicator_columns:
                # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
                # TODO: Add to features_bool?
                X_impute[self._indicator_columns] = X_impute[
                    self._indicator_columns
                ].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X

    def _set_default_params(self):
        default_params = {
            # copied from RealMLP
            "impute_bool": False,
            "name_categories": True,
            "bool_to_cat": False,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def _validate_fit_memory_usage(
        self,
        mem_error_threshold: float = 1.0,
        **kwargs,
    ) -> tuple[int | None, int | None]:
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, **kwargs
        )

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate.
        ```.

        """
        if hyperparameters is None:
            hyperparameters = {}

        num_features = len(X.columns)
        num_samples = len(X)

        # taken from pytabkit
        columns_mem_est = (
            6e4 * num_samples
            + 2e6 * num_features
            + 30.0 * num_samples**2
            + 30.0 * num_samples * num_features
        )

        dataset_size_mem_est = (
            5 * get_approximate_df_mem_usage(X).sum()
        )  # roughly 5x DataFrame memory size
        baseline_overhead_mem_est = 2e8  # 200 MB generic overhead

        model_mem_estimate = columns_mem_est + baseline_overhead_mem_est
        model_mem_estimate = min(
            model_mem_estimate, 3.8e10
        )  # using the tree strategy caps at <40 GB
        return model_mem_estimate + dataset_size_mem_est

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to mirror RealMLP learning rate scheduler while forcing stopping at a specific epoch?
        return {"can_refit_full": False}

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        y_pred_proba = super().predict_proba(*args, **kwargs)

        if self._device == "cuda":
            # backup free up VRAM after prediction
            import torch

            torch.cuda.empty_cache()

        return y_pred_proba
