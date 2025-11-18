from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]


def get_tabm_auto_batch_size(n_train: int) -> int:
    if n_train < 2_800:
        return 32
    if n_train < 4_500:
        return 64
    if n_train < 6_400:
        return 128
    if n_train < 32_000:
        return 256
    if n_train < 108_000:
        return 512
    return 1024


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    # adapted from pytabkit
    def __init__(
        self,
        noise=1e-5,
        random_state=None,
        n_quantiles=1000,
        subsample=1_000_000_000,
        output_distribution="normal",
    ):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)
        # show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        return X + np.random.default_rng(self.random_state).normal(0.0, 1e-5, X.shape).astype(X.dtype)


class ModernNCAImplementation:
    def __init__(self, early_stopping_metric: Scorer, save_path, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric

        self.cat_col_names_ = None
        self.n_classes_ = None
        self.task_type_ = None
        self.device_ = None
        self.n_train_ = None
        self.has_num_cols = None
        self.num_prep_ = None
        self.save_path_ = save_path

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_col_names: list[Any],
        time_to_fit_in_seconds: float | None = None,
    ):
        from tabarena.benchmark.models.ag.modernnca.modernnca_method import ModernNCAMethod

        seed: int | None = self.config.get("random_state", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if "n_threads" in self.config:
            torch.set_num_threads(self.config["n_threads"])

        if X_val is None or len(X_val) == 0:
            raise ValueError("Training without validation set is currently not implemented")

        problem_type = self.config["problem_type"]
        task_type: TaskType = "binclass" if problem_type == "binary" else problem_type
        n_classes = None
        self.cat_col_names_ = cat_col_names
        self.n_train_ = len(X_train)
        device = self.config["device"]
        device = torch.device(device)
        self.task_type_ = task_type
        self.device_ = device
        Path(self.save_path_).mkdir(parents=True, exist_ok=True)

        # hyperparams
        # defaults taken from https://github.com/LAMDA-Tabular/TALENT/blob/cb6cb0cc9d69ac75c467e8dae8ca5ac3d3beb2f2/TALENT/configs/default/modernNCA.json
        num_emb_type = self.config.get("num_emb_type", "plr")
        num_emb_n_frequencies = self.config.get("num_emb_n_frequencies", 77)
        num_emb_frequency_scale = self.config.get("num_emb_frequency_scale", 0.04431360576139521)
        num_emb_d_embedding = self.config.get("num_emb_d_embedding", 34)
        num_emb_lite = self.config.get("num_emb_lite", True)

        dim = self.config.get("dim", 128)
        sample_rate = self.config.get("sample_rate", 0.5)
        n_epochs = self.config.get("n_epochs", 200)
        # Dynamic batch size for smaller data as recommended by the authopr
        batch_size = self.config.get("batch_size", "auto")
        lr = self.config.get("lr", 2e-3)
        d_block = self.config.get("d_block", 512)
        dropout = self.config.get("dropout", 0.1)
        n_blocks = self.config.get("n_blocks", 0)
        weight_decay = self.config.get("weight_decay", 2e-4)
        temperature = self.config.get("temperature", 1.0)

        if batch_size == "auto":
            batch_size = get_tabm_auto_batch_size(n_train=len(X_train))

        self.num_prep_ = Pipeline(steps=[("qt", RTDLQuantileTransformer()), ("imp", SimpleImputer(add_indicator=True))])
        self.has_num_cols = bool(set(X_train.columns) - set(cat_col_names))

        ds_parts = dict()
        for part, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
            tensors = dict()

            tensors["x_cat"] = X[self.cat_col_names_].to_numpy()
            if self.has_num_cols:
                x_cont_np = X.drop(columns=cat_col_names).to_numpy(dtype=np.float32)
                if part == "train":
                    self.num_prep_.fit(x_cont_np)
                tensors["x_cont"] = self.num_prep_.transform(x_cont_np)
            else:
                tensors["x_cont"] = np.empty((len(X), 0), dtype=np.float32)

            if task_type == "regression":
                tensors["y"] = y.to_numpy(np.float32)
                if part == "train":
                    n_classes = 0
            else:
                tensors["y"] = y.to_numpy(np.int32)
                if part == "train":
                    n_classes = tensors["y"].max().item() + 1

            ds_parts[part] = tensors
        self.n_classes_ = n_classes

        config = {
            "training": {
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "model": {
                "dim": dim,
                "dropout": dropout,
                "d_block": d_block,
                "n_blocks": n_blocks,
                "temperature": temperature,
                "sample_rate": sample_rate,
                "num_embeddings": {
                    "type": "PLREmbeddings",
                    "n_frequencies": num_emb_n_frequencies,
                    "frequency_scale": num_emb_frequency_scale,
                    "d_embedding": num_emb_d_embedding,
                    "lite": num_emb_lite,
                },
            },
        }

        args = {
            "cat_policy": "tabr_ohe",  # mandatory
            "num_policy": "none",  # mandatory
            "use_float": True,
            "device": device,
            "num_nan_policy": self.config.get("num_nan_policy", "mean"),
            "cat_nan_policy": self.config.get("cat_nan_policy", "new"),
            "normalization": self.config.get("normalization", "standard"),
            "seed": seed,
            "batch_size": batch_size,
            "max_epoch": n_epochs,
            "time_to_fit_in_seconds": time_to_fit_in_seconds,
            "early_stopping_metric": self.early_stopping_metric,
            "save_path": self.save_path_,
            "config": config,
        }
        args = SimpleNamespace(**args)  # convert to an object with the keys as attributes
        info = {
            "task_type": task_type,
            "n_num_features": ds_parts["train"]["x_cont"].shape[1],
            "n_cat_features": ds_parts["train"]["x_cat"].shape[1],
        }

        # # set empty tensors to None
        # for tensors in ds_parts.values():
        #     for tens_name in ['x_cat', 'x_cont']:
        #         if tensors[tens_name].size == 0:
        #             tensors[tens_name] = None

        data = [
            {part: ds_parts[part][tens_name] for part in ["train", "val"]} for tens_name in ["x_cont", "x_cat", "y"]
        ]

        if info["n_num_features"] == 0:
            data[0] = None
        if info["n_cat_features"] == 0:
            data[1] = None

        data = tuple(data)

        assert num_emb_type == "plr"  # ohthers are currently not implemented

        # target standardization for regression is automatically done in "data_label_process"
        self.model_ = ModernNCAMethod(args=args, is_regression=task_type == "regression")
        self.model_.fit(data=data, info=info, train=True, config=config)

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        X = X.copy()
        tensors = dict()
        tensors["x_cat"] = X[self.cat_col_names_].to_numpy()
        tensors["x_cont"] = (
            self.num_prep_.transform(X.drop(columns=X[self.cat_col_names_]).to_numpy(dtype=np.float32))
            if self.has_num_cols
            else np.empty((len(X), 0), dtype=np.float32)
        )

        # for tens_name in ['x_cat', 'x_cont']:
        #     if tensors[tens_name].size == 0:
        #         tensors[tens_name] = None

        # generate dummy y
        if self.task_type_ == "regression":
            tensors["y"] = np.zeros(tensors["x_cat"].shape[0])
        else:
            tensors["y"] = np.zeros(tensors["x_cat"].shape[0], dtype=np.int32)

        data = [{"test": tensors[tens_name]} for tens_name in ["x_cont", "x_cat", "y"]]

        # eliminate empty tensors
        for i in range(2):
            if data[i]["test"].size == 0:
                data[i] = None
        data = tuple(data)

        # could use 'epoch-last' instead of 'best-val' for the last model
        y_pred = self.model_.predict(data=data, info=None, model_name="best-val")

        # print(f"predict: {X=}, {y_pred=}")

        return y_pred.cpu()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.predict_raw(X)
        if self.task_type_ == "regression":
            return y_pred.numpy()
        return y_pred.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = torch.softmax(self.predict_raw(X), dim=-1).numpy()
        if probas.shape[1] == 2:
            probas = probas[:, 1]
        return probas


class ModernNCAModel(AbstractModel):
    ag_key = "MNCA"
    ag_name = "ModernNCA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None
        self._feature_generator = None
        self._cat_features = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X_train, X_val, y_train, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)
        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        save_path = self.path + "/tmp_model"
        self.model = ModernNCAImplementation(
            n_threads=num_cpus,
            device=device,
            problem_type=self.problem_type,
            early_stopping_metric=self.stopping_metric,
            save_path=save_path,
            **hyp,
        )
        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=self._cat_features if self._cat_features is not None else [],
            time_to_fit_in_seconds=time_limit,
        )
        # Clean up tmp model folder
        shutil.rmtree(save_path, ignore_errors=True)

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])

        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X = X.copy()
            X[self._features_bool] = X[self._features_bool].astype("category")

        # Ordinal Encoding of cat features
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]

        return X

    def _set_default_params(self):
        default_params = dict(
            random_state=0,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 1 if torch.cuda.is_available() else 0
        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    # FIXME: Find a better estimate for memory usage of TabM. Currently borrowed from FASTAI estimate.
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        return 10 * get_approximate_df_mem_usage(X).sum()

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to force stopping at a specific epoch?
        return {"can_refit_full": False}
