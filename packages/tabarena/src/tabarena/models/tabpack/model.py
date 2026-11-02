"""AutoGluon model wrapper for TabPack, built on the official ``tabpackage`` scikit-learn API."""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)

#: AutoGluon scorer name to the TabPack metric used for each member's checkpoint selection and early stopping.
_METRIC_MAP = {
    "roc_auc": "roc_auc",
    "log_loss": "log_loss",
    "accuracy": "accuracy",
    "root_mean_squared_error": "mse",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "r2": "r2",
}
_CLASSIFICATION_METRICS = ("roc_auc", "log_loss", "accuracy")
_MISSING_CATEGORY = "__missing__"
_INDICATOR_PREFIX = "__tabpack_missing__"


def _drop_training_checkpoints(estimator) -> None:
    """Free the per-member training checkpoints a fitted TabPack estimator keeps but never predicts with.

    Developer fix. A fitted ``TabPackClassifier`` / ``TabPackRegressor`` (tabpackage 0.0.1) keeps
    ``tabpack_``, whose checkpoint store holds the best state of every member, padded to the pack's
    largest shape, for ``refit`` and inspection; prediction uses only the ensemble's ``_predictor``. On
    a 20-feature table with the default 64 members the store took a fitted model's pickle from 11 MB to
    159 MB, and every bagged fold model is saved. The wrapper never refits, so it clears the store's
    state dicts after the fit; predictions are unchanged. The library could offer this itself (an
    option to keep no training checkpoints, or a public method that drops them); to be asked upstream
    at yandex-research/tabpack together with the time budget.
    """
    estimator.tabpack_._checkpoints.state_dicts.clear()


class TabPackModel(AbstractTorchModel):
    """TabPack trains many MLPs with different hyperparameters in parallel and ensembles them greedily.

    One fit samples 64 MLPs (per-member depth, dropout, embedding and optimizer hyperparameters),
    trains them as one model pack with Muon + AdamW and early stopping on ``X_val``, and builds a
    greedy ensemble of their checkpoints on ``X_val`` while they train. Both steps use the validation
    rows TabArena passes to ``_fit``; the library never splits the training data itself.

    Not supported yet: ``time_limit`` and refit on train+val. ``tabpackage`` 0.0.1 has no time budget
    or stop hook, so a fit runs until its members and the ensemble stop early, and a bag whose first
    folds overrun their share fails with ``TimeLimitExceeded``; the library should take a
    ``time_limit`` or a per-epoch stop callback in ``tabpackage.train`` and ``fit`` (upstream ask to
    be filed at yandex-research/tabpack). The library's ``refit`` would put each fold's out-of-fold
    rows into the model that predicts them, so it is not used.

    Paper: TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning
    Authors: Yury Gorishniy, Akim Kotelnikov, Ivan Rubachev, Artem Babenko
    Codebase: https://github.com/yandex-research/tabpack
    License: Apache-2.0
    """

    ag_key = "TA-TABPACK"
    ag_name = "TA-TabPack"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    #: One fit trains all members at once and needs the whole GPU (the paper reports more than 12 GB
    #: for many members and features), so the bagged folds fit one after another.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"fold_fitting_strategy": "sequential_local"}
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabpackage", "tabpackage.sklearn", "sklearn.preprocessing")
    #: Knobs that make the warm-up's dummy fit and the smoke test cheap: two members, one epoch.
    cheap_hyperparameters: ClassVar[dict] = {"n_models": 2, "max_epochs": 1}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._num_features: list[str] | None = None
        self._cat_features: list[str] | None = None
        self._indicator_features: list[str] | None = None
        self._quantile_transformer = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        import torch
        from tabpackage.sklearn import TabPackClassifier, TabPackRegressor

        device = self._resolve_fit_device(num_gpus=num_gpus)
        torch.set_num_threads(num_cpus)  # restored after the fit by the reset_torch_threads class tag
        if time_limit is not None:
            logger.log(30, f"\tWarning: TabPack has no time budget; time_limit={time_limit:.1f}s is not enforced.")

        if X_val is None:
            # The library needs held-out rows for early stopping and for building the ensemble.
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.33, random_state=0
            )

        hps = self._get_model_params()
        criterion_metric = hps.pop("eval_criterion_metric", "auto")
        ensemble_metric = hps.pop("ensemble_metric", "auto")

        X = self.preprocess(X, is_train=True)
        X_val = self.preprocess(X_val)
        y_train, y_extra = self._encode_target(y, is_train=True), self._encode_target(y_val)
        is_classification = self.problem_type != "regression"
        if is_classification:
            # A class absent from this fold's training rows has no output column, and the library
            # rejects validation labels it has not seen; such rows cannot inform this fit.
            seen = np.isin(y_extra, np.unique(y_train))
            if not seen.all():
                X_val, y_extra = X_val[seen], y_extra[seen]

        if criterion_metric == "auto":
            criterion_metric = self._resolve_criterion_metric(y_extra)
        if ensemble_metric == "auto":
            # The authors' TabArena setting: the training loss as the greedy ensembling score.
            ensemble_metric = "log_loss" if is_classification else "mse"
        hps.setdefault("ensembles", {"default": {"type": "greedy", "algorithm_metric": ensemble_metric}})

        model_cls = TabPackClassifier if is_classification else TabPackRegressor
        self.model = model_cls(
            eval_criterion_metric=criterion_metric,
            device=device,
            print_progress=verbosity >= 3,
            show_progress_bar=False,
            **hps,
        )
        self.model.fit(X, y_train, extra_set=(X_val, y_extra))
        _drop_training_checkpoints(self.model)

    def _resolve_criterion_metric(self, y_val: np.ndarray) -> str:
        """The TabPack metric closest to the stopping metric, falling back to the training loss."""
        is_classification = self.problem_type != "regression"
        metric = _METRIC_MAP.get(self.stopping_metric.name)
        if metric is None or (metric in _CLASSIFICATION_METRICS) != is_classification:
            return "log_loss" if is_classification else "mse"
        if metric == "roc_auc" and (self.problem_type != "binary" or len(np.unique(y_val)) < 2):
            return "log_loss"
        return metric

    def _encode_target(self, y: pd.Series, is_train: bool = False) -> np.ndarray:
        """Class indices as int64, or a regression target standardized with the training statistics."""
        if self.problem_type != "regression":
            return y.to_numpy(dtype=np.int64)
        if is_train:
            self._y_mean = float(y.mean())
            std = float(y.std())
            self._y_std = std if np.isfinite(std) and std > 0 else 1.0
        return ((y.to_numpy(dtype=np.float64) - self._y_mean) / self._y_std).astype(np.float32)

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Shape the frame into ``tabpackage``'s input contract.

        Numerical features get the quantile transform of the authors' examples (normal output,
        ``n_quantiles = clip(n_train // 30, 10, 1000)``, a ``1e-5`` jitter to break ties), fit on the
        training rows only; a feature with missing training values then fills them with 0 and gets a
        categorical missing indicator. Categorical features and AutoGluon's bool features (the paper
        treats binary features as categorical) become string columns with missing values as their own
        category; the library ordinal-encodes them and maps unseen values to an extra index.
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            from sklearn.preprocessing import QuantileTransformer

            cat_features = self._feature_metadata.get_features(valid_raw_types=["category"])
            cat_features += self._feature_metadata.get_features(required_special_types=["bool"])
            self._cat_features = list(dict.fromkeys(cat_features))
            self._num_features = [f for f in X.columns if f not in self._cat_features]
            if self._num_features:
                X_num = self._numeric_frame(X)
                self._indicator_features = [f for f in self._num_features if X_num[f].isna().any()]
                jitter = np.random.default_rng(0).normal(0.0, 1e-5, X_num.shape).astype(np.float32)
                self._quantile_transformer = QuantileTransformer(
                    n_quantiles=max(min(len(X_num) // 30, 1000), 10),
                    output_distribution="normal",
                    subsample=10**9,
                    random_state=0,
                ).fit(X_num.to_numpy() + jitter)

        columns = {}
        if self._num_features:
            X_num = self._numeric_frame(X)
            transformed = self._quantile_transformer.transform(X_num.to_numpy()).astype(np.float32)
            for i, feature in enumerate(self._num_features):
                columns[feature] = np.nan_to_num(transformed[:, i], nan=0.0)
            for feature in self._indicator_features:
                columns[f"{_INDICATOR_PREFIX}{feature}"] = pd.array(
                    np.where(X_num[feature].isna(), "1", "0"), dtype="string"
                )
        for feature in self._cat_features:
            values = X[feature].astype(object)
            columns[feature] = values.where(values.notna(), _MISSING_CATEGORY).astype(str).astype("string")
        return pd.DataFrame(columns, index=X.index)

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """The numerical features as float32 with infinities marked missing."""
        return X[self._num_features].astype(np.float32).replace([np.inf, -np.inf], np.nan)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            return self.model.predict(X) * self._y_std + self._y_mean
        proba = self.model.predict_proba(X)
        if proba.shape[1] != self.num_classes:
            full = np.zeros((len(X), self.num_classes), dtype=proba.dtype)
            full[:, self.model.classes_] = proba
            proba = full
        return self._convert_proba_to_unified_form(proba)

    def get_device(self) -> str:
        return self.model._fitted_device.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model._fitted_device = device
        if self.model._predictor is not None:
            self.model._predictor.to(device)

    def _set_default_params(self):
        default_params = {
            # Per-member checkpoint selection and early stopping: the stopping metric.
            "eval_criterion_metric": "auto",
            # Greedy ensembling score: the training loss (log loss / MSE), as the authors ran TabArena.
            "ensemble_metric": "auto",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_stopping_metric(self):
        return self.eval_metric

    @classmethod
    def _class_tags(cls):
        return {"reset_torch_threads": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
