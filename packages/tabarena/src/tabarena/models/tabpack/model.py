"""AutoGluon model wrapper for TabPack, built on the official ``tabpackage`` scikit-learn API."""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.core.utils.exceptions import TimeLimitExceeded
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


def _category_strings(values: pd.Series) -> pd.Series:
    """A categorical feature's values as strings, with missing values as their own category."""
    values = values.astype(object)
    return values.where(values.notna(), _MISSING_CATEGORY).astype(str)


def _max_pack_size(X: pd.DataFrame, hps: dict, memory_budget: float) -> int | None:
    """How many models to train at once when the first layers of the whole pack would exceed ``memory_budget`` bytes.

    Developer fix. ``tabpackage`` trains all models of the pack at once, and each model's first layer takes
    ``n_numerical x d_embedding + n_categories`` inputs (numerical embeddings up to 32 wide by default, one
    input per category) and ``d_block`` (384) outputs, held as float32 weights, gradients and two AdamW
    moments. On BeyondArena's ``lung_cancer`` (197 rows, 12,600 features) that is 2.5 GB per model, 159 GB
    for 64, and the first backward pass ran out of memory on a 96 GB card. Above the budget, the library's
    ``max_pack_size`` trains as many models at once as fit and replaces them as they finish. The pack's
    stacked first-layer weights also stay below 2**31 elements: at 3.2e9 (38 models on a 6,771-feature table)
    the fit stopped with a CUDA illegal memory access, the mark of a kernel with 32-bit indexing. ``X`` is the
    preprocessed frame (categorical features as string columns); ``None`` means the whole pack fits.
    """
    n_models = hps.get("n_models") or 64
    is_categorical = [str(dtype) == "string" for dtype in X.dtypes]
    n_categories = sum(X[c].nunique() for c, cat in zip(X.columns, is_categorical, strict=True) if cat)
    n_numerical = len(is_categorical) - sum(is_categorical)
    d_embedding = hps.get("numerical_embeddings_d")
    if hps.get("numerical_embeddings") == "none":
        d_embedding = 1
    elif not isinstance(d_embedding, int):
        d_embedding = 32  # the top of the library's default range (8 to 32); the pack pads to its widest model
    d_block = hps.get("d_block") if isinstance(hps.get("d_block"), int) else 384
    weights_per_model = d_block * (n_numerical * d_embedding + n_categories)
    bytes_per_model = 16 * weights_per_model
    n_fit = min(int(memory_budget // bytes_per_model), (2**31 - 1) // weights_per_model)
    if n_fit >= n_models:
        return None
    logger.log(
        20,
        f"\tThe first layers of {n_models} models need ~{n_models * bytes_per_model / 1e9:.0f} GB: "
        f"training {max(n_fit, 1)} at once (max_pack_size).",
    )
    return max(n_fit, 1)


def _drop_training_checkpoints(estimator) -> None:
    """Free the per-member training checkpoints a fitted TabPack estimator keeps but never predicts with.

    Developer fix. A fitted ``TabPackClassifier`` / ``TabPackRegressor`` (tabpackage 0.0.2.dev0) keeps
    ``tabpack_``, whose checkpoint store holds the best state of every member, padded to the pack's
    largest shape, for ``refit`` and inspection; prediction uses only the ensemble's ``_predictor``. On
    a 20-feature table with the default 64 members the store took a fitted model's pickle from 11 MB to
    159 MB, and every bagged fold model is saved. The library has no option to skip or drop the store,
    and the wrapper never refits, so it clears the store's state dicts after the fit; predictions are
    unchanged.
    """
    estimator.tabpack_._checkpoints.state_dicts.clear()


class TabPackModel(AbstractTorchModel):
    """TabPack trains many MLPs with different hyperparameters in parallel and ensembles them greedily.

    One fit samples 64 MLPs (per-member depth, dropout, embedding and optimizer hyperparameters),
    trains them as one model pack with Muon + AdamW and early stopping on ``X_val``, and builds a
    greedy ensemble of their checkpoints on ``X_val`` while they train. Both steps use the validation
    rows TabArena passes to ``_fit``; the library never splits the training data itself.

    ``time_limit`` becomes the library's ``training_loop_time_budget``: what is left of it after the
    preprocessing, less 5% headroom for building the predictor. The library checks the budget after
    each epoch, so a fit can overrun it by up to one epoch, and the online greedy ensemble keeps the
    members it selected so far. From ``epoch_probe_min_rows`` training rows on, where an epoch can take
    minutes, a short probe fit estimates one epoch and the budget reserves it (``_estimate_epoch_seconds``). Refit on train+val is not supported: the library's ``refit`` would put
    each fold's out-of-fold rows into the model that predicts them.

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
        self._kept_categories: dict[str, list[str]] = {}
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
        start_time = time.time()
        import torch
        from tabpackage.sklearn import TabPackClassifier, TabPackRegressor

        device = self._resolve_fit_device(num_gpus=num_gpus)
        torch.set_num_threads(num_cpus)  # restored after the fit by the reset_torch_threads class tag

        if X_val is None:
            # The library needs held-out rows for early stopping and for building the ensemble.
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.33, random_state=0
            )

        hps = self._get_model_params()
        criterion_metric = hps.pop("eval_criterion_metric", "auto")
        ensemble_metric = hps.pop("ensemble_metric", "auto")
        max_total_categories = hps.pop("max_total_categories", None)
        max_pack_memory_fraction = hps.pop("max_pack_memory_fraction", None)
        epoch_probe_min_rows = hps.pop("epoch_probe_min_rows", None)

        X = self.preprocess(X, is_train=True, max_total_categories=max_total_categories)
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
        if max_pack_memory_fraction and "max_pack_size" not in hps and str(device).startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
            max_pack_size = _max_pack_size(X, hps, max_pack_memory_fraction * total_memory)
            if max_pack_size is not None:
                hps["max_pack_size"] = max_pack_size

        model_cls = TabPackClassifier if is_classification else TabPackRegressor
        model_kwargs = {
            "eval_criterion_metric": criterion_metric,
            "device": device,
            "print_progress": verbosity >= 3,
            "show_progress_bar": False,
            **hps,
        }
        if time_limit is not None:
            epoch_seconds = 0.0
            if epoch_probe_min_rows is not None and len(X) >= epoch_probe_min_rows:
                epoch_seconds = self._estimate_epoch_seconds(model_cls, model_kwargs, X, y_train, X_val, y_extra)
            budget = 0.95 * (time_limit - (time.time() - start_time)) - epoch_seconds
            if budget <= 0:
                raise TimeLimitExceeded
            model_kwargs["training_loop_time_budget"] = budget

        self.model = model_cls(**model_kwargs)
        self.model.fit(X, y_train, extra_set=(X_val, y_extra))
        _drop_training_checkpoints(self.model)

    def _estimate_epoch_seconds(
        self, model_cls, model_kwargs: dict, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
    ) -> float:
        """The seconds one training epoch of the full pack takes on these rows, from a short probe fit.

        Developer fix. ``tabpackage`` checks ``training_loop_time_budget`` only after a complete epoch, and on large
        tables an epoch takes minutes (about 10 minutes for 64 models on 875k rows x 988 features), so a fit can
        overrun its budget by up to one epoch. On BeyondArena's ``maps_router_eta_1m`` each of the eight folds then
        took slightly more than its share and AutoGluon stopped the bag on its time limit. The probe trains the same
        pack for one epoch on about 16 batches of training rows and the same share of validation rows, and scales the
        library's training-loop time by the inverse of that share; ``_fit`` reserves the estimate from the budget.
        """
        import torch

        n_train = len(X)
        # The library's batch size for this many rows (512 below 131,072, then 1024); the probe must use the same.
        batch_size = model_kwargs.get("batch_size") or (512 if n_train < 131_072 else 1024)
        share = min(1.0, 16 * batch_size / n_train)
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(n_train, size=int(np.ceil(share * n_train)), replace=False))
        val_idx = np.sort(rng.choice(len(X_val), size=max(1, int(np.ceil(share * len(X_val)))), replace=False))
        y_probe, X_val_probe, y_val_probe = y[train_idx], X_val.iloc[val_idx], y_val[val_idx]
        if self.problem_type != "regression":
            seen = np.isin(y_val_probe, np.unique(y_probe))
            X_val_probe, y_val_probe = X_val_probe[seen], y_val_probe[seen]
        if len(y_val_probe) == 0:
            return 0.0
        probe = model_cls(**{**model_kwargs, "max_epochs": 1, "batch_size": batch_size})
        probe.fit(X.iloc[train_idx], y_probe, extra_set=(X_val_probe, y_val_probe))
        seconds = probe.tabpack_.get_training_loop_time() / share
        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.log(20, f"\tOne epoch takes ~{seconds:.0f}s (probe on {len(train_idx)} rows); reserved from the budget.")
        return seconds

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

    def _preprocess(
        self, X: pd.DataFrame, is_train: bool = False, max_total_categories: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """Shape the frame into ``tabpackage``'s input contract.

        Numerical features get the quantile transform of the authors' examples (normal output,
        ``n_quantiles = clip(n_train // 30, 10, 1000)``, a ``1e-5`` jitter to break ties), fit on the
        training rows only; a feature with missing training values then fills them with 0 and gets a
        categorical missing indicator. Categorical features and AutoGluon's bool features (the paper
        treats binary features as categorical) become string columns with missing values as their own
        category; the library ordinal-encodes them and maps unseen values to an extra index. Above
        ``max_total_categories`` the largest categorical features keep only their most frequent training
        categories (see ``_select_categories``).
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            from sklearn.preprocessing import QuantileTransformer

            cat_features = self._feature_metadata.get_features(valid_raw_types=["category"])
            cat_features += self._feature_metadata.get_features(required_special_types=["bool"])
            self._cat_features = list(dict.fromkeys(cat_features))
            self._kept_categories = self._select_categories(X, max_total_categories)
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
            values = _category_strings(X[feature])
            if feature in self._kept_categories:
                values = values.where(values.isin(self._kept_categories[feature]), _MISSING_CATEGORY)
            columns[feature] = values.astype("string")
        return pd.DataFrame(columns, index=X.index)

    def _select_categories(self, X: pd.DataFrame, max_total_categories: int | None) -> dict[str, list[str]]:
        """The categories each trimmed feature keeps, when the training rows hold more than ``max_total_categories``.

        Developer fix. ``tabpackage`` one-hot encodes every training category into each member's first
        layer, so its memory grows with the total category count: members x batch x categories for the
        ``int64`` one-hot, plus a ``d_block``-wide weight row per category in each of the 64 members. On
        BeyondArena's ``ieee_fraud_detection`` (an ID-like ``uid`` with ~55k categories per fold) one step
        asked for 28 GB more on a 96 GB card. Above the budget, the largest features keep their most frequent
        training categories under one shared cap, the largest that keeps the total within the budget, and
        their other values join the missing category, as AutoGluon's ``maximum_num_cat`` does with rare
        values. Features below the cap stay whole.
        """
        if not max_total_categories or not self._cat_features:
            return {}
        counts = {f: _category_strings(X[f]).value_counts() for f in self._cat_features}
        sizes = np.array([len(c) for c in counts.values()])
        if sizes.sum() <= max_total_categories:
            return {}
        low, high = 1, int(sizes.max())
        while low < high:  # the largest per-feature cap whose total stays within the budget
            mid = (low + high + 1) // 2
            low, high = (mid, high) if np.minimum(sizes, mid).sum() <= max_total_categories else (low, mid - 1)
        kept = {}
        for feature, feature_counts in counts.items():
            if len(feature_counts) > low:
                frequent = feature_counts.index[feature_counts.index != _MISSING_CATEGORY][: low - 1]
                kept[feature] = [_MISSING_CATEGORY, *frequent]
        logger.log(
            20,
            f"\t{sizes.sum()} categories exceed max_total_categories={max_total_categories}: "
            f"{len(kept)} feature(s) keep their {low - 1} most frequent categories.",
        )
        return kept

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
            # Memory cap on the one-hot categoricals (see `_select_categories`): about 50 GB at the largest
            # batch, and above every TabArena-v0.1 table (kddcup09_appetency: ~25k categories per fold).
            "max_total_categories": 32768,
            # Wide tables (see `_max_pack_size`): the pack's first layers may take this share of the GPU's
            # memory, and stay below 2**31 weights. Together they engage above ~2,700 numerical features,
            # above every TabArena-v0.1 table.
            "max_pack_memory_fraction": 0.5,
            # Large tables (see `_estimate_epoch_seconds`): reserve one probed epoch from the time budget. Above
            # every TabArena-v0.1 fold (at most 87,500 training rows).
            "epoch_probe_min_rows": 100_000,
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
