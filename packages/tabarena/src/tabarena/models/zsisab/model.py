"""Zero-Shot ISAB (ZS-ISAB) model wrapper for AutoGluon / TabArena."""
from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


def apply_zsisab_shims():
    """Apply compatibility shims for PyTorch Optional and Scikit-Learn force_all_finite."""
    import typing
    import torch.nn.modules.transformer
    torch.nn.modules.transformer.Optional = typing.Optional

    import sklearn.utils.validation as val
    if not hasattr(val, "_zsisab_shimmed"):
        _orig_xy, _orig_arr = val.check_X_y, val.check_array
        val.check_X_y = lambda X, y, **kw: _orig_xy(X, y, **{('ensure_all_finite' if k == 'force_all_finite' else k): v for k, v in kw.items()})
        val.check_array = lambda X, **kw: _orig_arr(X, **{('ensure_all_finite' if k == 'force_all_finite' else k): v for k, v in kw.items()})
        val._zsisab_shimmed = True


def inject_zsisab_to_instance(tabpfn_model, num_prototypes: int = 512, chunk_size: int = 16384):
    """Injects ZS-ISAB forward attention into TabPFN's TransformerEncoderLayer."""
    apply_zsisab_shims()
    from zsisab.wrapper import inject_zsisab
    inject_zsisab(num_prototypes=num_prototypes, chunk_size=chunk_size)



class ZSISABModel(AbstractTorchModel):
    """AutoGluon model wrapper for Zero-Shot ISAB (ZS-ISAB)."""

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._discretizer = None
        self._bin_means = None
        self.model = None

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        return {
            "fold_fitting_strategy": "sequential_local",
            "raise_on_model_failure": False,
        }

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
            if X.shape[1] > 100:
                from sklearn.decomposition import TruncatedSVD
                n_comp = min(100, max(1, X.shape[0] - 1), X.shape[1])
                self._dim_reducer = TruncatedSVD(n_components=n_comp, random_state=42)
            else:
                self._dim_reducer = None

        if self._feature_generator is not None and getattr(self._feature_generator, "features_in", None):
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        if hasattr(self, "_dim_reducer") and self._dim_reducer is not None:
            X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0)
            if is_train:
                X_red = self._dim_reducer.fit_transform(X_arr)
            else:
                X_red = self._dim_reducer.transform(X_arr)
            X = pd.DataFrame(X_red, columns=[f"comp_{i}" for i in range(X_red.shape[1])], index=X.index)

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        apply_zsisab_shims()

        params = self._get_model_params()
        num_prototypes = params.get("num_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        n_ensemble = params.get("n_ensemble", 8)
        device = "cuda" if (num_gpus is not None and num_gpus > 0) else "cpu"

        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)

        # Inject ZS-ISAB into this specific instance only
        inject_zsisab_to_instance(self.model, num_prototypes=num_prototypes, chunk_size=chunk_size)

        X_processed = self.preprocess(X, y=y, is_train=True)

        if self.problem_type == "regression":
            from sklearn.preprocessing import KBinsDiscretizer
            y_arr = y.to_numpy(dtype=np.float32) if hasattr(y, "to_numpy") else np.array(y, dtype=np.float32)
            n_bins = min(10, max(2, len(np.unique(y_arr))))
            self._discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
            y_binned = self._discretizer.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(np.int64)
            self._bin_means = np.array(
                [y_arr[y_binned == k].mean() if np.any(y_binned == k) else 0.0 for k in range(n_bins)],
                dtype=np.float32,
            )
            y_fit = y_binned
        else:
            self._discretizer = None
            self._bin_means = None
            y_fit = y.to_numpy() if hasattr(y, "to_numpy") else y

        self.model.fit(X_processed, y_fit, overwrite_warning=True)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.problem_type == "regression":
            return self._predict(X, **kwargs)
        X_processed = self.preprocess(X, is_train=False)
        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(X_processed)
        else:
            preds = self.model.predict(X_processed)

        if self.problem_type == "binary":
            if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] >= 2:
                return preds[:, 1]
            elif isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 1:
                return preds.ravel()
            return preds
        return preds

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self.preprocess(X, is_train=False)
        if self.problem_type == "regression":
            if hasattr(self, "_discretizer") and self._discretizer is not None:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_processed)
                    if self._bin_means is not None:
                        k = min(probs.shape[1], len(self._bin_means))
                        return np.dot(probs[:, :k], self._bin_means[:k]).ravel()
                preds = self.model.predict(X_processed)
                if self._bin_means is not None:
                    return self._bin_means[np.clip(preds.astype(int), 0, len(self._bin_means) - 1)].ravel()
            preds = self.model.predict(X_processed)
            return preds.ravel() if isinstance(preds, np.ndarray) else np.array(preds).ravel()
        return super()._predict(X, **kwargs)

    def score_with_y_pred_proba(self, y, y_pred_proba, metric=None, **kwargs) -> float:
        try:
            return super().score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba, metric=metric, **kwargs)
        except ValueError:
            return 0.5

    def get_device(self) -> str:
        if self.model is not None and hasattr(self.model, "device"):
            return str(self.model.device)
        return "cpu"

    def _set_device(self, device: str):
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return int(min(1024 ** 3, X.memory_usage(deep=True).sum() * 10 + 200 * 1024 ** 2))
