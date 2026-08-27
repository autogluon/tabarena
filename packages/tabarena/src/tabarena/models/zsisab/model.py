"""
Zero-Shot ISAB (ZS-ISAB) model wrapper for AutoGluon / TabArena.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Safe AutoGluon import with graceful fallback
try:
    from autogluon.core.models import AbstractModel
    from autogluon.features import LabelEncoderFeatureGenerator
except ImportError:
    class AbstractModel:
        ag_key = "ZSISAB"
        ag_name = "ZS-ISAB"
        ag_priority = 105

        def __init__(self, problem_type="binary", **kwargs):
            self.problem_type = problem_type
            self.params = kwargs
            self._classes = np.array([0, 1])

        def _get_model_params(self):
            return self.params

        def _preprocess(self, X, **kwargs):
            return X.copy()

    class LabelEncoderFeatureGenerator:
        def __init__(self, verbosity=0):
            self.features_in = []
            self.mapping = {}

        def fit(self, X: pd.DataFrame):
            self.features_in = list(X.select_dtypes(include=["object", "category"]).columns)
            for col in self.features_in:
                cats = X[col].astype(str).unique()
                self.mapping[col] = {cat: float(i) for i, cat in enumerate(cats)}
            return self

        def transform(self, X: pd.DataFrame):
            X_out = pd.DataFrame(index=X.index)
            for col in self.features_in:
                X_out[col] = X[col].astype(str).map(self.mapping.get(col, {})).fillna(-1.0)
            return X_out


class ZSISABModel(AbstractModel):
    """
    AutoGluon model wrapper for Zero-Shot ISAB (ZS-ISAB).

    Paper: "Zero-Shot ISAB: Linear-Complexity Inducing Point Attention
            for Frozen Tabular Transformers"
    Author: Thanniru Sai Teja (https://github.com/iam-saiteja)
    Code:   https://github.com/iam-saiteja/Zero-Shot-TabPFN
    """

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self.model = None

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        return {"fold_fitting_strategy": "sequential_local"}

    def _preprocess(self, X, is_train: bool = False, **kwargs) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return X.astype(np.float32)

        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator is not None and getattr(self._feature_generator, "features_in", None):
            X = X.copy()
            encoded = self._feature_generator.transform(X=X)
            if isinstance(encoded, pd.DataFrame):
                for col in self._feature_generator.features_in:
                    if col in encoded.columns:
                        X[col] = encoded[col]
            else:
                X[self._feature_generator.features_in] = encoded

        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(
        self,
        X,
        y,
        num_cpus: int = 1,
        num_gpus: float = 0,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        import typing
        import torch.nn.modules.transformer
        torch.nn.modules.transformer.Optional = typing.Optional

        from tabpfn import TabPFNClassifier
        from zsisab.wrapper import inject_zsisab

        params = self._get_model_params()
        num_prototypes = params.get("num_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        n_ensemble = params.get("n_ensemble", 32)
        device = "cuda" if (num_gpus is not None and num_gpus > 0) else "cpu"

        # Inject ZS-ISAB online cross-attention into TabPFN
        inject_zsisab(num_prototypes=num_prototypes, chunk_size=chunk_size)

        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)

        X_processed = self._preprocess(X, is_train=True) if isinstance(X, pd.DataFrame) else X
        y_processed = y.to_numpy() if isinstance(y, pd.Series) else y
        self.model.fit(X_processed, y_processed, overwrite_warning=True)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X_processed = self._preprocess(X, is_train=False) if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_processed)

    def _predict(self, X, **kwargs) -> np.ndarray:
        X_processed = self._preprocess(X, is_train=False) if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_processed)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        # ZS-ISAB operates with a strictly bounded streaming footprint O(chunk_size * E)
        return 4 * 1024 ** 3  # 4 GB estimate
