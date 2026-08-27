"""Zero-Shot ISAB (ZS-ISAB) model wrapper for AutoGluon / TabArena."""
from __future__ import annotations

import types
from typing import TYPE_CHECKING

from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def inject_zsisab_to_instance(tabpfn_classifier, num_prototypes: int = 512, chunk_size: int = 16384):
    """Injects ZS-ISAB forward attention into only the specified TabPFNClassifier instance, avoiding global class mutation."""
    from zsisab.engine import get_zsisab_encoder_forward

    target_model = getattr(tabpfn_classifier, "model", tabpfn_classifier)
    if hasattr(target_model, "modules"):
        for module in target_model.modules():
            if module.__class__.__name__ == "TransformerEncoderLayer":
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                patched_fn = get_zsisab_encoder_forward(
                    module._original_forward,
                    num_prototypes=num_prototypes,
                    chunk_size=chunk_size,
                )
                module.forward = types.MethodType(patched_fn, module)


class ZSISABModel(AbstractTorchModel):
    """AutoGluon model wrapper for Zero-Shot ISAB (ZS-ISAB)."""

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass"]
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self.model = None

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        return {"fold_fitting_strategy": "sequential_local"}

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator is not None and getattr(self._feature_generator, "features_in", None):
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

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
        import typing

        import torch.nn.modules.transformer

        torch.nn.modules.transformer.Optional = typing.Optional

        from tabpfn import TabPFNClassifier

        params = self._get_model_params()
        num_prototypes = params.get("num_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        n_ensemble = params.get("n_ensemble", 32)
        device = "cuda" if (num_gpus is not None and num_gpus > 0) else "cpu"

        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)

        # Inject ZS-ISAB into this specific instance only
        inject_zsisab_to_instance(self.model, num_prototypes=num_prototypes, chunk_size=chunk_size)

        X_processed = self.preprocess(X, y=y, is_train=True)
        self.model.fit(X_processed, y.to_numpy() if hasattr(y, "to_numpy") else y, overwrite_warning=True)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self.preprocess(X, is_train=False)
        return self.model.predict_proba(X_processed)

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self.preprocess(X, is_train=False)
        return self.model.predict(X_processed)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return 4 * 1024 ** 3  # 4 GB estimate
