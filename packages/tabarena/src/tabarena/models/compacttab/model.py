"""AutoGluon model wrappers for CompactTab and CompactTab-Hybrid.

CompactTab is an open-source parameter-efficient tabular deep learning architecture
featuring Periodic-Quantile Piecewise-Linear Embeddings (PQ-PLE), Multiplicative
Feature Interaction Crossing, and BatchEnsemble multi-head rank-1 weight modulation.

CompactTab-Hybrid is a Tri-Engine ensemble combining CompactTab's continuous representations
with asymmetric leaf-wise (LightGBM) and symmetric oblivious (CatBoost) tree splits via
cross-validated convex SLSQP optimization over the probability simplex.

Repository: https://github.com/DakshBhalala/TBDML
License: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# Monkey-patch get_per_process_memory_fraction for PyTorch on Windows if absent
if torch.cuda.is_available() and not hasattr(torch.cuda, "get_per_process_memory_fraction"):
    torch.cuda.get_per_process_memory_fraction = lambda d=None: 1.0  # type: ignore

logger = logging.getLogger(__name__)


class CompactTabModel(AbstractModel):
    """AutoGluon model plugin for CompactTab deep learning architecture.

    Features Periodic Quantile Piecewise-Linear Embeddings, Multiplicative Feature Crossing,
    and BatchEnsemble multi-head weight generation.
    """

    ag_key = "TA-COMPACTTAB"
    ag_name = "TA-CompactTab"
    ag_priority = 60
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS, REGRESSION]

    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._estimator: Any = None

    def _get_default_auxiliary_params(self) -> dict[str, Any]:
        default_aux = super()._get_default_auxiliary_params()
        default_aux.update(
            {
                "valid_raw_types": ["int", "float", "category", "object"],
                "ignored_type_group_raw": [],
            }
        )
        return default_aux

    def _set_default_params(self) -> None:
        default_params = {
            "preset": "small",
            "max_epochs": 40,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
            "dropout": 0.15,
            "mixup_alpha": 0.1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs: Any,
    ) -> None:
        from compacttab.models.compacttab import CompactTabClassifier, CompactTabRegressor

        params = self._get_model_params()
        device = "cuda" if (num_gpus is not None and num_gpus > 0 and torch.cuda.is_available()) else "cpu"
        random_state = params.get("random_state", 42)

        if self.problem_type in (BINARY, MULTICLASS):
            self._estimator = CompactTabClassifier(
                preset=params.get("preset", "small"),
                hidden_dim=params.get("hidden_dim"),
                num_layers=params.get("num_layers"),
                num_heads=params.get("num_heads"),
                num_bins=params.get("num_bins"),
                embed_dim=params.get("embed_dim"),
                dropout=params.get("dropout"),
                learning_rate=params.get("learning_rate", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                max_epochs=params.get("max_epochs", 40),
                patience=params.get("patience", 10),
                batch_size=params.get("batch_size", 128),
                mixup_alpha=params.get("mixup_alpha", 0.1),
                device=device,
                random_state=random_state,
            )
        elif self.problem_type == REGRESSION:
            self._estimator = CompactTabRegressor(
                preset=params.get("preset", "small"),
                hidden_dim=params.get("hidden_dim"),
                num_layers=params.get("num_layers"),
                num_heads=params.get("num_heads"),
                num_bins=params.get("num_bins"),
                embed_dim=params.get("embed_dim"),
                dropout=params.get("dropout"),
                learning_rate=params.get("learning_rate", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                max_epochs=params.get("max_epochs", 40),
                patience=params.get("patience", 10),
                batch_size=params.get("batch_size", 128),
                mixup_alpha=params.get("mixup_alpha", 0.1),
                device=device,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        self._estimator.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("CompactTabModel is not fitted.")
        if self.problem_type == REGRESSION:
            return self._predict(X, **kwargs)
        return self._estimator.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("CompactTabModel is not fitted.")
        return self._estimator.predict(X)


class CompactTabHybridModel(AbstractModel):
    """AutoGluon model plugin for CompactTab-Hybrid Tri-Engine architecture.

    Ensembles CompactTab neural representations with LightGBM and CatBoost tree splits
    via out-of-fold convex SLSQP optimization over the probability simplex.
    """

    ag_key = "TA-COMPACTTAB-HYBRID"
    ag_name = "TA-CompactTabHybrid"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS, REGRESSION]

    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._estimator: Any = None

    def _get_default_auxiliary_params(self) -> dict[str, Any]:
        default_aux = super()._get_default_auxiliary_params()
        default_aux.update(
            {
                "valid_raw_types": ["int", "float", "category", "object"],
                "ignored_type_group_raw": [],
            }
        )
        return default_aux

    def _set_default_params(self) -> None:
        default_params = {
            "preset": "small",
            "cv_folds": 5,
            "max_epochs": 40,
            "batch_size": 128,
            "mixup_alpha": 0.1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs: Any,
    ) -> None:
        from compacttab.models.hybrid import CompactTabHybridClassifier, CompactTabHybridRegressor

        params = self._get_model_params()
        device = "cuda" if (num_gpus is not None and num_gpus > 0 and torch.cuda.is_available()) else "cpu"
        random_state = params.get("random_state", 42)

        if self.problem_type in (BINARY, MULTICLASS):
            self._estimator = CompactTabHybridClassifier(
                preset=params.get("preset", "small"),
                cv_folds=params.get("cv_folds", 5),
                mixup_alpha=params.get("mixup_alpha", 0.1),
                max_epochs=params.get("max_epochs", 40),
                batch_size=params.get("batch_size", 128),
                device=device,
                random_state=random_state,
            )
        elif self.problem_type == REGRESSION:
            self._estimator = CompactTabHybridRegressor(
                preset=params.get("preset", "small"),
                cv_folds=params.get("cv_folds", 5),
                mixup_alpha=params.get("mixup_alpha", 0.1),
                max_epochs=params.get("max_epochs", 40),
                batch_size=params.get("batch_size", 128),
                device=device,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        self._estimator.fit(X=X, y=y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("CompactTabHybridModel is not fitted.")
        if self.problem_type == REGRESSION:
            return self._predict(X, **kwargs)
        return self._estimator.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("CompactTabHybridModel is not fitted.")
        return self._estimator.predict(X)
