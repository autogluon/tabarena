from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


class FittedSupport:
    def __init__(self, xs, ys, task, device, network):
        self.xs, self.ys, self.task, self.device, self.network = xs, ys, task, device, network

    def __getstate__(self):
        return {**self.__dict__, "network": None}

    def get_network(self):
        from .network import load_network

        if self.network is None:
            self.network = load_network(self.task, self.device)
        return self.network


class KumoCellRidgeModel(AbstractTorchModel):
    """Kumo Small with four pretrained row blocks and per-table ridge regression.

    CellRidge-1024 uses the intermediate cell representations from a single Kumo
    Small estimator. The four added blocks act within each row. Only labeled
    support rows enter the affine ridge fit. Weights and development history:
    https://github.com/1587causalai/kumo-small-cellridge-1024
    """

    ag_key = "TA-KUMO-CELLRIDGE-1024"
    ag_name = "TA-Kumo-CellRidge-1024"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list] = [BINARY, MULTICLASS, REGRESSION]
    default_max_context_size: ClassVar[int] = 200_000
    default_num_gpus = 1
    minimum_num_gpus = 0
    default_resources_physical_cores_only = True
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    warmup_modules: ClassVar[tuple] = ("sdm", "sdm.models.kumo.tabular")

    def _set_default_params(self):
        super()._set_default_params()
        self._set_default_param_value("max_context_size", self.default_max_context_size)

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        import torch

        from .network import load_network

        self._fit_device = "cuda" if num_gpus > 0 else "cpu"
        self._threads = max(1, int(num_cpus or 1))
        self._task = "regression" if self.problem_type == REGRESSION else "classification"
        self._classes = 0 if self.problem_type == REGRESSION else self.num_classes
        if self._classes > 10:
            raise ValueError("The 1024 checkpoint supports up to ten classes in one pass")
        limit = self._get_model_params()["max_context_size"]
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
            raise ValueError("max_context_size must be a positive integer")
        network = load_network(self._task, self._fit_device)
        X = self.preprocess(X)
        # Draw only from this training fold; guarantee representation of each observed class.
        rng = np.random.default_rng(self.random_seed or 0)
        order = rng.permutation(len(X))
        labels = np.asarray(y)
        # Preserve the released 512-row ordering so the old support is a prefix.
        if self._classes and len(order) > min(512, limit):
            _, positions = np.unique(labels[order], return_index=True)
            keep = np.zeros(len(order), dtype=bool)
            keep[positions] = True
            order = np.concatenate((order[keep], order[~keep]))
        order = order[:limit]
        X = X.iloc[order]
        self._categories = {}
        for col in X:
            typ = X[col].dtype
            if (
                isinstance(typ, pd.CategoricalDtype)
                or not pd.api.types.is_numeric_dtype(typ)
                or pd.api.types.is_bool_dtype(typ)
            ):
                self._categories[col] = pd.unique(X[col].dropna()).tolist()
        self._nominal = [col in self._categories for col in X]
        self._columns = list(X)
        ys = labels[order].copy()
        self._observed = None
        if self._classes:
            self._observed, ys = np.unique(ys, return_inverse=True)
        self.model = FittedSupport(
            torch.from_numpy(self._encode(X)), torch.as_tensor(ys), self._task, self._fit_device, network
        )

    def _encode(self, X):
        matrix = np.empty((len(X), len(self._columns)), dtype=np.float32)
        for i, col in enumerate(self._columns):
            if col in self._categories:
                codes = pd.Categorical(X[col], categories=self._categories[col]).codes
                matrix[:, i] = np.where(codes < 0, np.nan, codes)
            else:
                matrix[:, i] = X[col].to_numpy(dtype=np.float32, na_value=np.nan)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    def _predict_proba(self, X, **kwargs):
        from threadpoolctl import threadpool_limits

        if len(X) == 0:
            shape = (0, self._classes) if self.problem_type == MULTICLASS else (0,)
            return np.empty(shape, dtype=np.float32)
        if self._classes and len(self._observed) == 1:
            values = np.zeros((len(X), self._classes), dtype=np.float32)
            values[:, self._observed[0]] = 1
            return self._convert_proba_to_unified_form(values)
        net = self.model.get_network()
        xs, ys = [v.detach().cpu().numpy() for v in (self.model.xs, self.model.ys)]
        query = self._encode(self.preprocess(X, **kwargs))
        pred = []
        # The CPU kernels show small batch-shape-dependent FP32 roundoff.
        # Fix the per-call shape on CPU to give each row identical arithmetic.
        batch_rows = 1 if str(self._fit_device).startswith("cpu") else 512
        with threadpool_limits(limits=self._threads):
            for i in range(0, len(query), batch_rows):
                rows = query[i : i + batch_rows]
                actual = len(rows)
                # Keep GPU GEMM dimensions fixed even for a short final batch.
                # Padding uses a training row; all query rows are independent.
                if actual < batch_rows:
                    rows = np.concatenate((rows, np.repeat(xs[:1], batch_rows - actual, axis=0)), axis=0)
                score = net.scores(xs, ys, rows, self._nominal, len(self._observed) if self._classes else 0)
                values = (score / 0.25).softmax(-1) if self._classes else score[:, 0]
                values = values[:actual]
                pred.append(values.cpu().numpy())
        values = np.concatenate(pred)
        if self._classes:
            proba = np.zeros((len(query), self._classes), dtype=np.float32)
            proba[:, self._observed] = values
            return self._convert_proba_to_unified_form(proba)
        return values

    def get_device(self):
        return self._fit_device

    def _set_device(self, device):
        self._fit_device = str(device)
        if self.model is not None:
            self.model.device = self._fit_device
            if self.model.network is not None:
                self.model.network.to(device)

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls):
        from .weights import checkpoint_path

        return [checkpoint_path(t) for t in ("classification", "regression")]
