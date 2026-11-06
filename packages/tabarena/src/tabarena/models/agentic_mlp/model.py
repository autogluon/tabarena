from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, ClassVar

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

#: The stage index fields of ``bin.mlp.hparams.INDEX_KEYS``.
INDEX_KEYS = (
    "numerical_preprocess_idx",
    "categorical_preprocess_idx",
    "target_preprocess_idx",
    "num_embedding_idx",
    "cat_embedding_idx",
    "model_idx",
    "train_idx",
    "optimizer_idx",
    "inference_idx",
    "loss_idx",
)
#: Stages with separate regression and classification modules; a config carries ``<key>_reg`` and ``<key>_clf``.
TASK_DISPATCHED_KEYS = ("target_preprocess_idx", "inference_idx", "loss_idx")

#: The fixed training settings of ``bin.mlp.hparams.sample_hps``, mid-range values for the sampled
#: hyperparameters, and every stage at index 0 (the base MLP with piecewise-linear embeddings).
DEFAULT_CONFIG = {
    "n_bins": 48,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "n_blocks": 3,
    "d_block": 512,
    "dropout": 0.1,
    "d_embedding": 24,
    "gradient_clip": 1.0,
    "max_epochs": 256,
    "patience": 16,
    "eval_batch_size": 32768,
    "bool_to_cat": True,
    "random_state": 0,
    **{
        name: 0
        for key in INDEX_KEYS
        for name in ((f"{key}_reg", f"{key}_clf") if key in TASK_DISPATCHED_KEYS else (key,))
    },
}


def tiered_batch_size(n_train: int) -> int:
    """The package fixes the batch size per dataset name (``bin.mlp.hparams.DATASET_TO_BATCH_SIZE``);
    these size tiers were fitted to that table.
    """
    for bound, batch_size in ((2_000, 16), (4_500, 32), (9_000, 64), (20_000, 128), (50_000, 256)):
        if n_train < bound:
            return batch_size
    return 512


class AgenticMLPModel(AbstractTorchModel):
    """An MLP split into a pipeline of stages (numerical, categorical and target preprocessing, numerical
    and categorical embeddings, model, train loop, optimizer, loss, inference), with agent-implemented
    candidate modules for each stage selected by per-stage indices in the search space. Index 0 in every
    stage is the base MLP with piecewise-linear embeddings.

    Paper: Agentic Search Spaces for Tabular Machine Learning (https://arxiv.org/abs/2609.16309)
    Authors: Renat Sergazinov, Artem Chistyakov, Sergey Pankevich, Artem Babenko
    Codebase: https://github.com/yandex-research/agentic-hpset
    License: MIT
    """

    ag_key = "TA-MLP-A"
    ag_name = "TA-MLP-A"
    ag_priority = 75
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("optuna", "bin.mlp.pipeline")
    cheap_hyperparameters: ClassVar[dict] = {"max_epochs": 1}
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"fold_fitting_strategy": "sequential_local"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._features_bool = None
        self._bool_to_cat = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        start_time = time.time()
        import torch

        from tabarena.models.agentic_mlp._internal.engine import MLPImplementation

        device = "cpu" if num_gpus == 0 else "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            raise AssertionError("Fit specified to use GPU, but CUDA is not available on this machine.")

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.33, random_state=0
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat")
        suffix, other = ("reg", "clf") if self.problem_type == "regression" else ("clf", "reg")
        indices = {}
        for key in INDEX_KEYS:
            if key in TASK_DISPATCHED_KEYS:
                hyp.pop(f"{key}_{other}")
                indices[key] = int(hyp.pop(f"{key}_{suffix}"))
            else:
                indices[key] = int(hyp.pop(key))
        batch_size = tiered_batch_size(len(y))
        # BatchNorm needs two or more rows per training batch, so step past a last batch of one row.
        while len(y) % batch_size == 1:
            batch_size += 1
        hyp["batch_size"] = batch_size
        X = self.preprocess(X, y=y, is_train=True, bool_to_cat=bool_to_cat)
        X_val = self.preprocess(X_val)
        self.model = MLPImplementation(
            problem_type=self.problem_type,
            n_classes=self.num_classes,
            device=device,
            n_threads=num_cpus,
            eval_metric=self.eval_metric,
            indices=indices,
            **hyp,
        )
        remaining = None if time_limit is None else max(0.0, time_limit - (time.time() - start_time))
        self.model.fit(X, y, X_val, y_val, time_limit=remaining)

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, bool_to_cat: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
        if self._bool_to_cat and self._features_bool:
            X = X.copy(deep=True)
            X[self._features_bool] = X[self._features_bool].astype("category")
        return X

    def _set_default_params(self):
        for param, value in DEFAULT_CONFIG.items():
            self._set_default_param_value(param, value)

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def get_device(self) -> str:
        return next(self.model.model_.parameters()).device.type

    def _set_device(self, device: str):
        self.model.device_ = self.to_torch_device(device)
        self.model.model_ = self.model.model_.to(self.model.device_)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = 1,
        **kwargs,
    ) -> int:
        """Parameters, optimizer state, one training batch of activations and the dataset."""
        p = {**DEFAULT_CONFIG, **(hyperparameters or {})}
        n_samples = len(X)
        cat_columns = X.select_dtypes(include=["category", "object", "bool"]).columns
        cat_sizes = (X[cat_columns].nunique() + 1).tolist()
        n_num = len(X.columns) - len(cat_columns)
        # Numerical preprocessing 5 appends a log copy of each skewed column, 7 three row statistics.
        if p["numerical_preprocess_idx"] == 5:
            n_num *= 2
        elif p["numerical_preprocess_idx"] == 7:
            n_num += 3
        d_embedding = int(p["d_embedding"])
        d_num = n_num * d_embedding + (d_embedding if p["num_embedding_idx"] == 5 else 0)
        # Categorical embeddings 0, 5 and 8 are one-hot based (5 and 8 add target-mean or frequency columns).
        cat_idx = p["cat_embedding_idx"]
        if cat_idx in {0, 5, 8}:
            d_cat = sum(cat_sizes) + (len(cat_sizes) if cat_idx in {5, 8} else 0)
        else:
            d_cat = len(cat_sizes) * d_embedding
        d_in = max(1, d_num + d_cat)
        d_block = int(p["d_block"])
        n_blocks = int(p["n_blocks"])
        n_out = max(1, int(num_classes or 1))
        # Residual and GLU blocks carry two d x d projections; the wide-narrow block d -> 2d -> d.
        width_factor = 4 if p["model_idx"] == 10 else 2 if p["model_idx"] in {1, 3} else 1
        n_params = width_factor * (d_in * d_block + max(0, n_blocks - 1) * d_block**2 + d_block * n_out)
        n_params += sum(cat_sizes) * d_embedding
        n_params += n_num * max(d_embedding, int(p["n_bins"])) * d_embedding
        activation_width = d_in + width_factor * n_blocks * d_block + n_out
        mem_dataset = 4 * n_samples * (d_in + n_out)
        mem_params = 4 * 5 * n_params
        mem_activation = 4 * 2 * tiered_batch_size(n_samples) * activation_width
        return math.ceil(mem_dataset + 1.2 * mem_params + 1.3 * mem_activation + 2.5 * 1024**3)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"reset_torch_threads": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
