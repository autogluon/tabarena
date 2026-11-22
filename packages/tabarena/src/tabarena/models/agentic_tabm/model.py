from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.tabm._internal.tabm_utils import get_tabm_auto_batch_size

if TYPE_CHECKING:
    import pandas as pd

#: The stage index fields of ``bin.tabm.hparams.INDEX_KEYS``.
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

#: The package has no defaults for its tuned hyperparameters, so these are TabM's own (TabM-mini with
#: piecewise-linear embeddings), with the training settings fixed in ``bin.tabm.hparams.sample_hps`` and every
#: stage at index 0 (the reference TabM).
DEFAULT_CONFIG = {
    "k": 32,
    "n_bins": 48,
    "d_embedding": 16,
    "n_blocks": 2,
    "d_block": 512,
    "dropout": 0.1,
    "lr": 2e-3,
    "weight_decay": 3e-4,
    "gradient_clip": 1.0,
    "max_epochs": 256,
    "patience": 16,
    "eval_batch_size": 32768,
    "batch_size": "auto",
    "random_state": 0,
    "bool_to_cat": True,
    **{
        name: 0
        for key in INDEX_KEYS
        for name in ((f"{key}_reg", f"{key}_clf") if key in TASK_DISPATCHED_KEYS else (key,))
    },
}


class AgenticTabMModel(AbstractTorchModel):
    """TabM split into a pipeline of stages (preprocessing, embeddings, model, train loop, optimizer, loss,
    inference), with agent-implemented candidate modules for each stage selected by per-stage indices in the
    search space. Index 0 in every stage is the reference TabM (TabM-mini with piecewise-linear embeddings).

    Paper: Agentic Search Spaces for Tabular Machine Learning (https://arxiv.org/abs/2609.16309)
    Authors: Renat Sergazinov, Artem Chistyakov, Sergey Pankevich, Artem Babenko
    Codebase: https://github.com/yandex-research/agentic-hpset
    License: MIT
    """

    ag_key = "TA-TABM-A"
    ag_name = "TA-TabM-A"
    ag_priority = 75
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "bin.tabm.pipeline",
        "tabarena.models.agentic_tabm._internal.engine",
    )
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
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()
        import torch

        from tabarena.models.agentic_tabm._internal.engine import AgenticTabMImplementation

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
        X = self.preprocess(X, y=y, is_train=True, bool_to_cat=bool_to_cat)
        X_val = self.preprocess(X_val)
        if hyp["batch_size"] == "auto":
            hyp["batch_size"] = get_tabm_auto_batch_size(n_samples=len(X), n_features=X.shape[1])
        self.model = AgenticTabMImplementation(
            problem_type=self.problem_type,
            n_classes=self.num_classes,
            device=device,
            n_threads=num_cpus,
            verbosity=verbosity,
            eval_metric=self.eval_metric,
            indices=indices,
            **hyp,
        )
        # 5% headroom for the closing evaluation and the out-of-fold prediction after the train loop.
        remaining = None if time_limit is None else max(0.0, 0.95 * (time_limit - (time.time() - start_time)))
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
        return self.model.device_.type

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
        """The official TabM wrapper's estimate, with this model's hyperparameter names."""
        from tabarena.models.tabm.model import TabMModel

        p = {**DEFAULT_CONFIG, **(hyperparameters or {})}
        tabm_hyperparameters = {
            "num_emb_n_bins": p["n_bins"],
            "d_embedding": p["d_embedding"],
            "d_block": p["d_block"],
            "n_blocks": p["n_blocks"],
            "tabm_k": p["k"],
            "batch_size": p["batch_size"],
        }
        return TabMModel._estimate_memory_usage_static(
            X=X, hyperparameters=tabm_hyperparameters, num_classes=num_classes, **kwargs
        )

    @classmethod
    def _class_tags(cls) -> dict:
        return {"reset_torch_threads": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
