from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, ClassVar

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

#: The pipeline stages; the hyperparameter ``<stage>_idx`` picks the module of each.
STAGES = (
    "numerical_preprocess",
    "categorical_preprocess",
    "target_preprocess",
    "num_embedding",
    "cat_embedding",
    "model",
    "train",
    "optimizer",
    "inference",
    "loss",
)

#: RealMLP-TD with every stage at index 0.
DEFAULT_CONFIG = {
    "n_hidden_layers": 3,
    "hidden_sizes": "rectangular",
    "hidden_width": 256,
    "p_drop": 0.15,
    "act": "mish",
    "plr_sigma": 0.1,
    "plr_lr_factor": 0.1,
    "scale_lr_factor": 6.0,
    "first_layer_lr_factor": 1.0,
    "ls_eps_sched": "coslog4",
    "ls_eps": 0.1,
    "p_drop_sched": "flat_cos",
    "lr": 0.04,
    "wd": 0.02,
    "use_ls": True,
    "embedding_size": 8,
    "n_ens": 8,
    "ens_av_before_softmax": False,
    "one_minus_sq_mom": 0.05,
    "max_one_hot_cat_size_raw": 9.0,
    "is_large": False,
    "plr_hidden_1": 16,
    "plr_hidden_2": 4,
    "n_epochs": 256,
    "use_early_stopping": False,
    "early_stopping_multiplicative_patience": 3,
    "early_stopping_additive_patience": 40,
    "bool_to_cat": True,
    "random_state": 0,
    "batch_size": 256,
    **{f"{stage}_idx": 0 for stage in STAGES},
}


class AgenticRealMLPModel(AbstractTorchModel):
    """RealMLP split into a pipeline of stages (preprocessing, embeddings, model, train loop, optimizer,
    loss, inference), with agent-implemented candidate modules for each stage selected by per-stage
    indices in the search space. Index 0 in every stage is the RealMLP-TD default.

    Paper: Agentic Search Spaces for Tabular Machine Learning (https://arxiv.org/abs/2609.16309)
    Authors: Renat Sergazinov, Artem Chistyakov, Sergey Pankevich, Artem Babenko
    Codebase: https://github.com/yandex-research/agentic-hpset
    License: MIT
    """

    ag_key = "TA-REALMLP-A"
    ag_name = "TA-RealMLP-A"
    ag_priority = 75
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("delu", "loguru", "bin.realmlp.modules")
    cheap_hyperparameters: ClassVar[dict] = {"n_epochs": 1}
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

        from tabarena.models.agentic_realmlp._internal.engine import RealMLPImplementation

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
        indices = {stage: int(hyp.pop(f"{stage}_idx")) for stage in STAGES}
        X = self.preprocess(X, y=y, is_train=True, bool_to_cat=bool_to_cat)
        X_val = self.preprocess(X_val)
        self.model = RealMLPImplementation(
            problem_type=self.problem_type,
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
        return next(self.model.member_.model.parameters()).device.type

    def _set_device(self, device: str):
        self.model.device_ = self.to_torch_device(device)
        self.model.member_.model = self.model.member_.model.to(self.model.device_)

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
        max_one_hot = math.floor(p["max_one_hot_cat_size_raw"])
        large_cards = [card for card in cat_sizes if card > max_one_hot]
        n_one_hot = sum(card for card in cat_sizes if card <= max_one_hot)
        embedding_size = int(p["embedding_size"])
        plr_hidden_2 = int(p["plr_hidden_2"])
        d_in = max(1, n_num * plr_hidden_2 + n_one_hot + len(large_cards) * embedding_size)
        width = int(p["hidden_width"])
        layers = int(p["n_hidden_layers"])
        n_out = max(1, int(num_classes or 1))
        n_ens = int(p["n_ens"])
        n_params_single = (
            d_in * width
            + max(0, layers - 1) * width**2
            + width * n_out
            + sum(large_cards) * embedding_size
            + n_num * int(p["plr_hidden_1"]) * plr_hidden_2
        )
        activation_width = n_ens * (d_in + layers * width + n_out)
        mem_dataset = 4 * n_samples * (n_num + n_one_hot + len(large_cards))
        mem_params = 4 * 5 * n_ens * n_params_single
        mem_activation = 4 * 2 * int(p["batch_size"]) * activation_width
        return math.ceil(mem_dataset + 1.2 * mem_params + 1.3 * mem_activation + 2.5 * 1024**3)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"reset_torch_threads": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
