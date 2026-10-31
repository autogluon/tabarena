from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

#: The stage index fields of ``bin.tabicl.pipeline.implementation_choices``.
INDEX_KEYS = (
    "model_idx",
    "feature_preproc_basket_idx",
    "augmentation_basket_idx",
    "postproc_basket_idx",
    "feature_permutation_idx",
    "ensemble_idx",
)
#: Stages with separate regression and classification modules; a config carries ``<key>_reg`` and ``<key>_clf``.
TASK_DISPATCHED_KEYS = ("postproc_basket_idx", "ensemble_idx")

#: Every stage at index 0, the reference TabICLv2 recipe (8 members; the package fixes the ensemble size).
DEFAULT_CONFIG = {
    "random_state": 0,
    **{
        name: 0
        for key in INDEX_KEYS
        for name in ((f"{key}_reg", f"{key}_clf") if key in TASK_DISPATCHED_KEYS else (key,))
    },
}


class AgenticTabICLv2Model(AbstractTorchModel):
    """The frozen TabICLv2 with its inference split into stages (feature preprocessing, augmentation, feature
    permutation, target postprocessing, ensemble aggregation), with agent-implemented candidate modules for
    each stage selected by per-stage indices in the search space. Index 0 in every stage is the reference
    TabICLv2 recipe. Like TabICLv2 on TabArena, the bag's folds only give the out-of-fold predictions and one
    model refit on all training rows predicts the test data.

    Paper: Agentic Search Spaces for Tabular Machine Learning (https://arxiv.org/abs/2609.16309)
    Authors: Renat Sergazinov, Artem Chistyakov, Sergey Pankevich, Artem Babenko
    Codebase: https://github.com/yandex-research/agentic-hpset
    License: MIT
    """

    ag_key = "TA-TABICLV2-A"
    ag_name = "TA-TabICLv2-A"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "bin.tabicl.pipeline",
        "tabarena.models.agentic_tabiclv2._internal.engine",
    )
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: The class cap is the checkpoint's head width.
    _default_auxiliary_params_extra: ClassVar[dict] = {"max_classes": 10}
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: float = 0, **kwargs):
        import torch

        from tabarena.models.agentic_tabiclv2._internal.engine import AgenticTabICLv2Implementation

        device = "cpu" if num_gpus == 0 else "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            raise AssertionError("Fit specified to use GPU, but CUDA is not available on this machine.")

        hyp = self._get_model_params()
        suffix, other = ("reg", "clf") if self.problem_type == "regression" else ("clf", "reg")
        indices = {}
        for key in INDEX_KEYS:
            if key in TASK_DISPATCHED_KEYS:
                hyp.pop(f"{key}_{other}")
                indices[key] = int(hyp.pop(f"{key}_{suffix}"))
            else:
                indices[key] = int(hyp.pop(key))
        self.model = AgenticTabICLv2Implementation(
            problem_type=self.problem_type,
            n_classes=self.num_classes,
            device=device,
            n_threads=num_cpus,
            indices=indices,
            seed=int(hyp.pop("random_state") or 0),
        )
        self.model.fit(self.preprocess(X, y=y), y)

    def _set_default_params(self):
        for param, value in DEFAULT_CONFIG.items():
            self._set_default_param_value(param, value)

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        self.model.device_ = self.to_torch_device(device)
        self.model.model_ = self.model.model_.to(self.model.device_)

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """The official TabICLv2 wrapper's estimate (data and overhead only; ``refit_folds`` keeps one model)."""
        return 3 * get_approximate_df_mem_usage(X).sum() + 1e9

    @classmethod
    def _class_tags(cls) -> dict:
        return {"reset_torch_threads": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
