from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_HF_REPO_ID = "tababax-ai/TabAbax"
#: Released checkpoint per task type; the library selects the same two files by default.
_HF_CHECKPOINTS = {
    "classification": "tababax-classifier.ckpt",
    "regression": "tababax-regressor.ckpt",
}
#: Commit that uploaded the released weights. Later commits only touched the model card, but the
#: library resolves ``main``, so pinning here keeps a republished checkpoint from changing the run.
#: Public so a run script can pin the same commit without importing a private name.
HF_REPO_ID = _HF_REPO_ID
HF_CHECKPOINTS = _HF_CHECKPOINTS
CHECKPOINT_REVISION = "039e4f239d06a4d631e38c15bc23ef195ed8cf79"


class TabAbaxModel(AbstractTorchModel):
    """TabAbax, an in-context foundation model for classification and regression.

    ``fit`` stores the training table. Each ensemble member runs one forward pass.
    The default estimator uses eight dataset-view ensemble members. The weights
    are not updated on the task.

    Omitting ``model_path`` loads the released checkpoint from
    ``tababax-ai/TabAbax`` at the commit this wrapper pins.
    Classification uses ``tababax-classifier.ckpt``
    and regression uses ``tababax-regressor.ckpt``. Pass ``model_path`` to load a
    local file instead. The weights are CC-BY-NC-4.0 and are limited to non-commercial use.

    Recommended defaults, applied unless a hyperparameter overrides them:

    - ``n_estimators=8``: ensemble views. ``1`` is a single forward pass.
    - ``batch_size=1``: ensemble members forwarded together. ``None`` sends all of them at once.
    - ``offload_mode="auto"``: puts the large column embedding on GPU, CPU, or disk according to free memory.
    - ``use_amp="auto"``: mixed precision from the table size. Off for small tables and on CPU.
    - ``use_fa3=False``: FlashAttention-3. Leave off unless the runtime provides it.
    - ``allow_auto_download=True``: download the checkpoint when ``model_path`` is missing.
      ``False`` refuses the download.

    ``random_state`` is the TabArena seed (``seed_name``). ``device`` follows the GPUs AutoGluon
    allocated: CUDA when ``num_gpus`` is not 0, otherwise CPU.

    The TabArena integration imports TabAbax, whose inference implementation
    contains portions derived from TabICLv2 and distributed under BSD-3-Clause.
    The additional TabAbax packaging and integration code is Apache-2.0.
    The default checkpoints are separate CC-BY-NC-4.0 artifacts and are not
    commercially licensed.

    Authors: TabAbax, based on TabICL by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan
    Codebase: https://github.com/tababax-research/TabAbax
    Upstream: https://github.com/soda-inria/tabicl
    Weights: https://huggingface.co/tababax-ai/TabAbax
    License: BSD-3-Clause for the TabICL-derived code. Apache-2.0 for TabAbax packaging. Weights: CC-BY-NC-4.0.
    """

    ag_key = "TA-TABABAX"
    ag_name = "TA-TabAbax"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tababax",)
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: The estimator builds its network inside ``_load_model``, which ``fit`` calls. The first
    #: fit of a checkpoint in the process builds it; every later estimator of that checkpoint
    #: and device reuses it.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=(
            "tababax:TabAbaxClassifier._load_model",
            "tababax:TabAbaxRegressor._load_model",
        ),
        key=("checkpoint_version", "model_path"),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def get_model_cls(self):
        if self.problem_type in ["binary", "multiclass"]:
            from tababax import TabAbaxClassifier

            return TabAbaxClassifier
        from tababax import TabAbaxRegressor

        return TabAbaxRegressor

    @classmethod
    def download_checkpoint(cls, task: str) -> str:
        """Resolve one released checkpoint (``"classification"`` / ``"regression"``) at the pinned revision.

        The Hub cache answers first (no network round trip on a node that ran ``prefetch_weights``),
        a download otherwise.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = {"repo_id": _HF_REPO_ID, "filename": _HF_CHECKPOINTS[task], "revision": CHECKPOINT_REVISION}
        try:
            return hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            return hf_hub_download(**kwargs)

    @classmethod
    def prefetch_weights(cls) -> dict[str, str]:
        """Pre-download both released checkpoints; return ``{task: local path}``.

        Classification and regression are separate files, so a run covering both problem types
        needs both warmed before the jobs are dispatched.
        """
        return {task: cls.download_checkpoint(task) for task in _HF_CHECKPOINTS}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            import tababax  # noqa: F401
        except ImportError as err:
            logger.log(
                40,
                "\tFailed to import tababax! Install it with `pip install tabarena[tababax]`. "
                "The package is derived from TabICL (github.com/soda-inria/tabicl, BSD-3-Clause).",
            )
            raise err

        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()
        task = "classification" if self.problem_type in ["binary", "multiclass"] else "regression"
        if "checkpoint_version" not in hyp:
            hyp["checkpoint_version"] = _HF_CHECKPOINTS[task]
        # Pin the revision the library resolves the checkpoint against. Without it the library follows
        # the Hub repo's default branch, so the bytes behind a published score could change under a
        # rerun. An explicit hyperparameter still wins, for a caller loading its own checkpoint.
        if hyp.get("checkpoint_revision") is None:
            hyp["checkpoint_revision"] = CHECKPOINT_REVISION
        disk_offload_dir = tempfile.mkdtemp(prefix="tababax_")
        self.model = self.get_model_cls()(
            **hyp,
            device=device,
            n_jobs=num_cpus,
            disk_offload_dir=disk_offload_dir,
            verbose=X.shape[0] > 250_000,
            inference_config=dict(COL_CONFIG=dict(cpu_safety_factor=0.75)),
        )
        X = self.preprocess(X, y=y)
        self.model = self.model.fit(X=X, y=y)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Host-memory estimate until the checkpoint's GPU offload behaviour is measured."""
        del hyperparameters, kwargs
        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()
        baseline_overhead_mem_est = 1e9
        return dataset_size_mem_est + baseline_overhead_mem_est

    def _set_default_params(self):
        """Defaults for a released checkpoint. A hyperparameter with the same name overrides these."""
        self._set_default_param_value("n_estimators", 8)
        self._set_default_param_value("batch_size", 1)
        self._set_default_param_value("offload_mode", "auto")
        self._set_default_param_value("use_amp", "auto")
        self._set_default_param_value("use_fa3", False)
        self._set_default_param_value("allow_auto_download", True)

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.device = self.model.device_.type
        self.model.model_ = self.model.model_.to(self.model.device_)
        self.model.inference_config_.COL_CONFIG.device = self.model.device_
        self.model.inference_config_.ROW_CONFIG.device = self.model.device_
        self.model.inference_config_.ICL_CONFIG.device = self.model.device_

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
