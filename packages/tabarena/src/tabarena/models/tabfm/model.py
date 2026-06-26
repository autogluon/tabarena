from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


class TabFMModel(AbstractModel):
    """TabFM: a tabular foundation model that predicts via in-context learning.

    TabFM is a JAX/Flax pre-trained model: at inference time it is shown the
    training data as context and predicts on the test rows without any
    per-dataset gradient training. It handles mixed numerical/categorical columns
    and missing values natively (via its own internal preprocessing pipeline), so
    the AutoGluon-side preprocessing is left as a no-op and the typed DataFrame is
    passed straight through.

    Because TabFM is JAX-based (not torch), this wraps ``AbstractModel`` rather
    than ``AbstractTorchModel``: JAX manages device placement at the process level
    (via ``CUDA_VISIBLE_DEVICES`` / ``jax.devices()``), so there is no per-model
    ``.to(device)`` to implement.

    Accepts an optional ``device`` hyperparameter: ``None`` (default) auto-selects
    GPU when one is available, ``"cpu"`` forces CPU execution (sets
    ``JAX_PLATFORMS=cpu`` process-wide), and ``"gpu"``/``"cuda"`` requires a GPU.

    Paper: TabFM (Tabular Foundation Model)
    Authors: Google Research
    Codebase: https://github.com/google-research/tabfm
    License: Apache-2.0

    Install:
        pip install "tabfm @ git+https://github.com/google-research/tabfm.git"
    Add the ``[cuda]`` extra for GPU execution:
        pip install "tabfm[cuda] @ git+https://github.com/google-research/tabfm.git"
    """

    ag_key = "TA-TABFM"
    ag_name = "TA-TabFM"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import os

        # `random_state` is injected by AutoGluon via `seed_name`; both the
        # classifier and the regressor accept it (and TabFM's other knobs default
        # sensibly), so the remaining params are forwarded as-is. `device` is a
        # wrapper-only knob (TabFM/JAX has no device arg), so it is popped here.
        hps = self._get_model_params()
        device = hps.pop("device", None)
        if device is not None:
            device = str(device).lower()

        if device == "cpu":
            # Force JAX onto CPU for the whole process (covers both the in-context
            # fit and later predict). Must be set before JAX initialises its
            # backend; `_fit` is where this wrapper first imports jax, so it takes
            # effect here.
            os.environ["JAX_PLATFORMS"] = "cpu"

        import jax

        gpu_visible = any(d.platform != "cpu" for d in jax.devices())
        if device in ("gpu", "cuda") and not gpu_visible:
            raise AssertionError(
                "Fit specified device='gpu', but JAX sees no GPU device. Install "
                "the CUDA build (`pip install tabfm[cuda]`) and ensure CUDA is "
                "available, or set device='cpu'.",
            )
        if device is None and num_gpus and not gpu_visible:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but JAX sees no GPU device. "
                "Install the CUDA build (`pip install tabfm[cuda]`) and ensure CUDA "
                "is available, set device='cpu', or switch to CPU usage.",
            )

        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0

        if self.problem_type in ["binary", "multiclass"]:
            model_type = "classification"
            model_cls = TabFMClassifier
        elif self.problem_type == "regression":
            model_type = "regression"
            model_cls = TabFMRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Downloads the pre-trained checkpoint from Hugging Face on first use
        # (see `prefetch_weights`); a no-op once cached.
        base_model = tabfm_v1_0_0.load(model_type=model_type)

        self.model = model_cls(model=base_model, **hps)

        # Does nothing (TabFM handles categoricals/missing natively); kept for
        # future preprocessing extensions and parity with the other wrappers.
        X = self.preprocess(X, y=y)
        self.model = self.model.fit(X=X, y=y)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks.
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(
        self,
        is_gpu_available: bool = False,
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        refit_folds avoids storing one in-context model per fold (each carries the
        full training context), refitting a single model on all data instead.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            },
        )
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: support memory estimate!
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def prefetch_weights() -> None:
    """Pre-download the TabFM v1.0.0 checkpoint from Hugging Face.

    Warms the local cache (``google/tabfm-v1-0-0``) so parallel / offline fits do
    not race on the download.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="google/tabfm-v1-0-0")
