from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


@contextmanager
def _preserved_cudnn_flags():
    """Keep the library's seeding, which pins cuDNN to deterministic mode, out of the process."""
    import torch

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


class TabSTARv1p1Model(AbstractTorchModel):
    """TabSTAR v1.1: https://arxiv.org/abs/2505.18125.

    Second generation of TabSTAR, pretrained on a larger corpus and shipped as a separate estimator
    in the same ``tabstar`` package. Classification and regression load different checkpoints.
    """

    ag_key = "TA-TABSTAR-V1.1"
    ag_name = "TA-TabSTAR-v1.1"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        # Parallel folding crashes if the weights aren't pre-downloaded.
        # TODO: switch to parallel fitting on one GPU once VRAM memory estimation is supported.
        "fold_fitting_strategy": "sequential_local",
        # No `refit_folds`: unlike the in-context wrappers, each fold fine-tunes its own adapters,
        # so the children differ and the bag's average is worth its inference cost.
    }
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "torch",
        "tabstar.tabstarv1p1_model",
        "tabstar.tabstarv1p1.finetune.hyperparams",
    )
    #: The library builds the base model through ``TabSTARModel.from_pretrained`` twice per fit (to
    #: train, and to attach the saved adapters). The checkpoint is read once per process; every
    #: fit fine-tunes its own deep copy and pickles it.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabstar.tabstarv1p1.arch.model:TabSTARModel.from_pretrained",
        key=("pretrained_model_name_or_path",),
        copy_per_fit=True,
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"max_epochs": 1}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabstar.tabstarv1p1.finetune.hyperparams import LORA_BATCH, VAL_BATCH
        from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier, TabSTARv1p1Regressor

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = TabSTARv1p1Classifier
        elif self.problem_type in ["regression"]:
            model_cls = TabSTARv1p1Regressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Simple heuristic for batch size
        train_batch_size = LORA_BATCH
        predict_batch_size = VAL_BATCH
        if X.shape[1] > 200:
            if X.shape[0] > 50_000:
                train_batch_size = 16
                predict_batch_size = 16
            else:
                train_batch_size = 64
                predict_batch_size = 64

        hps = self._get_model_params()
        with _preserved_cudnn_flags():
            self.model = model_cls(
                **hps,
                lora_batch=train_batch_size,
                val_batch_size=predict_batch_size,
                time_limit=time_limit,
                device=device,
                output_dir=self.path + "/model_checkpoints",
            )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.33,
                random_state=0,
            )

        # Does nothing but might be used for future extensions
        X = self.preprocess(X, y=y)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # TabSTAR reads class names as text, so the original labels are restored when AutoGluon
        # provides a cleaner. ``refit_full`` passes none; the encoded labels are then fit as-is,
        # which keeps the class order the predictions are expected in and loses only the names.
        if self.problem_type in ["binary", "multiclass"]:
            label_cleaner = getattr(self, "label_cleaner", None)
            if label_cleaner is None:
                logger.log(30, "Warning: no label cleaner available, fitting on the encoded class labels.")
            else:
                y = label_cleaner.inverse_transform(y)
                if y_val is not None:
                    y_val = label_cleaner.inverse_transform(y_val)

        with _preserved_cudnn_flags():
            self.model.fit(
                X=X,
                y=y,
                x_val=X_val,
                y_val=y_val,
            )

    # ``max_epochs`` is deliberately left at the library default: v1.1 builds its OneCycle schedule
    # over ``steps_per_epoch * max_epochs``, so inflating it to stop on time limit or early stopping
    # instead would keep every real epoch inside the warm-up ramp and never anneal the LR.

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _class_tags(cls):
        # The estimator places its model itself; AutoGluon moves nothing on save or load.
        return {
            **super()._class_tags(),
            "can_set_device": False,
            "set_device_on_save_to": None,
            "set_device_on_load": False,
        }


def prefetch_weights() -> None:
    """Pre-download the two checkpoints and the tokenizer (warms the cache for offline fits)."""
    from tabstar.tabstarv1p1.tabstar_checkpoints import download_tabstar

    download_tabstar()
