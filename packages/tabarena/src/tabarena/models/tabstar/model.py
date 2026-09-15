from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

from tabarena.models import prefetch as _hub
from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    import pandas as pd


logger = logging.getLogger(__name__)

#: The library's pretrained base model (``tabstar.training.utils.TABSTAR_REPO_ID``), resolved from its
#: default branch as the library does; the fit records the snapshot commit it used.
HF_REPO_ID = "alana89/TabSTAR"
#: Files of the base checkpoint ``TabStarModel.from_pretrained`` reads.
CHECKPOINT_FILES = ("config.json", "model.safetensors")
WEIGHTS_FILE = CHECKPOINT_FILES[1]
#: The frozen text encoder ``TabStarModel.__init__`` loads (``tabstar.arch.config.E5_SMALL``) and the
#: files ``AutoModel`` and ``AutoTokenizer`` need from it.
TEXT_ENCODER_REPO_ID = "intfloat/e5-small-v2"
TEXT_ENCODER_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)
#: The library's own override for the text encoder location; an explicit value is respected.
TEXT_ENCODER_ENV_VAR = "E5_SMALL_LOCAL_PATH"
_ESTIMATORS_MODULE = "tabarena.models.tabstar._estimators"


def _estimators() -> ModuleType:
    """The library-importing half of the wrapper, imported on first use."""
    return importlib.import_module(_ESTIMATORS_MODULE)


def _pretrain_path_is_not_a_checkpoint_dir(hyperparameters: Mapping[str, Any]) -> bool:
    """True for a ``pretrain_dataset_or_path`` that is not a local checkpoint directory (a pretraining dataset name or a Hub repo id), which the library resolves itself."""
    value = hyperparameters.get("pretrain_dataset_or_path")
    return value is not None and not os.path.isfile(os.path.join(str(value), WEIGHTS_FILE))


def resolve_base_model_dir(*, allow_download: bool = True) -> str:
    """Local snapshot directory of the TabSTAR base checkpoint: the Hugging Face cache first, the Hub when allowed.

    The same coordinates the shared-weights key resolves; the estimator's ``pretrain_dataset_or_path``
    takes this directory whether or not the fit shares its weights.

    Raises:
        WeightsUnavailableError: The checkpoint is not cached and ``allow_download`` is False.
    """
    spec = TabSTARModel.shared_weights_spec
    variant = spec.variant_for("binary")
    resolved = spec.checkpoint_for(variant).resolve(hyperparameters={}, variant=variant, allow_download=allow_download)
    return str(Path(resolved.path).parent)


def resolve_text_encoder_dir(*, allow_download: bool = True) -> str:
    """Local snapshot directory of the ``e5-small-v2`` text encoder, resolved like :func:`resolve_base_model_dir`."""
    return _hub.resolve_hf_snapshot(
        TEXT_ENCODER_REPO_ID,
        allow_patterns=list(TEXT_ENCODER_FILES),
        required_files=TEXT_ENCODER_FILES,
        allow_download=allow_download,
    )


# TODO:
#   - support for metric_name was rolled back, so maybe in the future add support for AG metrics again.
class TabSTARModel(SharedWeightsModelMixin, AbstractModel):
    """TabSTAR Model: https://arxiv.org/abs/2505.18125.

    TabSTAR fine-tunes LoRA adapters on a frozen pretrained ``TabStarModel`` (a 47M-parameter model
    around the ``e5-small-v2`` text encoder). The bag children share the base checkpoint's fp32
    state dict through the weights registry (:mod:`tabarena.models._shared_weights_model`,
    ``mode="state_dict"``) and each builds its own module from it through
    :mod:`tabarena.models.tabstar._estimators` instead of reading the 189 MB checkpoint twice per
    fit; the text encoder is loaded from its cached snapshot instead of being revalidated through
    the Hub at every construction. The fine-tuning, the adapters each child trains and owns, and the
    predictions are unchanged, and a fitted child pickles whole as before (the base weights are
    frozen and travel with the adapters). A ``pretrain_dataset_or_path`` that is not a local
    checkpoint directory keeps the library's own checkpoint load.
    """

    ag_key = "TABSTAR"
    #: Modules the timed fit would otherwise import for the first time: the library estimators (which
    #: pull in transformers and peft), the seam module and the Hub client the resolvers use.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "torch",
        "tabstar.tabstar_model",
        "tabstar.training.hyperparams",
        _ESTIMATORS_MODULE,
        "huggingface_hub",
        "safetensors.torch",
    )
    #: Cheapness knob for the warm-up dummy fit: one fine-tuning epoch. Never a checkpoint-relevant
    #: key, so the primed registry key is the one the real fit uses.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"max_epochs": 1}
    ag_name = "TabSTAR"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    #: One base checkpoint serves classification and regression; the shared payload is its fp32
    #: state dict on the CPU whatever device the fit runs on.
    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabstar",
        mode="state_dict",
        checkpoint=CheckpointSpec(
            repo_id=HF_REPO_ID,
            kind="snapshot",
            required_files=CHECKPOINT_FILES,
            weights_file=WEIGHTS_FILE,
            user_path_param="pretrain_dataset_or_path",
        ),
        variant="base",
        default_params={"pretrain_dataset_or_path": None},
        disable_when=(_pretrain_path_is_not_a_checkpoint_dir,),
        unshareable_examples=({"pretrain_dataset_or_path": "BIN_adult"},),
        cache_device="cpu",
        state_dict_format="safetensors",
        pin_memory=True,
    )

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """The base checkpoint (declared in the spec) plus the text encoder snapshot the library loads beside it."""
        return [*super().prefetch_weights(), resolve_text_encoder_dir(allow_download=True)]

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        # num_cpus: int = 1,  # Not used
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
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        est = _estimators()
        hps = self._get_model_params()

        # Both checkpoints are resolved local-first before the library runs, so no Hub request lands
        # in the timed fit; a fetch the `ag.fetch_pretrained_weights` policy forbids raises here.
        allow_fetch = fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="fit")
        try:
            text_encoder_dir = resolve_text_encoder_dir(allow_download=allow_fetch)
            base_model_dir = hps.get("pretrain_dataset_or_path") or resolve_base_model_dir(allow_download=allow_fetch)
        except _hub.WeightsUnavailableError as exc:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=self.name, stage="fit", location=str(exc))
            ) from exc
        est.use_local_text_encoder(text_encoder_dir)
        hps["pretrain_dataset_or_path"] = base_model_dir

        # The registry's CPU state dict (a hit after the warm-up or the first child); the estimator
        # borrows it for this fit and builds its own base model from it.
        key, state_dict = self._acquire_shared_weights(device=device)
        if self.problem_type in ["binary", "multiclass"]:
            model_cls = est.SharedTabSTARClassifier if key is not None else est.TabSTARClassifier
        elif self.problem_type in ["regression"]:
            model_cls = est.SharedTabSTARRegressor if key is not None else est.TabSTARRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        # Simple heuristic for batch size
        train_batch_size = est.LORA_BATCH
        predict_batch_size = est.VAL_BATCH
        if X.shape[1] > 200:
            if X.shape[0] > 50_000:
                train_batch_size = 16
                predict_batch_size = 16
            else:
                train_batch_size = 64
                predict_batch_size = 64

        self.model = model_cls(
            **hps,
            lora_batch=train_batch_size,
            val_batch_size=predict_batch_size,
            time_limit=time_limit,
            device=device,
            output_dir=self.path + "/model_checkpoints",
        )
        if key is not None:
            self.model.configure_shared_weights(state_dict)

        if X_val is None:
            # FIXME: make this a general utility function in autogluon that also handles
            #  ratio better! Or handle it before _fit based on `can_refit_full`
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
        # Inverse label transformation to retain original semantics for classification
        #   - hasattr for backward compatibility
        if self.problem_type in ["binary", "multiclass"]:
            if (not hasattr(self, "label_cleaner")) or (self.label_cleaner is None):
                raise ValueError("Label cleaner missing from AbstractModel!")

            y = self.label_cleaner.inverse_transform(y)
            if y_val is not None:
                y_val = self.label_cleaner.inverse_transform(y_val)

        try:
            # FIXME: .fit does not return self as expected from sklearn API
            self.model.fit(
                X=X,
                y=y,
                x_val=X_val,
                y_val=y_val,
            )
        finally:
            if key is not None:
                self.model.configure_shared_weights(None)

    def _set_default_params(self):
        # Default values from the current version of the code base
        default_params = {
            # Large max epochs, we want to stop based on time limit or early stopping
            "max_epochs": 10_000,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
