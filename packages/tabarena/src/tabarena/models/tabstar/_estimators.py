"""TabSTAR estimators whose base model comes from the shared checkpoint state dict.

TabSTAR fine-tunes LoRA adapters on top of a frozen pretrained ``TabStarModel``. In the library
every fit reads the 189 MB base checkpoint twice (``TabStarTrainer.__init__`` builds the model to
train, ``TabStarTrainer.load_model`` rebuilds it to attach the saved adapters), and every build
downloads or revalidates the ``intfloat/e5-small-v2`` text encoder through the Hub. The classes here
keep the library's construction path, ``TabStarModel.from_pretrained``, but feed it the state dict
the wrapper took from the shared-weights registry instead of a file, and point the text encoder at
its cached snapshot. The fine-tuning itself, the checkpoint averaging and the prediction code are
the library's.

What the state-dict build changes and what it keeps (verified against tabstar 1.1.15 with
transformers 4.57): the module is constructed by the same ``from_pretrained`` code, so no random
initialization runs and no global random generator advances; the resulting state dict equals a
file-based build tensor for tensor; the LoRA adapters created on top of it are identical, so the
fine-tuning consumes the random stream exactly as before. ``from_pretrained`` assigns the tensors
of a passed state dict to the module instead of copying them, so on a CPU fit
:func:`build_base_model` gives every aliased parameter its own storage before anything trains
(on a CUDA fit ``.to(device)`` creates the device copies and leaves the cached host tensors alone).

The three replicated bodies (``TabStarTrainer.__init__``, ``TabStarTrainer.load_model``,
``BaseTabSTAR.fit``) follow tabstar 1.1.15 line by line except for the base-model source; the
package version is pinned by the ``tabstar`` extra.

Imports tabstar, peft, transformers and torch, so it is only imported from the wrapper's warm-up
and fit paths, never at module discovery time.
"""

from __future__ import annotations

import gc
import itertools
import os
from os.path import exists

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tabstar.arch import arch as _arch
from tabstar.arch.arch import TabStarModel
from tabstar.arch.config import E5_SMALL, TabStarConfig
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabstar.training.checkpoint_averaging import CheckpointManager
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.hyperparams import LORA_BATCH, VAL_BATCH, set_accumulation_steps
from tabstar.training.optimizer import get_optimizer, get_scheduler
from tabstar.training.trainer import TabStarTrainer
from tabstar.training.utils import TABSTAR_REPO_ID
from torch.amp import GradScaler

from tabarena.models._shared_estimators import SharedStateDictEstimatorMixin
from tabarena.models._weights import normalize_device
from tabarena.models.tabstar.model import HF_REPO_ID, TEXT_ENCODER_ENV_VAR, TEXT_ENCODER_REPO_ID

__all__ = [
    "LORA_BATCH",
    "VAL_BATCH",
    "SharedTabSTARClassifier",
    "SharedTabSTARRegressor",
    "SharedTabStarTrainer",
    "TabSTARClassifier",
    "TabSTARRegressor",
    "build_base_model",
    "load_finetuned_shared",
    "load_pretrained_shared",
    "use_local_text_encoder",
]

if TABSTAR_REPO_ID != HF_REPO_ID or E5_SMALL != TEXT_ENCODER_REPO_ID:
    raise ImportError(
        f"tabstar resolves its checkpoints from {TABSTAR_REPO_ID!r} and {E5_SMALL!r}; the TabArena wrapper expects "
        f"{HF_REPO_ID!r} and {TEXT_ENCODER_REPO_ID!r}. Update tabarena.models.tabstar.model for this tabstar version."
    )

#: LoRA target modules and the frozen text-encoder layers of ``tabstar.training.lora.load_pretrained``.
_LORA_MODULES = ["query", "key", "value", "out_proj", "linear1", "linear2", "cls_head.layers.0", "reg_head.layers.0"]
_FROZEN_TEXT_ENCODER_LAYERS = range(6)


def use_local_text_encoder(snapshot_dir: str) -> None:
    """Point ``TabStarModel`` at the cached ``intfloat/e5-small-v2`` snapshot for this process.

    ``TabStarModel.__init__`` loads the text encoder and its tokenizer with
    ``AutoModel.from_pretrained(E5_SMALL_LOCAL_PATH or "intfloat/e5-small-v2")``; with the repo id
    every construction (two per child) revalidates the files through the Hub. The library reads its
    override from the ``E5_SMALL_LOCAL_PATH`` environment variable once at import, so this sets the
    module attribute that read it. The snapshot holds the same files the repo id resolves to, so the
    encoder and tokenizer are unchanged. An explicit ``E5_SMALL_LOCAL_PATH`` in the environment wins
    and is left alone. Idempotent.
    """
    if os.environ.get(TEXT_ENCODER_ENV_VAR):
        return
    _arch.E5_SMALL_LOCAL_PATH = snapshot_dir


def build_base_model(
    base_model_dir: str, state_dict: dict[str, torch.Tensor], *, device: str | torch.device
) -> TabStarModel:
    """A ``TabStarModel`` holding ``state_dict``, built as ``TabStarModel.from_pretrained(base_model_dir)`` would.

    The config comes from ``base_model_dir`` and the weights from ``state_dict`` (a shallow copy, so
    the registry's dict is never mutated), through the same ``from_pretrained`` machinery the
    library uses: the module is created without random initialization, the tensors are loaded,
    the model is put in eval mode. ``name_or_path`` (on the model and its config) is set to the
    directory, as a file-based build records it, so the ``base_model_name_or_path`` peft writes into
    the LoRA adapter config during fine-tuning is unchanged.

    ``from_pretrained`` assigns the passed tensors to the module's parameters. When ``device`` is a
    CPU device the model would keep training-time references into the shared dict (the checkpoint
    averaging step writes the model's state back into its own parameters), so every parameter or
    buffer that aliases a cached tensor receives its own copy. On a CUDA device the library's
    ``.to(device)`` creates the device copies and the host tensors stay untouched.
    """
    config = TabStarConfig.from_pretrained(base_model_dir, local_files_only=True)
    model = TabStarModel.from_pretrained(None, config=config, state_dict=dict(state_dict))
    model.config.name_or_path = base_model_dir
    model.name_or_path = base_model_dir
    if normalize_device(device) == "cpu":
        _own_aliased_tensors(model, state_dict)
    return model


def _own_aliased_tensors(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Give every parameter or buffer of ``model`` that shares storage with ``state_dict`` its own copy."""
    shared = {tensor.untyped_storage().data_ptr() for tensor in state_dict.values()}
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.untyped_storage().data_ptr() in shared:
            tensor.data = tensor.data.clone()


def load_pretrained_shared(
    base_model_dir: str,
    state_dict: dict[str, torch.Tensor],
    *,
    lora_r: int,
    lora_alpha: int,
    dropout: float,
    device: str | torch.device,
) -> PeftModel:
    """``tabstar.training.lora.load_pretrained`` with the base model built from the shared state dict."""
    model = build_base_model(base_model_dir, state_dict, device=device)
    prefixes = tuple(f"text_encoder.encoder.layer.{i}." for i in _FROZEN_TEXT_ENCODER_LAYERS)
    to_exclude = [name for name, _ in model.named_modules() if name.startswith(prefixes)]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * lora_alpha,
        target_modules=_LORA_MODULES,
        exclude_modules=to_exclude,
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def load_finetuned_shared(
    save_dir: str,
    base_model_dir: str,
    state_dict: dict[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> PeftModel:
    """``tabstar.training.lora.load_finetuned`` with the base model built from the shared state dict."""
    if not exists(save_dir):
        raise FileNotFoundError(f"Checkpoint path {save_dir} does not exist.")
    base_model = build_base_model(base_model_dir, state_dict, device=device)
    return PeftModel.from_pretrained(base_model, save_dir, device_map="cpu", local_files_only=True)


class SharedTabStarTrainer(TabStarTrainer):
    """``TabStarTrainer`` whose base model is built from the shared state dict instead of the checkpoint files.

    ``model_version`` must be the local directory holding the checkpoint's ``config.json`` (the
    wrapper passes the resolved snapshot directory). Training, checkpoint averaging and early
    stopping are inherited unchanged.
    """

    def __init__(
        self,
        max_epochs: int,
        lora_lr: float,
        lora_wd: float,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_batch: int,
        patience: int,
        global_batch: int,
        device: torch.device,
        model_version: str,
        cp_average: bool,
        time_limit: int,
        output_dir: str | None,
        val_batch_size: int,
        base_state_dict: dict[str, torch.Tensor],
    ):
        self.lora_batch = lora_batch
        self.global_batch = global_batch
        self.val_batch_size = val_batch_size
        self.accumulation_steps = set_accumulation_steps(global_batch=global_batch, batch_size=lora_batch)
        self.max_epochs = max_epochs
        self.device = device
        self.cp_average = cp_average
        self.model_version = model_version
        self._base_state_dict = base_state_dict
        self.model = load_pretrained_shared(
            model_version, base_state_dict, lora_r=lora_r, lora_alpha=lora_alpha, dropout=lora_dropout, device=device
        )
        self.model.to(self.device)
        self.optimizer = get_optimizer(model=self.model, lr=lora_lr, wd=lora_wd)
        self.scheduler = get_scheduler(optimizer=self.optimizer, max_lr=lora_lr, epochs=self.max_epochs)
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = EarlyStopping(patience=patience)
        self.cp_manager = CheckpointManager(do_average=self.cp_average, output_dir=output_dir)
        self.steps: int = 0
        self.time_limit = time_limit or 60 * 60 * 10

    def load_model(self) -> PeftModel:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.model = load_finetuned_shared(
            self.cp_manager.to_load_dir, self.model_version, self._base_state_dict, device=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        return self.model


class SharedTabSTARMixin(SharedStateDictEstimatorMixin):
    """Estimator plumbing shared by :class:`SharedTabSTARClassifier` and :class:`SharedTabSTARRegressor`.

    ``configure_shared_weights`` attaches the registry's state dict before ``fit``; the fit then
    runs :class:`SharedTabStarTrainer`. Without an attached dict ``fit`` is the library's.
    """

    def configure_shared_weights(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        """Attach the base checkpoint state dict the next ``fit`` builds from (``None`` detaches it).

        ``pretrain_dataset_or_path`` must name the local directory holding that checkpoint's
        ``config.json``; the wrapper passes the resolved snapshot directory.
        """
        if state_dict is not None and not os.path.isdir(self.model_version):
            raise ValueError(
                f"Shared TabSTAR weights need a local checkpoint directory as `pretrain_dataset_or_path`, "
                f"got {self.model_version!r}."
            )
        super().configure_shared_weights(state_dict)

    def fit(self, X, y, x_val=None, y_val=None):
        """``BaseTabSTAR.fit`` running :class:`SharedTabStarTrainer` when a state dict is attached."""
        if self._shared_state_dict is None:
            return super().fit(X, y, x_val=x_val, y_val=y_val)
        if self.model_ is not None:
            raise ValueError("Model is already trained. Call fit() only once.")
        self.download_base_model()
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        train_data, val_data = self._prepare_for_train(X, y, x_val, y_val)
        self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
        trainer = SharedTabStarTrainer(
            lora_lr=self.lora_lr,
            lora_wd=self.lora_wd,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_batch=self.lora_batch,
            global_batch=self.global_batch,
            max_epochs=self.max_epochs,
            patience=self.patience,
            device=self.device,
            model_version=self.model_version,
            cp_average=self.cp_average,
            time_limit=self.time_limit,
            output_dir=self.output_dir,
            val_batch_size=self.val_batch_size,
            base_state_dict=self._shared_state_dict,
        )
        trainer.train(train_data, val_data)
        self.model_ = trainer.load_model()
        if not self.keep_model:
            trainer.delete_model()
        return None


class SharedTabSTARClassifier(SharedTabSTARMixin, TabSTARClassifier):
    """``TabSTARClassifier`` fine-tuned from the shared base checkpoint."""


class SharedTabSTARRegressor(SharedTabSTARMixin, TabSTARRegressor):
    """``TabSTARRegressor`` fine-tuned from the shared base checkpoint."""
