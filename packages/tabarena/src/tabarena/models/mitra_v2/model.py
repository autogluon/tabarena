from __future__ import annotations

import contextlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular.models.mitra.mitra_model import MitraModel

from tabarena.models.mitra_v2._internal import recipe

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class MitraV2Model(MitraModel):
    """Mitra-v2: the second-generation Mitra tabular foundation model, fine-tuned per bag child.

    Mitra-v2 keeps Mitra's 12-layer 2D-attention backbone (77M parameters) and is pretrained on
    a far larger and more diverse synthetic prior. It is deployed as a fine-tuned, bagged model:
    every bag child fine-tunes the checkpoint on its fit fold for 50 steps, validates on its
    held-out fold, and predicts in context. This wrapper runs one such child; TabArena's
    standard 8-fold bagging is the protocol behind the reported numbers, so nothing is bagged
    or refit inside the wrapper.

    Paper: Mitra-v2 Technical Report (arXiv:2609.04540)
    Authors: Yefan Tao, Xiyuan Zhang, Xinyi Liu, Boran Han, Danielle Maddix, Haoyang Fang, Zhen Han,
        Jiading Gai, Xuanqing Liu, Michael Bohlke-Schneider, Yuyang (Bernie) Wang, Gerald Friedland,
        Kevan Mah, Chris Lee, Chris Kong (Amazon)
    Codebase: https://huggingface.co/autogluon/mitra-finetune (fine-tuning recipe and results),
        https://huggingface.co/autogluon/mitra-classifier-2 and
        https://huggingface.co/autogluon/mitra-regressor-2 (weights)
    License: Apache-2.0

    The wrapper reproduces the frozen deployment recipe of the ``mitra-finetune`` package on top
    of AutoGluon's Mitra implementation, as subclasses rather than the package's process-global
    patches (see ``_internal/recipe.py`` for the values and their source):

    * Pinned Mitra-v2 checkpoints; the regressor's 1,000-bin cross-entropy head is decoded to
      the mean of the predicted distribution.
    * Fine-tuning schedule: 50 steps, AdamW with learning rate 1e-5 (3e-6 on binary tasks with at
      most 16,384 training rows), 10 warm-up steps, weight decay 0.3, a 250 s wall-clock budget
      per child, and an in-context support of up to 16,384 (classification) or 20,480
      (regression) rows.
    * Prediction: in-context support of up to 16,384 (binary) or 32,768 (multiclass, regression)
      rows, class-balanced on binary tasks, halved under GPU memory pressure.
    * Heldout in support: after its out-of-fold predictions, a bag child predicts with its fit
      fold plus its held-out fold as support, so the bag needs no refit. The held-out labels are
      used only as fine-tuning validation and as support rows, never for test information.
    * Wide tables (more than 256 features) are reduced on the training rows alone: top-256
      ANOVA-F columns for predominantly continuous classification tables, a 256-component PCA
      for regression.

    Not covered: the package's hierarchical decomposition for more than ten classes (no TabArena
    dataset exceeds ten, so ``max_classes`` stays at the checkpoint's head width), and its
    truncation of a bag whose time limit is hit; AutoGluon fails the bag instead.

    Requires a CUDA GPU. ``flash-attn`` is optional and speeds up attention; the reported numbers
    were calibrated without it. Install with ``pip install tabarena[mitra_v2]``.
    """

    ag_key = "TA-MITRA-V2"
    ag_name = "TA-Mitra-v2"
    ag_priority = 65
    minimum_num_gpus = 1

    #: Lifts the stock Mitra caps (10,000 rows, 500 features): the recipe subsamples the support
    #: and reduces wide tables itself. The class cap is the checkpoint's head width.
    _default_auxiliary_params_extra = {
        "max_rows": 1_000_000,
        "max_features": 50_000,
        "max_classes": 10,
    }
    #: No refit: the bag children are the model (heldout-in-support gives each child the whole
    #: training table as context), and a refit would drop the fine-tuning validation split.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wide_table_reducer: recipe.WideTableReducer | None = None

    def get_model_cls(self):
        from tabarena.models.mitra_v2._internal.estimators import MitraV2Classifier, MitraV2Regressor

        if self.problem_type in ["binary", "multiclass"]:
            return MitraV2Classifier
        if self.problem_type == "regression":
            return MitraV2Regressor
        raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

    def _set_default_params(self):
        default_params = {
            "n_estimators": 1,
            "fine_tune": True,
            "fine_tune_steps": recipe.FINE_TUNE_STEPS,
            "lr": recipe.LEARNING_RATE,
            "warmup_steps": recipe.WARMUP_STEPS,
            # --- recipe values consumed by this wrapper, not by AutoGluon's Mitra constructor ---
            "small_binary_lr": recipe.SMALL_BINARY_LEARNING_RATE,
            "small_binary_max_rows": recipe.SMALL_BINARY_MAX_ROWS,
            "weight_decay": recipe.WEIGHT_DECAY,
            "fine_tune_budget": recipe.FINE_TUNE_BUDGET_S,
            "finetune_support_cap": None,  # None: the recipe's task-dependent value
            "predict_support_cap": None,  # None: the recipe's task-dependent value
            "predict_support_floor": recipe.PREDICT_SUPPORT_FLOOR,
            "predict_query_chunk": recipe.PREDICT_QUERY_CHUNK,
            "predict_query_chunk_floor": recipe.PREDICT_QUERY_CHUNK_FLOOR,
            "balanced_binary_support": True,
            "max_features_budget": recipe.MAX_FEATURES_BUDGET,
            # "auto": the held-out fold of a bag child joins its support; an external validation
            # set (holdout fit) does not. True / False force either behavior.
            "heldout_in_support": "auto",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    _WRAPPER_PARAMS = (
        "small_binary_lr",
        "small_binary_max_rows",
        "weight_decay",
        "fine_tune_budget",
        "finetune_support_cap",
        "predict_support_cap",
        "predict_support_floor",
        "predict_query_chunk",
        "predict_query_chunk_floor",
        "balanced_binary_support",
        "max_features_budget",
        "heldout_in_support",
    )

    def _preprocess(
        self, X: pd.DataFrame, is_train: bool = False, y: pd.Series | None = None, **kwargs
    ) -> pd.DataFrame:
        """Stock Mitra preprocessing (label-encoded categoricals), then the wide-table reduction.

        The reduction is fit on the training rows only and applied unchanged afterwards. It is
        skipped on tables with categorical columns, as in the reference pipeline, which only
        reduces tables it can read as a float matrix.
        """
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        if is_train:
            self._wide_table_reducer = None
            budget = self.params.get("max_features_budget")
            has_categoricals = bool(self._label_encoder is not None and self._label_encoder.features_in)
            if budget and y is not None and not has_categoricals:
                self._wide_table_reducer = recipe.WideTableReducer.fit(
                    X.to_numpy(dtype=float),
                    np.asarray(y),
                    problem_type=self.problem_type,
                    budget=budget,
                )
                if self._wide_table_reducer is not None:
                    logger.log(
                        20,
                        f"\tMitra-v2: reducing {X.shape[1]} features to "
                        f"{self._wide_table_reducer.n_features_out} ({self._wide_table_reducer.kind}).",
                    )
        if self._wide_table_reducer is not None:
            X = pd.DataFrame(self._wide_table_reducer.transform(X.to_numpy(dtype=float)), index=X.index)
        return X

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
        import torch

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        wrapper_params = {name: hyp.pop(name) for name in self._WRAPPER_PARAMS}

        # The recipe conditions the learning rate on the whole outer training table, which for a
        # bag child is its fit fold plus its held-out fold.
        n_outer_rows = len(X) + (len(X_val) if X_val is not None else 0)
        if self.problem_type == "binary" and n_outer_rows <= wrapper_params["small_binary_max_rows"]:
            hyp["lr"] = wrapper_params["small_binary_lr"]

        checkpoint_dir = hyp.pop("hf_model", None) or resolve_checkpoint_dir(self.problem_type)
        hyp["hf_model"] = checkpoint_dir
        for deprecated_key in ["hf_cls_model", "hf_reg_model", "hf_general_model"]:
            hyp.pop(deprecated_key, None)

        self._log_cpu_fallback_warning(num_gpus=num_gpus)
        if hyp.get("device", None) is None:
            hyp["device"] = "cpu" if num_gpus == 0 else self._get_default_device()
        if hyp["device"] == "cpu":
            logger.log(30, "\tWarning: fine-tuning Mitra-v2 on CPU is very slow; the recipe assumes a CUDA GPU.")
        if "verbose" not in hyp:
            hyp["verbose"] = verbosity >= 3

        heldout_in_support = wrapper_params["heldout_in_support"]
        if heldout_in_support == "auto":
            heldout_in_support = recipe.is_bagged_child_name(self.name)
        settings = recipe.RecipeSettings.for_problem_type(
            self.problem_type,
            weight_decay=wrapper_params["weight_decay"],
            finetune_support_cap=wrapper_params["finetune_support_cap"],
            predict_support_cap=wrapper_params["predict_support_cap"],
            predict_support_floor=wrapper_params["predict_support_floor"],
            predict_query_chunk=wrapper_params["predict_query_chunk"],
            predict_query_chunk_floor=wrapper_params["predict_query_chunk_floor"],
            balanced_binary_support=wrapper_params["balanced_binary_support"],
            fine_tune_budget=wrapper_params["fine_tune_budget"],
            heldout_in_support=bool(heldout_in_support),
            n_bins=_head_width(checkpoint_dir) if self.problem_type == "regression" else None,
        )

        torch_threads = torch.get_num_threads()
        with _seeded_global_rngs(hyp.get(self.seed_name)):
            try:
                if isinstance(num_cpus, (int, float)) and num_cpus != torch_threads:
                    torch.set_num_threads(int(num_cpus))
                self.model = model_cls(**hyp)
                self.model.configure_recipe(settings)

                X = self.preprocess(X, y=y, is_train=True)
                if X_val is not None:
                    X_val = self.preprocess(X_val)
                self.model = self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit)
                for trainer in self.model.trainers:
                    trainer.post_fit_optimize()
            finally:
                torch.set_num_threads(torch_threads)

    def reduce_memory_size(
        self, remove_fit: bool = True, remove_info: bool = False, requires_save: bool = True, **kwargs
    ):
        """Stock clean-up, plus the switch to heldout-in-support.

        AutoGluon calls this with ``remove_fit=True`` once a model's validation predictions are
        done and before it is saved: for a bag child right after its out-of-fold predictions
        (``FoldFittingStrategy._predict_oof``), for a standalone model after its validation score
        (``AbstractTrainer.save_model``). That is the moment the held-out rows may join the
        child's support without influencing any validation, and because the switch lands in the
        saved child, every later prediction sees it, including TabArena's per-child test
        predictions from children reloaded from disk.
        """
        super().reduce_memory_size(
            remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs
        )
        if remove_fit and self.model is not None:
            self.model.activate_heldout_in_support()

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        problem_type: str | None = None,
        **kwargs,
    ) -> int:
        """Rough peak-memory estimate for the capped in-context regime.

        Stock Mitra's estimates grow quadratically with the row count and were fit for tables
        of at most 10,000 rows; here the support is capped at prediction time and wide tables are
        reduced, so the dominant term is the activations over the capped support: about eight
        live ``rows x features x 512`` bf16 tensors. Plus the data, and a flat 3 GB for the
        weights, AdamW state, and the CUDA context. TabArena runs pass the GPU's memory as the
        available memory, so this is read against VRAM in practice.
        """
        hyperparameters = hyperparameters or {}
        cap = hyperparameters.get("predict_support_cap") or (
            recipe.default_predict_support_cap(problem_type) if problem_type else recipe.PREDICT_SUPPORT_CAP_MULTICLASS
        )
        budget = hyperparameters.get("max_features_budget", recipe.MAX_FEATURES_BUDGET) or X.shape[1]
        n_rows = min(X.shape[0], cap)
        n_features = min(X.shape[1], budget)
        activation_mem = 8 * n_rows * n_features * 512 * 2
        data_mem = 4 * get_approximate_df_mem_usage(X).sum()
        return int(activation_mem + data_mem + 3e9)

    def _more_tags(self) -> dict:
        # No refit: a refit would drop the fine-tuning validation split the recipe relies on.
        return {"can_refit_full": False}

    @classmethod
    def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
        """Warm torch (plus the CUDA context) and the heavy Mitra imports, untimed and data-free.

        AutoGluon's Mitra interface pulls in ``transformers`` (through its scheduler helpers),
        ``loguru`` and ``einops``, which take seconds on a cold process.
        """
        from tabarena.models.warmup import warmup_imports, warmup_torch

        warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)
        warmup_imports("autogluon.tabular.models.mitra.sklearn_interface")


def resolve_checkpoint_dir(problem_type: str) -> str:
    """Local directory of the pinned Mitra-v2 checkpoint for a task (downloaded if missing)."""
    if problem_type == "regression":
        return _download_checkpoint(recipe.HF_REGRESSOR_REPO, recipe.HF_REGRESSOR_REVISION)
    return _download_checkpoint(recipe.HF_CLASSIFIER_REPO, recipe.HF_CLASSIFIER_REVISION)


def prefetch_weights() -> None:
    """Pre-download both Mitra-v2 checkpoints (classifier and regressor) at their pinned revisions."""
    _download_checkpoint(recipe.HF_CLASSIFIER_REPO, recipe.HF_CLASSIFIER_REVISION)
    _download_checkpoint(recipe.HF_REGRESSOR_REPO, recipe.HF_REGRESSOR_REVISION)


def _download_checkpoint(repo_id: str, revision: str) -> str:
    """Fetch a checkpoint's files at ``revision`` into the Hugging Face cache; return their directory.

    Tries the local cache first so offline compute nodes skip the etag request that
    ``hf_hub_download`` otherwise makes.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    paths = []
    for filename in recipe.CHECKPOINT_FILES:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        paths.append(Path(path).parent)
    if len(set(paths)) != 1:
        raise RuntimeError(f"Checkpoint files of {repo_id}@{revision} resolved to different directories: {paths}")
    return str(paths[0])


def _head_width(checkpoint_dir: str) -> int:
    """The output width (``dim_output``) recorded in a checkpoint directory's ``config.json``."""
    with open(Path(checkpoint_dir) / "config.json") as f:
        return int(json.load(f)["dim_output"])


@contextlib.contextmanager
def _seeded_global_rngs(seed: int | None) -> Iterator[None]:
    """Seed the Python, NumPy and torch global RNGs for a fit and restore their states after.

    Mitra draws its random feature mirror from NumPy's global RNG and its fine-tuning consumes
    torch's global stream, which stock AutoGluon leaves unseeded; the reference evaluation seeds
    them once per process. Seeding per fit makes a child reproducible from its AutoGluon seed,
    and restoring keeps the host process's RNG state untouched.
    """
    if seed is None:
        yield
        return
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
