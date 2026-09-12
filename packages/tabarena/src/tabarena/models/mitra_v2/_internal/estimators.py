"""Mitra-v2 estimators: AutoGluon's Mitra sklearn interface running the frozen recipe.

Imports torch and AutoGluon's Mitra internals, so it is only imported from the wrapper's
fit path, never at module discovery time.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from autogluon.common.utils.random import get_numpy_seed
from autogluon.tabular.models.mitra._internal.config.enums import LossName, Task
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier, MitraRegressor

from tabarena.models.mitra_v2._internal.recipe import RecipeSettings
from tabarena.models.mitra_v2._internal.trainer import MitraV2Trainer, is_cuda_oom

#: Fine-tuning context sizes that fit the GPU, learned from the out-of-memory ratchet in
#: :meth:`MitraV2Mixin._train_ensemble` and keyed on the table shape. The eight children of a bag run
#: one after another in one process on tables of the same shape, so once the first child has found
#: the context that fits, the others start there instead of repeating the attempts that failed.
#: Process-local, like the ratchet itself; the ratchet stays in place as the fallback.
_FITTED_CONTEXT_MEMO: dict[tuple, tuple[int, int]] = {}


def _context_memo_key(task, n_rows: int, n_features: int, support_cap: int, query_cap: int) -> tuple:
    """Memo key: task, feature count, row count to the nearest 256 (bag folds differ by a row), requested caps."""
    return (str(task), int(n_features), round(n_rows / 256), int(support_cap), int(query_cap))


class MitraV2Mixin:
    """Recipe plumbing shared by :class:`MitraV2Classifier` and :class:`MitraV2Regressor`.

    The recipe is attached after construction (:meth:`configure_recipe`) so the sklearn
    constructor keeps the stock signature. The mixin also implements heldout-in-support: the
    validation rows passed to ``fit`` are kept, and once :meth:`activate_heldout_in_support` is
    called they join the in-context support of every later prediction. Until then predictions
    condition on the fit rows alone, which is what keeps validation and out-of-fold predictions
    honest.
    """

    recipe: RecipeSettings = RecipeSettings()
    _heldout: tuple[np.ndarray, np.ndarray] | None = None
    _heldout_in_support_active: bool = False

    def configure_recipe(self, recipe: RecipeSettings) -> None:
        """Attach the per-fit recipe values (call before ``fit``)."""
        self.recipe = recipe

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        result = super().fit(X, y, X_val=X_val, y_val=y_val, time_limit=time_limit)
        if self.recipe.heldout_in_support and X_val is not None and y_val is not None:
            self._heldout = (_values(X_val), _values(y_val))
        return result

    def activate_heldout_in_support(self) -> None:
        """From now on predict with the fit rows plus the stashed validation rows as support."""
        self._heldout_in_support_active = True

    @property
    def heldout_in_support_active(self) -> bool:
        """Whether predictions currently condition on the validation rows as well."""
        return self._heldout_in_support_active and self._heldout is not None

    def _support(self) -> tuple[np.ndarray, np.ndarray]:
        """The in-context support for a prediction: fit rows, plus held-out rows once active."""
        if self.heldout_in_support_active:
            X_extra, y_extra = self._heldout
            return (
                np.concatenate([np.asarray(self.X), X_extra], axis=0),
                np.concatenate([np.asarray(self.y), y_extra], axis=0),
            )
        return self.X, self.y

    def _create_config(self, task, dim_output, time_limit=None):
        cfg, model_cls = super()._create_config(task, dim_output, time_limit)
        recipe = self.recipe
        cfg.hyperparams["weight_decay"] = recipe.weight_decay
        cfg.hyperparams["max_samples_support"] = recipe.finetune_support_cap
        if recipe.fine_tune_budget is not None:
            cfg.hyperparams["budget"] = recipe.fine_tune_budget
        if cfg.task == Task.REGRESSION:
            if recipe.n_bins is None:
                raise ValueError("Mitra-v2 regression needs `n_bins`, the head width read from the checkpoint config.")
            # The released regressor is a cross-entropy head over target bins; stock AutoGluon
            # would build a one-output MSE head that does not match the checkpoint.
            cfg.hyperparams["regression_loss"] = LossName.CROSS_ENTROPY
            cfg.hyperparams["dim_output"] = recipe.n_bins
        return cfg, model_cls

    def _train_ensemble(self, X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=0, time_limit=None):
        """Stock training loop, constructing :class:`MitraV2Trainer` instead of the stock trainer."""
        cfg, model_cls = self._create_config(task, dim_output, time_limit)
        rng = np.random.RandomState(get_numpy_seed(cfg.seed))

        hp = cfg.hyperparams
        memo_key = _context_memo_key(
            task, len(X_train), X_train.shape[1], hp["max_samples_support"], hp["max_samples_query"]
        )
        memo = _FITTED_CONTEXT_MEMO.get(memo_key)
        if memo is not None and memo < (hp["max_samples_support"], hp["max_samples_query"]):
            print(
                f"Starting fine-tuning at max_samples_support={memo[0]}, max_samples_query={memo[1]}: the context that fit this table shape earlier in this process."
            )
            hp["max_samples_support"], hp["max_samples_query"] = memo

        success = False
        while not success and cfg.hyperparams["max_samples_support"] > 0 and cfg.hyperparams["max_samples_query"] > 0:
            model = None
            trainer = None
            try:
                self.trainers.clear()
                self.train_time = 0
                for _ in range(self.n_estimators):
                    model = model_cls.from_pretrained(self.hf_model, device=self.device)
                    trainer = MitraV2Trainer(
                        cfg,
                        model,
                        n_classes=n_classes,
                        device=self.device,
                        rng=rng,
                        verbose=self.verbose,
                        recipe=self.recipe,
                    )
                    start_time = time.time()
                    trainer.train(X_train, y_train, X_valid, y_valid)
                    self.trainers.append(trainer)
                    self.train_time += time.time() - start_time
                success = True
                _FITTED_CONTEXT_MEMO[memo_key] = (hp["max_samples_support"], hp["max_samples_query"])
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                self.trainers.clear()
                trainer = None
                model = None
                torch.cuda.empty_cache()
                # Same fallback as stock AutoGluon: shrink the fine-tuning context and retry.
                old_support = cfg.hyperparams["max_samples_support"]
                cfg.hyperparams["max_samples_support"] = old_support // 2
                print(f"Reducing max_samples_support from {old_support} to {old_support // 2} due to OOM error.")
                if old_support < 2048:
                    old_query = cfg.hyperparams["max_samples_query"]
                    cfg.hyperparams["max_samples_query"] = old_query // 2
                    print(f"Reducing max_samples_query from {old_query} to {old_query // 2} due to OOM error.")

        if not success:
            raise RuntimeError("Failed to train Mitra-v2 after multiple attempts due to out of memory error.")
        return self


class MitraV2Classifier(MitraV2Mixin, MitraClassifier):
    """Mitra-v2 classifier: stock AutoGluon ``MitraClassifier`` under the frozen recipe."""

    def predict_proba(self, X):
        X = _values(X)
        X_support, y_support = self._support()
        n_classes = len(np.unique(y_support))
        probabilities = []
        for trainer in self.trainers:
            logits = trainer.predict(X_support, y_support, X)[..., :n_classes]
            exp_logits = np.exp(logits)
            probabilities.append(exp_logits / exp_logits.sum(axis=1, keepdims=True))
        return sum(probabilities) / len(probabilities)


class MitraV2Regressor(MitraV2Mixin, MitraRegressor):
    """Mitra-v2 regressor: stock AutoGluon ``MitraRegressor`` under the frozen recipe."""

    def predict(self, X):
        X = _values(X)
        X_support, y_support = self._support()
        predictions = [trainer.predict(X_support, y_support, X) for trainer in self.trainers]
        return sum(predictions) / len(predictions)


def _values(data) -> np.ndarray:
    return data.values if isinstance(data, pd.DataFrame | pd.Series) else np.asarray(data)
