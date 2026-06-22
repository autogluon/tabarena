"""Driver that trains LightAutoML's DenseLight neural network end to end.

This module reuses LightAutoML's *own* neural-net training stack — imported from an
installed ``lightautoml`` package — rather than vendoring a copy. Only the DenseLight
model and the surrounding training pieces are touched:

* ``lightautoml.ml_algo.torch_based.nn_models.DenseLightModel`` — the network,
* ``lightautoml.text.embed.{ContEmbedder,CatEmbedder}`` — the default flat embedders,
* ``lightautoml.text.nn_model.{TorchUniversalModel,UniversalDataset}`` — the model wrapper + dataset,
* ``lightautoml.text.trainer.Trainer`` (+ its ``SnapshotEns``) — the training loop / SWA / early stop,
* ``lightautoml.tasks.Task`` — supplies the torch loss + metric for the problem type.

None of LightAutoML's AutoML / reader / preset / report machinery is imported, so a
``pip install lightautoml --no-deps`` install (relying on torch/numpy/pandas/scikit-learn,
which TabArena already provides) is sufficient. The import chain of the modules above pulls
in only those already-present packages.

The defaults below mirror LightAutoML's ``tabular_config.yml`` ``nn_params`` for ``denselight``
(``hidden_size=[512, 256]``, ``LeakyReLU``, ``n_epochs=50``, snapshot/SWA early stopping by
validation loss, StandardScaler numerics via ``use_qnt: false``) plus the code defaults from
``TorchModel._default_params`` (Adam ``lr=3e-4``, ``ReduceLROnPlateau``, last-layer bias init).

The per-epoch loop is driven here (using ``Trainer``'s public ``train`` / ``test`` /
``SnapshotEns`` pieces) instead of via ``Trainer.fit`` so that TabArena's ``time_limit`` can
break training between epochs — ``Trainer`` itself has no time budget.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

logger = logging.getLogger(__name__)

# AutoGluon problem_type -> LightAutoML task name.
_PROBLEM_TYPE_TO_TASK = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}


def _auto_batch_size(rows_num: int) -> int:
    """Replicates ``TorchModel.init_params_on_input`` adaptive batch size."""
    bs = 256
    if rows_num > 50_000:
        bs = 512
    if rows_num > 100_000:
        bs = 1024
    if rows_num > 1_000_000:
        bs = 2048
    return bs


def _get_mean_target(target: np.ndarray, task_name: str, n_out: int) -> np.ndarray:
    """Last-layer bias init, mirroring ``TorchModel.get_mean_target``.

    For multiclass we use ``np.bincount(..., minlength=n_out)`` instead of upstream's
    ``np.unique(..., return_counts=True)`` so the bias vector always has length ``n_out``
    even when a fold is missing a class (equivalent when every class is present).
    """
    from lightautoml.text.utils import inv_sigmoid, inv_softmax

    target = np.asarray(target)
    if task_name == "multiclass":
        bias = np.bincount(target.astype(int).reshape(-1), minlength=n_out).astype(float)
        bias = inv_softmax(bias)
    else:
        bias = np.nanmean(target.astype(float).reshape(target.shape[0], -1), axis=0)
        if task_name in ("binary", "multilabel"):
            bias = inv_sigmoid(bias)

    bias[np.isposinf(bias)] = np.nanmax(bias[~np.isposinf(bias)])
    bias[np.isneginf(bias)] = np.nanmin(bias[~np.isneginf(bias)])
    bias[np.isnan(bias)] = np.nanmean(bias[~np.isnan(bias)])
    return bias


class DenseLightRunner:
    """Fits and predicts with LightAutoML's DenseLight network on preprocessed arrays.

    The AutoGluon wrapper (:class:`tabarena.models.denselight.model.DenseLightModel`) hands
    this runner an already-``_preprocess``-ed pandas frame (categoricals as ``category`` dtype,
    everything else numeric). Numeric standardization (StandardScaler) and categorical integer
    encoding (``0`` reserved for unknown / unseen, matching LightAutoML's ``LabelEncoder``) are
    done here so they fit on train and apply unchanged at predict time.

    Args:
        problem_type: AutoGluon problem type (``binary`` / ``multiclass`` / ``regression``).
        num_classes: Number of classes (used for multiclass output dim); ``None`` otherwise.
        device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        n_threads: Torch intra-op thread count (wired from ``num_cpus``); ``None`` to leave as is.
        stopping_metric: AutoGluon scorer (kept for parity; LightAutoML early-stops by val loss).
        **config: Hyperparameters overriding the LightAutoML denselight defaults.
    """

    def __init__(
        self,
        *,
        problem_type: str,
        num_classes: int | None,
        device: str,
        n_threads: int | None,
        stopping_metric: Scorer | None = None,
        **config: Any,
    ):
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.device = device
        self.n_threads = n_threads
        self.stopping_metric = stopping_metric
        self.config = config

        # Fitted state.
        self.torch_model_ = None
        self.task_name_: str | None = None
        self.n_out_: int | None = None
        self.num_features_: list = []
        self.cat_features_: list = []
        self.cat_dims_: list[int] = []
        self.cat_maps_: list[dict] = []
        self.num_imputer_ = None
        self.num_scaler_ = None
        self.bs_: int = 256

    def _hp(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    @staticmethod
    def _resolve_act_fun(act_fun: Any):
        from torch import nn

        return getattr(nn, act_fun) if isinstance(act_fun, str) else act_fun

    # ------------------------------------------------------------------ preprocessing
    def _fit_transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if not self.num_features_:
            return np.empty((len(X), 0), dtype=np.float32)
        arr = X[self.num_features_].to_numpy(dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan  # FillInf
        self.num_imputer_ = SimpleImputer(strategy="mean", keep_empty_features=True)  # FillnaMean
        self.num_scaler_ = StandardScaler()  # use_qnt: false -> StandardScaler
        arr = self.num_scaler_.fit_transform(self.num_imputer_.fit_transform(arr))
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        if not self.num_features_:
            return np.empty((len(X), 0), dtype=np.float32)
        arr = X[self.num_features_].to_numpy(dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan
        arr = self.num_scaler_.transform(self.num_imputer_.transform(arr))
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _fit_transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        self.cat_maps_ = []
        self.cat_dims_ = []
        if not self.cat_features_:
            return np.empty((len(X), 0), dtype=np.int64)
        cols = []
        for c in self.cat_features_:
            cats = X[c].astype("category").cat.categories
            mapping = {v: i + 1 for i, v in enumerate(cats)}  # 0 reserved for unknown / NaN
            self.cat_maps_.append(mapping)
            self.cat_dims_.append(len(mapping) + 1)
            cols.append(self._encode_cat_col(X[c], mapping))
        return np.stack(cols, axis=1)

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        if not self.cat_features_:
            return np.empty((len(X), 0), dtype=np.int64)
        cols = [
            self._encode_cat_col(X[c], mapping) for c, mapping in zip(self.cat_features_, self.cat_maps_, strict=True)
        ]
        return np.stack(cols, axis=1)

    @staticmethod
    def _encode_cat_col(s: pd.Series, mapping: dict) -> np.ndarray:
        codes = s.map(mapping).to_numpy(dtype="float64")
        codes[np.isnan(codes)] = 0  # unseen / missing -> unknown bin
        return codes.astype(np.int64)

    def _encode_target(self, y) -> np.ndarray:
        arr = np.asarray(y)
        return arr.astype(np.float32) if self.problem_type == "regression" else arr.astype(np.int64)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_val: pd.DataFrame,
        y_val,
        num_features: list,
        cat_features: list,
        time_to_fit_in_seconds: float | None = None,
    ) -> None:
        import torch

        # LightAutoML's own neural-net stack. Its net class is also named ``DenseLightModel``; alias
        # it to ``DenseLightNet`` to avoid confusion with the AutoGluon wrapper of the same name.
        from lightautoml.ml_algo.torch_based.nn_models import DenseLightModel as DenseLightNet
        from lightautoml.tasks import Task
        from lightautoml.text.embed import CatEmbedder, ContEmbedder
        from lightautoml.text.nn_model import TorchUniversalModel
        from lightautoml.text.trainer import Trainer
        from lightautoml.text.utils import seed_everything

        start_time = time.time()
        seed = int(self._hp("random_state", 42))
        deterministic = bool(self._hp("deterministic", True))
        seed_everything(seed, deterministic)
        if self.n_threads is not None:
            torch.set_num_threads(self.n_threads)

        device = torch.device(self.device)
        self.task_name_ = _PROBLEM_TYPE_TO_TASK[self.problem_type]
        self.n_out_ = int(self.num_classes) if self.problem_type == "multiclass" else 1
        self.num_features_ = list(num_features)
        self.cat_features_ = list(cat_features)
        is_cont = len(self.num_features_) > 0
        is_cat = len(self.cat_features_) > 0

        x_cont_tr = self._fit_transform_numeric(X_train)
        x_cat_tr = self._fit_transform_categorical(X_train)
        x_cont_val = self._transform_numeric(X_val)
        x_cat_val = self._transform_categorical(X_val)
        y_tr = self._encode_target(y_train)
        y_va = self._encode_target(y_val)

        task = Task(self.task_name_)
        loss = task.losses["torch"].loss
        metric = task.losses["torch"].metric_func

        bias = None
        if self._hp("init_bias", True):
            try:
                bias = _get_mean_target(y_tr, self.task_name_, self.n_out_)
            except Exception:  # bias init is best effort, mirroring LightAutoML's own try/except
                bias = None

        # Param dict consumed by TorchUniversalModel (+ embedders, + the DenseLight backbone
        # via **kwargs forwarding). Keys mirror LightAutoML's denselight ``nn_params``.
        embedding_size = int(self._hp("embedding_size", 10))
        net_params: dict[str, Any] = {
            "task": task,
            "loss": loss,
            "n_out": self.n_out_,
            "loss_on_logits": True,
            "bias": bias,
            "torch_model": DenseLightNet,
            "cont_embedder_": ContEmbedder if is_cont else None,
            "cont_params": {
                "num_dims": len(self.num_features_),
                "input_bn": bool(self._hp("input_bn", False)),
                "embedding_size": embedding_size,
                "bins": None,
            }
            if is_cont
            else None,
            "cat_embedder_": CatEmbedder if is_cat else None,
            "cat_params": {
                "cat_dims": self.cat_dims_,
                "emb_dropout": float(self._hp("emb_dropout", 0.1)),
                "emb_ratio": int(self._hp("emb_ratio", 3)),
                "max_emb_size": int(self._hp("max_emb_size", 256)),
                "embedding_size": embedding_size,
            }
            if is_cat
            else None,
            "text_embedder": None,
            "text_params": None,
            # DenseLight backbone kwargs (forwarded by TorchUniversalModel).
            "hidden_size": list(self._hp("hidden_size", [512, 256])),
            "drop_rate": self._hp("drop_rate", 0.1),
            "act_fun": self._resolve_act_fun(self._hp("act_fun", "LeakyReLU")),
            "noise_std": float(self._hp("noise_std", 0.05)),
            "num_init_features": self._hp("num_init_features", None),
            "use_bn": bool(self._hp("use_bn", True)),
            "use_noise": bool(self._hp("use_noise", False)),
            "concat_input": bool(self._hp("concat_input", True)),
            "dropout_first": bool(self._hp("dropout_first", True)),
            "bn_momentum": float(self._hp("bn_momentum", 0.1)),
            "ghost_batch": self._hp("ghost_batch", None),
            "use_skip": bool(self._hp("use_skip", False)),
            "leaky_gate": bool(self._hp("leaky_gate", False)),
            "weighted_sum": bool(self._hp("weighted_sum", True)),
            "device": device,
        }

        rows_num = len(y_tr) + len(y_va)
        bs = self._hp("bs", "auto")
        bs = _auto_batch_size(rows_num) if bs in ("auto", None) else int(bs)
        self.bs_ = int(min(bs, max(1, len(y_tr))))

        trainer = Trainer(
            net=TorchUniversalModel,
            net_params=net_params,
            opt=torch.optim.Adam,
            opt_params={"lr": float(self._hp("lr", 3e-4)), "weight_decay": float(self._hp("weight_decay", 0.0))},
            n_epochs=int(self._hp("n_epochs", 50)),
            device=device,
            device_ids=None,
            metric=metric,
            snap_params={
                "k": int(self._hp("snap_k", 3)),
                "early_stopping": True,
                "patience": int(self._hp("patience", 10)),
                "swa": bool(self._hp("swa", True)),
            },
            is_snap=False,
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"patience": 10, "factor": 1e-2, "min_lr": 1e-5},
            verbose=None,
            verbose_bar=False,
            apex=False,
            stop_by_metric=False,
        )

        dataloaders = {
            "train": self._make_loader(x_cont_tr, x_cat_tr, y_tr, stage="train"),
            "val": self._make_loader(x_cont_val, x_cat_val, y_va, stage="val"),
        }

        # Per-epoch loop mirroring Trainer.fit, but with a wall-clock budget check between epochs.
        trainer._init()  # the model/optimizer/scheduler init step Trainer.fit performs internally
        budget = time_to_fit_in_seconds * 0.95 if time_to_fit_in_seconds is not None else None
        for epoch in range(trainer.n_epochs):
            trainer.epoch = epoch
            trainer.train(dataloaders=dataloaders)
            val_loss, _, _ = trainer.test(dataloader=dataloaders["val"])
            cond = float(np.mean(val_loss))
            trainer.se.update(trainer.model, cond)
            if trainer.scheduler is not None:
                trainer.scheduler.step(cond)
            if trainer.se.early_stop:
                logger.log(15, f"DenseLight: early stopping at epoch {epoch}.")
                break
            if budget is not None and (time.time() - start_time) >= budget:
                logger.log(15, f"DenseLight: stopping at epoch {epoch} (time_limit reached).")
                break

        trainer.se.set_best_params(trainer.model)  # load SWA-averaged best weights into trainer.model
        self.torch_model_ = trainer.model.to(device)
        self.torch_model_.eval()
        trainer.clean()  # drop optimizer / scheduler / snapshot copies; we keep only the fitted module

    def _make_loader(self, x_cont: np.ndarray, x_cat: np.ndarray, y: np.ndarray, stage: str):
        import torch
        from lightautoml.text.nn_model import UniversalDataset
        from lightautoml.text.utils import collate_dict

        data: dict[str, np.ndarray] = {}
        if self.num_features_:
            data["cont"] = x_cont
        if self.cat_features_:
            data["cat"] = x_cat
        dataset = UniversalDataset(
            data=data,
            y=y,
            w=np.ones(len(y), dtype=np.float32),
            tokenizer=None,
            max_length=256,
            stage=stage,
        )
        is_train = stage == "train"
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs_,
            shuffle=is_train,
            num_workers=int(self._hp("num_workers", 0)),
            collate_fn=collate_dict,
            drop_last=is_train and len(y) > self.bs_,
        )

    # ------------------------------------------------------------------ predict
    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        from lightautoml.text.nn_model import UniversalDataset
        from lightautoml.text.utils import _dtypes_mapping, collate_dict

        x_cont = self._transform_numeric(X)
        x_cat = self._transform_categorical(X)
        n = len(X)
        data: dict[str, np.ndarray] = {}
        if self.num_features_:
            data["cont"] = x_cont
        if self.cat_features_:
            data["cat"] = x_cat
        dataset = UniversalDataset(
            data=data,
            y=np.ones(n, dtype=np.float32),
            w=np.ones(n, dtype=np.float32),
            tokenizer=None,
            max_length=256,
            stage="test",
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.bs_),
            shuffle=False,
            num_workers=int(self._hp("num_workers", 0)),
            collate_fn=collate_dict,
        )
        device = next(self.torch_model_.parameters()).device
        self.torch_model_.eval()
        preds = []
        with torch.no_grad():
            for sample in loader:
                batch = {
                    k: (v.long().to(device) if _dtypes_mapping[k] == "long" else v.to(device))
                    for k, v in sample.items()
                }
                preds.append(self.torch_model_.predict(batch).detach().cpu().numpy())
        return np.vstack(preds) if preds and preds[0].ndim == 2 else np.hstack(preds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regression predictions, shape ``(n,)``."""
        return self._predict_raw(X).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities: ``(n,)`` positive-class for binary, ``(n, n_classes)`` for multiclass."""
        pred = self._predict_raw(X)
        return pred if self.problem_type == "multiclass" else pred.reshape(-1)
