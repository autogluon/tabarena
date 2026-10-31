"""Fit and predict one agentic RealMLP with the modules of ``bin.realmlp.modules``.

``_fit_member`` and the predict methods follow ``_fit_realmlp`` and ``_predict`` in ``bin/realmlp/pipeline.py``
(agentic-tabular 3d7349b), with the time limit passed to the train module and a preprocessor that pickles.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import delu
import numpy as np
import pandas as pd
import torch
from bin.realmlp.modules import TabPreprocessor, resolve
from bin.realmlp.modules.inference_clf.realmlp import score_classifier
from bin.realmlp.modules.inference_reg.realmlp import score_regressor
from loguru import logger

# Rows used for the data-dependent initialization; halved with the predict batch size on CUDA OOM.
_INIT_BATCH_SIZE = 4096


@contextmanager
def torch_state(device: torch.device, n_threads: int):
    """Use ``n_threads`` torch threads, and TF32 on CUDA, inside the block; restore both afterwards."""
    old = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_num_threads(n_threads)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.set_num_threads(old[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[1:]


@dataclass
class FittedPreprocessor:
    """A fitted ``TabPreprocessor`` without its stage closures, which do not pickle."""

    task_type: str
    cfg: dict[str, Any]
    state: dict[str, Any]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        indices = self.cfg["implementation_indices"]
        preprocessor = TabPreprocessor(
            resolve("numerical_preprocess", indices["numerical_preprocess"], self.task_type)(self.cfg),
            resolve("categorical_preprocess", indices["categorical_preprocess"], self.task_type)(self.cfg),
        )
        vars(preprocessor).update(self.state)
        return preprocessor.transform(frame)


@dataclass
class FittedMember:
    preprocessor: FittedPreprocessor
    model: torch.nn.Module
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    y_min: np.ndarray | None = None
    y_max: np.ndarray | None = None


class RealMLPImplementation:
    """One fitted agentic RealMLP (a single bag child)."""

    def __init__(self, *, problem_type: str, device: str, n_threads: int, eval_metric, indices: dict, **params):
        self.problem_type = problem_type
        self.task_type = "regression" if problem_type == "regression" else "multiclass"
        self.device_ = torch.device(device)
        self.n_threads = n_threads
        self.seed = int(params.pop("random_state") or 0)
        cfg = dict(params, implementation_indices=indices)
        cfg.pop("is_large")  # a sampling switch; no module reads it
        cfg["sq_mom"] = 1.0 - cfg.pop("one_minus_sq_mom")
        cfg["max_one_hot_cat_size"] = math.floor(cfg.pop("max_one_hot_cat_size_raw"))
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg.setdefault("predict_batch_size", 8192)
        if problem_type != "regression":
            cfg["val_metric_name"] = "class_error" if eval_metric.name == "accuracy" else "cross_entropy"
        self.cfg = cfg
        self.member_ = None

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean-imputed float32 numerical columns, then the categorical columns as codes over the training
        categories, with unknown and missing values on the extra code ``len(categories)``.
        """
        frames = []
        if self.num_cols_:
            x_num = X[self.num_cols_].to_numpy(dtype=np.float32)
            x_num = np.where(np.isnan(x_num), self.num_fill_, x_num)
            frames.append(pd.DataFrame(x_num, columns=[f"num_{i}" for i in range(x_num.shape[1])]))
        if self.categories_:
            columns = []
            for col, cats in self.categories_.items():
                codes = cats.get_indexer(X[col])
                columns.append(np.where(codes < 0, len(cats), codes).astype(np.int64))
            x_cat = np.column_stack(columns)
            frames.append(pd.DataFrame(x_cat, columns=[f"cat_{i}" for i in range(x_cat.shape[1])]).astype("category"))
        return pd.concat(frames, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, time_limit: float | None):
        delu.random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        cat_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
        self.num_cols_ = [col for col in X.columns if col not in cat_cols]
        # A sliced Categorical can keep levels seen only outside this fold; drop them so they map to unknown.
        self.categories_ = {col: X[col].cat.remove_unused_categories().cat.categories for col in cat_cols}
        if self.num_cols_:
            x_num = X[self.num_cols_].to_numpy(dtype=np.float32)
            observed = ~np.isnan(x_num)
            sums = np.where(observed, x_num, 0.0).sum(axis=0, dtype=np.float64)
            counts = observed.sum(axis=0)
            fill = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
            self.num_fill_ = np.where(np.isnan(fill), np.float32(0.0), fill)
        x_train, x_val = self._frame(X), self._frame(X_val)
        cat_indicator = [False] * len(self.num_cols_) + [True] * len(cat_cols)

        if self.problem_type == "regression":
            y_train = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            out_dim = 1
        else:
            index = self.cfg["implementation_indices"]["target_preprocess"]
            y_int, classes, _ = resolve("target_preprocess", index, "multiclass")(
                np.concatenate([np.asarray(y), np.asarray(y_val)])
            )
            y_train, y_val, out_dim = y_int[: len(y)], y_int[len(y) :], len(classes)

        init_batch_size, start = _INIT_BATCH_SIZE, time.monotonic()
        with torch_state(self.device_, self.n_threads):
            while True:
                remaining = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - start))
                try:
                    self.member_ = self._fit_member(
                        x_train, x_val, y_train, y_val, cat_indicator, out_dim, init_batch_size, remaining
                    )
                    return
                except Exception as err:
                    predict_batch_size = int(self.cfg["predict_batch_size"])
                    oom = isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()
                    if not oom or predict_batch_size <= 1:
                        raise
                    logger.warning(f"fit OOM with predict_batch_size={predict_batch_size}; retrying with half")
                    self.cfg["predict_batch_size"] = predict_batch_size // 2
                    init_batch_size = max(1, init_batch_size // 2)
                    if self.device_.type == "cuda":
                        torch.cuda.empty_cache()

    def _fit_member(self, x_train, x_val, y_train, y_val, cat_indicator, out_dim, init_batch_size, time_limit):
        cfg, indices, task_type, device = self.cfg, self.cfg["implementation_indices"], self.task_type, self.device_
        prep = TabPreprocessor(
            resolve("numerical_preprocess", indices["numerical_preprocess"], task_type)(cfg),
            resolve("categorical_preprocess", indices["categorical_preprocess"], task_type)(cfg),
        )
        prep.fit(x_train, cat_indicator=cat_indicator)
        x_cont_train, x_cat_train = prep.transform(x_train)
        x_cont_val, x_cat_val = prep.transform(x_val)

        y_dtype = torch.float32 if task_type == "regression" else torch.long
        score_target = torch.as_tensor(y_val, dtype=y_dtype, device=device)  # untransformed targets
        y_stats = {}
        if task_type == "regression":
            target_fn = resolve("target_preprocess", indices["target_preprocess"], task_type)
            y_train, _, y_stats = target_fn(y_train, y_val, cfg)

        tx = torch.as_tensor(x_cont_train, dtype=torch.float32, device=device)
        tc = torch.as_tensor(x_cat_train, dtype=torch.long, device=device)
        vx = torch.as_tensor(x_cont_val, dtype=torch.float32, device=device)
        vc = torch.as_tensor(x_cat_val, dtype=torch.long, device=device)
        ty = torch.as_tensor(y_train, dtype=y_dtype, device=device)

        num_emb, num_emb_out = resolve("num_embedding", indices["num_embedding"], task_type)(prep.n_num_, cfg)
        cat_emb, cat_emb_out = resolve("cat_embedding", indices["cat_embedding"], task_type)(prep.large_cat_sizes_, cfg)
        model = resolve("model", indices["model"], task_type)(
            num_emb,
            cat_emb,
            num_embedding_out=num_emb_out,
            n_num=prep.n_num_,
            n_one_hot=prep.n_one_hot_,
            cat_embedding_out=cat_emb_out,
            cfg=cfg,
            out_dim=out_dim,
        ).to(device)
        init_idx = torch.randperm(tx.shape[0], device=device)[: min(init_batch_size, tx.shape[0])]
        model.initialize_from_data(tx[init_idx], tc[init_idx])

        optimizer = resolve("optimizer", indices["optimizer"], task_type)(model, cfg)
        loss_fn = resolve("loss", indices["loss"], task_type)

        def score_fn(m, xc, xt, yv):
            if task_type == "regression":
                return score_regressor(m, xc, xt, yv, cfg, **y_stats)
            return score_classifier(m, xc, xt, yv, cfg)

        train_fn = resolve("train", indices["train"], task_type)
        result = train_fn(
            model,
            optimizer,
            loss_fn,
            tx,
            tc,
            ty,
            vx,
            vc,
            score_target,
            score_fn,
            cfg,
            has_val=True,
            time_limit_s=time_limit,
        )
        model.load_state_dict(result.best_state)
        model.eval()
        model.to("cpu")
        state = {key: value for key, value in vars(prep).items() if key not in {"_num", "_cat"}}
        return FittedMember(FittedPreprocessor(task_type, dict(cfg), state), model, **y_stats)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        inference_fn = resolve("inference", self.cfg["implementation_indices"]["inference"], "regression")
        with torch_state(self.device_, self.n_threads):
            pred = inference_fn([self.member_], self._frame(X), self.cfg, is_y_1d=True, is_y_float64=False)
        return np.asarray(pred).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        inference_fn = resolve("inference", self.cfg["implementation_indices"]["inference"], "multiclass")
        with torch_state(self.device_, self.n_threads):
            proba = np.asarray(inference_fn([self.member_], self._frame(X), self.cfg))
        # Float32 averaging over the ensemble can exceed 1 by an ulp, which sklearn's log_loss rejects.
        proba = np.clip(proba, 0.0, 1.0)
        if self.problem_type == "binary":
            return np.column_stack([1.0 - proba[:, 1], proba[:, 1]])
        return proba
