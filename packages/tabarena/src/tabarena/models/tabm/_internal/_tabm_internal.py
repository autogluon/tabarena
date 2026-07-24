"""Training and preprocessing harness around the official ``tabm`` package.

The model itself (``tabm.TabM``) and the numerical embeddings
(``rtdl_num_embeddings``) come from the upstream PyPI packages
(https://github.com/yandex-research/tabm). This module only supplies the parts
the package leaves to the caller:

* preprocessing -- quantile-normalised numerical features and ordinal-encoded
  categorical features (with a reserved slot for unknown/missing categories);
* the training loop -- AdamW with early stopping and an optional time budget;
* ensemble-aware prediction -- averaging the ``k`` submodel outputs.

The original TabArena implementation was partially adapted from pytabkit's TabM
implementation.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import tabm
import torch
from autogluon.core.metrics import compute_metric
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .tabm_utils import get_tabm_auto_batch_size

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]

logger = logging.getLogger(__name__)


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transformer with the RTDL/pytabkit defaults.

    The number of quantiles adapts to the dataset size and a small amount of
    Gaussian noise can be added before fitting to break ties between identical
    values (this matches the RTDL preprocessing used by the TabM authors).
    """

    def __init__(
        self,
        noise: float = 1e-5,
        random_state: int | None = None,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
    ):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        normalizer.fit(self._add_noise(X) if self.noise > 0 else X)
        self.normalizer_ = normalizer
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        rng = np.random.default_rng(self.random_state)
        return X + rng.normal(0.0, self.noise, X.shape).astype(X.dtype)


class TabMOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encoder that maps unknown/missing values to a reserved index.

    Each categorical column with ``cardinality`` known categories is encoded to
    ``range(cardinality)``; every unknown or missing value is mapped to the extra
    index ``cardinality``. The model therefore needs to allocate ``cardinality + 1``
    slots per feature (see :meth:`get_cardinalities`).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        self.encoder_.fit(X)
        self.cardinalities_ = [len(cats) for cats in self.encoder_.categories_]
        return self

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "cardinalities_"])
        X_enc = self.encoder_.transform(pd.DataFrame(X))
        # NaN marks an unknown or missing value -> the reserved index ``cardinality``.
        for col_idx, cardinality in enumerate(self.cardinalities_):
            X_enc[np.isnan(X_enc[:, col_idx]), col_idx] = cardinality
        return X_enc.astype(np.int64)

    def get_cardinalities(self) -> list[int]:
        """Number of known categories per column (excluding the reserved slot)."""
        check_is_fitted(self, ["cardinalities_"])
        return self.cardinalities_


def make_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into weight-decayed and decay-free groups.

    Following TabM's default optimisation setup, bias parameters are excluded from
    weight decay. The supported TabM architectures use no normalisation layers, so
    biases are the only decay-free parameters in practice.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class TabMImplementation:
    """Fit/predict harness for :class:`tabm.TabM`."""

    def __init__(self, early_stopping_metric: Scorer, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_col_names: list[Any],
        time_to_fit_in_seconds: float | None = None,
    ):
        start_time = time.time()
        if X_val is None or len(X_val) == 0:
            raise ValueError("Training without validation set is currently not implemented")

        seed = self.config.get("random_state", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        if "n_threads" in self.config:
            torch.set_num_threads(self.config["n_threads"])

        problem_type = self.config["problem_type"]
        task_type: TaskType = "binclass" if problem_type == "binary" else problem_type
        device = torch.device(self.config["device"])
        n_train = len(X_train)
        self.task_type_ = task_type
        self.device_ = device
        self.cat_col_names_ = cat_col_names

        # -- Hyperparameters
        arch_type = self.config.get("arch_type", "tabm-mini")
        num_emb_type = self.config.get("num_emb_type", "pwl")
        n_epochs = self.config.get("n_epochs", 1_000_000_000)
        patience = self.config.get("patience", 16)
        batch_size = self.config.get("batch_size", "auto")
        compile_model = self.config.get("compile_model", False)
        lr = self.config.get("lr", 2e-3)
        d_embedding = self.config.get("d_embedding", 16)
        d_block = self.config.get("d_block", 512)
        dropout = self.config.get("dropout", 0.1)
        tabm_k = self.config.get("tabm_k", 32)
        n_blocks = self.config.get("n_blocks", "auto")
        num_emb_n_bins = self.config.get("num_emb_n_bins", 48)
        eval_batch_size = self.config.get("eval_batch_size", "auto")
        share_training_batches = self.config.get("share_training_batches", False)
        weight_decay = self.config.get("weight_decay", 3e-4)
        # Search-space default (the upstream example uses 'none').
        gradient_clipping_norm = self.config.get("gradient_clipping_norm", 1.0)
        # Mixed precision is GPU-only; ``allow_amp`` is kept as an alias.
        allow_amp = self.config.get("amp", self.config.get("allow_amp", False))

        # -- Resolve "auto" hyperparameters
        num_emb_n_bins = min(num_emb_n_bins, n_train - 1)
        if n_train <= 2:
            num_emb_type = "none"  # no valid number of bins for piecewise-linear embeddings
        if batch_size == "auto":
            batch_size = get_tabm_auto_batch_size(n_samples=n_train, n_features=X_train.shape[1])
        self.eval_batch_size_ = batch_size if eval_batch_size == "auto" else eval_batch_size

        # -- Fit preprocessing on the training split
        self.ord_enc_ = TabMOrdinalEncoder().fit(X_train[cat_col_names]) if cat_col_names else None
        self.has_num_cols_ = bool(set(X_train.columns) - set(cat_col_names))
        if self.has_num_cols_:
            self.num_prep_ = Pipeline(
                steps=[
                    ("qt", RTDLQuantileTransformer(random_state=seed)),
                    ("imp", SimpleImputer(add_indicator=True)),
                ]
            )
            x_num_train_np = self.num_prep_.fit_transform(self._numeric_array(X_train))
            # Drop columns that are near-constant on the training split (preprocessing can
            # re-introduce them); AutoGluon already removes the obvious ones. The tolerance
            # (vs exact equality) matters: float32 mean-imputation of a constant column yields
            # a second value one ulp off the constant, which would otherwise survive as a
            # "two-valued" column and build a single ~1e-9-wide piecewise-linear-embedding bin
            # whose unclamped encoding explodes on out-of-range values at predict time. The
            # pipeline's outputs are bounded (quantile-transformed |z| <= 5.2, indicators 0/1),
            # so an absolute tolerance is safe: legitimate columns have a range >= ~1e-3.
            col_range = x_num_train_np.max(axis=0) - x_num_train_np.min(axis=0)
            self.num_col_mask_ = col_range > 1e-6
            x_num_train_np = x_num_train_np[:, self.num_col_mask_]
        else:
            self.num_prep_ = None
            self.num_col_mask_ = None
            x_num_train_np = np.empty((n_train, 0), dtype=np.float32)
        self.n_num_features_ = x_num_train_np.shape[1]

        # -- Build input tensors
        x_num_train = self._to_num_tensor(x_num_train_np)
        x_cat_train = self._to_cat_tensor(X_train)
        x_num_val, x_cat_val = self._make_input_tensors(X_val)

        if task_type == "regression":
            self.n_classes_ = 0
            y_train_t = torch.as_tensor(y_train.to_numpy(np.float32), dtype=torch.float32, device=device)
            self.y_mean_ = y_train_t.mean().item()
            self.y_std_ = y_train_t.std(correction=0).item()
            y_train_t = (y_train_t - self.y_mean_) / (self.y_std_ + 1e-30)
        else:
            y_train_np = y_train.to_numpy(np.int64)
            self.n_classes_ = int(y_train_np.max()) + 1
            y_train_t = torch.as_tensor(y_train_np, dtype=torch.long, device=device)
        y_val_np = y_val.to_numpy()

        # -- Build the model
        self._compiled_ = compile_model
        self._amp_dtype, self._amp_enabled = self._resolve_amp(allow_amp)
        model = self._build_model(
            x_num_train=x_num_train,
            cat_cardinalities=self.ord_enc_.get_cardinalities() if self.ord_enc_ else [],
            num_emb_type=num_emb_type,
            num_emb_n_bins=num_emb_n_bins,
            d_embedding=d_embedding,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            arch_type=arch_type,
            tabm_k=tabm_k,
        )
        self.model_ = torch.compile(model) if compile_model else model
        k = model.k

        optimizer = torch.optim.AdamW(make_parameter_groups(model, weight_decay), lr=lr)
        grad_scaler = (
            torch.amp.GradScaler(device.type) if self._amp_enabled and self._amp_dtype is torch.float16 else None
        )
        logger.log(
            15,
            f"Device: {device.type.upper()} | AMP: {self._amp_enabled} ({self._amp_dtype}) "
            f"| torch.compile: {compile_model}",
        )

        loss_fn = torch.nn.functional.mse_loss if task_type == "regression" else torch.nn.functional.cross_entropy

        def compute_loss(x_num_b, x_cat_b, y_b: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device.type, enabled=self._amp_enabled, dtype=self._amp_dtype):
                # (batch, k) for regression, (batch, k, n_classes) for classification.
                y_pred = self.model_(x_num_b, x_cat_b).squeeze(-1).float()
            if share_training_batches:
                # The same batch is fed to every submodel -> broadcast its targets.
                y_b = y_b.unsqueeze(1).expand(-1, k)
            return loss_fn(y_pred.flatten(0, 1), y_b.flatten(0, 1))

        # -- Training loop with early stopping
        best_val = -np.inf
        best_epoch = -1
        best_params = [p.detach().clone() for p in model.parameters()]
        remaining_patience = patience
        progress = self._progress_bar()

        logger.log(15, "-" * 88)
        for epoch in range(n_epochs):
            if epoch > 0 and time_to_fit_in_seconds is not None:
                predicted_time = (epoch + 1) / epoch * (time.time() - start_time)
                if predicted_time >= time_to_fit_in_seconds:
                    break

            for idx in progress(self._train_batches(n_train, k, batch_size, share_training_batches, device)):
                model.train()
                optimizer.zero_grad()
                loss = compute_loss(
                    None if x_num_train is None else x_num_train[idx],
                    None if x_cat_train is None else x_cat_train[idx],
                    y_train_t[idx],
                )
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    if gradient_clipping_norm not in (None, "none"):
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if gradient_clipping_norm not in (None, "none"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
                    optimizer.step()

            val_score = self._score(x_num_val, x_cat_val, y_val_np)
            logger.log(15, f"Epoch {epoch}: (val) {val_score:.4f}")

            if val_score > best_val:
                best_val, best_epoch = val_score, epoch
                remaining_patience = patience
                with torch.no_grad():
                    for best, p in zip(best_params, model.parameters(), strict=False):
                        best.copy_(p)
            else:
                remaining_patience -= 1
            if remaining_patience < 0:
                break

        logger.log(15, f"Restoring best model (epoch {best_epoch}, val {best_val:.4f})")
        with torch.no_grad():
            for best, p in zip(best_params, model.parameters(), strict=False):
                p.copy_(best)

        return self

    # -- Model construction ------------------------------------------------------

    def _build_model(
        self,
        *,
        x_num_train: torch.Tensor | None,
        cat_cardinalities: list[int],
        num_emb_type: str,
        num_emb_n_bins: int,
        d_embedding: int,
        n_blocks: int | str,
        d_block: int,
        dropout: float,
        arch_type: str,
        tabm_k: int,
    ) -> tabm.TabM:
        num_embeddings = None
        if num_emb_type == "pwl" and self.n_num_features_ > 0:
            bins = rtdl_num_embeddings.compute_bins(x_num_train, n_bins=num_emb_n_bins)
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins,
                d_embedding=d_embedding,
                activation=False,
                version="B",
            )

        make_kwargs: dict[str, Any] = dict(
            n_num_features=self.n_num_features_,
            # ``+ 1`` reserves the one-hot slot the encoder uses for unknown/missing
            # categories (the package's one-hot encoding has no out-of-vocabulary handling).
            cat_cardinalities=[c + 1 for c in cat_cardinalities],
            d_out=1 if self.task_type_ == "regression" else self.n_classes_,
            num_embeddings=num_embeddings,
            d_block=d_block,
            dropout=dropout,
            arch_type=arch_type,
            k=tabm_k,
        )
        if n_blocks != "auto":
            # Otherwise ``TabM.make`` defaults to 2 with embeddings, 3 without.
            make_kwargs["n_blocks"] = n_blocks
        return tabm.TabM.make(**make_kwargs).to(self.device_)

    def _resolve_amp(self, allow_amp: bool) -> tuple[torch.dtype | None, bool]:
        if not torch.cuda.is_available():
            return None, False
        # bfloat16 is preferred; float16 needs a GradScaler (handled by the caller).
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return amp_dtype, allow_amp

    # -- Preprocessing helpers ---------------------------------------------------

    def _numeric_array(self, X: pd.DataFrame) -> np.ndarray:
        return X.drop(columns=self.cat_col_names_).to_numpy(dtype=np.float32)

    def _to_num_tensor(self, x_num_np: np.ndarray) -> torch.Tensor | None:
        if self.n_num_features_ == 0:
            return None
        return torch.as_tensor(x_num_np, dtype=torch.float32, device=self.device_)

    def _to_cat_tensor(self, X: pd.DataFrame) -> torch.Tensor | None:
        if self.ord_enc_ is None:
            return None
        return torch.as_tensor(self.ord_enc_.transform(X[self.cat_col_names_]), dtype=torch.long, device=self.device_)

    def _make_input_tensors(self, X: pd.DataFrame) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x_num = None
        if self.has_num_cols_:
            x_num_np = self.num_prep_.transform(self._numeric_array(X))[:, self.num_col_mask_]
            x_num = self._to_num_tensor(x_num_np)
        return x_num, self._to_cat_tensor(X)

    # -- Training/eval helpers ---------------------------------------------------

    @staticmethod
    def _train_batches(n_train, k, batch_size, share_training_batches, device):
        """Yield index tensors for one epoch.

        Shared batches give a 1-D index ``(batch,)``; otherwise a 2-D index
        ``(batch, k)`` with an independent permutation per submodel, which
        ``tabm.TabM`` interprets as one batch per submodel.
        """
        if share_training_batches:
            return torch.randperm(n_train, device=device).split(batch_size)
        return torch.rand((n_train, k), device=device).argsort(dim=0).split(batch_size, dim=0)

    def _progress_bar(self):
        if self.config.get("verbosity", 0) >= 1:
            try:
                from tqdm.std import tqdm

                return tqdm
            except ImportError:
                pass
        return lambda batches: batches

    def _forward_logits(self, x_num: torch.Tensor | None, x_cat: torch.Tensor | None) -> torch.Tensor:
        """Run the model over the data in eval batches; returns CPU logits.

        Shape ``(N, k)`` for regression, ``(N, k, n_classes)`` for classification.
        """
        self.model_.eval()
        n = len(x_num) if x_num is not None else len(x_cat)
        no_grad = torch.no_grad() if self._compiled_ else torch.inference_mode()
        outputs = []
        with no_grad, torch.autocast(self.device_.type, enabled=self._amp_enabled, dtype=self._amp_dtype):
            for idx in torch.arange(n, device=self.device_).split(self.eval_batch_size_):
                logits = (
                    self.model_(
                        None if x_num is None else x_num[idx],
                        None if x_cat is None else x_cat[idx],
                    )
                    .squeeze(-1)
                    .float()
                )
                outputs.append(logits)
        return torch.cat(outputs).cpu()

    def _aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        """Combine the ``k`` submodel outputs into a single prediction.

        Regression returns de-normalised targets ``(N,)``; classification returns
        class probabilities ``(N, n_classes)`` averaged in probability space (or in
        logit space when ``average_logits`` is set).
        """
        if self.task_type_ == "regression":
            return logits.mean(1) * self.y_std_ + self.y_mean_
        if self.config.get("average_logits", False):
            return torch.softmax(logits.mean(1), dim=-1)
        return torch.softmax(logits, dim=-1).mean(1)

    def _score(self, x_num, x_cat, y_true: np.ndarray) -> float:
        preds = self._aggregate(self._forward_logits(x_num, x_cat)).numpy()
        if self.task_type_ == "regression":
            y_pred, y_pred_proba = preds, preds
        else:
            y_pred = preds.argmax(1)
            y_pred_proba = preds[:, 1] if self.task_type_ == "binclass" else preds
        return compute_metric(
            y=y_true,
            metric=self.early_stopping_metric,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            silent=True,
        )

    # -- Public prediction API ---------------------------------------------------

    def _predict_aggregated(self, X: pd.DataFrame) -> np.ndarray:
        x_num, x_cat = self._make_input_tensors(X)
        return self._aggregate(self._forward_logits(x_num, x_cat)).numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._predict_aggregated(X)
        return preds if self.task_type_ == "regression" else preds.argmax(1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = self._predict_aggregated(X)
        return probas[:, 1] if probas.shape[1] == 2 else probas
