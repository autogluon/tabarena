"""AutoGluon's Mitra fine-tuning trainer with the Mitra-v2 prediction recipe.

Imports torch and AutoGluon's Mitra internals, so it is only imported from the wrapper's
fit path, never at module discovery time.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
from autogluon.tabular.models.mitra._internal.config.enums import LossName, ModelName, Task
from autogluon.tabular.models.mitra._internal.core.prediction_metrics import PredictionMetricsTracker
from autogluon.tabular.models.mitra._internal.core.trainer_finetune import TrainerFinetune
from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune

from tabarena.models.mitra_v2._internal.recipe import (
    RecipeSettings,
    balanced_binary_support_indices,
)

logger = logging.getLogger(__name__)


def is_cuda_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory error (either torch class or message form)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def set_attention_backend(model, backend: str):
    """Select the attention kernel of a loaded Tab2D backbone; returns a callable that restores the previous one.

    Tab2D picks ``flash_attn_varlen_func`` at construction whenever ``flash-attn`` imports and the
    device is CUDA, and otherwise runs its layers on ``torch.nn.functional.scaled_dot_product_attention``.
    The choice is an instance attribute of the model and of each layer. ``"sdpa"`` forces the latter,
    ``"stock"`` leaves the construction-time choice. The recipe uses ``"sdpa"`` for the fine-tuning
    loop only (see ``recipe.FINETUNE_ATTENTION_BACKEND``): at prediction shapes the flash path is
    both faster and far lighter on memory, so prediction keeps the stock choice.
    """
    if backend not in ("sdpa", "stock"):
        raise ValueError(f"attention backend must be 'sdpa' or 'stock', got {backend!r}")
    layers = [m for m in [model, *getattr(model, "layers", [])] if hasattr(m, "use_flash_attn")]
    previous = [(m, m.use_flash_attn) for m in layers]
    if backend == "sdpa":
        for m in layers:
            m.use_flash_attn = False

    def restore() -> None:
        for m, flag in previous:
            m.use_flash_attn = flag

    return restore


class PredictSupportDataset(DatasetFinetune):
    """Prediction-time dataset with a deterministic full support and a balanced binary subsample.

    When every support row fits in context one seeded permutation is reused: stock
    ``DatasetFinetune`` draws a fresh random permutation per query chunk, which changes nothing
    mathematically (Tab2D has no row positions) but makes predictions differ at bf16 precision
    between calls and between a single row and a batch. When the support exceeds the cap, the
    subsample is class-balanced on binary tasks, as the recipe prescribes; multiclass and
    regression keep the stock uniform draw.
    """

    def __init__(self, *args, balanced_binary: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.balanced_binary = balanced_binary

    def __getitem__(self, idx):
        if self.support_size >= self.n_samples_support:
            support_indices = getattr(self, "_full_support_indices", None)
            if support_indices is None:
                support_indices = self._full_support_indices = self.rng.choice(
                    self.n_samples_support,
                    size=self.n_samples_support,
                    replace=False,
                )
        else:
            support_indices = None
            if self.balanced_binary:
                support_indices = balanced_binary_support_indices(self.y_support, self.support_size, self.rng)
            if support_indices is None:
                support_indices = self.rng.choice(self.n_samples_support, size=self.support_size, replace=False)
        return {
            "x_support": torch.as_tensor(self.x_support[support_indices]),
            "y_support": torch.as_tensor(self.y_support[support_indices]),
            "x_query": torch.as_tensor(self.x_queries[idx]),
            "y_query": torch.as_tensor(self.y_queries[idx]),
        }


class MitraV2Trainer(TrainerFinetune):
    """Stock fine-tuning trainer plus the recipe's prediction-side changes and two cheaper-loop changes.

    1. Regression with the cross-entropy bin head decodes the softmax-weighted mean of the bin
       centers instead of the most likely bin, in both ``evaluate`` (drives checkpoint selection
       during fine-tuning) and ``predict``.
    2. ``predict`` conditions on up to ``recipe.predict_support_cap`` rows (the fine-tuning cap
       stays untouched), predicts in one wide query chunk when the whole support fits (then in
       one seeded order, so repeated predictions agree), balances the support draw on binary
       tasks, and on CUDA out-of-memory halves the query chunk first (quality-neutral) and then
       the support cap down to its floor.
    3. The validation pass after every fine-tuning step predicts the validation set in one wide
       query chunk (``recipe.finetune_eval_query_chunk``) instead of the stock 1,024-row chunks
       with a fresh support draw each, which made that pass cost as much as several steps on large
       tables; under out-of-memory the chunk is halved back to the stock size.
    4. Before the first validation pass, one throw-away forward and backward pass at the fine-tuning
       context size (:meth:`_memory_preflight`) makes a context that does not fit the GPU fail in
       seconds instead of after a full validation pass at that size.
    5. The fine-tuning loop runs attention on ``torch.nn.functional.scaled_dot_product_attention``
       (:func:`set_attention_backend`), which matches flash-attn 2 to bf16 rounding and is as fast
       per step on an H100 and 1.3 to 1.7 times faster on an RTX PRO 6000 Blackwell; prediction
       keeps the stock kernel, which at prediction shapes is faster and uses far less memory.
    6. ``set_device`` moves the weights with a blocking copy and synchronizes. The stock
       non-blocking move races the ``torch.save`` that follows it in ``MitraModel.save``: on
       torch 2.13 the destination is pinned memory, so the copy is truly asynchronous and a
       child can be serialized with a stale suffix of its tensors.
    """

    def __init__(
        self, cfg, model, n_classes: int, device: str, rng=None, verbose: bool = True, *, recipe: RecipeSettings
    ):
        super().__init__(cfg, model, n_classes=n_classes, device=device, rng=rng, verbose=verbose)
        self.recipe = recipe

    @property
    def regression_over_bins(self) -> bool:
        """Whether the model predicts a distribution over target bins (Mitra-v2's regression head)."""
        return self.cfg.task == Task.REGRESSION and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY

    def set_device(self, device: str) -> None:
        self.device = device
        self.model = self.model.to(device=device, non_blocking=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _forward(self, batch: dict) -> torch.Tensor:
        """One forward pass over a collated batch under autocast; returns raw model outputs."""
        with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams["precision"])):
            x_s = batch["x_support"].to(self.device, non_blocking=True)
            y_s = batch["y_support"].to(self.device, non_blocking=True)
            x_q = batch["x_query"].to(self.device, non_blocking=True)
            padding_features = batch["padding_features"].to(self.device, non_blocking=True)
            padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
            padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)

            if self.regression_over_bins:
                y_s = torch.bucketize(y_s, self.bins) - 1
                y_s = torch.clamp(y_s, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)

            if self.cfg.model_name == ModelName.TABPFN:
                return self.model(x_s, y_s, x_q, task=self.cfg.task).squeeze(-1)
            return self.model(x_s, y_s, x_q, padding_features, padding_obs_support, padding_obs_query)

    def _mean_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Torch twin of :func:`mean_decode_bins`, used on the validation path."""
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        centers = (self.bins[:-1] + self.bin_width / 2).to(logits.device)
        return (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)

    def train(self, x_train, y_train, x_val, y_val):
        restore = set_attention_backend(self.model, self.recipe.finetune_attention_backend)
        try:
            if self.recipe.finetune_memory_preflight and str(self.device).startswith("cuda"):
                self._memory_preflight(x_train)
            return super().train(x_train, y_train, x_val, y_val)
        finally:
            restore()

    def _memory_preflight(self, x_train) -> None:
        """One throw-away forward and backward pass at the fine-tuning context size.

        Stock fine-tuning validates the pretrained model on the whole validation set before its
        first step, so when the context does not fit the GPU the out-of-memory error arrives only
        after that full pass, about a minute per attempt on large tables, and the ratchet in
        ``_train_ensemble`` pays it at every context size it tries. This pass reproduces the shapes
        of a training step on synthetic data: the loop's 80/20 support and query split, capped as
        the loop caps them, over the training table's non-constant columns (the preprocessor drops
        constant ones). A context that cannot fit fails here in seconds. The model is left as it
        was: no optimizer step, gradients cleared, and neither the trainer's RNG nor the global
        RNGs are drawn from, so a fit that goes on is the same fit as without the pass.
        """
        hp = self.cfg.hyperparams
        x = np.asarray(x_train, dtype=np.float32)
        n_rows = x.shape[0]
        if n_rows < 2 or x.shape[1] == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            n_features = int((np.nanmax(x, axis=0) != np.nanmin(x, axis=0)).sum())
        if n_features == 0:
            return
        n_support_pool = int(np.ceil(0.8 * n_rows))
        n_support = min(int(hp["max_samples_support"]), n_support_pool)
        n_query = max(1, min(int(hp["max_samples_query"]), n_rows - n_support_pool))
        rng = np.random.RandomState(0)
        x_support = rng.standard_normal((n_support, n_features)).astype(np.float32)
        x_query = rng.standard_normal((n_query, n_features)).astype(np.float32)
        if self.cfg.task == Task.CLASSIFICATION:
            y_support = np.zeros(n_support, dtype=np.int64)
        else:
            y_support = np.zeros(n_support, dtype=np.float32)
        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=None,
            max_samples_support=n_support,
            max_samples_query=n_query,
            rng=rng,
        )
        batch = next(iter(self.make_loader(dataset, training=False)))
        was_training = self.model.training
        self.model.train()
        outputs = None
        try:
            with torch.enable_grad():
                outputs = self._forward(batch)
                outputs.float().mean().backward()
        finally:
            self.model.train(was_training)
            outputs = None
            batch = None
            self.optimizer.zero_grad(set_to_none=True)
            for parameter in self.model.parameters():
                parameter.grad = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def evaluate(self, x_support, y_support, x_query, y_query):
        x_support = self.preprocessor.transform_X(x_support)
        y_support = self.preprocessor.transform_y(y_support)
        x_query = self.preprocessor.transform_X(x_query)
        stock_chunk = int(self.cfg.hyperparams["max_samples_query"])
        query_chunk = max(stock_chunk, min(int(self.recipe.finetune_eval_query_chunk), len(x_query)))
        while True:
            try:
                return self._evaluate_once(x_support, y_support, x_query, y_query, query_chunk=query_chunk)
            except RuntimeError as exc:
                if not is_cuda_oom(exc) or query_chunk <= stock_chunk:
                    raise
                torch.cuda.empty_cache()
                query_chunk = max(stock_chunk, query_chunk // 2)
                logger.warning(
                    f"Mitra-v2 fine-tuning validation: CUDA out of memory, halving the query chunk to {query_chunk}."
                )

    def _evaluate_once(self, x_support, y_support, x_query, y_query, *, query_chunk: int):
        self.model.eval()
        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=y_query,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=query_chunk,
            rng=self.rng,
        )
        loader = self.make_loader(dataset, training=False)
        tracker = PredictionMetricsTracker(task=self.cfg.task, preprocessor=self.preprocessor)
        with torch.no_grad():
            for batch in loader:
                y_hat = self._forward(batch)
                y_q = batch["y_query"].to(self.device, non_blocking=True)
                if self.regression_over_bins:
                    y_hat = self._mean_decode(y_hat)
                tracker.update(y_hat.float(), y_q, train=False)
        return tracker.get_metrics()

    def predict(self, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        recipe = self.recipe
        n_support = len(x_support)
        support_cap = recipe.predict_support_cap
        stock_query_chunk = self.cfg.hyperparams["max_samples_query"]

        def query_chunk_for(cap: int) -> int:
            # One wide chunk when every support row fits in context; otherwise the stock chunking,
            # whose per-chunk support redraw acts as an implicit ensemble over capped subsamples.
            return recipe.predict_query_chunk if n_support <= cap else stock_query_chunk

        query_chunk = query_chunk_for(support_cap)
        while True:
            try:
                return self._predict_once(
                    x_support, y_support, x_query, support_cap=support_cap, query_chunk=query_chunk
                )
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                torch.cuda.empty_cache()
                if n_support <= support_cap and query_chunk > recipe.predict_query_chunk_floor:
                    query_chunk = max(recipe.predict_query_chunk_floor, query_chunk // 2)
                    logger.warning(f"Mitra-v2 predict: CUDA out of memory, halving the query chunk to {query_chunk}.")
                    continue
                if support_cap <= recipe.predict_support_floor:
                    raise
                support_cap = max(recipe.predict_support_floor, support_cap // 2)
                query_chunk = query_chunk_for(support_cap)
                logger.warning(f"Mitra-v2 predict: CUDA out of memory, halving the support cap to {support_cap}.")

    def _predict_once(self, x_support, y_support, x_query, *, support_cap: int, query_chunk: int) -> np.ndarray:
        dataset = PredictSupportDataset(
            self.cfg,
            x_support=self.preprocessor.transform_X(x_support),
            y_support=self.preprocessor.transform_y(y_support),
            x_query=self.preprocessor.transform_X(x_query),
            y_query=None,
            max_samples_support=support_cap,
            max_samples_query=query_chunk,
            rng=self.rng,
            balanced_binary=self.recipe.balanced_binary_support and self.cfg.task == Task.CLASSIFICATION,
        )
        loader = self.make_loader(dataset, training=False)
        self.model.eval()
        y_pred_list = []
        with torch.no_grad():
            for batch in loader:
                y_hat = self._forward(batch)[0].float().cpu()
                if self.regression_over_bins:
                    y_hat = self._mean_decode(y_hat)
                y_hat = y_hat.numpy()
                y_pred_list.append(self.preprocessor.inverse_transform_y(y_hat))
        return np.concatenate(y_pred_list, axis=0)
