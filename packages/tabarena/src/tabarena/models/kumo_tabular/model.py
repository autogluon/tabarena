from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_TARGET = "__target__"
#: Smallest query pass the out-of-memory fallback in ``_predict_values`` splits down to.
_MIN_QUERY_PASS_ROWS = 512
#: Large freed pinned blocks go back to the system, so they are allocated at their exact size and do not pile up.
_CUDA_ALLOC_CONF = "expandable_segments:True,pinned_max_cached_size_mb:64"


def _configure_allocator() -> None:
    """The shared allocator setup, plus releasing large pinned blocks when the allocator has not started yet."""
    import torch

    from tabarena.models.warmup import configure_cuda_allocator

    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None and not torch.cuda.is_initialized():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _CUDA_ALLOC_CONF
    configure_cuda_allocator()


def to_signed_integers(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` with its ``uint16`` / ``uint32`` / ``uint64`` columns cast to ``int64`` (``float64`` above its range).

    The library keeps the unsigned dtype for the category values of such columns when it infers them as
    categorical, and torch pickles those tensors with a storage its own unpickler cannot read back
    (``'UntypedStorage' has no attribute 'dtype'``), so the fitted child fails to load at refit:
    home_credit_default_stability_1m has 20 ``uint32`` columns, 17 of them inferred as categorical. The
    values and the inferred semantic types are unchanged.
    """
    casts = {}
    for column, dtype in X.dtypes.items():
        if dtype.kind == "u" and dtype.itemsize > 1:
            fits = dtype.itemsize < 8 or X[column].max() <= np.iinfo(np.int64).max
            casts[column] = "int64" if fits else "float64"
    return X.astype(casts) if casts else X


def context_subsample_index(n_rows: int, num_estimators: int, max_context_size: int | None, seed: int | None):
    """Row indices of each ensemble member's context, shape ``[num_estimators, max_context_size]``.

    ``None`` when the context fits within ``max_context_size``. Otherwise NVIDIA's adapter rule: the members
    take consecutive slices of concatenated random permutations, so every row is used about equally often
    (a member whose slice spans two permutations can draw a row twice).
    """
    import torch

    if max_context_size is None or n_rows <= max_context_size:
        return None
    generator = torch.Generator().manual_seed(seed) if isinstance(seed, int) else None
    num_repeats = -(-num_estimators * max_context_size // n_rows)
    perm = torch.cat([torch.randperm(n_rows, generator=generator) for _ in range(num_repeats)])
    return perm[: num_estimators * max_context_size].view(num_estimators, max_context_size)


def stack_member_contexts(x_context, y_context, context_index):
    """One context per ensemble member along a leading axis, which the library reads as the ensemble."""
    index = context_index.flatten()
    return x_context[index].unflatten(0, context_index.shape), y_context[index].unflatten(0, context_index.shape)


class KumoTabularModel(AbstractTorchModel):
    """Kumo Tabular: NVIDIA's pretrained in-context-learning tabular foundation model (large checkpoint).

    An interleaved row/column encoder (induced set attention across rows per feature group, then
    attention across a row's feature groups and readout tokens) turns cells into row embeddings, and a
    dataset-wise in-context-learning transformer predicts the query rows from the labeled context rows.
    Classification uses a 10-class head (error-correcting output codes above that); regression predicts
    999 quantiles, read out here as their mean. Classification and regression are separate checkpoints,
    released in three sizes; the medium and small subclasses below run the smaller ones.

    Paper: NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction
        (https://huggingface.co/blog/nvidia/kumo-tabular)
    Authors: Qu et al. (NVIDIA)
    Codebase: https://github.com/NVIDIA/structured-data-models (weights: https://huggingface.co/nvidia/Kumo-Tabular)
    License: code under Apache-2.0, weights under OpenMDW 1.1.

    ``fit`` fits the preprocessing recipe and records the context KV cache. ``predict`` reuses both in
    query-row batches, under float16 autocast on CUDA. The recipe computes numerical statistics in
    float64. ``estimator_batch_size`` and ``ag.max_batch_size`` default to conservative memory-based
    estimates; positive integer overrides fix either size. These estimates are not an OOM guarantee:
    fitting stores the ensemble's KV cache in host memory. Above ``max_context_size`` training rows
    (default 200,000), each member uses its own random context subsample. Prediction retries CUDA
    out-of-memory errors with smaller query passes; the allocator uses expandable segments.

    With ``cache_context=False``, ``fit`` keeps only the context rows and every predict call runs the
    library's stateless forward over context and query, so no KV cache is held in host memory; the
    predictions match the cached path up to floating-point rounding. TabArena's config fits the bagged fold
    models this way, since each predicts its out-of-fold rows once and is then dropped (``refit_folds``), and
    turns the cache back on for the refit model through ``ag.refit_hyperparameters``.
    """

    ag_key = "TA-KUMO-TABULAR"
    ag_name = "TA-Kumo-Tabular"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS, REGRESSION]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    warmup_modules: ClassVar[tuple[str, ...]] = ("sdm", "sdm.models.kumo.tabular")
    #: The library loads inside its constructor, so the loading half is replicated in
    #: ``_estimators.load_network`` (a developer fix, see that module); one build per task, size and
    #: device per process.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabarena.models.kumo_tabular._estimators:load_network", key=("task", "size")
    )
    #: The three sizes are registered separately; each owns its ``share_weights`` class setting.
    class_settings_per_subclass = True
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"num_estimators": 1}

    #: The checkpoint size this class runs.
    size: ClassVar[str] = "large"
    #: The library's default for the large model; NVIDIA's adapter uses 8 for the smaller two.
    default_num_estimators: ClassVar[int] = 16
    default_max_context_size: ClassVar[int] = 200_000

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        """Fit preprocessing and cache the context, or keep only the context rows (``cache_context=False``).

        An in-context-learning model without a training loop, so ``X_val`` / ``y_val`` and
        ``time_limit`` are unused, like in the other foundation-model wrappers.
        """
        import sdm
        import torch

        from tabarena.models.kumo_tabular import _estimators

        _configure_allocator()
        task = "regression" if self.problem_type == REGRESSION else "classification"
        network = _estimators.load_network(task=task, size=self.size, device=self._resolve_fit_device(num_gpus))
        device = next(network.parameters()).device
        self.model = _estimators.FittedNetwork(task=task, size=self.size, network=network)

        X = self.preprocess(X, y=y)
        # AutoGluon hands binary columns over as integers; the library then treats them as categorical.
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes)
        target_stype = "numerical" if task == "regression" else "categorical"
        y_context = sdm.TableTensor.from_pandas(df=y.rename(_TARGET).to_frame(), stypes={_TARGET: target_stype})
        # Include target classes absent from this fold in batch estimates.
        self._y_metadata = y_context[:0].cpu()
        if task == "classification":
            self._y_metadata = sdm.TableTensor(
                columns=y_context.columns,
                categorical=sdm.CategoricalTensor(
                    code=self._y_metadata.categorical.code,
                    categories=(torch.arange(self.num_classes),),
                ),
            )
        params = self._get_model_params()
        context_index = context_subsample_index(
            n_rows=len(X),
            num_estimators=params["num_estimators"],
            max_context_size=params["max_context_size"],
            seed=self.random_seed,
        )
        num_estimators = params["num_estimators"]
        self._context_subsampled = context_index is not None
        self._num_cpus = num_cpus
        estimator = self.model.estimator()
        if not params["cache_context"]:
            # Indices rather than the stacked subsamples, so the context is held once.
            self.model.context = (x_context, y_context, context_index)
            member_context = x_context if context_index is None else x_context[: context_index.size(1)]
            self._estimator_batch_size = self._resolve_estimator_batch_size(estimator, member_context, device)
            return
        if self._context_subsampled:
            x_context, y_context = stack_member_contexts(x_context, y_context, context_index)
            num_estimators = None
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        batch_size = self._resolve_estimator_batch_size(estimator, x_context, device)
        self._estimator_batch_size = batch_size
        generator = torch.Generator(device).manual_seed(self.random_seed) if isinstance(self.random_seed, int) else None
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            with (
                torch.inference_mode(),
                torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                estimator.fit(
                    x=x_context,
                    y=y_context,
                    num_estimators=num_estimators,
                    estimator_batch_size=batch_size,
                    generator=generator,
                )
            # Multi-member fits already offload to pinned CPU memory; also offload singleton fits.
            self.model.cache = estimator._cache.cpu()
            # Query batch estimation only needs feature and target metadata.
            self._x_metadata = sdm.TableTensor.from_tensor(torch.empty(0, x_context.size(-1), device="cpu"))
        finally:
            torch.set_num_threads(previous_threads)

    def _resolve_estimator_batch_size(self, estimator, x_context, device) -> int:
        """``estimator_batch_size``, else SDM's memory-based estimate on CUDA and one member at a time on the CPU.

        The estimate reads only the shape of one member's context, ``x_context.shape[-2:]``.
        """
        from sdm.models.kumo.tabular import estimate_fit_batch_size

        from tabarena.models.kumo_tabular import _estimators

        params = self._get_model_params()
        if params["estimator_batch_size"] is not None:
            return params["estimator_batch_size"]
        if device.type != "cuda":
            return 1
        return estimate_fit_batch_size(
            model=estimator,
            x=x_context,
            y=self._y_metadata,
            num_estimators=params["num_estimators"],
            memory_budget=_estimators.available_memory(device),
        )

    def _get_max_batch_size(self) -> int | None:
        import torch
        from sdm._memory import chunk_memory_limit
        from sdm.models.kumo.tabular import estimate_predict_batch_size

        from tabarena.models.kumo_tabular import _estimators

        batch_size = super()._get_max_batch_size()
        device = torch.device(self.get_device())
        # Without the cache every call encodes the context again, so the query goes in one call.
        if batch_size is not None or device.type != "cuda" or not self._get_model_params()["cache_context"]:
            return batch_size
        estimator = self.model.estimator()
        params = self._get_model_params()
        cache = self.model.cache
        # SDM overlaps transfer of the next estimator batch with execution of the current one.
        staging_bytes = 2 * max(cache[i].size() for i in range(cache["num_batches"]))
        budget = min(chunk_memory_limit(device), max(0, _estimators.available_memory(device) - staging_bytes) // 2)
        return estimate_predict_batch_size(
            model=estimator,
            x=self._x_metadata,
            y=self._y_metadata,
            num_estimators=params["num_estimators"],
            estimator_batch_size=self._estimator_batch_size,
            memory_budget=2 * budget,
        )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import sdm
        import torch

        device = torch.device(self.get_device())
        X = self.preprocess(X, **kwargs)
        query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)
        labels, values = self._predict_values(query, device=device)
        if self.problem_type == REGRESSION:
            return values[:, 0]
        predictions = np.zeros((len(X), self.num_classes), dtype=np.float32)
        predictions[:, [int(label) for label in labels]] = values
        return self._convert_proba_to_unified_form(predictions)

    def _predict_values(self, x_query, device) -> tuple[list, np.ndarray]:
        """The output columns and values for ``x_query``, in one pass or, after a CUDA out-of-memory error, two.

        Each query row is predicted from the context alone, so splitting the query rows changes the
        predictions only by floating-point rounding.
        """
        import torch

        try:
            return self._forward(x_query, device=device)
        except torch.OutOfMemoryError:
            if len(x_query) <= _MIN_QUERY_PASS_ROWS:
                raise
        # Outside the except block, so the failed pass's tensors are released before the retry.
        torch.cuda.empty_cache()
        half = len(x_query) // 2
        labels, first = self._predict_values(x_query[:half], device=device)
        _, second = self._predict_values(x_query[half:], device=device)
        return labels, np.concatenate([first, second])

    def _forward(self, x_query, device) -> tuple[list, np.ndarray]:
        import sdm
        import torch

        params = self._get_model_params()
        if self._context_subsampled:
            x_query = x_query.expand(params["num_estimators"], *x_query.size())
        # Built outside inference mode, where ``unflatten`` reaches ``TableTensor`` undecomposed and fails.
        uncached = None if params["cache_context"] else self._uncached_context(device)
        estimator = self.model.estimator()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_cpus)
            with (
                torch.inference_mode(),
                torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                out = estimator.predict(x_query) if uncached is None else estimator(x_query=x_query, **uncached)
                if self.problem_type == REGRESSION:
                    values = out.numerical.float().mean(dim=-1, keepdim=True)
                    return [_TARGET], values.cpu().numpy()
                return list(out.columns[sdm.Stype.numerical]), out.numerical.float().cpu().numpy()
        finally:
            torch.set_num_threads(previous_threads)

    def _uncached_context(self, device) -> dict:
        """Keyword arguments of the library's stateless forward, which encodes the context with the fit's seed per call."""
        import torch

        x_context, y_context, context_index = self.model.context
        num_estimators = self._get_model_params()["num_estimators"]
        if context_index is not None:
            x_context, y_context = stack_member_contexts(x_context, y_context, context_index)
            num_estimators = None
        generator = torch.Generator(device).manual_seed(self.random_seed) if isinstance(self.random_seed, int) else None
        return {
            "x_context": x_context.to(device),
            "y_context": y_context.to(device),
            "num_estimators": num_estimators,
            "estimator_batch_size": self._estimator_batch_size,
            "generator": generator,
        }

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return to_signed_integers(super()._preprocess(X, **kwargs))

    def _set_default_params(self):
        self._set_default_param_value("num_estimators", self.default_num_estimators)
        self._set_default_param_value("estimator_batch_size", None)
        self._set_default_param_value("max_context_size", self.default_max_context_size)
        self._set_default_param_value("cache_context", True)

    def get_device(self) -> str:
        param = next(self.model.network.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        self.model.network.to(device)

    def set_device(self, device: str):
        # AutoGluon's shared-weight device walker would also move the offloaded KV cache (or the raw context) onto the GPU.
        cache, context = self.model.cache, self.model.context
        self.model.cache = self.model.context = None
        try:
            super().set_device(device)
        finally:
            self.model.cache, self.model.context = cache, context
        self.model.move_processors(device)

    #: The KV cache is saved next to ``model.pkl`` rather than inside it.
    cache_file_name: ClassVar[str] = "kv_cache.pt"

    @contextlib.contextmanager
    def _without_cache(self):
        """Detach the KV cache while ``self`` is pickled, so the pickle never copies it."""
        cache = self.model.cache
        self.model.cache = None
        try:
            yield cache
        finally:
            self.model.cache = cache

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Pickle the model without the KV cache and write the cache with ``torch.save``, which needs no copy."""
        if self.model is None or self.model.cache is None:
            return super().save(path=path, verbose=verbose)
        import torch

        with self._without_cache() as cache:
            path = super().save(path=path, verbose=verbose)
        cache_path = os.path.join(path, self.cache_file_name)
        # Written beside the file and swapped in: a cache loaded on the CPU stays memory-mapped from this file,
        # and truncating it under the mapping kills the process with SIGBUS (refit_full re-saves a loaded model).
        torch.save(cache, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the model and its KV cache, memory-mapped and then pinned for transfers on CUDA."""
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        cache_path = os.path.join(path, cls.cache_file_name)
        if os.path.exists(cache_path):
            import torch

            # The cache file is as trusted as model.pkl.
            cache = torch.load(cache_path, mmap=True, weights_only=False)
            model.model.cache = cache.pin_memory() if torch.cuda.is_available() else cache
        return model

    def _get_pickled_size(self) -> int:
        """The pickle size without the KV cache plus the cache's bytes."""
        if self.model is None:
            return super()._get_pickled_size()
        with self._without_cache() as cache:
            size = super()._get_pickled_size()
        return size + (cache.size() if cache is not None else 0)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
        """Configure the CUDA allocator before the CUDA context exists, then create that context.

        The allocator reads ``PYTORCH_CUDA_ALLOC_CONF`` when it first runs, so this runs before the
        generic torch layer of ``warmup_model_cls`` and calls :func:`warmup_torch` itself (idempotent).
        """
        from tabarena.models.warmup import warmup_torch

        _configure_allocator()
        warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)

    @classmethod
    def prefetch_weights(cls) -> list:
        """Download this size's classification and regression checkpoints; return their cache paths."""
        from tabarena.models.kumo_tabular._estimators import download_checkpoint

        return [download_checkpoint(task, cls.size) for task in ("classification", "regression")]


class KumoTabularMediumModel(KumoTabularModel):
    """Kumo Tabular with the medium checkpoint and 8 estimators."""

    ag_key = "TA-KUMO-TABULAR-MEDIUM"
    ag_name = "TA-Kumo-Tabular-Medium"
    size = "medium"
    default_num_estimators = 8


class KumoTabularSmallModel(KumoTabularModel):
    """Kumo Tabular with the small checkpoint and 8 estimators."""

    ag_key = "TA-KUMO-TABULAR-SMALL"
    ag_name = "TA-Kumo-Tabular-Small"
    size = "small"
    default_num_estimators = 8
