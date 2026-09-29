from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_TARGET = "__target__"
logger = logging.getLogger(__name__)


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
    fitting still preprocesses the full context and stores the ensemble's KV cache in host memory.
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

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        """Fit preprocessing and record the context once, without updating checkpoint weights.

        An in-context-learning model without a training loop, so ``X_val`` / ``y_val`` and
        ``time_limit`` are unused, like in the other foundation-model wrappers.
        """
        import sdm
        import torch

        from tabarena.models.kumo_tabular import _estimators

        device = torch.device(self._resolve_fit_device(num_gpus))
        task = "regression" if self.problem_type == REGRESSION else "classification"
        network = _estimators.load_network(task=task, size=self.size, device=str(device))
        device = next(network.parameters()).device
        self.model = _estimators.FittedNetwork(task=task, size=self.size, network=network)

        X = self.preprocess(X, y=y)
        # AutoGluon hands binary columns over as integers; the library then treats them as categorical.
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)
        target_stype = "numerical" if task == "regression" else "categorical"
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(_TARGET).to_frame(), stypes={_TARGET: target_stype}, device=device
        )
        params = self._get_model_params()
        self._num_cpus = num_cpus
        self._num_columns = X.shape[1]
        self._row_bytes, cache_bytes = _estimators.row_bytes(network, X.shape[1], self.num_classes or 0)
        batch_size = params["estimator_batch_size"]
        if batch_size is None:
            batch_size = 1
            if device.type == "cuda":
                budget = _estimators.available_memory(device) // 2
                batch_size = max(1, min(params["num_estimators"], budget // (len(X) * (self._row_bytes + cache_bytes))))
        self._estimator_batch_size = batch_size
        logger.info("\tKumo estimator batch size: %s", batch_size)
        generator = torch.Generator(device).manual_seed(self.random_seed) if isinstance(self.random_seed, int) else None
        estimator = self.model.estimator()
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
                    num_estimators=params["num_estimators"],
                    estimator_batch_size=batch_size,
                    generator=generator,
                )
            # Multi-member fits already offload to pinned CPU memory; also offload singleton fits.
            self.model.cache = estimator._cache.cpu()
        finally:
            torch.set_num_threads(previous_threads)

    def _get_max_batch_size(self) -> int | None:
        import torch
        from sdm._memory import chunk_memory_limit

        from tabarena.models.kumo_tabular import _estimators

        batch_size = super()._get_max_batch_size()
        device = torch.device(self.get_device())
        if batch_size is not None or device.type != "cuda":
            return batch_size
        self.model.move_processors(device)
        params = self._get_model_params()
        cache = self.model.cache
        # SDM overlaps transfer of the next estimator batch with execution of the current one.
        staging_bytes = 2 * max(cache[i].size() for i in range(cache["num_batches"]))
        budget = min(chunk_memory_limit(device), max(0, _estimators.available_memory(device) - staging_bytes) // 2)
        output_columns = 999 if self.problem_type == REGRESSION else self.num_classes
        # Query preprocessing and output reduction keep tensors for all members, not just a network batch.
        bytes_per_row = self._row_bytes * self._estimator_batch_size + params["num_estimators"] * 8 * (
            8 * self._num_columns + 4 * output_columns
        )
        batch_size = max(1, budget // bytes_per_row)
        logger.info("\tKumo query batch size: %s", batch_size)
        return batch_size

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import sdm
        import torch

        device = torch.device(self.get_device())
        X = self.preprocess(X, **kwargs)
        estimator = self.model.estimator()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_cpus)
            with (
                torch.inference_mode(),
                torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)
                out = estimator.predict(query)
                if self.problem_type == REGRESSION:
                    return out.numerical.float().mean(dim=-1).cpu().numpy()
                # A class missing from the fitted context gets zero probability.
                predictions = np.zeros((len(X), self.num_classes), dtype=np.float32)
                labels = [int(label) for label in out.columns[sdm.Stype.numerical]]
                predictions[:, labels] = out.numerical.float().cpu().numpy()
                return self._convert_proba_to_unified_form(predictions)
        finally:
            torch.set_num_threads(previous_threads)

    def _set_default_params(self):
        self._set_default_param_value("num_estimators", self.default_num_estimators)
        self._set_default_param_value("estimator_batch_size", None)

    def get_device(self) -> str:
        param = next(self.model.network.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        self.model.network.to(device)

    def set_device(self, device: str):
        # AutoGluon's shared-weight device walker would also move the offloaded KV cache onto the GPU.
        cache, self.model.cache = self.model.cache, None
        try:
            super().set_device(device)
        finally:
            self.model.cache = cache
        self.model.move_processors(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

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
