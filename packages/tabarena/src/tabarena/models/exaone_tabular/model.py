from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.abstract import SharedWeights
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


class EXAONETabularModel(AbstractTorchModel):
    """EXAONE Tabular: an in-context-learning tabular foundation model from LG AI Research.

    A Cross-axis Summary Transformer (CAST) that conditions on the training rows at inference time,
    with no per-dataset gradient training. Classification and regression are two separate released
    checkpoints of the same architecture, each loaded by its own estimator: a 20.8M-parameter
    10-class head (ECOC above that) and a 21.1M-parameter 999-quantile head read out as a trimmed
    mean over the quantile function.

    Paper: technical report not yet released (the repository's citation block is a placeholder).
    Authors: LG AI Research
    Codebase: https://github.com/LGAI-Research/EXAONE-Tabular
    License: code under the BSD-3-Clause-LG AI Research License; the released weights under the
        EXAONE AI Model License 1.2 - NC, which permits non-commercial use only.

    The wrapper turns on expandable segments in torch's CUDA caching allocator for its process (see
    :func:`tabarena.models.warmup.configure_cuda_allocator`): on wide medium-size tables the chunked
    attention passes otherwise strand a third of the card in fragmented reserves.
    """

    ag_key = "TA-EXAONE-TABULAR"
    warmup_modules: ClassVar[tuple[str, ...]] = ("exaonetabular.classifier", "exaonetabular.regressor")
    ag_name = "TA-EXAONE-Tabular"
    ag_priority = 65
    seed_name = "seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: ``from_pretrained`` builds and loads the network inside one classmethod, so the loading half is
    #: replicated in ``_estimators.load_network`` (a developer fix, see that module); one build per
    #: task, compute dtype and device per process. A fit with custom weights or Hub coordinates runs
    #: the library's ``from_pretrained`` and builds its own.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabarena.models.exaone_tabular._estimators:load_network", key=("task", "compute_dtype")
    )
    #: Budget on the in-context workload, ``support rows x min(columns, feature_limit)``, above which the
    #: support set is subsampled to ``max_support_cells // min(columns, feature_limit)`` rows through the
    #: runtime's ``support_row_limit``. The largest workload that fit the 95 GB RTX PRO 6000 in the
    #: BeyondArena runs was california_house_prices_2020 r1 (20,764 x 657 regression columns, 13.6M); its
    #: r2 (31,146 x 657, 20.5M) exhausted the card even with expandable segments. Every other BeyondArena
    #: table up to 100k training rows stays below 10M (classification keeps at most 100 columns), so the
    #: guard engages only above a shape that fails. The ``max_support_cells`` hyperparameter overrides it;
    #: ``None`` disables it.
    max_support_cells: ClassVar[int] = 14_000_000
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"ensemble_count": 1}
    # Sequential fold fitting avoids contention on the shared Hugging Face checkpoint cache.
    # ``refit_folds=True`` matches the other TFM wrappers (TabICL, TabSwift, TabPFN-3, ...): for
    # an in-context-learning model, refitting one model on all data gives faster inference at
    # similar quality to the bagged ensemble.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> np.ndarray:
        """Produce the dense real-valued matrix EXAONE Tabular's estimators require.

        The estimators accept only a 2-D NumPy array of a real numeric dtype, so categoricals are
        ordinal-encoded (the encoding the upstream README asks callers to apply). Missing cells stay
        as NaN: the library's own preprocessor mean-imputes them and keeps a missing mask, which is
        strictly more informative than imputing here. Infinities are folded into NaN because the
        library rejects them outright.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        X = np.asarray(X.to_numpy(), dtype=np.float32)
        X[~np.isfinite(X)] = np.nan
        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fit EXAONE Tabular, loading the checkpoint that matches the problem type.

        As an in-context-learning foundation model there is no training loop and no early stopping,
        so (like the other TFM wrappers) ``X_val`` / ``y_val`` and ``time_limit`` are intentionally
        ignored — fitting loads the pre-trained checkpoint and stores the support set. ``num_cpus``
        is likewise unused: the library exposes no thread-count knob.

        Regression does one extra thing inside ``fit``: from ~10k support rows up, it holds out a
        fifth of them to solve for non-negative ensemble-member weights, which costs an additional
        forward pass. Below that threshold the members stay uniformly weighted.
        """
        import torch

        from tabarena.models.warmup import configure_cuda_allocator

        configure_cuda_allocator()
        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from tabarena.models.exaone_tabular._estimators import estimator_cls, load_network, released_manifest

        hps = self._get_model_params()
        max_support_cells = hps.pop("max_support_cells", self.max_support_cells)
        if device == "cpu" and hps.get("compute_dtype") == "float16":
            # Half precision is a GPU choice; several torch CPU kernels have no half
            # implementation, so the CPU fallback path runs in float32 instead.
            logger.log(15, "Running on CPU: overriding compute_dtype 'float16' with 'float32'.")
            hps["compute_dtype"] = "float32"

        X_np = self.preprocess(X, y=y, is_train=True)
        # Passed through unscaled for both tasks: the regressor standardizes the target against its
        # own support set and maps its predictions back, and the classifier encodes the labels.
        y_np = np.asarray(y.to_numpy())

        task = "regression" if self.problem_type == "regression" else "classification"
        if set(hps) - {"ensemble_count", "compute_dtype", "seed", "max_vram_bytes"}:
            # Custom weights or Hub coordinates: the library resolves, builds and loads for this fit alone.
            self.model = estimator_cls(task).from_pretrained(device=device, **hps)
        else:
            network = load_network(task, device, hps.get("compute_dtype"))
            manifest = released_manifest(
                task,
                ensemble_count=hps.get("ensemble_count"),
                compute_dtype=hps.get("compute_dtype"),
                seed=hps.get("seed"),
            )
            support_row_limit = self._support_row_limit(
                X_np.shape, manifest.runtime.feature_limit, manifest.runtime.support_row_limit, max_support_cells
            )
            if support_row_limit is not None:
                manifest = released_manifest(
                    task,
                    ensemble_count=hps.get("ensemble_count"),
                    compute_dtype=hps.get("compute_dtype"),
                    seed=hps.get("seed"),
                    support_row_limit=support_row_limit,
                )
            self.model = estimator_cls(task)(
                manifest, device=device, model=network, max_vram_bytes=hps.get("max_vram_bytes")
            )
        self.model.fit(X_np, y_np)

    @staticmethod
    def _support_row_limit(
        shape: tuple[int, int], feature_limit: int, current_limit: int, max_support_cells: int | None
    ) -> int | None:
        """The support-row cap that keeps ``rows x min(columns, feature_limit)`` within ``max_support_cells``.

        ``None`` when the workload fits the budget (or the budget is disabled), so the runtime keeps its own
        ``support_row_limit``; the cap never raises that limit.
        """
        if max_support_cells is None:
            return None
        rows, columns = shape
        effective_columns = max(1, min(columns, feature_limit))
        if rows * effective_columns <= max_support_cells:
            return None
        limit = min(current_limit, max(1, max_support_cells // effective_columns))
        logger.log(
            20,
            f"\tSubsampling the support set to {limit} of {rows} rows: {rows} x {effective_columns} effective "
            f"columns exceeds max_support_cells={max_support_cells}.",
        )
        return limit

    def _set_default_params(self):
        # The released checkpoints' runtime defaults, identical for both (exaonetabular.presets).
        default_params = {
            "ensemble_count": 8,
            "compute_dtype": "float16",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_device(self) -> str:
        return self.model.device.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device = device
        self.model.model = self.model.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
        """Configure the CUDA allocator before the CUDA context exists, then create that context.

        The allocator reads ``PYTORCH_CUDA_ALLOC_CONF`` when it first runs, so this classmethod runs
        before the generic torch layer of ``warmup_model_cls`` and calls :func:`warmup_torch` itself
        (idempotent, the generic layer repeats it harmlessly).
        """
        from tabarena.models.warmup import configure_cuda_allocator, warmup_torch

        configure_cuda_allocator()
        warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)

    @classmethod
    def download_checkpoint(cls, task: str) -> str:
        """Download one released checkpoint (``"classification"`` / ``"regression"``), return its path.

        The Hub coordinates come from ``exaonetabular.presets`` rather than being hardcoded here, so
        a repo or revision bump in the library is picked up automatically.

        Deliberately no ``local_files_only`` fast path: the released files are served from a mutable
        ``main`` revision and have been republished in place at least once, so trusting a cache hit
        would pin a superseded checkpoint forever and silently benchmark the wrong weights. The
        normal call revalidates the etag and re-downloads when the bytes changed.
        """
        from exaonetabular import released_checkpoint
        from huggingface_hub import hf_hub_download

        checkpoint = released_checkpoint(task)
        # The library's weight resolution also requests the repo's ``config.json`` (its download
        # tracking), so an offline fit needs that file in the cache next to the weights.
        hf_hub_download(repo_id=checkpoint.repo_id, filename="config.json", revision=checkpoint.revision)
        return hf_hub_download(
            repo_id=checkpoint.repo_id,
            filename=checkpoint.filename,
            revision=checkpoint.revision,
        )

    @classmethod
    def prefetch_weights(cls) -> dict[str, str]:
        """Pre-download both released checkpoints; return ``{task: local path}``.

        Classification and regression are separate files, so a run covering both problem types
        needs both warmed before the jobs are dispatched.
        """
        return {task: cls.download_checkpoint(task) for task in ("classification", "regression")}
