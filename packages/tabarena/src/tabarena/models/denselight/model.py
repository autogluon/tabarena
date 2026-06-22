from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DenseLightModel(AbstractTorchModel):
    """DenseLight: the default tabular neural network of the LightAutoML AutoML system.

    Only LightAutoML's DenseLight model and its surrounding neural-net training stack
    (``Trainer`` / ``TorchUniversalModel`` / the flat embedders) are used — none of
    LightAutoML's AutoML, reader, preset or report machinery is imported.

    LightAutoML should be installed *without* its heavy transitive tree:

        pip install lightautoml --no-deps

    Defaults mirror LightAutoML's ``tabular_config.yml`` ``denselight`` preset. See
    :class:`tabarena.models.denselight._internal.denselight_runner.DenseLightRunner`.

    Paper: LightAutoML: AutoML Solution for a Large Financial Services Ecosystem
    Authors: Anton Vakhrushev, Alexander Ryzhkov, Dmitry Simakov, Gleb Gusev, et al. (Sber AI Lab)
    Codebase: https://github.com/sb-ai-lab/LightAutoML
    License: Apache-2.0
    """

    ag_key = "TA-DENSELIGHT"
    ag_name = "TA-DenseLight"
    ag_priority = 65
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cat_features = None
        self._num_features = None
        self._features_bool = None
        self._bool_to_cat = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        start_time = time.time()

        try:
            import lightautoml  # noqa: F401
            import torch

            from tabarena.models.denselight._internal.denselight_runner import DenseLightRunner
        except ImportError as err:
            logger.log(
                40,
                "\tFailed to import lightautoml/torch! To use the DenseLight model, install LightAutoML "
                "(recommended: `pip install lightautoml --no-deps`, which avoids its heavy dependency tree "
                "since the DenseLight code path only needs torch/numpy/pandas/scikit-learn).",
            )
            raise err

        device = "cpu" if num_gpus == 0 else "cuda:0"
        if (device == "cuda:0") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.1,
                random_state=0,
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)

        X = self.preprocess(X, y=y, is_train=True, bool_to_cat=bool_to_cat)
        X_val = self.preprocess(X_val)

        # The inner fit mutates global torch state — torch.set_num_threads(num_cpus) and, via
        # LightAutoML's seed_everything, torch.backends.cudnn.deterministic. AutoGluon expects a
        # model's fit to leave global state unchanged, so snapshot and restore both afterwards.
        original_num_threads = torch.get_num_threads()
        original_cudnn_deterministic = torch.backends.cudnn.deterministic
        try:
            self.model = DenseLightRunner(
                problem_type=self.problem_type,
                num_classes=self.num_classes,
                device=device,
                n_threads=num_cpus,
                stopping_metric=self.stopping_metric,
                **hyp,
            )
            self.model.fit(
                X_train=X,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
                num_features=self._num_features,
                cat_features=self._cat_features,
                time_to_fit_in_seconds=(time_limit - (time.time() - start_time)) if time_limit is not None else None,
            )
        finally:
            torch.set_num_threads(original_num_threads)
            torch.backends.cudnn.deterministic = original_cudnn_deterministic

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Convert booleans to categoricals and record numeric/categorical feature names.

        The actual numeric standardization and categorical integer-encoding happen inside the
        runner (so they fit on train and apply unchanged at predict time); here we only split
        the columns by dtype the way LightAutoML's neural-net pipeline expects.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])

        if self._bool_to_cat and self._features_bool:
            X = X.copy(deep=True)
            X[self._features_bool] = X[self._features_bool].astype("category")

        if is_train:
            self._cat_features = X.select_dtypes(include="category").columns.tolist()
            self._num_features = [c for c in X.columns if c not in self._cat_features]

        return X

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        if self.model is None or self.model.torch_model_ is None:
            return "cpu"
        return next(self.model.torch_model_.parameters()).device.type

    def _set_device(self, device: str):
        if self.model is not None and self.model.torch_model_ is not None:
            self.model.torch_model_ = self.model.torch_model_.to(device)
            self.model.device = device

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = None,
        **kwargs,
    ) -> int:
        """Coarse first estimate of peak **GPU / VRAM** usage (bytes) for a DenseLight fit.

        Sums the three terms that dominate device memory and applies a safety factor:

        1. **Optimizer state** — Adam keeps ``param + grad + 2 moments`` = 4 fp32 copies of every
           weight. Weights come from the categorical embedding tables plus the dense blocks
           (``Linear`` + ``BatchNorm``, with ``concat_input`` widening each block's input by ``n_in``).
        2. **Activations** — forward activations of one batch kept for the backward pass (~3x for the
           activations, their gradients, and the concatenated-input copies).
        3. **CUDA / cuDNN context** — a fixed allocator + context overhead.

        DenseLight is a small MLP, so for typical tabular data the fixed context overhead dominates;
        the parameter term only grows with wide inputs or high-cardinality embedding tables. The
        SnapshotEns / SWA weight copies live on CPU, so they are intentionally excluded from VRAM.
        This is deliberately a rough upper-ish estimate, not a tight measurement.
        """
        if hyperparameters is None:
            hyperparameters = {}

        # Feature structure: categoricals (incl. bools, see `_preprocess`) get embeddings; the rest
        # are numeric inputs fed straight into the first dense block.
        cat_cols = X.select_dtypes(include=["category", "object", "bool"]).columns
        cat_cardinalities = [int(X[c].nunique(dropna=False)) + 1 for c in cat_cols]  # +1 unknown bin
        n_numeric = X.shape[1] - len(cat_cols)

        emb_ratio = int(hyperparameters.get("emb_ratio", 3))
        max_emb_size = int(hyperparameters.get("max_emb_size", 256))
        emb_dims = [min(max_emb_size, max(1, (card + 1) // emb_ratio)) for card in cat_cardinalities]
        n_in = n_numeric + sum(emb_dims)

        hidden_size = list(hyperparameters.get("hidden_size", [512, 256]))
        concat_input = bool(hyperparameters.get("concat_input", True))
        n_out = num_classes if (num_classes is not None and num_classes > 2) else 1

        # Parameter count: embeddings + dense blocks (Linear + BatchNorm affine) + final layer.
        n_emb_params = sum(card * dim for card, dim in zip(cat_cardinalities, emb_dims, strict=True))
        n_dense_params = 0
        in_dim = n_in
        act_per_sample = n_in
        for hid in hidden_size:
            n_dense_params += in_dim * hid + 2 * hid  # Linear (bias-free under BN) + BatchNorm
            act_per_sample += in_dim + hid  # layer input (kept for backward) + output
            in_dim = (n_in + hid) if concat_input else hid
        n_dense_params += hidden_size[-1] * n_out + n_out  # final fc (with bias)
        act_per_sample += n_out
        n_params = n_emb_params + n_dense_params

        # Batch size: mirror the runner's adaptive heuristic.
        n_samples = len(X)
        bs = hyperparameters.get("bs", "auto")
        if bs in ("auto", None):
            bs = 256
            if n_samples > 50_000:
                bs = 512
            if n_samples > 100_000:
                bs = 1024
            if n_samples > 1_000_000:
                bs = 2048
        bs = min(int(bs), max(1, n_samples))

        bytes_per_float = 4
        mem_params = 4 * n_params * bytes_per_float
        mem_activations = 3 * bs * act_per_sample * bytes_per_float
        mem_batch = bs * (n_numeric * 4 + len(cat_cardinalities) * 8)
        mem_overhead = 1_000_000_000  # fixed CUDA + cuDNN context / allocator overhead

        total = int((mem_params + mem_activations + mem_batch) * 1.3 + mem_overhead)
        logger.log(15, f"Estimated DenseLight GPU memory usage: {total / 1e9:.2f} GB")
        return total

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # SWA-averaged weights + LightAutoML's early stopping make exact-epoch refit unsupported.
        return {"can_refit_full": False}
