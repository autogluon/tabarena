from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

_TARGET = "__target__"


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

    ``fit`` stores the context; the library's preprocessing recipe and the forward pass over context and
    query rows run at predict time, under float16 autocast on CUDA as in NVIDIA's own TabArena adapter.
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

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_gpus: int = 0, **kwargs):
        """Load the pretrained network and store the context as table tensors on the CPU.

        An in-context-learning model without a training loop, so ``X_val`` / ``y_val`` and
        ``time_limit`` are unused, like in the other foundation-model wrappers; the library has no
        thread-count knob for ``num_cpus``.
        """
        import sdm

        from tabarena.models.kumo_tabular import _estimators

        device = self._resolve_fit_device(num_gpus)
        task = "regression" if self.problem_type == REGRESSION else "classification"
        network = _estimators.load_network(task=task, size=self.size, device=device)
        self.model = _estimators.FittedNetwork(task=task, size=self.size, network=network)

        X = self.preprocess(X, y=y)
        # AutoGluon hands binary columns over as integers; the library then treats them as categorical.
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        self._x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes)
        target_stype = "numerical" if task == "regression" else "categorical"
        self._y_context = sdm.TableTensor.from_pandas(df=y.rename(_TARGET).to_frame(), stypes={_TARGET: target_stype})

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import sdm
        import torch

        device = torch.device(self.get_device())
        X = self.preprocess(X, **kwargs)
        x_query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)
        generator = None
        if isinstance(self.random_seed, int):
            generator = torch.Generator(device).manual_seed(self.random_seed)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = self.model.estimator()(
                x_context=self._x_context.to(device),
                y_context=self._y_context.to(device),
                x_query=x_query,
                num_estimators=self._get_model_params()["num_estimators"],
                estimator_batch_size=self._estimator_batch_size(n_query=len(X), device=device),
                generator=generator,
            )
        values = out.numerical.float().cpu().numpy()
        if self.problem_type == REGRESSION:
            return values.mean(axis=-1)
        # Output columns are the class labels seen in the context; a class missing there gets zero.
        proba = np.zeros((len(X), self.num_classes), dtype=np.float32)
        proba[:, [int(label) for label in out.columns[sdm.Stype.numerical]]] = values
        return self._convert_proba_to_unified_form(proba)

    def _estimator_batch_size(self, n_query: int, device) -> int | None:
        """How many ensemble members run through the network together (``None``: all of them).

        NVIDIA's adapter heuristic: batch all members on CUDA while context plus query rows stay within
        3,000 rows and 50,000 cells, else one at a time.
        """
        n_rows, n_cols = self._x_context.shape[-2:]
        n_rows += n_query
        if device.type != "cuda" or n_rows > 3_000 or n_rows * n_cols > 50_000:
            return 1
        return None

    def _set_default_params(self):
        self._set_default_param_value("num_estimators", self.default_num_estimators)

    def get_device(self) -> str:
        param = next(self.model.network.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        self.model.network.to(device)

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
