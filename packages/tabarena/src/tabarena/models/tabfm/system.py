from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.models.tabfm.model import _build_tabfm_estimator, _resolve_device

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


class TabFMPlusSystemModel(ExternalSystemModel):
    """TabFM+ — TabFM run through its heavier ``ensemble`` interface, benchmarked as a system.

    Init hyperparameters (each a per-config knob for the system generator):

    * ``interface`` — ``"ensemble"`` (default) or ``"default"`` (plain constructor).
    * ``device`` — ``None`` (default: a GPU when available/allocated, else CPU), ``"cpu"`` to force
      CPU, or ``"gpu"`` / ``"cuda"`` to require a GPU.

    The TabFM estimator's ensemble seed is not an init knob: it is the per-split ``random_state``
    the runner threads into :meth:`_fit_system` (see the base ``ExternalSystemModel``), so each split
    gets distinct but reproducible randomness.

    Codebase: https://github.com/google-research/tabfm
    """

    def __init__(
        self,
        *,
        interface: str = "ensemble",
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interface = interface
        self.device = device
        self._estimator = None

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        """Fit a single TabFM estimator (``interface`` preset) on all the training data.

        ``random_state`` is the per-split seed threaded in by the runner; it is used as the TabFM
        estimator's ensemble seed, falling back to ``0`` when ``None`` (a direct fit outside the
        runner) so the fit stays deterministic.
        """
        import torch

        cuda_available = torch.cuda.is_available()
        # When the runner leaves the GPU budget unconstrained (``None``), fall back to using a GPU
        # whenever CUDA is present -- TabFM is a GPU foundation model.
        effective_num_gpus = num_gpus if num_gpus is not None else int(cuda_available)
        device = _resolve_device(self.device, effective_num_gpus, cuda_available=cuda_available)

        estimator = _build_tabfm_estimator(
            problem_type=problem_type,
            device=device,
            interface=self.interface,
            random_state=random_state if random_state is not None else 0,
        )
        # TabFM does its own preprocessing/label handling, so the raw frames are passed through.
        self._estimator = estimator.fit(X=X, y=y)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._estimator.predict(X), index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # `classes_` is the original label space (the estimator inverse-transforms its own
        # internal encoding), so the columns line up with the task's labels.
        proba = self._estimator.predict_proba(X)
        return pd.DataFrame(proba, index=X.index, columns=self._estimator.classes_)
