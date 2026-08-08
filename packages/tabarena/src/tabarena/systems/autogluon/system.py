from __future__ import annotations

import shutil
import tempfile
from typing import TYPE_CHECKING

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


class AutoGluonSystemModel(ExternalSystemModel):
    """AutoGluon's ``TabularPredictor`` benchmarked as a system.

    Init hyperparameters (each a per-config knob for the system generator):

    * ``preset`` — the AutoGluon preset to fit, e.g. ``"best_quality"`` / ``"extreme_quality"``.
    * ``path`` — where the predictor writes its artifacts. ``None`` (default) uses a temp dir
      that :meth:`cleanup` removes after the fit.

    The compute and time budgets are not init knobs: the runner passes them per split into
    :meth:`_fit_system`, so every system on the leaderboard is held to the same constraints.

    Codebase: https://github.com/autogluon/autogluon
    """

    def __init__(self, *, preset: str = "best_quality", path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preset = preset
        self.path = path
        self._predictor = None
        self._predictor_path: str | None = None

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
        """Fit a ``TabularPredictor`` on all the training data.

        See the parent ``ExternalSystemModel._fit_system`` docstring for the full argument
        contract. AutoGluon carves its own validation split out of ``X``, so none is passed in.
        """
        from autogluon.tabular import TabularPredictor

        # Materialize the label as a column named `target_name` -- the task's real target name
        # when known -- so its semantic meaning is preserved. `X` is ours to edit in place (the
        # base handled the copy-vs-in-place decision) and `y` shares its index.
        X[target_name] = y

        self._predictor_path = self.path or tempfile.mkdtemp(prefix="tabarena_autogluon_")
        self._predictor = TabularPredictor(
            label=target_name,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=self._predictor_path,
            verbosity=0,
        ).fit(
            X,
            presets=self.preset,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_limit=memory_limit,
            time_limit=time_limit,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return self._predictor.predict(X)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._predictor.predict_proba(X)

    def cleanup(self):
        # Only remove a directory we created: an explicit `path` belongs to the caller.
        if self._predictor_path and self.path is None:
            shutil.rmtree(self._predictor_path, ignore_errors=True)
