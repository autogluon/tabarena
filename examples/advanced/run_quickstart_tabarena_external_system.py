"""Benchmark a Predictive Machine Learning System (like AutoGluon) on TabArena.

Define your system as an ``ExternalSystemModel`` subclass (a minimal ``AutoGluonSystemModel`` is
shown below), pass it to a ``SystemConfigGenerator``, and run the usual quickstart flow with the
bundle flipped to ``system_experiments=True``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


class AutoGluonSystemModel(ExternalSystemModel):
    """Demo system: fit an AutoGluon ``TabularPredictor`` on all the data (see ``ExternalSystemModel``)."""

    def __init__(self, *, preset: str = "medium_quality", path: str | None = None, **kwargs):
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
        target_name: str,  # name for the label column (the task's real target name when known)
        problem_type: str,  # "binary" | "multiclass" | "regression"
        eval_metric: Scorer,  # the metric the task is scored with
        validation_metadata: ValidationMetadata,  # split metadata (group_on / time_on / ...)
        num_cpus: int | None,  # CPU budget for the fit
        num_gpus: int | None,  # GPU budget for the fit
        memory_limit: float | None,  # memory budget in GB
        time_limit: float | None,  # wall-clock budget in seconds
    ):
        """Fit a ``TabularPredictor`` on all the data.

        See the parent ``ExternalSystemModel._fit_system`` docstring for the full argument
        contract — types, semantics, and the in-place / no-validation guarantees.
        """
        from autogluon.tabular import TabularPredictor

        # Materialize the label as a column named `target_name` — the task's real target name when
        # known — so its semantic meaning is preserved
        # `X` is ours to edit in place (the base handled the copy-vs-in-place decision) and `y` shares its index.
        X[target_name] = y

        self._predictor_path = self.path or tempfile.mkdtemp(prefix="tabarena_system_")
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
        import shutil

        if self._predictor_path:
            shutil.rmtree(self._predictor_path, ignore_errors=True)


if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "external_system"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: build the experiments. A `SystemConfigGenerator` pairs the system's exec-model class with a
    #    `name` (a system has no AutoGluon registry name) and its configs; `system_experiments=True`
    #    emits one no-validation `ExternalSystemExperiment` per config. Here we run two configs of the
    #    demo system — the `medium_quality` and `high_quality` presets — yielding
    #    `DemoAutoGluonSystem_c1` / `DemoAutoGluonSystem_c2` (plus the bundle's per-pipeline tag).
    #    A small `time_limit` (seconds) keeps this quick to run; it is forwarded to the system.
    generator = SystemConfigGenerator(
        model_cls=AutoGluonSystemModel,
        name="DemoAutoGluonSystem",
        manual_configs=[{}, {"preset": "high_quality"}],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        system_experiments=True,
    ).build_experiments(time_limit=30)

    # 2: the context is the hub. build_and_run_jobs scopes to a single tiny dataset's first split
    #    via a typed `TaskSubset` (one dataset + `subset="lite"` == r0f0), pairs each config with that
    #    split, runs them locally, and registers the configs as in-memory methods. debug_mode=True ->
    #    in-process native backend.
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=["small", "lite"],
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 3: compare against the cached TabArena baselines; the registered configs are picked up
    #    automatically and carried into the website-format leaderboard with their metadata.
    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
