"""Run the BeyondArena loop with *outer* (no-validation) experiments — no bagging.

Counterpart to ``run_quickstart_beyondarena.py``, but the bundle runs each model as an
``AGModelWrapper`` that trains on all the data with no train/val split, bagging, or ensemble
(``BeyondArenaExperimentBundle(..., outer_experiments=True)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.evaluation.context.beyond_arena import BeyondArenaContext
from tabarena.utils.config_utils import ConfigGenerator

if TYPE_CHECKING:
    import pandas as pd


class DummyPredictorModel(AbstractModel):
    """A minimal custom AutoGluon model: a scikit-learn dummy (constant-baseline) predictor.

    NOTE: defined in ``__main__`` only because the runner runs with ``debug_mode=True`` (in-process
    native backend). Ray-backed runs (``debug_mode=False``) need the class in an importable module.
    """

    ag_key = "DUMMYPREDICTOR"
    ag_name = "DummyPredictor"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from sklearn.dummy import DummyClassifier, DummyRegressor

        self.model = DummyRegressor() if self.problem_type == "regression" else DummyClassifier()
        self.model.fit(self.preprocess(X), y)

    # `_predict_proba` is inherited from AbstractModel (uses `self.model.predict[_proba]`).

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """Default-only bundle entry for this model (see the sibling bagging example)."""
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{}])


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "beyondarena_new_model_without_bagging"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname`
    eval_dir = here / "eval" / run_name  # leaderboard `output_dir`

    # 1: suite metadata -> filter
    subset = ["core", "tiny", "!high-dim"]

    # 2: build the experiments. `outer_experiments=True` makes the bundle emit no-validation
    # `AGModelWrapper` fits (no train/val split, no bagging) for each model, instead of bagged
    # experiments. `verbosity` is forwarded to each model's preprocessing feature generator, so
    # its output is printed during fit (2 = the usual key prints, 3+ = more detail; default 2).
    experiments = BeyondArenaExperimentBundle(
        models=[(DummyPredictorModel.config_generator(), 0)],
        outer_experiments=True,
        # verbosity=3,  # set to log default preprocessing
    ).build_experiments()

    # 3: build_and_run_jobs scopes the context's BeyondArena task metadata to `subset`, pairs each
    #    config with each split, materializes the selected tasks, runs the model locally, and registers
    #    the results as in-memory methods
    context = BeyondArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 4: compare against the cached BeyondArena baselines; the registered method is picked up
    #    automatically and the leaderboard is scoped to the tasks just run.
    leaderboard = context.compare(output_dir=eval_dir)
    print(leaderboard.to_markdown())
