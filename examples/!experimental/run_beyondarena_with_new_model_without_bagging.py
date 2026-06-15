"""Run the BeyondArena loop with *outer* (no-validation) experiments — no bagging.

Counterpart to ``run_beyondarena_with_new_model.py``, but the bundle runs each model as an
``AGModelWrapper`` that trains on all the data with no train/val split, bagging, or ensemble
(``BeyondArenaExperimentBundle(..., outer_experiments=True)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import (
    BeyondArenaExperimentBundle,
    ExperimentBatchRunner,
    build_jobs,
)
from tabarena.benchmark.task.metadata import BeyondArenaTaskMetadataCollection
from tabarena.evaluation.context.beyond_arena import BeyondArenaContext
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.utils.config_utils import ConfigGenerator

if TYPE_CHECKING:
    import pandas as pd


class DummyPredictorModel(AbstractModel):
    """A minimal custom AutoGluon model: a scikit-learn dummy (constant-baseline) predictor."""

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

    # 1: suite metadata -> filter. Same subsets as the bagging example: small + first split
    # (`lite`) + no high-dimensional datasets. See `BeyondArenaContext.SUBSET_PREDICATES`.
    subset = ["lite", "tiny", "!high-dim"]
    task_collection = BeyondArenaTaskMetadataCollection().subset_tasks(subset=subset)

    # 2: materialize -- ensure the relevant data is on the system.
    task_collection = task_collection.materialize()

    # 3: build the experiments. `outer_experiments=True` makes the bundle emit no-validation
    # `AGModelWrapper` fits (no train/val split, no bagging) for each model, instead of bagged
    # experiments. `verbosity` is forwarded to each model's preprocessing feature generator, so
    # its output is printed during fit (2 = the usual key prints, 3+ = more detail; default 2).
    bundle = BeyondArenaExperimentBundle(
        models=[(DummyPredictorModel.config_generator(), 0)],
        outer_experiments=True,
        verbosity=3,  # set to log default preprocessing
    )
    experiments = bundle.build_experiments()

    # 4: experiments x the collection's splits.
    jobs = build_jobs(experiments, task_collection)

    # 5: run
    runner = ExperimentBatchRunner(expname=results_dir, task_metadata=task_collection, debug_mode=True)
    results_lst = runner.run_jobs(jobs)

    # 6: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    df_results = EndToEnd.from_raw_to_results_df(
        results_lst=results_lst,
        task_metadata=task_collection,
        new_result_prefix="[New] ",
    )
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].head().to_string(index=False))

    # 7: compare against the BeyondArena baselines (same subsets, to restrict baselines to the
    # demo's tasks).
    beyond_context = BeyondArenaContext(task_metadata=task_collection)
    beyond_leaderboard = beyond_context.compare(
        output_dir=eval_dir,
        new_results=df_results,
        subset=subset,
    )
