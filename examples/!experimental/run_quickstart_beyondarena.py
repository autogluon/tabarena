"""Run the BeyondArena benchmark loop on a custom (non-registry) model defined in this file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import BeyondArenaTaskMetadataCollection
from tabarena.evaluation.context.beyond_arena import BeyondArenaContext
from tabarena.utils.config_utils import ConfigGenerator

if TYPE_CHECKING:
    import pandas as pd


class DummyPredictorModel(AbstractModel):
    """A minimal custom model: an AutoGluon wrapper around scikit-learn's dummy predictors.

    ``ag_key`` / ``ag_name`` are required by ``ConfigGenerator`` (which uses them to name
    the generated configs, e.g. ``DummyPredictor_c1_BAG_L1``).

    NOTE: it is defined here in ``__main__`` only because the runner runs with ``debug_mode=True``
    (in-process "native" backend). For large-scale, Ray-backed runs (``debug_mode=False``) the
    model class MUST live in a separate importable module, since Ray workers cannot unpickle a
    class defined in ``__main__``.
    """

    ag_key = "DUMMYPREDICTOR"
    ag_name = "DummyPredictor"

    def preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Features are ignored by the dummy predictor, so no preprocessing is needed.
        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        X = self.preprocess(X, y=y)

        from sklearn.dummy import DummyClassifier, DummyRegressor

        self.model = DummyRegressor() if self.problem_type == "regression" else DummyClassifier()
        self.model.fit(X, y)

    # `_predict_proba` is inherited from AbstractModel: it calls `self.model.predict_proba`
    # (classification) / `self.model.predict` (regression), relying on the sklearn API.

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """The bundle entry for this model: a default-only config generator.

        With `manual_configs=[{}]` (+ `n_configs=0` in the bundle's `models`), only the
        default config is built; the empty search space is never sampled. To run HPO,
        give the generator a real `search_space` and pass `n_configs > 0`.
        """
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{}])


if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "beyondarena_new_model"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname`
    eval_dir = here / "eval" / run_name  # leaderboard `output_dir`

    # 1: suite metadata -> filter
    # All BeyondArena subset predicate names (pass any to `subset=`, combine with `|`=OR,
    # `!`=NOT within an expression, AND across a list); see `BeyondArenaContext.SUBSET_PREDICATES`:
    #   problem type : binary, multiclass, classification, regression
    #   size bucket  : tiny, small, medium, large           (on max_train_rows)
    #   split regime : iid (== random), temporal, grouped
    #   features     : low-dim, high-dim, text, high-cardinality
    #   split        : core, lite (first split == r0f0 of each dataset), all
    #       `core` is the recommended evaluation protocol.
    subset = ["core", "tiny", "!high-dim"]
    task_collection = BeyondArenaTaskMetadataCollection().subset_tasks(subset=subset)

    # 2: build the model experiment configs.
    experiments = BeyondArenaExperimentBundle(
        models=[(DummyPredictorModel.config_generator(), 0)],
    ).build_experiments()

    # 3: run_experiments materializes the selected tasks, runs the model
    #    locally, and registers the results as in-memory methods
    context = BeyondArenaContext(task_metadata=task_collection)
    context.run_experiments(
        experiments,
        expname=results_dir,
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 4: compare against the cached BeyondArena baselines; the registered method is picked up
    #    automatically and the leaderboard is scoped to the tasks just run.
    leaderboard = context.compare(output_dir=eval_dir)
    print(leaderboard.to_markdown())
