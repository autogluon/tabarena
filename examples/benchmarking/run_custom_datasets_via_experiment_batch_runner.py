"""End-to-end example: two custom datasets through ``ExperimentBatchRunner``.

This walks the full local-benchmark loop on datasets you define yourself (no OpenML
download), and showcases each of these steps:

1. Implement two custom datasets as ``UserTask``s (one classification, one regression);
   ``create_task`` computes each task's native ``TabArenaTaskMetadata`` on the way.
2. Cache them to disk (``save_task``) so they load locally at run time.
3. Register them in ``ExperimentBatchRunner`` via the ``user_tasks=`` argument.
4. Run them on two models, using ``run_jobs`` for a *non-rectangular* sweep (the
   classification task has 3 folds, the regression task has 1).
5. Aggregate the raw results with ``EndToEnd``.
6. Compute a leaderboard via an ``AbstractArenaContext.compare`` call (with no baseline methods).

Run with::

    python examples/benchmarking/run_custom_datasets_via_experiment_batch_runner.py

----------------------------------------------------------------------------------------
Note on step 3 (what this required in the source):
``ExperimentBatchRunner`` resolves every dataset name to a ``TaskSpec`` — the handle that
owns the task's vending logic. An OpenML task resolves to an ``OpenMLTaskSpec`` (an OpenML
*download* on load); a custom ``UserTask`` lives only on local disk, so it must resolve to
the ``UserTask`` itself (which loads locally). The ``user_tasks=`` argument used below
registers the tasks for that resolution. See ``ExperimentBatchRunner._resolve_task``.
----------------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import (
    AGModelBagExperiment,
    ExperimentBatchRunner,
    Job,
)
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.end_to_end import EndToEnd


def _toy_frame(*, classification: bool) -> pd.DataFrame:
    """A tiny mixed numeric/categorical frame with a ``target`` column."""
    maker = make_classification if classification else make_regression
    kwargs = {"n_classes": 2} if classification else {}
    X, y = maker(n_samples=120, n_features=8, n_informative=5, random_state=0, **kwargs)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = pd.Categorical(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    return df.assign(target=y)


def make_classification_task(task_cache_dir: Path) -> tuple[UserTask, TabArenaTaskMetadata]:
    """A 3-fold classification ``UserTask`` plus its native ``TabArenaTaskMetadata``.

    Steps 1 + 2: build the dataset and stratified splits, create the task (its exact
    metadata — problem type, sizes, dtype flags, per-split stats, and the task's
    identity — is computed by ``create_task``), then cache it to disk.
    """
    dataset = _toy_frame(classification=True)
    n_splits = 3
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(
            dataset.drop(columns="target"),
            dataset["target"],
        ),
        n_splits=n_splits,
    )

    task = UserTask(task_name="toy_classification", task_cache_path=task_cache_dir)
    task_wrapper = task.create_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    task.save_task(task_wrapper)  # cache to disk
    return task, task_wrapper.metadata


def make_regression_task(task_cache_dir: Path) -> tuple[UserTask, TabArenaTaskMetadata]:
    """A 1-fold (holdout) regression ``UserTask`` plus its native ``TabArenaTaskMetadata``."""
    dataset = _toy_frame(classification=False)
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.33,
        random_state=0,
        shuffle=True,
    )
    splits = {0: {0: (train_idx, test_idx)}}

    task = UserTask(task_name="toy_regression", task_cache_path=task_cache_dir)
    task_wrapper = task.create_task(
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits=splits,
    )
    task.save_task(task_wrapper)  # cache to disk
    return task, task_wrapper.metadata


if __name__ == "__main__":
    here = Path(__file__).parent
    results_dir = str(here / "experiments" / "custom_ebr")
    eval_dir = here / "eval" / "custom_ebr"
    task_cache_dir = here / "task_cache" / "custom_ebr"

    # 1 + 2: build and cache the two custom datasets, with native metadata throughout —
    # no legacy task_metadata DataFrame anywhere. The collection drives every consumer
    # below: ExperimentBatchRunner (step 3), EndToEnd (step 5), and the leaderboard
    # context (step 6).
    clf_task, clf_meta = make_classification_task(task_cache_dir)
    reg_task, reg_meta = make_regression_task(task_cache_dir)
    tasks = [clf_task, reg_task]
    task_collection = TaskMetadataCollection.from_source([clf_meta, reg_meta])

    # Sanity: the stored metadata matches each task as it will actually load at run
    # time — validate_metadata recomputes the metadata from the loaded task (available
    # on every task wrapper) and raises listing any diverging field.
    for task, meta in [(clf_task, clf_meta), (reg_task, reg_meta)]:
        task.with_task_metadata(meta).load().validate_metadata()

    # Two models. (See `tabarena.models.utils.get_configs_generator_from_name` for the
    # model-registry / random-search route; here we keep it to two explicit configs.)
    from autogluon.tabular.models import LGBModel, RFModel

    methods = [
        AGModelBagExperiment(name="LightGBM", model_cls=LGBModel, model_hyperparameters={}, num_bag_folds=2),
        AGModelBagExperiment(name="RandomForest", model_cls=RFModel, model_hyperparameters={}, num_bag_folds=2),
    ]

    # 3: register the custom tasks so the runner resolves them to local UserTasks.
    runner = ExperimentBatchRunner(
        expname=results_dir,
        task_metadata=task_collection,
        user_tasks=tasks,
        debug_mode=True,
    )

    # 4: run a non-rectangular sweep with run_jobs — clf has 3 folds, reg has 1, and each
    # job names its own (experiment, dataset, fold). A rectangular `run(folds=[0,1,2])`
    # could not express this (reg has no folds 1/2).
    jobs: list[Job] = []
    for method in methods:
        for fold in range(clf_meta.n_splits):
            jobs.append(Job.create(method, clf_task.tabarena_task_name, fold=fold, repeat=0))
        jobs.append(Job.create(method, reg_task.tabarena_task_name, fold=0, repeat=0))
    results_lst = runner.run_jobs(jobs)

    # 5: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_collection,
        cache=False,
        cache_raw=False,
        backend="native",
    )
    df_results = end_to_end.to_results().get_results()
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].to_string(index=False))

    # 6: leaderboard via the context's `compare`, on the custom task metadata. The generic
    # arena context is instantiated directly (no TabArena presets involved); with
    # `methods=[]` it contributes no baseline results (`load_results` returns an empty
    # frame), so the leaderboard is computed purely from the results passed as
    # `new_results`.
    context = AbstractArenaContext(
        task_metadata=task_collection,
        methods=[],
    )
    leaderboard = context.compare(output_dir=eval_dir, new_results=df_results)
    print("\n=== leaderboard ===")
    print(leaderboard.to_string())
