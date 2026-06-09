"""End-to-end example: two custom datasets through ``ExperimentBatchRunner``.

This walks the full local-benchmark loop on datasets you define yourself (no OpenML
download), and showcases each of these steps:

1. Implement two custom datasets as ``UserTask``s (one classification, one regression).
2. Cache them to disk (``save_local_openml_task``) so they load locally at run time.
3. Register them in ``ExperimentBatchRunner`` via the ``user_tasks=`` argument.
4. Run them on two models, using ``run_jobs`` for a *non-rectangular* sweep (the
   classification task has 3 folds, the regression task has 1).
5. Aggregate the raw results with ``EndToEnd``.
6. Compute a leaderboard via a subclassed ``TabArenaContext.compare`` call.

Run with::

    python examples/benchmarking/run_custom_datasets_via_experiment_batch_runner.py

----------------------------------------------------------------------------------------
Note on step 3 (what this required in the source):
``ExperimentBatchRunner`` historically resolved every dataset name to an *integer* OpenML
tid and handed those ints to ``run_experiments_new``, which loads them via
``OpenMLTaskWrapper.from_task_id`` — i.e. an OpenML *download*. A custom ``UserTask`` lives
only on local disk, so that path cannot run it. The ``user_tasks=`` argument used below was
added for exactly this: a registered ``UserTask`` is resolved to its live object (loaded
locally) instead of its tid. See ``ExperimentBatchRunner._resolve_task``.
----------------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import (
    AGModelBagExperiment,
    ExperimentBatchRunner,
    Job,
)
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def _toy_frame(*, classification: bool) -> pd.DataFrame:
    """A tiny mixed numeric/categorical frame with a ``target`` column."""
    maker = make_classification if classification else make_regression
    kwargs = {"n_classes": 2} if classification else {}
    X, y = maker(n_samples=120, n_features=8, n_informative=5, random_state=0, **kwargs)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = pd.Categorical(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    return df.assign(target=y)


def make_classification_task(task_cache_dir: Path) -> tuple[UserTask, dict]:
    """A 3-fold classification ``UserTask`` plus its legacy-metadata row.

    Steps 1 + 2: build the dataset and stratified splits, then cache the task to disk.
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
    oml_task = task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    task.save_local_openml_task(oml_task)  # cache to disk
    return task, _metadata_row(task, dataset, splits, problem_type="binary", n_classes=2)


def make_regression_task(task_cache_dir: Path) -> tuple[UserTask, dict]:
    """A 1-fold (holdout) regression ``UserTask`` plus its legacy-metadata row."""
    dataset = _toy_frame(classification=False)
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.33,
        random_state=0,
        shuffle=True,
    )
    splits = {0: {0: (train_idx, test_idx)}}

    task = UserTask(task_name="toy_regression", task_cache_path=task_cache_dir)
    oml_task = task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits=splits,
    )
    task.save_local_openml_task(oml_task)  # cache to disk
    return task, _metadata_row(task, dataset, splits, problem_type="regression", n_classes=0)


def _metadata_row(
    task: UserTask,
    dataset: pd.DataFrame,
    splits: dict,
    *,
    problem_type: str,
    n_classes: int,
) -> dict:
    """Legacy one-row ``task_metadata`` for a custom task.

    The dataset name is the task's ``tabarena_task_name`` (also the results ``dataset`` key
    and the tid-map key); ``tid`` must equal ``UserTask.task_id``. Includes every column
    ``TaskMetadataCollection.from_legacy_df`` requires so the same frame drives the runner,
    ``EndToEnd``, and the leaderboard context.
    """
    fold_pairs = [splits[r][f] for r in splits for f in splits[r]]
    return {
        "tid": task.task_id,
        "name": task.tabarena_task_name,
        "dataset": task.tabarena_task_name,
        "problem_type": problem_type,
        "n_folds": len(splits[0]),
        "n_repeats": len(splits),
        "n_features": dataset.shape[1] - 1,
        "n_classes": n_classes,
        "NumberOfInstances": len(dataset),
        "n_samples_train_per_fold": float(np.mean([len(tr) for tr, _ in fold_pairs])),
        "n_samples_test_per_fold": float(np.mean([len(te) for _, te in fold_pairs])),
        "target_feature": "target",
    }


class CustomBenchmarkContext(TabArenaContext):
    """A ``TabArenaContext`` for a self-contained custom benchmark (no paper baselines).

    The base ``compare`` pulls the official TabArena paper results to compare against (via
    ``load_results_paper``). A custom benchmark has no such baselines, so we override it to
    contribute nothing — the leaderboard is then computed purely from the results we pass as
    ``new_results``, against the custom task metadata the context was constructed with.
    """

    def load_results_paper(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


if __name__ == "__main__":
    here = Path(__file__).parent
    results_dir = str(here / "experiments" / "custom_ebr")
    eval_dir = here / "eval" / "custom_ebr"
    task_cache_dir = here / "task_cache" / "custom_ebr"

    # 1 + 2: build and cache the two custom datasets.
    clf_task, clf_meta = make_classification_task(task_cache_dir)
    reg_task, reg_meta = make_regression_task(task_cache_dir)
    tasks = [clf_task, reg_task]
    task_metadata = pd.DataFrame([clf_meta, reg_meta])
    # The native task-metadata representation, shared by EndToEnd (step 5) and the
    # leaderboard context (step 6). `ExperimentBatchRunner` (step 3) still accepts the
    # legacy DataFrame directly.
    task_collection = TaskMetadataCollection.from_legacy_df(task_metadata)

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
        for fold in range(clf_meta["n_folds"]):
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

    # 6: leaderboard via the subclassed context's `compare`, on the custom task metadata.
    context = CustomBenchmarkContext(
        task_metadata=task_collection,
        methods=[],
        fillna_method=None,
        calibration_method=None,
    )
    leaderboard = context.compare(output_dir=eval_dir, new_results=df_results)
    print("\n=== leaderboard ===")
    print(leaderboard.to_string())
