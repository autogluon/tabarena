"""Quickstart: benchmark TabArena models on your own (custom / private) datasets.

Same workflow as ``run_quickstart_tabarena.py`` (collection -> bundle -> build_jobs ->
run_jobs -> EndToEnd -> compare); the tasks are datasets you define yourself rather than
the committed TabArena suite:

1. Implement each dataset as a ``UserTask`` (one classification, one regression);
   ``create_task`` computes its native ``TabArenaTaskMetadata`` (problem type, sizes,
   dtype flags, per-split stats) — no legacy task_metadata DataFrame anywhere.
2. Cache them to disk (``save_task``) so they load locally at run time.
3. Collect them into a ``TaskMetadataCollection`` and register them with the runner via
   ``user_tasks=`` (so each dataset name resolves to the local task, not an OpenML download).
4. Run a (non-rectangular) sweep: ``build_jobs`` pairs each experiment with exactly the
   collection's splits, so the 3-fold classification task and the 1-fold regression task
   each get the right number of jobs automatically.
5. Aggregate with ``EndToEnd.from_raw_to_results_df``.
6. Compute a leaderboard. Your own data has no TabArena baselines, so a generic
   ``AbstractArenaContext`` (``methods=[]``) computes it purely from your results.

Run with::

    python examples/benchmarking/run_quickstart_tabarena_custom_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import (
    ExperimentBatchRunner,
    TabArenaV0pt1ExperimentBundle,
    build_jobs,
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
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "quickstart_custom_datasets"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`
    task_cache_dir = here / "task_cache" / run_name

    # 1 + 2: build and cache the two custom datasets, with native metadata throughout.
    clf_task, clf_meta = make_classification_task(task_cache_dir)
    reg_task, reg_meta = make_regression_task(task_cache_dir)
    tasks = [clf_task, reg_task]
    # 3: collect into the native collection that drives every consumer below.
    task_collection = TaskMetadataCollection.from_source([clf_meta, reg_meta])

    # Sanity: the stored metadata matches each task as it will actually load at run time —
    # validate_metadata recomputes the metadata from the loaded task and raises listing any
    # diverging field.
    for task, meta in [(clf_task, clf_meta), (reg_task, reg_meta)]:
        task.with_task_metadata(meta).load().validate_metadata()

    # 4: models to run, each at its default config. See `run_quickstart_tabarena.py` for
    #    custom models + HPO. Registry names: `tabarena.models.utils.get_configs_generator_from_name`.
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[
            ("LightGBM", 0),
            ("RandomForest", 0),
        ],
    )
    experiments = bundle.build_experiments()

    # 5: experiments x the collection's (non-rectangular) splits -> jobs. The clf task has 3
    #    folds and the reg task has 1; build_jobs produces exactly those — no manual job loop.
    jobs = build_jobs(experiments, task_collection)

    # 6: run. Register the custom tasks via `user_tasks=` so the runner resolves each dataset
    #    name to the local UserTask (rather than attempting an OpenML download).
    runner = ExperimentBatchRunner(
        expname=results_dir,
        task_metadata=task_collection,
        user_tasks=tasks,
        debug_mode=True,
    )
    results_lst = runner.run_jobs(jobs)

    # 7: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    df_results = EndToEnd.from_raw_to_results_df(
        results_lst=results_lst,
        task_metadata=task_collection,
        new_result_prefix="[New] ",
    )
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].to_string(index=False))

    # 8: leaderboard via a generic arena context (no TabArena presets / baselines): with
    #    methods=[] the leaderboard is computed purely from the results passed as new_results.
    context = AbstractArenaContext(task_metadata=task_collection, methods=[])
    leaderboard = context.compare(output_dir=eval_dir, new_results=df_results)
    print("\n=== leaderboard ===")
    print(leaderboard.to_string())
