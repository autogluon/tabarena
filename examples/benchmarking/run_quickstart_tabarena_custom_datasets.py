"""Quickstart: benchmark TabArena models on your own (custom / private) datasets.

1. Implement each dataset as a ``UserTask`` (one classification, one regression);
   ``create_task`` computes its native ``TabArenaTaskMetadata`` (problem type, sizes,
   dtype flags, per-split stats) — no legacy task_metadata DataFrame anywhere.
2. Cache them to disk (``save_task``) so they load locally at run time.
3. Collect them into a ``TaskMetadataCollection`` and build a generic ``AbstractArenaContext``
   over it. Your own data has no TabArena baselines, so ``methods=[]`` — the leaderboard is
   computed purely from your results.
4. ``context.run_experiments(..., user_tasks=tasks)`` runs the (non-rectangular) sweep —
   ``user_tasks=`` resolves each dataset name to the local task (not an OpenML download), and
   the runner runs exactly the collection's splits (so the 3-fold classification task and the
   1-fold regression task each get the right number of jobs automatically) — then registers the
   results as in-memory methods.
5. ``compare`` computes the leaderboard from the registered methods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext


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
    # 3: collect into the native collection that drives the context below.
    task_collection = TaskMetadataCollection.from_source([clf_meta, reg_meta])

    # Sanity: the stored metadata matches each task as it will actually load at run time —
    # validate_metadata recomputes the metadata from the loaded task and raises listing any
    # diverging field.
    for task, meta in [(clf_task, clf_meta), (reg_task, reg_meta)]:
        task.with_task_metadata(meta).load().validate_metadata()

    # 4: models to run, each at its default config. See `run_quickstart_tabarena.py` for
    #    custom models + HPO. Registry names: `tabarena.models.utils.get_configs_generator_from_name`.
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            ("LightGBM", 0),
            ("RandomForest", 0),
        ],
    ).build_experiments()

    # 5: a generic arena context over the custom collection. `methods=[]` -> no TabArena presets /
    #    baselines, so the leaderboard is computed purely from our own results.
    context = AbstractArenaContext(task_metadata=task_collection, methods=[])

    # 6: run + register.
    context.run_experiments(
        experiments,
        expname=results_dir,
        user_tasks=tasks,
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 7: compute the leaderboard from the registered methods.
    leaderboard = context.compare(output_dir=eval_dir)
    print("\n=== leaderboard ===")
    print(leaderboard.to_markdown())
    print(f"\nView saved figures in {eval_dir}")
