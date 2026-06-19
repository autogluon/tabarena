"""Integration test for the arena-context API.

    1. ``ConfigGenerator(search_space={}, model_cls=..., manual_configs=[...])`` + a
       ``TabArenaV0pt1ExperimentBundle(models=[(generator, 0)], outer_experiments=True)`` for the
       single candidate model under eval,
    2. ``context.build_jobs(experiments, ..., pre_materialize=True)`` to enumerate the jobs,
    3. ``context.metadata_for_jobs(jobs)`` -> per-job ``split_metadata.num_instances_train`` /
       ``num_features_train`` for the backend's load-balancing cost,
    4. ``context.run_job(job, expname=None, register=False, debug_mode=True)`` per job (gather all,
       register once),
    5. ``context.register(raw_results, new_result_prefix=...)``,
    6. ``context.compare(..., return_results=True, return_single=True)`` -> a single leaderboard
       row (``pd.Series``) + the per-split results frame.

This test pins that exact call sequence against a fully-local, network-free stand-in: toy ``UserTask``
datasets (as in ``run_quickstart_tabarena_custom_datasets.py``), a base ``AbstractArenaContext``
holding one synthetic in-memory baseline (standing in for the downloaded suite, so the leaderboard's
pairwise metrics are computable), and AutoGluon's ``DummyModel`` (a constant/majority predictor —
sklearn-backed, deterministic, no GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from autogluon.core.models import DummyModel
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import Job, TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.utils.config_utils import ConfigGenerator

# The example script tags its in-memory method with this prefix, then picks its row out of the
# leaderboard via ``return_single``.
_NEW_RESULT_PREFIX = "[New] "

# Columns the example script reads off the ``compare(return_results=True)`` frame to surface its
# per-split table; the structural contract this test pins.
_RESULT_COLS = ["dataset", "fold", "method", "metric_error"]

# Leaderboard column -> scalar key (a subset of the leaderboard row's columns). The extraction
# "degrades gracefully" — only columns actually present are emitted — so this test does not assert
# any *value*, only that pulling them off the single row works the same way.
_SCALAR_COLUMNS = {"elo": "elo", "rank": "rank", "winrate": "winrate", "mrr": "mrr"}


@dataclass(frozen=True)
class _Future:
    """Opaque handle returned by :meth:`_LocalBackend.run`; its work runs when waited on."""

    thunk: Callable[[], Any]


class _LocalBackend:
    """Single-process twin of the example script's ``LocalBackend``.

    Mirrors the future-based surface the real adapter fans jobs across (``run`` -> a future,
    ``wait_for_futures`` -> the results in submission order).
    """

    world_size = 1
    global_rank = 0

    def run(self, fn: Callable[..., Any], *args: Any, cost_estimate: int = 1) -> _Future:
        return _Future(thunk=lambda: fn(*args))

    def wait_for_futures(self, futures: list[_Future]) -> list[Any]:
        return [future.thunk() for future in futures]


def _toy_frame(*, classification: bool) -> pd.DataFrame:
    """A tiny mixed numeric/categorical frame with a ``target`` column."""
    maker = make_classification if classification else make_regression
    kwargs = {"n_classes": 2} if classification else {}
    X, y = maker(n_samples=120, n_features=8, n_informative=5, random_state=0, **kwargs)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = pd.Categorical(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    return df.assign(target=y)


def _make_classification_task(task_cache_dir) -> tuple[UserTask, TabArenaTaskMetadata]:
    """A 3-fold classification ``UserTask`` plus its native metadata, cached to disk."""
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
    task.save_task(task_wrapper)
    return task, task_wrapper.metadata


def _make_regression_task(task_cache_dir) -> tuple[UserTask, TabArenaTaskMetadata]:
    """A 1-fold (holdout) regression ``UserTask`` plus its native metadata, cached to disk."""
    dataset = _toy_frame(classification=False)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.33, random_state=0, shuffle=True)
    splits = {0: {0: (train_idx, test_idx)}}
    task = UserTask(task_name="toy_regression", task_cache_path=task_cache_dir)
    task_wrapper = task.create_task(
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits=splits,
    )
    task.save_task(task_wrapper)
    return task, task_wrapper.metadata


def _comparison_baseline(collection: TaskMetadataCollection) -> InMemoryMethodMetadata:
    """One in-memory baseline with a result on every (dataset, split) of ``collection``.

    The example script scores its candidate against TabArena's *downloaded* baseline suite, which
    is what makes the leaderboard's pairwise metrics (elo / rank / win-rate) computable. A toy
    local context has no such suite, so this synthesizes a single stand-in baseline straight from
    the collection's task grid (so its tasks line up with the candidate's). ``return_single`` then
    still collapses to the one *registered* candidate row — the baseline is the comparison set,
    not a "new" method.
    """
    grid = collection.task_grid()
    rows = [
        {
            "method": "ToyBaseline (default)",
            "dataset": dataset,
            "fold": int(split),  # a results frame's "fold" is the split index
            "metric_error": 0.4,
            "metric_error_val": 0.4,
            "metric": "rmse" if problem_type == "regression" else "roc_auc",
            "problem_type": problem_type,
            "method_type": "config",
            "method_subtype": "default",
            "config_type": "ToyBaseline",
            "ta_name": "ToyBaseline",
            "ta_suite": "ToyBaseline",
            # Positive train/infer times so the leaderboard's log-scale runtime figures render.
            "time_train_s": 1.0,
            "time_infer_s": 0.1,
        }
        for dataset, split, problem_type in zip(grid["dataset"], grid["split"], grid["problem_type"], strict=True)
    ]
    return InMemoryMethodMetadata(
        results=pd.DataFrame(rows),
        method="ToyBaseline",
        artifact_name="ToyBaseline",
        method_type="config",
        model_key="ToyBaseline",
    )


def test_async_tabarena_api(tmp_path):
    """The async/fan-out build_jobs -> metadata_for_jobs -> run_job -> register -> compare path
    round-trips end-to-end against a toy local setup and a dummy predictor.
    """
    task_cache_dir = tmp_path / "task_cache"
    clf_task, clf_meta = _make_classification_task(task_cache_dir)
    reg_task, reg_meta = _make_regression_task(task_cache_dir)
    tasks = [clf_task, reg_task]
    collection = TaskMetadataCollection.from_source([clf_meta, reg_meta])

    # Score against a single local stand-in baseline (no network) instead of TabArena's
    # downloaded suite; the registered candidate stays the one "new" method ``return_single`` picks.
    context = AbstractArenaContext(methods=[_comparison_baseline(collection)], task_metadata=collection)

    # Step 1: the single candidate model under eval, built exactly as the example script builds it
    # (empty search space + a single manual default config; outer_experiments -> no bagging).
    generator = ConfigGenerator(search_space={}, model_cls=DummyModel, manual_configs=[{}])
    experiments = TabArenaV0pt1ExperimentBundle(models=[(generator, 0)], outer_experiments=True).build_experiments()

    # Step 2: enumerate experiment x split jobs (3 clf folds + 1 reg holdout = 4 jobs for the one
    # method). pre_materialize is a no-op for already-local user tasks.
    jobs = context.build_jobs(experiments, pre_materialize=True)
    assert jobs, "build_jobs produced no jobs"
    assert all(isinstance(job, Job) for job in jobs)
    assert len(jobs) == 4

    # Step 3: the backend load-balancing cost loop, verbatim from the example script — pins that
    # metadata_for_jobs yields one per-job metadata exposing a single split's
    # num_instances_train / num_features_train.
    metas = context.metadata_for_jobs(jobs)
    assert len(metas) == len(jobs)
    costs = []
    for task_meta in metas:
        split_meta = task_meta.split_metadata
        costs.append(max(int(split_meta.num_instances_train * split_meta.num_features_train), 1))
    assert all(cost >= 1 for cost in costs)

    # Step 4: fan the jobs across the (single-process) backend; each runs in-process with
    # register=False so the whole gather is registered once below. user_tasks resolves each
    # dataset name to its local cached task instead of an OpenML download — run_job scopes the
    # collection to the single job's dataset, so each job is handed only its own local task.
    backend = _LocalBackend()
    tasks_by_dataset = {task.tabarena_task_name: task for task in tasks}

    def _run(job: Job) -> list[dict]:
        job_task = tasks_by_dataset[job.task.dataset]
        return context.run_job(job, expname=None, register=False, debug_mode=True, user_tasks=[job_task])

    futures = [backend.run(_run, job, cost_estimate=cost) for job, cost in zip(jobs, costs, strict=True)]
    raw_results = [r for batch in backend.wait_for_futures(futures) for r in batch]
    assert raw_results, "running the jobs produced no results"

    # Step 5: register the gather as one new in-memory method.
    new_methods = context.register(raw_results, new_result_prefix=_NEW_RESULT_PREFIX)
    assert len(new_methods) == 1

    # Step 6: one call -> our method's single leaderboard row + the per-split frame backing it.
    leaderboard_row, new_results = context.compare(
        output_dir=tmp_path / "eval",
        return_results=True,
        return_single=True,
    )

    # return_single collapses to exactly one leaderboard row, as a Series — what the example
    # script returns and reads its scalars off.
    assert isinstance(leaderboard_row, pd.Series)
    # The per-split frame carries the columns the example script surfaces, scoped to our one method.
    assert set(_RESULT_COLS).issubset(new_results.columns)
    assert not new_results.empty
    assert set(new_results["method"]) == {leaderboard_row["method"]}
    # Both problem types ran (the 3 clf folds + the 1 reg fold).
    assert set(new_results["dataset"]) == {clf_task.tabarena_task_name, reg_task.tabarena_task_name}

    # The scalar extraction reads only whatever leaderboard columns exist — assert it round-trips
    # to a dict of floats without raising (values are not meaningful with a single method).
    scalars = {
        key: float(leaderboard_row[col])
        for col, key in _SCALAR_COLUMNS.items()
        if col in leaderboard_row.index and pd.notna(leaderboard_row[col])
    }
    assert isinstance(scalars, dict)
