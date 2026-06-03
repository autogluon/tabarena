from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import UserTask
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionPickle

if TYPE_CHECKING:
    import pandas as pd


def _clean_repetitions_mode_args_for_matrix(
    repetitions_mode_args: tuple,
) -> tuple[list[int], list[int]]:
    # Ensure input is a tuple of two elements
    assert isinstance(repetitions_mode_args, tuple), "Input must be tuple!"
    assert len(repetitions_mode_args) == 2, (
        "If `repetitions_mode_args` for 'matrix' is a tuple, it must contain two elements: (folds, repeats)"
    )

    a, b = repetitions_mode_args

    # If both are ints -> convert to ranges
    if isinstance(a, int) and isinstance(b, int):
        return (list(range(a)), list(range(b)))

    # If one is int and other not, error
    if isinstance(a, int) or isinstance(b, int):
        raise AssertionError(
            "If `repetitions_mode_args` for 'matrix' is a tuple with integers, both elements must be integers.",
        )

    # Now both must be lists of ints
    assert isinstance(a, list) and isinstance(b, list), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, both elements must be a list."
    )
    assert (len(a) > 0) and all(isinstance(x, int) for x in a), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, the first list must contain at least one integer for folds."
    )
    assert (len(b) > 0) and all(isinstance(x, int) for x in b), (  # noqa: PT018
        "If `repetitions_mode_args` for 'matrix' is a tuple with lists, the second list must contain at least one integer for repeats."
    )

    return (a, b)


def _parse_repetitions_mode_and_args(
    *,
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    repetitions_mode_args: tuple | list | None,
    tasks: list[int | UserTask],
    tasks_metadata: pd.DataFrame | None = None,
) -> list[list[tuple[int, int]]]:
    """Parse the `repetitions_mode` and `repetitions_mode_args` parameters to determine
    which folds and repeats to run per dataset.

    Returns a standardized format: a list of elements, where each element corresponds
    to the repetitions to run for the task; where each element is a list of tuples,
    each tuple represents a fold-repeat pair; and where each tuple contains two
    integers, the first one is the fold index second one is the repeat index.
    """
    if repetitions_mode == "TabArena":
        if tasks_metadata is None:
            from tabarena.nips2025_utils.fetch_metadata import (
                load_curated_task_metadata,
            )

            tasks_metadata = load_curated_task_metadata()
        else:
            # Verify user metadata
            req_columns = [
                "tabarena_num_repeats",
                "num_folds",
                "task_id",
            ]
            for col in req_columns:
                assert col in tasks_metadata.columns, (
                    f"`tasks_metadata` must contain the column '{col}' when `repetitions_mode` is 'TabArena'"
                )
        fold_repeat_pairs_per_task = []
        metadata_task_ids = tasks_metadata["task_id"].astype(str)
        metadata_task_id_set = set(metadata_task_ids.tolist())
        for task in tasks:
            t_id = task.task_id_str if isinstance(task, UserTask) else str(task)
            assert t_id in metadata_task_id_set, f"Task ID '{t_id}' from `tasks` not found in `tasks_metadata`"

            task_meta = tasks_metadata[metadata_task_ids == t_id].iloc[0]
            n_folds = int(task_meta["num_folds"])
            n_repeats = int(task_meta["tabarena_num_repeats"])
            fold_repeat_pairs = [(f, r) for r in range(n_repeats) for f in range(n_folds)]
            fold_repeat_pairs_per_task.append(fold_repeat_pairs)
        return fold_repeat_pairs_per_task

    if repetitions_mode == "TabArena-Lite":
        # Run only the first fold of the first repeat for each task
        return [[(0, 0)] for _ in range(len(tasks))]

    if repetitions_mode == "matrix":
        assert repetitions_mode_args is not None, (
            "If `repetitions_mode` is 'matrix', `repetitions_mode_args` must be provided"
        )
        if isinstance(repetitions_mode_args, list):
            assert len(repetitions_mode_args) == len(tasks), (
                "If `repetitions_mode_args` for 'matrix' is a list, it must have the same length as `tasks`"
            )
            assert all(isinstance(rep, tuple) for rep in repetitions_mode_args), (
                "If `repetitions_mode_args` for 'matrix' is a list, all elements must be tuples"
            )
            repetitions_mode_args = [_clean_repetitions_mode_args_for_matrix(rep) for rep in repetitions_mode_args]
        else:
            assert isinstance(repetitions_mode_args, tuple), (
                "If `repetitions_mode_args` for 'matrix' is not a list, it must be a tuple"
            )
            repetitions_mode_args = [_clean_repetitions_mode_args_for_matrix(repetitions_mode_args)] * len(tasks)
        return [[(f, r) for f in e[0] for r in e[1]] for e in repetitions_mode_args]

    if repetitions_mode == "individual":
        assert repetitions_mode_args is not None, (
            "If `repetitions_mode` is 'individual', `repetitions_mode_args` must be provided"
        )
        assert isinstance(repetitions_mode_args, list), (
            "If `repetitions_mode` is 'individual', `repetitions_mode_args` must be a list"
        )
        assert len(repetitions_mode_args) > 0, "`repetitions_mode_args` for 'individual' must not be empty"

        if isinstance(repetitions_mode_args[0], tuple):
            assert all(
                isinstance(rep, tuple) and (len(rep) == 2) and all(isinstance(i, int) for i in rep)
                for rep in repetitions_mode_args
            ), (
                "If `repetitions_mode_args` for 'individual' is a list of tuples, all elements must be tuples of integers of (fold_index, repeat_index) pairs"
            )
            repetitions_mode_args = [repetitions_mode_args] * len(tasks)

        # At this point, repetitions_mode_args must be list of lists
        assert len(repetitions_mode_args) == len(tasks), (
            "If `repetitions_mode_args` for 'individual' is a list, it must have the same length as `tasks`"
        )
        assert isinstance(repetitions_mode_args[0], list), (
            "Elements of `repetitions_mode_args` for 'individual' must be a list"
        )
        assert all(isinstance(rep, list) for rep in repetitions_mode_args), (
            "If `repetitions_mode_args` for 'individual' is a list, all elements must be lists"
        )
        for e in repetitions_mode_args:
            assert len(e) > 0, (
                "In `repetitions_mode_args`, each task's repetition list must contain at least one (fold, repeat) tuple."
            )
            for rep in e:
                assert isinstance(rep, tuple) and len(rep) == 2, (  # noqa: PT018
                    "In `repetitions_mode_args`, each repetition entry must be a tuple of (fold_index, repeat_index)."
                )
                assert all(isinstance(i, int) for i in rep), (
                    "In `repetitions_mode_args`, each element of a repetition tuple must be an integer."
                )

        return repetitions_mode_args

    raise ValueError(f"Unknown `repetitions_mode` str: {repetitions_mode}")


def _task_cache_key(task: int | UserTask) -> int | str:
    """Canonical, filesystem-safe identifier for a task, used to key its cache.

    Returns the OpenML task id for an integer task, or the ``UserTask.slug`` for a
    local task. This is the per-task component of the results cache path (see
    ``_build_cache_prefix``) and the same key under which the task's text-embedding
    cache is stored (mirrors
    ``tabarena.benchmark.preprocessing.text_cache.text_cache_key``), so a task's
    results and text caches stay consistently keyed off one identifier.
    """
    return task if isinstance(task, int) else task.slug


def _build_cache_prefix(
    *,
    method_name: str,
    cache_task_key: int | str,
    fold: int,
    repeat: int,
) -> str:
    """Build the cache directory prefix (relative to the base cache path).

    The subtask component is always ``{repeat}_{fold}``.
    """
    subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)
    return f"data/{method_name}/{cache_task_key}/{subtask_cache_name}"


def _build_results_cacher(
    *,
    cache_cls: type[AbstractCacheFunction],
    cache_cls_kwargs: dict | None,
    base_cache_path: str,
    method_name: str,
    cache_task_key: int | str,
    fold: int,
    repeat: int,
) -> AbstractCacheFunction:
    """Construct the cacher for a single (method, task, fold, repeat) ``results`` artifact.

    Centralizes the results cache layout so the run loop never assembles cache paths by
    hand: the artifact is named ``results`` and lives at
    ``{base_cache_path}/data/{method_name}/{cache_task_key}/{repeat}_{fold}``. Note the
    task identity enters only as the pre-derived ``cache_task_key`` (see
    ``_task_cache_key``); the cache class itself stays task-agnostic and reusable.

    ``include_self_in_call`` passes the cacher into the runner so a failed fit (in
    benchmark mode) can drop a ``model_failures`` artifact next to the results. It
    defaults to False (no such artifact) and does not affect the ``results`` cache
    file's path, format, or hit logic. An explicit
    ``cache_cls_kwargs["include_self_in_call"]`` takes precedence (e.g.
    ``ExperimentBatchRunner`` sets it True to keep the legacy artifact).
    """
    cache_prefix = _build_cache_prefix(
        method_name=method_name,
        cache_task_key=cache_task_key,
        fold=fold,
        repeat=repeat,
    )
    cache_path = f"{base_cache_path}/{cache_prefix}"
    return cache_cls(
        cache_name="results",
        cache_path=cache_path,
        **{"include_self_in_call": False, **(cache_cls_kwargs or {})},
    )


def run_experiments_new(
    *,
    output_dir: str,
    model_experiments: list[Experiment],
    tasks: list[int | UserTask],
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"],
    tasks_metadata: pd.DataFrame | None = None,
    repetitions_mode_args: tuple | list | None = None,
    cache_mode: Literal["default", "ignore", "only", "only_strict"] = "default",
    cache_cls: type[AbstractCacheFunction] = CacheFunctionPickle,
    cache_cls_kwargs: dict | None = None,
    raise_on_failure: bool = True,
    debug_mode: bool = False,
    failure_on_non_finite_metric_error: bool = False,
) -> list[dict]:
    """Run model experiments for a set of tasks.

    Parameters
    ----------
    output_dir: str
        Path to the local directory where experiment artifacts are cached.
    model_experiments: list[Experiment]
        List of model experiments to run. Each element must be an instance of the
        Experiment class. Each instance contains the configuration of the model
        and experiment to run.
    tasks: list[int | UserTask]
        The OpenML task IDs or UserTask instances to run the experiments on.
        See `tabarena.benchmark.task.user_task` for more details on how to define
        UserTask.
    repetitions_mode: Literal["TabArena-Lite", "TabArena", "matrix", "individual"]
        Determines how to run repeats of experiments:
            - "TabArena-Lite": Preset setting, run the first fold of the first repeat
                for all tasks.
            - "TabArena": Full TabArena benchmark setting. Recommended for the final
                evaluation. Requires `tasks_metadata` to be set.
            - "matrix": Allows you to specify a matrix of folds and repeats to run all
                combinations. See `repetitions_mode_args`.
            - "individual": Allows you to specific individual fold-repeats pairs to run.
                See `repetitions_mode_args`.
    tasks_metadata: pd.DataFrame | None, default None
        Metadata about each task in `tasks`. Required if `repetitions_mode="TabArena"`.

        If None, we assume that the `tasks` contain tasks from the official curated
        TabArena benchmark and load the metadata internally. If it contains tasks
        not in the official benchmark, an error will be raised.

        If pd.DataFrame, we assume the users passes custom metadata. This dataframe
        must contain the following columns:
            "task_id": str
                The task ID for the task as an int.
                If a local task, we assume this to be `UserTask.task_id_str`.
            "tabarena_num_repeats": int
                The number of repeats for the task based on the protocol from TabArena.
                See tabarena.nips2025_utils.fetch_metadata._get_n_repeats for details.
            "num_folds": int
                The number of folds for the task.
    repetitions_mode_args: list | tuple | None, default None
        Determine how many repetitions of the experiments to run per task, i.e., how
        many folds and repeats to run for each task. Note, all tasks come with
        pre-defined splits, so this parameter does not control how the data is split
        and will error if the numbers are not compatible with the pre-defined splits.
        This parameter's behavior depends on the `repetitions_mode`.

        If `repetitions_mode` is "TabArena-Lite", this parameter is ignored.

        If `repetitions_mode` is "matrix", this parameter defines a list of folds and
        a list of repeats for which we run all combinations. For example, if you pass
        `repetitions_mode_args = ([0, 1, 2], [0, 1])`, we will run the folds 0, 1,
        and 2 for repeats 0 and 1. The options to specify the folds and repeats are:
            - tuple[list[int], list[int]]: A tuple of two lists, where the first list
                contains the folds to run, and the second the repeats to run.
                For example, ([0, 2], [0, 3]) will run the first and third fold of the
                first and third repeats for each task. We start counting from 0.
                Set (list[int], [0]) to run folds for the first repeat.
            - tuple[int, int]: The first element is the number of folds to run, and
                the second the number of repeats to run. For example, (5, 3) will run
                the first 5 folds of the first 3 repeats for each task.
                We start counting from 0, so (2, 3) will run folds 0-1 and repeats 0-2.
                Set (X, 1) to run only the first X folds of the first repeat.
            - list[Any]: A list of tuples, where each of the elements follows one of
                the above formats, that specifies the repeat and fold pairs to run for
                each task. We assume the list is ordered the same as the tasks, so the
                first tuple corresponds to the first task, and so on.

        If `repetitions_mode` is "individual", this parameter defines a list of
        individual folds-repeat to run. For example, if you pass
        `repetitions_mode_args = [(0,0), (2,3)]`, we will run the first fold of
        the first repeat and the third fold of the fourth repeats for each task.
        The options to specify individual folds and repeats are:
            - list[tuple[int,int]]: A list of tuples, where the each tuple
                represents one fold-repeat pair to run for all tasks. Each
                tuple contains two integers, the first one is the fold index and
                second one is the repeat index.
            - list[Any]: A list of lists, where each of the elements follows the
                above format, that specifies the repeat and fold pairs to run for
                each task. We assume the list is ordered the same as the tasks, so the
                first tuple corresponds to the first task, and so on.
    cache_mode: Literal["default", "ignore", "only", "only_strict"], default "default"
        Determines how to handle the cache:
            - "default": Skip experiment if cache exists, otherwise run the experiment.
            - "ignore": Ignore the cache and always run the experiment. This will
                overwrite the cache file upon completion.
            - "only": Only load results from cache. This does not run the experiment
                if cache does not exist; missing experiments are silently skipped.
            - "only_strict": Like "only", but raise if any requested experiment is
                missing from the cache (after listing all of the missing experiments).
    cache_cls: type[AbstractCacheFunction], default CacheFunctionPickle
        The cache class used to read/write each experiment's `results`. Must accept
        `cache_name` and `cache_path` constructor arguments (e.g. `CacheFunctionPickle`
        or a drop-in replacement).
    cache_cls_kwargs: dict | None, default None
        Extra keyword arguments forwarded to `cache_cls(...)` (e.g. `compress`).
    raise_on_failure: bool, default True
        If True, will raise exceptions that occur during experiments, stopping all runs.
        If False, will ignore exceptions and continue fitting queued experiments.
        Experiments with exceptions will not be included in the output list.
    debug_mode: bool, default False
        Determine how to run the experiments:
            - If True, operates in a manner best suited for local model development.
                This mode is friendly to local debuggers and avoids subprocesses/threads
                and complex try/except logic.
            - If False, operates in a manner best suited for large-scale benchmarking.
                This mode tries to record information when method's fail and might not
                work well with local debuggers.
    failure_on_non_finite_metric_error: bool, default False
        If True, we count a non-finite final metric error as a failure and delete
        the cache file produced for the run. This is useful to ensure that such
        errors resulting from models running into overflows are not ignored silently.
        Moreover, if `raise_on_failure` is also True, the exception will be raised.

    Each experiment carries its own `dynamic_tabarena_validation_protocol` flag
    (see `Experiment`): when True, that experiment's validation split is
    dynamically configured based on the task type and dataset type at run time.

    Returns:
    -------
    result_lst: list[dict]
        Containing all metrics from fit() and predict() of all the given tasks
    """
    base_cache_path = output_dir

    assert all(isinstance(exp, Experiment) for exp in model_experiments), (
        "All `model_experiments` elements must be instances of Experiment class"
    )
    assert len({exp.name for exp in model_experiments}) == len(model_experiments), (
        "Duplicate experiment name found in `model_experiments`. All names must be unique."
    )
    assert all(isinstance(task, (int, UserTask)) for task in tasks), (
        f"Not all tasks are int or UserTask instances! Got: {tasks}"
    )

    fold_repeat_pairs_per_task = _parse_repetitions_mode_and_args(
        repetitions_mode=repetitions_mode,
        repetitions_mode_args=repetitions_mode_args,
        tasks=tasks,
        tasks_metadata=tasks_metadata,
    )
    n_splits = sum(len(pairs) for pairs in fold_repeat_pairs_per_task)

    print(
        f"Running Experiments, saving to: '{output_dir}'..."
        f"\n\tFitting {len(tasks)} tasks with a total of {n_splits} fold-repeat pairs"
        f"\n\tFitting {len(model_experiments)} methods with {n_splits} fold-repeat pairs for a total of {n_splits * len(model_experiments)} jobs..."
        f"\n\tTIDs    : {tasks}"
        f"\n\tRepeat-Fold-Pairs-Per-Task (first 20): {fold_repeat_pairs_per_task[:20]}"
        f"\n\tMethods : {[method.name for method in model_experiments]}",
    )

    result_lst = []
    cur_experiment_idx, experiment_success_count, experiment_fail_count = -1, 0, 0
    experiment_missing_count, experiment_cache_exists_count = 0, 0
    # (method, task, fold, repeat) tuples that were requested but absent from the cache,
    # collected only for `cache_mode="only_strict"` to raise after the full sweep.
    missing_cached_experiments: list[tuple] = []
    experiment_count_total = n_splits * len(model_experiments)
    for dataset_index, task_id_or_object in enumerate(tasks):
        task, eval_metric_name = None, None
        # Display name used as the results `dataset` key. Derived from the task object
        # (OpenML name) / UserTask slug when the task is loaded below.
        tabarena_task_name = None
        print(f"Starting Dataset {dataset_index + 1}/{len(tasks)}...")

        for split_index, (fold, repeat) in enumerate(fold_repeat_pairs_per_task[dataset_index], start=1):
            print(
                f"Starting Split {split_index}/{len(fold_repeat_pairs_per_task[dataset_index])} (Fold {fold}, Repeat {repeat})...",
            )

            for me_index, model_experiment in enumerate(model_experiments, start=1):
                cur_experiment_idx += 1
                cache_task_key = _task_cache_key(task_id_or_object)
                print(
                    f"Starting Model {me_index}/{len(model_experiments)}..."
                    f"\n\t"
                    f"{cur_experiment_idx}/{experiment_count_total} ran | "
                    f"{experiment_success_count} success | "
                    f"{experiment_fail_count} fail | "
                    f"{experiment_cache_exists_count} cache_exists | "
                    f"{experiment_missing_count} missing | "
                    f"Fitting {cache_task_key} on repeat {repeat}, fold {fold} for method {model_experiment.name}",
                )

                # Setup Cache for this (method, task, fold, repeat) `results` artifact.
                cacher = _build_results_cacher(
                    cache_cls=cache_cls,
                    cache_cls_kwargs=cache_cls_kwargs,
                    base_cache_path=base_cache_path,
                    method_name=model_experiment.name,
                    cache_task_key=cache_task_key,
                    fold=fold,
                    repeat=repeat,
                )
                cache_exists = cacher.exists
                load_only = cache_mode in ("only", "only_strict")

                # Check cache state
                if cache_exists and (cache_mode != "ignore"):
                    experiment_cache_exists_count += 1
                elif (not cache_exists) and load_only:
                    experiment_missing_count += 1
                    if cache_mode == "only_strict":
                        missing_cached_experiments.append((model_experiment.name, cache_task_key, fold, repeat))
                    continue

                if load_only:
                    out = cacher.load_cache()
                else:
                    if (task is None) and ((cache_mode == "ignore") or (not cache_exists)):
                        if isinstance(task_id_or_object, int):
                            task = OpenMLTaskWrapper.from_task_id(task_id=task_id_or_object)
                            if tabarena_task_name is None:
                                tabarena_task_name = task.task.get_dataset().name
                        else:
                            if tabarena_task_name is None:
                                tabarena_task_name = task_id_or_object.tabarena_task_name
                            task = OpenMLTaskWrapper(
                                task=task_id_or_object.load_local_openml_task(),
                                use_task_eval_metric=True,
                                lazy_load_data=True,
                            )

                        eval_metric_name = task.eval_metric
                        print(f"Using eval metric: {eval_metric_name}")

                    task_cache_cm = model_experiment.prepare_for_task(
                        task=task,
                        cache_task_key=cache_task_key,
                    )

                    try:
                        with task_cache_cm:
                            out = model_experiment.run(
                                task=task,
                                fold=fold,
                                cacher=cacher,
                                ignore_cache=cache_mode == "ignore",
                                debug_mode=debug_mode,
                                repeat=repeat,
                                # TODO: remove task_name as required parameter in .run()
                                #   - also unclear how this is used in only cache case,
                                #     where we don't have the task object.
                                task_name=tabarena_task_name,  # used in eval as name later.
                                eval_metric_name=eval_metric_name,
                            )
                    except Exception as exc:
                        if raise_on_failure:
                            raise
                        print(exc.__class__)
                        out = None

                # Safety check for results with non-finite metric errors
                if out is not None:
                    for metric_error_key in ["metric_error", "metric_error_val"]:
                        if metric_error_key not in out:
                            continue

                        if not np.isfinite(out[metric_error_key]):
                            print(
                                "Non-finite final metric error detected: "
                                f"\t{metric_error_key}={out[metric_error_key]}. ",
                            )
                            if failure_on_non_finite_metric_error:
                                print("\tDeleting cache file and counting as failure.")
                                # TODO: fix for s3
                                cacher.delete_cache()
                                out = None
                                if raise_on_failure:
                                    raise RuntimeError(
                                        f"Non-finite metric error detected for key {metric_error_key!r}.",
                                    )
                            break

                if out is not None:
                    experiment_success_count += 1
                    result_lst.append(out)
                else:
                    experiment_fail_count += 1

    if cache_mode == "only_strict" and missing_cached_experiments:
        missing_str = "\n\t".join(str(m) for m in missing_cached_experiments)
        raise AssertionError(
            f"cache_mode='only_strict': missing cached results for "
            f"{len(missing_cached_experiments)}/{experiment_count_total} experiment(s).\n"
            f"Missing (method, task, fold, repeat):\n\t{missing_str}",
        )

    return result_lst
