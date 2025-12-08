from __future__ import annotations

from typing import Any, Literal, Type

import json
import pandas as pd
from pathlib import Path
import traceback

from tabarena.benchmark.result import BaselineResult, ExperimentResults
from tabarena.benchmark.task.openml import OpenMLTaskWrapper, OpenMLS3TaskWrapper
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionPickle, CacheFunctionDummy
from tabarena.repository import EvaluationRepository
from tabarena.benchmark.experiment.experiment_constructor import Experiment


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def __init__(
        self,
        expname: str,
        task_metadata: pd.DataFrame,
        cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict | None = None,
        cache_path_format: Literal["name_first", "task_first"] = "name_first",
        only_cache: bool = False,
        mode: str = "local",
        s3_bucket: str | None = None,
        debug_mode: bool = True,
        s3_dataset_cache: str | None = None,
    ):
        """

        Parameters
        ----------
        expname
        cache_cls
        cache_cls_kwargs
        cache_path_format: {"name_first", "task_first"}, default "name_first"
            Determines the folder structure for artifacts.
            "name_first" -> {expname}/data/{method}/{tid}/{fold}/
            "task_first" -> {expname}/data/tasks/{tid}/{fold}/{method}/
        mode: str, default "local"
            Either "local" or "aws". In "aws" mode, s3_bucket must be provided.
        s3_bucket: str, optional
            Required when mode="aws". The S3 bucket where artifacts will be stored.
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        s3_dataset_cache: str, optional
            Full S3 URI to the openml dataset cache (format: s3://bucket/prefix)
            If None, skip S3 download attempt
        """
        cache_cls = CacheFunctionDummy if cache_cls is None else cache_cls
        cache_cls_kwargs = {"include_self_in_call": True} if cache_cls_kwargs is None else cache_cls_kwargs

        self.expname = expname
        self.task_metadata = task_metadata
        self.cache_cls = cache_cls
        self.cache_cls_kwargs = cache_cls_kwargs
        self.cache_path_format = cache_path_format
        self.only_cache = only_cache
        self._dataset_to_tid_dict = self.task_metadata[['tid', 'dataset']].drop_duplicates(['tid', 'dataset']).set_index('dataset')['tid'].to_dict()
        self.mode = mode
        self.s3_bucket = s3_bucket
        self.debug_mode = debug_mode
        self.s3_dataset_cache = s3_dataset_cache

    @property
    def datasets(self) -> list[str]:
        return list(self._dataset_to_tid_dict.keys())

    def run_w_folds_per_dataset(
        self,
        methods: list[Experiment],
        dataset_folds_repeats_lst: list[tuple[str, list[int], list[int] | None]],
        ignore_cache: bool = False,
        raise_on_failure: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Similar to `run` but with the ability to specify folds and repeats on a per-dataset basis.

        Parameters
        ----------
        methods
        dataset_folds_repeats_lst
        ignore_cache: bool, default False
            If True, will run the experiments regardless if the cache exists already, and will overwrite the cache file upon completion.
            If False, will load the cache result if it exists for a given experiment, rather than running the experiment again.
        raise_on_failure

        Returns
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        results_lst = []
        len_datasets = len(dataset_folds_repeats_lst)
        for i, (dataset, folds, repeats) in enumerate(dataset_folds_repeats_lst):
            print(f"Fitting dataset {i+1}/{len_datasets}... (dataset={dataset}, folds={folds}, repeats={repeats})")
            results_lst_cur = self.run(
                methods=methods,
                datasets=[dataset],
                folds=folds,
                repeats=repeats,
                ignore_cache=ignore_cache,
                raise_on_failure=raise_on_failure,
            )
            results_lst += results_lst_cur
        return results_lst

    def run(
        self,
        methods: list[Experiment],
        datasets: list[str],
        folds: list[int],
        repeats: list[int] | None = None,
        ignore_cache: bool = False,
        raise_on_failure: bool = True,
    ) -> list[dict[str, Any]]:
        """

        Parameters
        ----------
        methods
        datasets
        folds
        repeats
        ignore_cache: bool, default False
            If True, will run the experiments regardless if the cache exists already, and will overwrite the cache file upon completion.
            If False, will load the cache result if it exists for a given experiment, rather than running the experiment again.

        Returns
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        self._validate_datasets(datasets=datasets)
        self._validate_folds(folds=folds)
        self._validate_repeats(repeats=repeats)

        tids = [self._dataset_to_tid_dict[dataset] for dataset in datasets]
        return run_experiments(
            expname=self.expname,
            tids=tids,
            folds=folds,
            repeats=repeats,
            methods=methods,
            task_metadata=self.task_metadata,
            ignore_cache=ignore_cache,
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            cache_path_format=self.cache_path_format,
            mode=self.mode,
            s3_bucket=self.s3_bucket,
            only_cache=self.only_cache,
            raise_on_failure=raise_on_failure,
            debug_mode=self.debug_mode,
            s3_dataset_cache=self.s3_dataset_cache
        )

    def load_results(
        self,
        methods: list[Experiment | str],
        datasets: list[str],
        folds: list[int],
        repeats: list[int] | None = None,
        require_all: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Load results from the cache.

        Parameters
        ----------
        methods
        datasets
        folds
        repeats
        require_all: bool, default True
            If True, will raise an exception if not all methods x datasets x folds have a cached result to load.
            If False, will return only the list of results with a cached result. This can be an empty list if no cached results exist.

        Returns
        -------
        results_lst
            The same output format returned by `self.run`

        """
        results_lst = []
        results_lst_exists = []
        results_lst_missing = []
        if repeats is not None:
            repeat_fold_pairs = [(r, f) for r in repeats for f in folds]
        else:
            repeat_fold_pairs = [(None, f) for f in folds]
        for method in methods:
            if isinstance(method, Experiment):
                method_name = method.name
            else:
                method_name = method
            for dataset in datasets:
                for repeat, fold in repeat_fold_pairs:
                    cache_exists = self._cache_exists(method_name=method_name, dataset=dataset, fold=fold)
                    cache_args = (method_name, dataset, fold, repeat)
                    if cache_exists:
                        results_lst_exists.append(cache_args)
                        print(method.name, dataset, fold)
                        print(f"\t{cache_exists}")
                    else:
                        results_lst_missing.append(cache_args)
        if require_all and results_lst_missing:
            raise AssertionError(
                f"Missing cached results for {len(results_lst_missing)}/{len(results_lst_exists) + len(results_lst_missing)} experiments! "
                f"\nTo load only the {len(results_lst_exists)} existing experiments, set `require_all=False`, "
                f"or call `exp_batch_runner.run(methods=methods, datasets=datasets, folds=folds)` to run the missing experiments."
                f"\nMissing experiments:\n\t{results_lst_missing}"
            )
        for method_name, dataset, fold, repeat in results_lst_exists:
            results_lst.append(self._load_result(method_name=method_name, dataset=dataset, fold=fold, repeat=repeat))
        return results_lst

    def repo_from_results(
        self,
        results_lst: list[dict[str, Any] | BaselineResult],
        calibrate: bool = False,
        include_holdout: bool = False,
    ) -> EvaluationRepository:
        experiment_results = ExperimentResults(task_metadata=self.task_metadata)
        repo = experiment_results.repo_from_results(
            results_lst=results_lst,
            calibrate=calibrate,
            include_holdout=include_holdout,
        )
        return repo

    @classmethod
    def _subtask_name(cls, fold: int, repeat: int | None = None) -> str:
        if repeat is None:
            subtask_name = f"{fold}"
        else:
            subtask_name = f"{repeat}_{fold}"
        return subtask_name

    def _cache_name(self, method_name: str, dataset: str, fold: int, repeat: int | None = None) -> str:
        subtask_name = self._subtask_name(fold=fold, repeat=repeat)
        # TODO: Windows? Use Path?
        tid = self._dataset_to_tid_dict[dataset]
        if self.cache_path_format == "name_first":
            cache_name = f"data/{method_name}/{tid}/{subtask_name}/results"
        elif self.cache_path_format == "task_first":
            # Legacy format from early prototyping
            cache_name = f"data/tasks/{tid}/{subtask_name}/{method_name}/results"
        else:
            raise ValueError(f"Unknown cache_path_format: {self.cache_path_format}")
        return cache_name

    def _cache_exists(self, method_name: str, dataset: str, fold: int, repeat: int | None = None) -> bool:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold, repeat=repeat)
        return cacher.exists

    def _load_result(self, method_name: str, dataset: str, fold: int, repeat: int | None = None) -> dict[str, Any]:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold, repeat=repeat)
        return cacher.load_cache()

    def _get_cacher(self, method_name: str, dataset: str, fold: int, repeat: int | None = None) -> AbstractCacheFunction:
        cache_name = self._cache_name(method_name=method_name, dataset=dataset, fold=fold, repeat=repeat)
        cacher = self.cache_cls(cache_name=cache_name, cache_path=self.expname, **self.cache_cls_kwargs)
        return cacher

    def _validate_datasets(self, datasets: list[str]):
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {self.datasets}"
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError(f"Duplicate datasets present! Ensure all datasets are unique.")

    def _validate_folds(self, folds: list[int]):
        if len(folds) != len(set(folds)):
            raise AssertionError(f"Duplicate folds present! Ensure all folds are unique.")

    def _validate_repeats(self, repeats: list[int] | None):
        if repeats is None:
            return
        if len(repeats) != len(set(repeats)):
            raise AssertionError(f"Duplicate repeats present! Ensure all repeats are unique.")

def check_cache_hit(
    *,
    result_dir: str,
    method_name: str,
    task_id: int,
    fold: int,
    repeat: int | None,
    cache_path_format: Literal["name_first", "task_first"],
    cache_cls: Type[AbstractCacheFunction] | None,
    cache_cls_kwargs: dict | None = None,
    mode: Literal["local", "s3"],
    s3_bucket: str | None = None,
    delete_cache: bool = False,
) -> bool:
    """Returns true if cache exists for the given experiment."""
    base_cache_path = result_dir if mode == "local" else f"s3://{s3_bucket}/{result_dir}"

    subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)

    if cache_path_format == "name_first":
        cache_prefix = f"data/{method_name}/{task_id}/{subtask_cache_name}"
        cache_name = "results"
    elif cache_path_format == "task_first":
        # Legacy format from early prototyping
        cache_prefix = f"data/tasks/{task_id}/{subtask_cache_name}/{method_name}"
        cache_name = "results"
    else:
        raise ValueError(f"Invalid cache_path_format: {cache_path_format}")

    cache_path = f"{base_cache_path}/{cache_prefix}"

    cacher = cache_cls(cache_name=cache_name, cache_path=cache_path, **cache_cls_kwargs)

    if delete_cache:
        Path(cacher.cache_file).unlink(missing_ok=True)

    return cacher.exists


def run_experiments(
    expname: str,
    tids: list[int],
    folds: list[int] | None,
    methods: list[Experiment],
    task_metadata: pd.DataFrame | dict,
    ignore_cache: bool,
    repeats: list[int] | None = None,
    cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
    cache_cls_kwargs: dict = None,
    cache_path_format: Literal["name_first", "task_first"] = "name_first",
    mode: str = "local",
    s3_bucket: str | None = None,
    only_cache: bool = False,
    raise_on_failure: bool = True,
    debug_mode: bool = True,
    s3_dataset_cache: str | None = None,
    repeat_fold_pairs: list[tuple[int | None, int]] | None = None,
) -> list[dict]:
    """

    Parameters
    ----------
    expname: str, Name of the experiment given by the user
    tids: list[int], List of OpenML task IDs given by the user
    folds: list[int], Number of folds present for the given task
    repeats: list[int] | None, Number of repeats present for the given task. If None, defaults to [0]
    methods: list[Experiment], Models used for fit() and predict() in this experiment
    task_metadata: pd.DataFrame or None, OpenML task metadata
         If dict, it is a map from TID to task name. As only task name is used here.
    ignore_cache: bool, whether to use cached results (if present)
    cache_cls: WIP
    cache_cls_kwargs: WIP
    cache_path_format: {"name_first", "task_first"}, default "name_first"
    mode: {"local", "aws"}, default "local"
    s3_bucket: str, default None
        Required when mode="aws". The S3 bucket where artifacts will be stored.
    raise_on_failure: bool, default True
        If True, will raise exceptions that occur during experiments, stopping all runs.
        If False, will ignore exceptions and continue fitting queued experiments. Experiments with exceptions will not be included in the output list.
    s3_dataset_cache: str, default None
        Full S3 URI to the openml dataset cache (format: s3://bucket/prefix)
        If None, skip S3 download attempt
    repeat_fold_pairs: list[tuple[int | None, int]] | None, default None
        alternative to `repeats` and `folds` parameters to specify the repeat and fold pairs to run.

    Returns
    -------
    result_lst: list[dict], containing all metrics from fit() and predict() of all the given OpenML tasks
    """
    if mode == "aws" and (s3_bucket is None or s3_bucket == ""):
        raise ValueError(f"s3_bucket parameter is required when mode is 'aws', got {s3_bucket}")

    if cache_cls is None:
        cache_cls = CacheFunctionDummy
    if cache_cls_kwargs is None:
        cache_cls_kwargs = {}

    if folds is None:
        assert repeat_fold_pairs is not None, "If folds is None, repeat_fold_pairs must be provided"

    # Modify cache path based on mode
    if mode == "local":
        base_cache_path = expname
    else:
        base_cache_path = f"s3://{s3_bucket}/{expname}"

    methods_og = methods
    methods = []
    for method in methods_og:
        # TODO: remove tuple input option, doing it to keep old scripts working
        if not isinstance(method, Experiment):
            method = Experiment(name=method[0], method_cls=method[1], method_kwargs=method[2])
        methods.append(method)

    unique_names = set()
    for method in methods:
        if method.name in unique_names:
            raise AssertionError(f"Duplicate experiment name found. All names must be unique. name: {method.name}")
        unique_names.add(method.name)

    # FIXME: dataset or name? Where does `dataset` come from, why can it be different from `name`?
    #  Using dataset for now because for some datasets like "GAMETES", the name is slightly different with `.` in `name` being replaced with `_` in `dataset`.
    #  This is probably because `.` isn't a valid name in a file in s3.
    #  TODO: What if `dataset` doesn't exist as a column? Maybe fallback to `name`? Or do the `name` -> `dataset` conversion, or use tid.
    dataset_name_column = "dataset"
    if isinstance(task_metadata, dict):
        dataset_names = [task_metadata[tid] for tid in tids]
    else:
        dataset_names = [task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0] for tid in tids]

    if repeat_fold_pairs is None:
        n_splits = len(folds)
        if repeats is not None:
            n_splits *= len(repeats)
    else:
        n_splits = len(repeat_fold_pairs)
    if repeat_fold_pairs is None:
        if repeats is not None:
            repeat_fold_pairs = [(r, f) for r in repeats for f in folds]
        else:
            repeat_fold_pairs = [(None, f) for f in folds]
    print(
        f"Running Experiments for expname: '{expname}'..."
        f"\n\tFitting {len(tids)} datasets and {n_splits} repeat-fold splits for a total of {len(tids) * n_splits} tasks"
        f"\n\tFitting {len(methods)} methods on {len(tids) *n_splits} tasks for a total of {len(tids) * n_splits * len(methods)} jobs..."
        f"\n\tTIDs    : {tids}"
        f"\n\tDatasets: {dataset_names}"
        f"\n\tFolds   : {folds}"
        f"\n\tRepeats : {repeats}"
        f"\n\tRepeat-Fold-Pairs: {repeat_fold_pairs}"
        f"\n\tMethods : {[method.name for method in methods]}"
    )
    result_lst = []
    num_datasets = len(tids)
    missing_tasks = []
    cur_experiment_idx = -1
    experiment_success_count = 0
    experiment_fail_count = 0
    experiment_missing_count = 0
    experiment_cache_exists_count = 0
    experiment_count_total = len(tids) * len(methods) * n_splits
    for i, tid in enumerate(tids):
        task = None  # lazy task loading
        if isinstance(task_metadata, dict):
            task_name = task_metadata[tid]
        else:
            task_name = task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0]
        print(f"Starting Dataset {i+1}/{num_datasets}...")
        for repeat, fold in repeat_fold_pairs:
            subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)
            for method in methods:
                cur_experiment_idx += 1
                if cache_path_format == "name_first":
                    cache_prefix = f"data/{method.name}/{tid}/{subtask_cache_name}"
                    cache_name = "results"
                elif cache_path_format == "task_first":
                    # Legacy format from early prototyping
                    cache_prefix = f"data/tasks/{tid}/{subtask_cache_name}/{method.name}"
                    cache_name = "results"
                else:
                    raise ValueError(f"Invalid cache_path_format: {cache_path_format}")
                print(
                    f"\t"
                    f"{cur_experiment_idx}/{experiment_count_total} ran | "
                    f"{experiment_success_count} success | "
                    f"{experiment_fail_count} fail | "
                    f"{experiment_cache_exists_count} cache_exists | "
                    f"{experiment_missing_count} missing | "
                    f"Fitting {task_name} on repeat {repeat}, fold {fold} for method {method.name}"
                )
                cache_path = f"{base_cache_path}/{cache_prefix}"

                cacher = cache_cls(cache_name=cache_name, cache_path=cache_path, **cache_cls_kwargs)

                cache_exists = cacher.exists
                if cache_exists and not ignore_cache:
                    experiment_cache_exists_count += 1

                if only_cache:
                    if not cache_exists:
                        missing_tasks.append(cache_name)
                        experiment_missing_count += 1
                        continue
                    else:
                        out = cacher.load_cache()
                else:
                    if task is None:
                        if ignore_cache or not cache_exists:
                            if s3_dataset_cache:
                                task = OpenMLS3TaskWrapper.from_task_id(task_id=tid, s3_dataset_cache=s3_dataset_cache)
                            else:
                                task = OpenMLTaskWrapper.from_task_id(task_id=tid)

                    if repeat is not None:
                        run_kwargs = {
                            "repeat": repeat,
                        }
                    else:
                        run_kwargs = {}

                    try:
                        out = method.run(
                            task=task,
                            fold=fold,
                            task_name=task_name,
                            cacher=cacher,
                            ignore_cache=ignore_cache,
                            debug_mode=debug_mode,
                            **run_kwargs,
                        )
                    except Exception as exc:
                        error_cacher = cache_cls(cache_name="error", cache_path=cache_path, **cache_cls_kwargs)
                        exception_info = json.dumps({"error": str(exc), "traceback": traceback.format_exc()})
                        error_cacher.save_cache(exception_info)
                        if raise_on_failure:
                            raise
                        print(exc.__class__)
                        print(f"Exception Info:")
                        print(exception_info)
                        out = None
                if out is not None:
                    experiment_success_count += 1
                    result_lst.append(out)
                else:
                    experiment_fail_count += 1

    return result_lst
