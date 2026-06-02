from __future__ import annotations

from typing import Any, Literal, Type

import pandas as pd
from pathlib import Path

from tabarena.benchmark.result import BaselineResult, ExperimentResults
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

        # Lazy import to avoid a circular import (experiment_runner_api imports this module).
        from tabarena.benchmark.experiment.experiment_runner_api import run_experiments_new

        tids = [int(self._dataset_to_tid_dict[dataset]) for dataset in datasets]

        # Translate folds/repeats into explicit (fold, repeat) pairs run for every task.
        if repeats is None:
            # Legacy single-repeat layout: run repeat 0 and cache under `{fold}` (no repeat
            # in the path), matching the cache files written before this delegation.
            fold_repeat_pairs = [(fold, 0) for fold in folds]
            include_repeat_in_cache_name = False
        else:
            fold_repeat_pairs = [(fold, repeat) for repeat in repeats for fold in folds]
            include_repeat_in_cache_name = True

        if self.only_cache:
            cache_mode = "only"
        elif ignore_cache:
            cache_mode = "ignore"
        else:
            cache_mode = "default"

        if self.mode == "aws":
            s3_kwargs = {"bucket": self.s3_bucket}
            if self.s3_dataset_cache is not None:
                s3_kwargs["dataset_cache"] = self.s3_dataset_cache
        else:
            s3_kwargs = None

        return run_experiments_new(
            output_dir=self.expname,
            model_experiments=methods,
            tasks=tids,
            tasks_metadata=self.task_metadata,
            # Legacy behavior: the results `dataset` key comes from the metadata.
            use_metadata_task_name=True,
            repetitions_mode="individual",
            repetitions_mode_args=fold_repeat_pairs,
            run_mode=self.mode,
            cache_mode=cache_mode,
            cache_path_format=self.cache_path_format,
            include_repeat_in_cache_name=include_repeat_in_cache_name,
            # Forward the configured cache backend. The default `cache_cls_kwargs`
            # carries `include_self_in_call=True`, preserving the legacy
            # `model_failures` artifact on failure.
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            s3_kwargs=s3_kwargs,
            raise_on_failure=raise_on_failure,
            debug_mode=self.debug_mode,
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
