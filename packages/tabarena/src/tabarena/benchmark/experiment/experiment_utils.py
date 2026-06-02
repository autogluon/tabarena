from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy, CacheFunctionPickle

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.benchmark.experiment.experiment_constructor import Experiment


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def __init__(
        self,
        expname: str,
        task_metadata: pd.DataFrame,
        dataset_fold_repeats: list[tuple[str, int, int]] | None = None,
        cache_cls: type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict | None = None,
        cache_mode: Literal["default", "ignore", "only", "only_strict"] = "default",
        debug_mode: bool = True,
        raise_on_failure: bool = True,
    ):
        """Parameters
        ----------
        expname
        dataset_fold_repeats: list[tuple[str, int, int]] | None, default None
            The allowed (dataset, fold, repeat) triplets that `run_all` executes. If
            None, `run_all` uses the full grid implied by the `n_folds` and `n_repeats`
            columns of `task_metadata`. Each triplet is validated against `task_metadata`
            (dataset must be present; fold < n_folds; repeat < n_repeats).
        cache_cls
        cache_cls_kwargs
        cache_mode: {"default", "ignore", "only", "only_strict"}, default "default"
            How to handle the experiment cache:
                - "default": skip an experiment if its cache exists, otherwise run it.
                - "ignore": always run the experiment, overwriting any existing cache.
                - "only": only load results from cache; never run an experiment, and
                    silently skip experiments whose cache is missing.
                - "only_strict": like "only", but raise if any requested experiment is
                    missing from the cache.
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        raise_on_failure: bool, default True
            If True, exceptions raised during an experiment propagate and stop the run.
            If False, failures are recorded and the remaining experiments continue.
        """
        cache_cls = CacheFunctionDummy if cache_cls is None else cache_cls
        cache_cls_kwargs = {"include_self_in_call": True} if cache_cls_kwargs is None else cache_cls_kwargs

        self.expname = expname
        self.task_metadata = task_metadata
        self.cache_cls = cache_cls
        self.cache_cls_kwargs = cache_cls_kwargs
        self.cache_mode = cache_mode
        self._dataset_to_tid_dict = (
            self.task_metadata[["tid", "dataset"]]
            .drop_duplicates(["tid", "dataset"])
            .set_index("dataset")["tid"]
            .to_dict()
        )
        self.debug_mode = debug_mode
        self.raise_on_failure = raise_on_failure
        if dataset_fold_repeats is not None:
            self._validate_dataset_fold_repeats(dataset_fold_repeats)
        self._dataset_fold_repeats = dataset_fold_repeats

    @property
    def datasets(self) -> list[str]:
        return list(self._dataset_to_tid_dict.keys())

    def run(
        self,
        methods: list[Experiment],
        datasets: list[str],
        folds: list[int],
        repeats: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Parameters
        ----------

        Methods:
        datasets
        folds
        repeats

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        self._validate_datasets(datasets=datasets)
        self._validate_folds(folds=folds)
        self._validate_repeats(repeats=repeats)

        tids = [int(self._dataset_to_tid_dict[dataset]) for dataset in datasets]

        # Translate folds/repeats into explicit (fold, repeat) pairs run for every task.
        # When `repeats` is unspecified, default to repeat 0. The cache path always
        # includes the repeat (`{repeat}_{fold}`).
        if repeats is None:
            fold_repeat_pairs = [(fold, 0) for fold in folds]
        else:
            fold_repeat_pairs = [(fold, repeat) for repeat in repeats for fold in folds]

        return self._run_individual(
            methods=methods,
            tids=tids,
            repetitions_mode_args=fold_repeat_pairs,
        )

    def run_dataset_fold_repeats(
        self,
        methods: list[Experiment],
        dataset_fold_repeats: list[tuple[str, int, int]],
    ) -> list[dict[str, Any]]:
        """Run an explicit list of (dataset, fold, repeat) tasks.

        Unlike `run`, which runs the cartesian product of `datasets` x `folds` x
        `repeats`, this runs exactly the (dataset, fold, repeat) triples provided.

        Parameters
        ----------

        Methods:
        dataset_fold_repeats: list[tuple[str, int, int]]
            The (dataset, fold, repeat) triples to run. Must not contain duplicates.

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        if len(dataset_fold_repeats) != len(set(dataset_fold_repeats)):
            raise AssertionError("Duplicate (dataset, fold, repeat) triples present! Ensure all triples are unique.")

        # Group the (fold, repeat) pairs by dataset, preserving first-seen order, so each
        # task gets its own list of pairs (the per-task "individual" arg format).
        pairs_per_dataset: dict[str, list[tuple[int, int]]] = {}
        for dataset, fold, repeat in dataset_fold_repeats:
            pairs_per_dataset.setdefault(dataset, []).append((fold, repeat))

        datasets = list(pairs_per_dataset.keys())
        self._validate_datasets(datasets=datasets)

        tids = [int(self._dataset_to_tid_dict[dataset]) for dataset in datasets]
        repetitions_mode_args = [pairs_per_dataset[dataset] for dataset in datasets]

        return self._run_individual(
            methods=methods,
            tids=tids,
            repetitions_mode_args=repetitions_mode_args,
        )

    def run_all(
        self,
        methods: list[Experiment],
    ) -> list[dict[str, Any]]:
        """Run the configured (dataset, fold, repeat) triplets.

        If `dataset_fold_repeats` was passed at init, runs exactly those triplets.
        Otherwise, for each dataset runs all folds in `range(n_folds)` x repeats in
        `range(n_repeats)`, taken from the `n_folds` and `n_repeats` columns of
        `task_metadata`. Delegates to `run_dataset_fold_repeats`.

        Parameters
        ----------

        Methods:

        Returns:
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        dataset_fold_repeats = self._dataset_fold_repeats
        if dataset_fold_repeats is None:
            for col in ("dataset", "n_folds", "n_repeats"):
                if col not in self.task_metadata.columns:
                    raise AssertionError(f"`task_metadata` must contain the column '{col}' to use `run_all`.")

            metadata = self.task_metadata.drop_duplicates(subset="dataset")
            dataset_fold_repeats = []
            for dataset, n_folds, n_repeats in zip(
                metadata["dataset"], metadata["n_folds"], metadata["n_repeats"], strict=False
            ):
                for repeat in range(int(n_repeats)):
                    for fold in range(int(n_folds)):
                        dataset_fold_repeats.append((dataset, fold, repeat))

        return self.run_dataset_fold_repeats(
            methods=methods,
            dataset_fold_repeats=dataset_fold_repeats,
        )

    def _run_individual(
        self,
        methods: list[Experiment],
        tids: list[int],
        repetitions_mode_args: list,
    ) -> list[dict[str, Any]]:
        """Invoke `run_experiments_new` in 'individual' repetitions mode.

        `repetitions_mode_args` is either a single list of (fold, repeat) pairs applied
        to every task, or one (fold, repeat) list per task (aligned with `tids`). See
        `run_experiments_new` for the 'individual' arg format.
        """
        # Lazy import to avoid a circular import (experiment_runner_api imports this module).
        from tabarena.benchmark.experiment.experiment_runner_api import run_experiments_new

        return run_experiments_new(
            output_dir=self.expname,
            model_experiments=methods,
            tasks=tids,
            tasks_metadata=self.task_metadata,
            repetitions_mode="individual",
            repetitions_mode_args=repetitions_mode_args,
            cache_mode=self.cache_mode,
            # Forward the configured cache backend. The default `cache_cls_kwargs`
            # carries `include_self_in_call=True`, preserving the legacy
            # `model_failures` artifact on failure.
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            raise_on_failure=self.raise_on_failure,
            debug_mode=self.debug_mode,
        )

    @classmethod
    def _subtask_name(cls, fold: int, repeat: int | None = None) -> str:
        return f"{fold}" if repeat is None else f"{repeat}_{fold}"

    def _validate_datasets(self, datasets: list[str]):
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {self.datasets}",
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError("Duplicate datasets present! Ensure all datasets are unique.")

    def _validate_folds(self, folds: list[int]):
        if len(folds) != len(set(folds)):
            raise AssertionError("Duplicate folds present! Ensure all folds are unique.")

    def _validate_repeats(self, repeats: list[int] | None):
        if repeats is None:
            return
        if len(repeats) != len(set(repeats)):
            raise AssertionError("Duplicate repeats present! Ensure all repeats are unique.")

    def _validate_dataset_fold_repeats(self, dataset_fold_repeats: list[tuple[str, int, int]]):
        """Verify each (dataset, fold, repeat) is possible according to `task_metadata`.

        Checks the dataset is present, and (when the `n_folds`/`n_repeats` columns exist)
        that `0 <= fold < n_folds` and `0 <= repeat < n_repeats` for that dataset.
        """
        has_counts = {"n_folds", "n_repeats"}.issubset(self.task_metadata.columns)
        counts: dict[str, tuple[int, int]] = {}
        if has_counts:
            metadata = self.task_metadata.drop_duplicates(subset="dataset")
            counts = {
                dataset: (int(n_folds), int(n_repeats))
                for dataset, n_folds, n_repeats in zip(
                    metadata["dataset"], metadata["n_folds"], metadata["n_repeats"], strict=False
                )
            }
        invalid = []
        for dataset, fold, repeat in dataset_fold_repeats:
            if dataset not in self._dataset_to_tid_dict:
                invalid.append((dataset, fold, repeat, "unknown dataset"))
            elif has_counts:
                n_folds, n_repeats = counts[dataset]
                if not (0 <= fold < n_folds) or not (0 <= repeat < n_repeats):
                    invalid.append((dataset, fold, repeat, f"out of range (n_folds={n_folds}, n_repeats={n_repeats})"))
        if invalid:
            invalid_str = "\n\t".join(str(x) for x in invalid)
            raise AssertionError(
                f"`dataset_fold_repeats` contains {len(invalid)} entry(ies) not valid for `task_metadata`:\n\t{invalid_str}",
            )


def check_cache_hit(
    *,
    result_dir: str,
    method_name: str,
    task_id: int,
    fold: int,
    repeat: int | None,
    cache_cls: type[AbstractCacheFunction] | None,
    cache_cls_kwargs: dict | None = None,
    delete_cache: bool = False,
) -> bool:
    """Returns true if cache exists for the given experiment."""
    base_cache_path = result_dir

    subtask_cache_name = ExperimentBatchRunner._subtask_name(fold=fold, repeat=repeat)

    cache_prefix = f"data/{method_name}/{task_id}/{subtask_cache_name}"
    cache_name = "results"

    cache_path = f"{base_cache_path}/{cache_prefix}"

    cacher = cache_cls(cache_name=cache_name, cache_path=cache_path, **cache_cls_kwargs)

    if delete_cache:
        Path(cacher.cache_file).unlink(missing_ok=True)

    return cacher.exists
