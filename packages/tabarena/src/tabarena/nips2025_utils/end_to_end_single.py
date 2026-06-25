from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Literal, Self

import pandas as pd
from autogluon.common.savers import save_pd

from tabarena.benchmark.result import BaselineResult, ConfigResult
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts.tabarena_context import TabArenaContext
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_simulator import MethodSimulator
from tabarena.nips2025_utils.fetch_metadata import task_metadata_collection_from_openml
from tabarena.nips2025_utils.method_processor import (
    load_all_artifacts,
    load_raw,
)
from tabarena.utils.pickle_utils import fetch_all_pickles
from tabarena.utils.ray_utils import ray_map_list

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.repository import EvaluationRepository


_LEGACY_TASK_METADATA_REJECTED = (
    "EndToEnd/EndToEndSingle no longer accept a legacy task_metadata DataFrame. Pass a "
    "TaskMetadataCollection (e.g. TaskMetadataCollection.from_legacy_df(df)) or None to "
    "auto-infer from OpenML."
)


def _reject_legacy_task_metadata(task_metadata: TaskMetadataCollection | None) -> None:
    """Raise if a legacy DataFrame is passed; ``None`` (auto-infer) and a collection are OK."""
    if task_metadata is not None and not isinstance(task_metadata, TaskMetadataCollection):
        raise TypeError(_LEGACY_TASK_METADATA_REJECTED)


class EndToEndSingle:
    """End-to-end pipeline for processing and evaluating a **single** method's results.

    This class orchestrates:
      1. Inferring method metadata from raw per-task results.
      2. Building an :class:`EvaluationRepository` with processed artifacts.
      3. Simulating HPO and ensembling under TabArena protocols.
      4. Producing per-task evaluation tables (e.g., metrics, train/infer times).

    Most users should not instantiate this class directly. Prefer
    :meth:`EndToEndSingle.from_raw` or :meth:`EndToEndSingle.from_path_raw`.
    If you are evaluating multiple methods, use :class:`EndToEnd`, which
    manages a list of ``EndToEndSingle`` instances.

    Parameters
    ----------
    method_metadata : MethodMetadata
        Resolved metadata describing the method, its cache locations, and naming.
    repo : EvaluationRepository
        Repository of processed artifacts built from the raw runs and task metadata.
    model_results : pd.DataFrame or None
        Raw per-task model results prior to HPO / model selection.
        These are the original results without simulation on TabArena.
        ``None`` if not yet computed.
    hpo_results : pd.DataFrame or None
        TabArena HPO simulation results (one row per (task, config, seed)).
        ``None`` if not yet computed.

    Attributes:
    ----------
    method_metadata : MethodMetadata
        Method identity and on-disk artifact layout (e.g., ``path``, ``path_raw``).
    repo : EvaluationRepository
        Processed repository backing downstream analyses and comparisons.
    model_results : pandas.DataFrame or None
        Raw per-task model results prior to HPO / model selection.
    hpo_results : pandas.DataFrame or None
        Output of TabArena HPO and ensemble simulation for this method.
    task_metadata : pd.DataFrame
        (Property) Task-level metadata table provided by the repository.

    Notes:
    -----
    **Caching & Side Effects**
    - The factory constructors (:meth:`from_raw`, :meth:`from_path_raw`) can
      write artifacts to disk when ``cache=True`` and/or ``cache_raw=True``:
        * Method metadata YAML to ``method_metadata.path_metadata``.
        * Raw run pickles under ``method_metadata.path_raw``.
        * Processed repository files under ``method_metadata.path_processed``
    - Naming overrides (``name`` / ``name_suffix``) are applied to all configs
      for consistency. If a unique name cannot be assigned (e.g., multiple
      distinct configs while forcing a single ``name``), underlying helpers may raise.

    See Also:
    --------
    EndToEnd
        Multi-method manager that constructs and coordinates multiple
        ``EndToEndSingle`` pipelines.
    EndToEndResultsSingle
        Lightweight container returned by :meth:`to_results`.
    MethodMetadata
        Serialized description of a method and its artifact layout.
    TabArenaContext
        Simulator used internally for HPO/model selection under TabArena.

    Examples:
    --------
    Basic usage from raw objects::

        results = [BaselineResult(...), ...]  # one per task/run
        e2e = EndToEndSingle.from_raw(results, cache=True, cache_raw=True)
        res = e2e.to_results()
        print(res.model_results.head())

    From an on-disk raw directory of per-run ``results.pkl`` files::

        e2e = EndToEndSingle.from_path_raw("artifacts/my_method/raw", cache=True)
        res = e2e.to_results()

    Loading a previously cached method::

        e2e = EndToEndSingle.from_cache("MyMethodName")
        res = e2e.to_results()

    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        repo: EvaluationRepository,
        model_results: pd.DataFrame | None,
        hpo_results: pd.DataFrame | None,
    ):
        self.method_metadata = method_metadata
        self.repo = repo
        self.model_results = model_results
        self.hpo_results = hpo_results

    @property
    def task_metadata(self) -> pd.DataFrame:
        """The legacy task-metadata frame the repo was built from (tabrepo stores a DataFrame,
        regardless of whether a collection or a DataFrame was passed to construction).
        """
        return self.repo.task_metadata

    def configs_hyperparameters(self) -> dict[str, dict | None]:
        return self.repo.configs_hyperparameters()

    @classmethod
    def clean_raw(cls, results_lst: list[BaselineResult | dict]) -> list[BaselineResult]:
        return [BaselineResult.from_dict(result=r) for r in results_lst]

    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult | dict],
        method_metadata: MethodMetadata | None = None,
        task_metadata: TaskMetadataCollection | None = None,
        cache: bool = True,
        cache_raw: bool = True,
        cache_hpo_trajectories: bool = False,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        method: str | None = None,
        suite: str | None = None,
        backend: Literal["ray", "native"] = "ray",
        verbose: bool = True,
    ) -> Self:
        """Run logic end-to-end and cache all results:
        1. (only if using `from_path_raw`) load raw artifacts
            path_raw should be a directory containing `results.pkl` files for each run.
            In the current code, we require `path_raw` to contain the results of only 1 type of method.
        2. infer method_metadata
        3. infer task_metadata
        4. generate repo (processed data)
        5. generate results (per-task metric scores, train time, infer time, etc.).

        Parameters
        ----------
        results_lst : list[BaselineResult | dict]
            The raw results of the method on all tasks and configs.
        method_metadata : MethodMetadata or None = None
            The method_metadata containing information about the method.
            If unspecified, will be inferred from ``results_lst``.
            If specified, ``method`` and ``suite`` will be ignored.
        task_metadata : pd.DataFrame or None = None
            The task_metadata containing information for each task,
            such as the target evaluation metric and problem_type.
            If unspecified, will be inferred from ``results_lst``.
        cache : bool = True
            If True, will cache method metadata, processed data, and results to disk.
        cache_raw : bool = True
            If True, will cache raw data to disk.
        cache_hpo_trajectories : bool = False
            If True, will also generate and cache HPO trajectories to disk via
            :meth:`MethodMetadata.generate_hpo_trajectories`. Only applies when
            ``method_metadata.method_type == "config"``; silently skipped otherwise.
        name : str or None = None
            If specified, will overwrite the name of the method.
            Will raise an exception if more than one config is present.
        name_prefix : str or None = None
            If specified, will be prepended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
            Will also update the model_key.
        name_suffix : str or None = None
            If specified, will be appended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
            Will also update the model_key.
        model_key : str or None = None
            If specified, will overwrite the model_key of the method (result.model_type).
            This is the `ag_key` value, used to distinguish between different config families
            during portfolio simulation.
        method : str or None = None
            The name of the lower directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, will default to ``{name_prefix}`` for configs or ``{name}`` for baselines.
        suite : str or None = None
            The name of the upper directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, will default to ``{method}``
        backend : "ray" or "native" = "ray"
            If "ray", will parallelize the calculation of hpo_results and model_results.
            If "native", will sequentially compute hpo_results and model_results.
        verbose : bool = True
            If True will log info about the data processing and simulation.

        Returns:
        -------
        EndToEndSingle
            An initialized EndToEndSingle class based on the provided raw results_lst.
        """
        _reject_legacy_task_metadata(task_metadata)
        log = print if verbose else (lambda *a, **k: None)

        # raw
        results_lst: list[BaselineResult] = cls.clean_raw(results_lst=results_lst)
        if method_metadata is not None and model_key is None:
            model_key = method_metadata.model_key
        results_lst = cls._rename(
            results_lst=results_lst,
            name=name,
            name_prefix=name_prefix,
            name_suffix=name_suffix,
            model_key=model_key,
        )
        if method_metadata is None:
            method_metadata: MethodMetadata = MethodMetadata.from_raw(
                results_lst=results_lst,
                method=method,
                suite=suite,
            )

        log(
            f"{method_metadata.method}: Creating EndToEndSingle from raw results... "
            f"(cache={cache}, cache_raw={cache_raw})",
        )

        if cache or cache_raw:
            log(f'\tArtifacts will be saved to "{method_metadata.path}"')
        if cache:
            method_metadata.to_yaml()

        if cache_raw:
            log(f'\tCaching raw results to "{method_metadata.path_raw}" ({len(results_lst)} task results)')
            method_metadata.cache_raw(results_lst=results_lst)

        if task_metadata is None:
            tids = list({r.task_metadata["tid"] for r in results_lst})
            task_metadata = cls.fetch_task_metadata(tids=tids, verbose=verbose)

        log("\tConverting raw results into an EvaluationRepository...")
        # processed
        repo: EvaluationRepository = method_metadata.generate_repo(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
        )

        if cache:
            # reload into mem-map mode, otherwise can be very slow for large datasets
            repo = method_metadata.load_processed()

        log("\tSimulating HPO...")
        hpo_results, model_results = MethodSimulator(method_metadata).generate_results(
            repo=repo,
            cache=cache,
            backend=backend,
        )

        if cache_hpo_trajectories:
            if method_metadata.method_type == "config":
                log("\tGenerating and caching HPO trajectories...")
                MethodSimulator(method_metadata).generate_hpo_trajectories(
                    repo=repo,
                    backend=backend,
                    cache=True,
                )
            else:
                log(
                    f"\tSkipping HPO trajectories (method_type={method_metadata.method_type!r}, requires 'config')",
                )

        log("\tComplete!")
        return cls(
            method_metadata=method_metadata,
            repo=repo,
            model_results=model_results,
            hpo_results=hpo_results,
        )

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path | list[str | Path],
        method_metadata: MethodMetadata | None = None,
        task_metadata: TaskMetadataCollection | None = None,
        cache: bool = True,
        cache_raw: bool = True,
        cache_hpo_trajectories: bool = False,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        method: str | None = None,
        suite: str | None = None,
        name_prefix_raw: str | None = None,
        backend: Literal["ray", "native"] = "ray",
        num_cpus: int | None = None,
        verbose: bool = True,
    ) -> Self:
        """Create and cache all artifacts for the method in the given directory.

        Parameters
        ----------
        path_raw : str | Path | list[str | Path]
            Path to the directory containing raw results.
        method_metadata
        task_metadata
        cache
        cache_raw
        name
        name_prefix
        name_suffix
        model_key
        method
        suite
        backend
        verbose

        Returns:
        -------

        """
        _reject_legacy_task_metadata(task_metadata)
        if num_cpus is None:
            num_cpus = len(os.sched_getaffinity(0))

        engine = "ray" if backend == "ray" else "sequential"
        results_lst: list[BaselineResult] = load_raw(
            path_raw=path_raw,
            engine=engine,
            name_pattern=name_prefix_raw,
            num_workers=num_cpus,
        )
        return cls.from_raw(
            results_lst=results_lst,
            method_metadata=method_metadata,
            task_metadata=task_metadata,
            cache=cache,
            cache_raw=cache_raw,
            cache_hpo_trajectories=cache_hpo_trajectories,
            name=name,
            name_prefix=name_prefix,
            name_suffix=name_suffix,
            model_key=model_key,
            method=method,
            suite=suite,
            backend=backend,
            verbose=verbose,
        )

    @classmethod
    def _rename(
        cls,
        results_lst: list[BaselineResult],
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        inplace: bool = True,
    ) -> list[BaselineResult]:
        if not inplace:
            results_lst = copy.deepcopy(results_lst)
        rename = any([name, name_prefix, name_suffix])
        rename_model_key = any([model_key, name_prefix, name_suffix])
        if rename or rename_model_key:
            for r in results_lst:
                if rename:
                    r.update_name(name=name, name_prefix=name_prefix, name_suffix=name_suffix)
                if rename_model_key and isinstance(r, ConfigResult):
                    r.update_model_type(name=model_key, name_prefix=name_prefix, name_suffix=name_suffix)
        return results_lst

    @classmethod
    def from_cache(cls, method: str | MethodMetadata, suite: str | None = None) -> Self:
        if isinstance(method, MethodMetadata):
            method_metadata = method
        else:
            if suite is None:
                suite = method
            method_metadata = MethodMetadata.from_yaml(
                method=method,
                suite=suite,
            )
        repo = method_metadata.load_processed()
        end_to_end_results_single = EndToEndResultsSingle(method_metadata=method_metadata)
        return cls(
            method_metadata=method_metadata,
            repo=repo,
            model_results=end_to_end_results_single.model_results,
            hpo_results=end_to_end_results_single.hpo_results,
        )

    def to_results(self) -> EndToEndResultsSingle:
        return EndToEndResultsSingle(
            method_metadata=self.method_metadata,
            model_results=self.model_results,
            hpo_results=self.hpo_results,
        )

    @staticmethod
    def fetch_task_metadata(tids: list[int], verbose: bool = True) -> TaskMetadataCollection:
        """Auto-infer task metadata for ``tids`` as a (lossy) ``TaskMetadataCollection``.

        Thin wrapper around
        :func:`~tabarena.nips2025_utils.fetch_metadata.task_metadata_collection_from_openml`.
        """
        return task_metadata_collection_from_openml(tids=tids, verbose=verbose)

    @staticmethod
    def from_path_raw_to_results(
        path_raw: str | Path | list[str | Path],
        method_metadata: MethodMetadata | None = None,
        task_metadata: TaskMetadataCollection | None = None,
        cache: bool = True,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        method: str | None = None,
        suite: str | None = None,
        num_cpus: int | None = None,
        name_prefix_raw: str | None = None,
        verbose: bool = True,
    ) -> EndToEndResultsSingle:
        """Create and cache end-to-end results for the method in the given directory.
        Will not cache raw or processed data. To cache all artifacts, call `from_path_raw` instead.
        This is ~10x+ faster than calling `from_path_raw` for large artifacts, but will not cache raw or processed data.

        Will process each task separately in parallel using ray, minimizing disk operations and memory burden.

        Parameters
        ----------
        path_raw : str | Path | list[str | Path]
            Path to the directory containing raw results.
        method_metadata : MethodMetadata or None = None
            The method_metadata containing information about the method.
            If unspecified, will be inferred from ``results_lst``.
            If specified, ``method`` and ``suite`` will be ignored.
        task_metadata : pd.DataFrame or None = None
            The task_metadata containing information for each task,
            such as the target evaluation metric and problem_type.
            If unspecified, will be inferred from ``results_lst``.
        cache : bool = True
            If True, will cache method metadata and results to disk.
            This function will never cache raw and processed data.
        name : str or None = None
            If specified, will overwrite the name of the method.
            Will raise an exception if more than one config is present.
        name_suffix : str or None = None
            If specified, will be appended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
        method : str or None = None
            The name of the lower directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, will default to ``{name_prefix}`` for configs or ``{name}`` for baselines.
        suite : str or None = None
            The name of the upper directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, will default to ``{method}``
        num_cpus : int or None = None
            Number of CPUs to use for parallel processing.
            If None, it will use all available CPUs.
        name_prefix_raw: str | None = None
            If specified, we only search for results in subdirectories starting with this prefix.
            Useful when `path_raw` contains results for multiple methods. This should be the
            `ag_name` of the method's AbstractModel class.
        """
        _reject_legacy_task_metadata(task_metadata)
        if num_cpus is None:
            num_cpus = len(os.sched_getaffinity(0))

        print("Get results paths...")
        file_paths = fetch_all_pickles(
            dir_path=path_raw,
            suffix="results.pkl",
            name_pattern=name_prefix_raw,
            num_workers=num_cpus,
        )
        if len(file_paths) == 0:
            raise ValueError(f"No results.pkl files found in {path_raw} with name prefix {name_prefix_raw}!")

        all_file_paths_method = {}
        for file_path in file_paths:
            did_sid = f"{file_path.parts[-3]}/{file_path.parts[-2]}"
            if did_sid not in all_file_paths_method:
                all_file_paths_method[did_sid] = []
            all_file_paths_method[did_sid].append(file_path)

        if task_metadata is None:
            # Only integer (OpenML) task dirs can be fetched from OpenML; UserTask slugs
            # require an explicitly-provided ``task_metadata``.
            tids = list({int(_task_dir(k)) for k in all_file_paths_method if _task_dir(k).isdigit()})
            task_metadata = EndToEndSingle.fetch_task_metadata(tids=tids, verbose=verbose)

        all_file_paths_method = _filter_file_paths_by_task_metadata(all_file_paths_method, task_metadata)

        import ray

        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)

        results: list[EndToEndResultsSingle] = ray_map_list(
            list_to_map=list(all_file_paths_method.values()),
            func=_process_result_list,
            func_element_key_string="file_paths_method",
            num_workers=num_cpus,
            num_cpus_per_worker=1,
            func_put_kwargs=dict(
                method_metadata=method_metadata,
                task_metadata=task_metadata,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                method=method,
                suite=suite,
            ),
            track_progress=True,
            tqdm_kwargs={"desc": "Processing Results"},
            ray_remote_kwargs={"max_calls": 0},
        )

        print("Merging results...")

        e2e_results: EndToEndResultsSingle = EndToEndResultsSingle.concat(results)
        method_metadata = e2e_results.method_metadata

        if cache:
            print(f"Caching metadata and results to {method_metadata.path}...")
            e2e_results.cache()
        return e2e_results


class EndToEndResultsSingle:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        *,
        model_results: pd.DataFrame = None,
        hpo_results: pd.DataFrame = None,
    ):
        self.method_metadata = method_metadata
        if model_results is None:
            model_results = self.method_metadata.load_model_results()
        if hpo_results is None and self.method_metadata.method_type == "config":
            hpo_results = self.method_metadata.load_hpo_results()
        self.model_results = model_results
        self.hpo_results = hpo_results

    def to_method_metadata(
        self,
        *,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ):
        """Vend this method as an :class:`InMemoryMethodMetadata` (metadata + in-memory results).

        The returned object can be passed to an arena context's ``methods=`` /
        ``extra_methods=`` so the method is registered at init and flows through
        ``compare`` and the leaderboard/website machinery like a cached baseline.

        ``use_suite_in_prefix`` / ``use_model_results`` mirror
        :meth:`get_results` (forwarded through :meth:`InMemoryMethodMetadata.from_results_single`).
        """
        from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata

        return InMemoryMethodMetadata.from_results_single(
            self,
            new_result_prefix=new_result_prefix,
            use_suite_in_prefix=use_suite_in_prefix,
            use_model_results=use_model_results,
        )

    def get_results(
        self,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
        fillna: bool = False,
    ) -> pd.DataFrame:
        """Get data to compare results on TabArena leaderboard.

        Args:
                new_result_prefix (str | None): If not None, add a prefix to the new
                    results to distinguish new results from the original TabArena results.
                    Use this, for example, if you re-run a model from TabArena.
        """
        use_model_results = self.method_metadata.method_type != "config" or use_model_results

        df_results = copy.deepcopy(self.model_results) if use_model_results else copy.deepcopy(self.hpo_results)

        if use_suite_in_prefix:
            if new_result_prefix is None:
                new_result_prefix = ""
            new_result_prefix = new_result_prefix + f"[{self.method_metadata.suite}] "
        if new_result_prefix is not None:
            df_results = self.add_prefix_to_results(results=df_results, prefix=new_result_prefix, inplace=True)

        if fillna:
            df_results = self.fillna_results_on_tabarena(df_results=df_results)

        return df_results

    @classmethod
    def add_prefix_to_results(cls, results: pd.DataFrame, prefix: str, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            results = results.copy()
        for col in ["method", "config_type", "ta_name", "ta_suite"]:
            if col in results:
                results[col] = prefix + results[col]
        return results

    def cache(self):
        self.method_metadata.to_yaml()
        if self.hpo_results is not None:
            save_pd.save(path=str(self.method_metadata.path_results_hpo()), df=self.hpo_results)
        if self.model_results is not None:
            save_pd.save(path=str(self.method_metadata.path_results_model()), df=self.model_results)

    @classmethod
    def fillna_results_on_tabarena(cls, df_results: pd.DataFrame) -> pd.DataFrame:
        tabarena_context = TabArenaContext()
        fillna_method = "RF (default)"
        fillna_method_name = "RandomForest"

        df_fillna = tabarena_context.load_results(methods=[fillna_method_name])
        df_fillna = df_fillna[df_fillna["method"] == fillna_method]
        assert not df_fillna.empty

        return TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=df_fillna,
        )

    @classmethod
    def concat(cls, e2e_lst: list[EndToEndResultsSingle]) -> EndToEndResultsSingle:
        method_metadata = copy.deepcopy(e2e_lst[0].method_metadata)
        hpo_results = copy.deepcopy(e2e_lst[0].hpo_results)
        model_results = copy.deepcopy(e2e_lst[0].model_results)
        for e2e_other in e2e_lst[1:]:
            method_metadata_other = e2e_other.method_metadata
            hpo_results_other = e2e_other.hpo_results
            model_results_other = e2e_other.model_results

            # Capture the any() in metadata creation.
            if method_metadata.is_bag or method_metadata_other.is_bag:
                method_metadata.is_bag = True
                method_metadata_other = copy.deepcopy(method_metadata_other)
                method_metadata_other.is_bag = True

            if method_metadata.config_default != method_metadata_other.config_default:
                # The two sides disagree on the default config. Either one already spans multiple
                # configs (config_default None), or — when merging per-task results from partially
                # completed runs — different tasks each finished a single, *different* config, which
                # MethodMetadata.from_raw infers as that task's config_default. Across the merged
                # set the method therefore has multiple configs and no single default: defer it to
                # None (resolved later via get_config_default(use_first_if_missing=True)) and mark
                # the method HPO-capable. This matches what a single-pass from_path_raw over all of
                # the method's files produces, and keeps the merge order-independent (a later
                # single-config task can never overwrite the deferred None).
                method_metadata.config_default = None
                method_metadata_other.config_default = None
                method_metadata.can_hpo = True
                method_metadata_other.can_hpo = True
            if method_metadata.can_hpo != method_metadata_other.can_hpo:
                method_metadata.can_hpo = True
                method_metadata_other.can_hpo = True
            if method_metadata.__dict__ != method_metadata_other.__dict__:
                diffs = {
                    k: (v, method_metadata_other.__dict__.get(k))
                    for k, v in method_metadata.__dict__.items()
                    if v != method_metadata_other.__dict__.get(k)
                }
                diff_str = "\n".join(f"  {k}: {v1!r} != {v2!r}" for k, (v1, v2) in diffs.items())
                raise ValueError(
                    f"Method metadata mismatch! The following fields differ:\n{diff_str}",
                )

            # merge results
            hpo_results_to_concat = [r for r in [hpo_results, hpo_results_other] if r is not None]
            if hpo_results_to_concat:
                hpo_results = pd.concat(hpo_results_to_concat, ignore_index=True)
            model_results_to_concat = [r for r in [model_results, model_results_other] if r is not None]
            if model_results_to_concat:
                model_results = pd.concat(model_results_to_concat, ignore_index=True)
        return EndToEndResultsSingle(
            method_metadata=method_metadata,
            hpo_results=hpo_results,
            model_results=model_results,
        )


def _task_dir(file_path_key: str) -> str:
    """Extract the task-directory component from a ``"{task}/{split}"`` grouping key.

    The task directory is either an OpenML integer task id or a UserTask slug
    (== ``tabarena_task_name``, e.g. ``"emscad-1790bb44ad91"``), so it must be
    treated as an opaque string rather than coerced to ``int``.
    """
    return file_path_key.split("/", maxsplit=1)[0]


def _filter_file_paths_by_task_metadata(
    all_file_paths_method: dict[str, list[Path]],
    task_metadata: TaskMetadataCollection,
) -> dict[str, list[Path]]:
    """Drop grouped file paths whose task is absent from ``task_metadata``.

    Matches each task directory against both the integer ``tid`` (from ``dataset_to_tid()``)
    and the slug ``tabarena_task_name`` (from ``dataset_names()``) so local/user tasks (whose
    directories are slugs, not integers) are not erroneously removed.
    """
    valid_task_keys = {str(t) for t in task_metadata.dataset_to_tid().values()}
    valid_task_keys |= {str(n) for n in task_metadata.dataset_names()}

    removed = [k for k in all_file_paths_method if _task_dir(k) not in valid_task_keys]
    for task_key in sorted({_task_dir(k) for k in removed}):
        print(f"Removing file paths for task not in task_metadata: task={task_key}")
    return {k: v for k, v in all_file_paths_method.items() if k not in removed}


def _process_result_list(
    *,
    file_paths_method: list[Path],
    method_metadata: MethodMetadata | None = None,
    task_metadata: TaskMetadataCollection,
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    suite: str | None = None,
) -> EndToEndResultsSingle:
    results_lst = load_all_artifacts(
        file_paths=file_paths_method,
        engine="sequential",
        progress_bar=False,
    )

    e2e = EndToEndSingle.from_raw(
        results_lst=results_lst,
        method_metadata=method_metadata,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
        name=name,
        name_prefix=name_prefix,
        name_suffix=name_suffix,
        model_key=model_key,
        method=method,
        suite=suite,
        backend="native",
        verbose=False,
    )
    return e2e.to_results()
