"""``EndToEnd`` — the raw-results -> TabArena-artifacts pipeline, and its results container.

One interface for turning a benchmark run's raw ``results.pkl`` artifacts into TabArena
results and (optionally) cached artifact tiers, for any number of methods:

* :meth:`EndToEnd.from_path_raw` — process a raw-results directory task-by-task in parallel.
  Only small, prediction-free frames ever reach the driver; the heavy tiers (raw copies,
  processed repo) are written incrementally by the workers when requested.
* :meth:`EndToEnd.from_raw` — same processing for an already-loaded results list (e.g. what
  ``ExperimentBatchRunner.run_jobs`` returns).
* :meth:`EndToEnd.from_cache` / :meth:`EndToEndResults.from_cache` — load previously cached
  methods.

Every entry point returns an :class:`EndToEndResults`: one :class:`MethodResults` per method.

Artifact tiers are opt-in per call: ``cache`` (metadata.yaml + results parquets, on by
default), ``cache_raw`` (raw ``results.pkl`` copies under the cache layout), and
``cache_processed`` (the processed ``EvaluationRepository``); ``cache_hpo_trajectories``
additionally simulates + caches HPO trajectories for config methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pandas as pd

from tabarena.end_to_end._pipeline import (
    Backend,
    fetch_task_metadata,
    process_path_raw,
    process_raw_results,
)
from tabarena.end_to_end.method_results import MethodResults
from tabarena.models._method_metadata import MethodMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.benchmark.result import BaselineResult
    from tabarena.benchmark.task.metadata import TaskMetadataCollection


class EndToEnd:
    """Entry points of the end-to-end pipeline; every constructor returns :class:`EndToEndResults`."""

    fetch_task_metadata = staticmethod(fetch_task_metadata)

    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult | dict],
        method_metadata: MethodMetadata | None = None,
        task_metadata: TaskMetadataCollection | None = None,
        cache: bool = True,
        cache_raw: bool = False,
        cache_processed: bool = False,
        cache_hpo_trajectories: bool = False,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        method: str | None = None,
        suite: str | None = None,
        artifact_dir: str | Path | None = None,
        backend: Backend = "ray",
        verbose: bool = True,
    ) -> EndToEndResults:
        """Process already-loaded raw results end to end (all methods found in ``results_lst``).

        Parameters
        ----------
        results_lst : list[BaselineResult | dict]
            The raw results of one or more methods on all tasks and configs.
        method_metadata : MethodMetadata or None = None
            The method_metadata containing information about the method.
            If unspecified, it is inferred per method from ``results_lst``.
            If specified, all results are treated as this single method and
            ``method`` / ``suite`` / ``artifact_dir`` are ignored.
        task_metadata : TaskMetadataCollection or None = None
            The tasks the results were produced on (target metric, problem_type, ...).
            If unspecified, inferred from ``results_lst`` (fetched from OpenML by tid).
        cache : bool = True
            If True, caches each method's metadata yaml and results parquets.
        cache_raw : bool = False
            If True, additionally copies the raw results into each method's cache layout.
        cache_processed : bool = False
            If True, additionally caches each method's processed ``EvaluationRepository``.
        cache_hpo_trajectories : bool = False
            If True, also generates and caches HPO trajectories (config methods only; needs the
            processed repo, i.e. ``cache_processed=True`` or an in-memory repo from this call).
        name : str or None = None
            If specified, overwrites the name of the method.
            Raises if more than one config is present.
        name_prefix : str or None = None
            If specified, prepended to the name of every method/config. Useful for ensuring a
            unique name compared to prior results for a given model type, such as when
            re-running LightGBM. Also updates the model_key.
        name_suffix : str or None = None
            If specified, appended to the name of every method/config (see ``name_prefix``).
        model_key : str or None = None
            If specified, overwrites the model_key of the method (result.model_type).
            This is the `ag_key` value, used to distinguish between different config families
            during portfolio simulation.
        method : str or None = None
            The name of the lower directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, defaults to ``{name_prefix}`` for configs or ``{name}`` for
            baselines. Single-method only.
        suite : str or None = None
            The name of the upper directory in the cache:
                ~/.cache/tabarena/artifacts/{suite}/methods/{method}/
            If unspecified, defaults to ``{method}``.
        artifact_dir : str | Path or None = None
            If specified (and ``method_metadata`` is inferred), the method's artifacts are
            cached directly under this directory (``metadata.yaml`` + ``processed/`` +
            ``results/``) instead of the ``{suite}/methods/{method}`` layout under the global
            cache root. Reload with ``MethodMetadata.from_yaml(path=artifact_dir)``.
            Single-method only; ignored if ``method_metadata`` is given.
        backend : "ray" or "native" = "ray"
            Parallelization backend for the HPO/model-result simulation.
        verbose : bool = True
            If True, logs info about the data processing and simulation.
        """
        return EndToEndResults(
            method_results_lst=process_raw_results(
                results_lst,
                task_metadata=task_metadata,
                method_metadata=method_metadata,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                method=method,
                suite=suite,
                artifact_dir=artifact_dir,
                cache=cache,
                cache_raw=cache_raw,
                cache_processed=cache_processed,
                cache_hpo_trajectories=cache_hpo_trajectories,
                backend=backend,
                verbose=verbose,
            ),
        )

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path | list[str | Path],
        method_metadata: MethodMetadata | None = None,
        task_metadata: TaskMetadataCollection | None = None,
        cache: bool = True,
        cache_raw: bool = False,
        cache_processed: bool = False,
        cache_hpo_trajectories: bool = False,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        method: str | None = None,
        suite: str | None = None,
        artifact_dir: str | Path | None = None,
        name_prefix_raw: str | None = None,
        file_paths: list[str | Path] | None = None,
        backend: Backend = "ray",
        num_cpus: int | None = None,
        verbose: bool = True,
    ) -> EndToEndResults:
        """Process a directory of raw ``results.pkl`` files end to end, task-by-task in parallel.

        Each (task, split) group of files is loaded, simulated, and — per the ``cache_*``
        flags — written to disk inside its own worker, so driver memory stays flat regardless
        of run size (predictions never accumulate in the driver).

        Accepts the same arguments as :meth:`from_raw` plus the ones below; see
        :meth:`from_raw` for the shared parameter documentation.

        Parameters
        ----------
        path_raw : str | Path | list[str | Path]
            Directory (or directories) containing the raw ``results.pkl`` files, searched
            recursively.
        name_prefix_raw : str or None = None
            If specified, only results in subdirectories starting with this prefix are used.
            Useful when ``path_raw`` contains results for multiple methods; this should be the
            ``ag_name`` of the method's AbstractModel class.
        file_paths : list[str | Path] or None = None
            Pre-discovered ``results.pkl`` paths under ``path_raw``. Skips the recursive
            directory walk (useful when a preceding inspect step already discovered them);
            mutually exclusive with ``name_prefix_raw``.
        num_cpus : int or None = None
            Number of parallel workers (``backend="ray"``). If None, uses all available CPUs.
        """
        return EndToEndResults(
            method_results_lst=process_path_raw(
                path_raw,
                task_metadata=task_metadata,
                method_metadata=method_metadata,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                method=method,
                suite=suite,
                artifact_dir=artifact_dir,
                name_prefix_raw=name_prefix_raw,
                file_paths=file_paths,
                cache=cache,
                cache_raw=cache_raw,
                cache_processed=cache_processed,
                cache_hpo_trajectories=cache_hpo_trajectories,
                backend=backend,
                num_cpus=num_cpus,
                verbose=verbose,
            ),
        )

    @classmethod
    def from_cache(
        cls,
        methods: list[str | MethodMetadata | tuple[str, str]],
        *,
        default_suite: str | None = None,
    ) -> EndToEndResults:
        """Load previously cached methods; see :meth:`EndToEndResults.from_cache`."""
        return EndToEndResults.from_cache(methods=methods, default_suite=default_suite)

    @classmethod
    def from_raw_to_methods(
        cls,
        results_lst: list[BaselineResult | dict],
        task_metadata: TaskMetadataCollection | None = None,
        *,
        new_result_prefix: str | None = None,
        debug_mode: bool = True,
        verbose: bool = True,
    ) -> list[MethodMetadata]:
        """Turn raw experiment results into a list of ``InMemoryMethodMetadata`` (one per method).

        The registration-first sibling of :meth:`from_raw_to_results_df`: instead of a tidy
        results DataFrame you must thread into ``compare(new_results=...)``, this returns
        method objects to register at context init via ``extra_methods=`` so they flow
        through every context operation, not just ``compare``.
        """
        cache = not debug_mode
        backend: Backend = "native" if debug_mode else "ray"
        results = cls.from_raw(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
            backend=backend,
            verbose=verbose,
        )
        return results.to_method_metadata_lst(new_result_prefix=new_result_prefix)

    @classmethod
    def from_raw_to_results_df(
        cls,
        results_lst: list[BaselineResult | dict],
        task_metadata: TaskMetadataCollection | None = None,
        *,
        new_result_prefix: str | None = None,
        debug_mode: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Turn raw experiment results into a tidy per-(method, dataset, fold) results DataFrame.

        One-call shorthand for the common
        ``EndToEnd.from_raw(...).get_results(...)`` chain.

        Args:
            results_lst: Raw results, e.g. the list returned by
                :meth:`ExperimentBatchRunner.run_jobs`.
            task_metadata: The tasks the results were produced on (inferred from the results
                if omitted).
            new_result_prefix: Prefix added to each method name in the output (e.g. ``"[New] "``),
                handy to distinguish your results from cached baselines when comparing.
            debug_mode: If True (default), use the lightweight settings suited to local runs /
                examples — no caching and the in-process ``"native"`` backend. If False, cache
                metadata + results and simulate via ray, suited to large-scale benchmarking.
            verbose: Whether to print progress.

        Returns:
            The results DataFrame. For finer control (e.g. ``use_model_results``) call
            ``from_raw(...).get_results(...)`` directly.
        """
        cache = not debug_mode
        backend: Backend = "native" if debug_mode else "ray"
        results = cls.from_raw(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
            backend=backend,
            verbose=verbose,
        )
        return results.get_results(new_result_prefix=new_result_prefix)


class EndToEndResults:
    """Results of one pipeline run: one :class:`MethodResults` per method."""

    def __init__(
        self,
        method_results_lst: list[MethodResults],
    ):
        self.method_results_lst = method_results_lst

    @property
    def method_metadata_lst(self) -> list[MethodMetadata]:
        """Each method's (disk-backed) ``MethodMetadata``, e.g. to register on an arena context
        after a cached run. For an in-memory registration use :meth:`to_method_metadata_lst`.
        """
        return [results.method_metadata for results in self.method_results_lst]

    def to_method_metadata_lst(
        self,
        *,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ) -> list[MethodMetadata]:
        """Vend each method as an :class:`InMemoryMethodMetadata` for context registration.

        ``use_suite_in_prefix`` / ``use_model_results`` are forwarded to each method's
        :meth:`MethodResults.to_method_metadata`.
        """
        return [
            results.to_method_metadata(
                new_result_prefix=new_result_prefix,
                use_suite_in_prefix=use_suite_in_prefix,
                use_model_results=use_model_results,
            )
            for results in self.method_results_lst
        ]

    def get_results(
        self,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ) -> pd.DataFrame:
        df_results_lst = []
        for results in self.method_results_lst:
            df_results_lst.append(
                results.get_results(
                    new_result_prefix=new_result_prefix,
                    use_suite_in_prefix=use_suite_in_prefix,
                    use_model_results=use_model_results,
                ),
            )
        return pd.concat(df_results_lst, ignore_index=True)

    @classmethod
    def from_cache(
        cls, methods: list[str | MethodMetadata | tuple[str, str]], *, default_suite: None | str = None
    ) -> Self:
        method_results_lst = []
        for method in methods:
            if isinstance(method, tuple):
                method, suite = method
            else:
                suite = default_suite
            if isinstance(method, MethodMetadata):
                method_metadata = method
            else:
                method_metadata = MethodMetadata.from_yaml(
                    method=method,
                    suite=suite if suite is not None else method,
                )
            method_results_lst.append(MethodResults(method_metadata=method_metadata))
        return cls(method_results_lst=method_results_lst)

    def cache(self):
        for method_results in self.method_results_lst:
            method_results.cache()
