"""Internals of the raw-results -> artifacts pipeline behind :class:`~tabarena.end_to_end.EndToEnd`.

The pipeline operates per task: each (task, split) group of ``results.pkl`` files is loaded,
renamed, turned into a small in-memory repo, and simulated in its own worker. Workers write the
heavy artifact tiers (raw copies, processed per-task predictions) straight to disk and return
only prediction-free frames, so driver memory stays flat no matter how large the run is. The
driver then merges the per-task pieces per method: result frames via
:meth:`MethodResults.concat`, and the processed tier's task-independent context files via
:func:`~tabarena.repository.evaluation_repository.write_processed_context`.

The same per-method core (:func:`process_results_in_memory`) also runs in-process on an
already-loaded results list (``EndToEnd.from_raw``), so both entry points share one backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import pandas as pd

from tabarena.benchmark.result.baseline_result import BaselineResult
from tabarena.benchmark.result.config_result import ConfigResult
from tabarena.benchmark.result.raw_loading import fetch_raw_result_paths, load_all_artifacts
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.metadata.fetch_metadata import task_metadata_collection_from_openml
from tabarena.end_to_end.method_results import MethodResults
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_simulator import MethodSimulator
from tabarena.utils.ray_utils import ray_map_list

if TYPE_CHECKING:
    from tabarena.repository import EvaluationRepository

_LEGACY_TASK_METADATA_REJECTED = (
    "EndToEnd no longer accepts a legacy task_metadata DataFrame. Pass a "
    "TaskMetadataCollection (e.g. TaskMetadataCollection.from_legacy_df(df)) or None to "
    "auto-infer from OpenML."
)

#: Simulation / parallelization backend: ``"ray"`` (parallel) or ``"native"`` (in-process).
Backend = Literal["ray", "native"]


def _log_fn(verbose: bool):
    return print if verbose else (lambda *a, **k: None)


def reject_legacy_task_metadata(task_metadata: TaskMetadataCollection | None) -> None:
    """Raise if a legacy DataFrame is passed; ``None`` (auto-infer) and a collection are OK."""
    if task_metadata is not None and not isinstance(task_metadata, TaskMetadataCollection):
        raise TypeError(_LEGACY_TASK_METADATA_REJECTED)


def fetch_task_metadata(tids: list[int], verbose: bool = True) -> TaskMetadataCollection:
    """Auto-infer task metadata for ``tids`` as a (lossy) ``TaskMetadataCollection``.

    Thin wrapper around
    :func:`~tabarena.benchmark.task.metadata.fetch_metadata.task_metadata_collection_from_openml`.
    """
    return task_metadata_collection_from_openml(tids=tids, verbose=verbose)


def clean_raw(results_lst: list[BaselineResult | dict]) -> list[BaselineResult]:
    return [BaselineResult.from_dict(result=r) for r in results_lst]


def rename_results(
    results_lst: list[BaselineResult],
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
) -> list[BaselineResult]:
    """Apply the naming overrides to each result in-place (config family keys included)."""
    rename = any([name, name_prefix, name_suffix])
    rename_model_key = any([model_key, name_prefix, name_suffix])
    if rename or rename_model_key:
        for r in results_lst:
            if rename:
                r.update_name(name=name, name_prefix=name_prefix, name_suffix=name_suffix)
            if rename_model_key and isinstance(r, ConfigResult):
                r.update_model_type(name=model_key, name_prefix=name_prefix, name_suffix=name_suffix)
    return results_lst


def _task_dir(file_path_key: str) -> str:
    """Extract the task-directory component from a ``"{task}/{split}"`` grouping key.

    The task directory is either an OpenML integer task id or a UserTask slug
    (== ``tabarena_task_name``, e.g. ``"emscad-1790bb44ad91"``), so it must be
    treated as an opaque string rather than coerced to ``int``.
    """
    return file_path_key.split("/", maxsplit=1)[0]


def group_file_paths_by_task(file_paths: list[Path]) -> dict[str, list[Path]]:
    """Group ``.../{task}/{split}/results.pkl`` paths into per-(task, split) lists."""
    groups: dict[str, list[Path]] = {}
    for file_path in file_paths:
        did_sid = f"{file_path.parts[-3]}/{file_path.parts[-2]}"
        if did_sid not in groups:
            groups[did_sid] = []
        groups[did_sid].append(file_path)
    return groups


def filter_file_paths_by_task_metadata(
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


@dataclass
class _RepoFragment:
    """Prediction-free slice of a processed repo, mergeable across tasks.

    Carries exactly what :func:`~tabarena.repository.evaluation_repository.write_processed_context`
    needs; the per-task prediction/ground-truth files themselves are written to disk by the
    worker that owned the repo (:meth:`EvaluationRepository.to_dir_task_data`).
    """

    df_configs: pd.DataFrame | None
    df_baselines: pd.DataFrame | None
    configs_hyperparameters: dict | None
    dataset_fold_lst_pp: list[tuple[str, int]]
    dataset_fold_lst_gt: list[tuple[str, int]]

    @classmethod
    def from_repo(cls, repo: EvaluationRepository) -> Self:
        # The aligned zeroshot-context frames are exactly what repo.to_dir would serialize.
        zeroshot_context = repo._zeroshot_context
        return cls(
            df_configs=zeroshot_context.df_configs,
            df_baselines=zeroshot_context.df_baselines,
            configs_hyperparameters=zeroshot_context.configs_hyperparameters,
            dataset_fold_lst_pp=repo._tabular_predictions.dataset_fold_lst(),
            dataset_fold_lst_gt=repo._ground_truth.dataset_fold_lst(),
        )

    @classmethod
    def merge(cls, fragments: list[Self]) -> Self:
        def _concat(frames: list[pd.DataFrame | None]) -> pd.DataFrame | None:
            frames = [f for f in frames if f is not None and len(f) > 0]
            return pd.concat(frames, ignore_index=True) if frames else None

        configs_hyperparameters: dict = {}
        for fragment in fragments:
            if fragment.configs_hyperparameters:
                for config, hyperparameters in fragment.configs_hyperparameters.items():
                    configs_hyperparameters.setdefault(config, hyperparameters)
        return cls(
            df_configs=_concat([f.df_configs for f in fragments]),
            df_baselines=_concat([f.df_baselines for f in fragments]),
            configs_hyperparameters=configs_hyperparameters or None,
            dataset_fold_lst_pp=sorted({tuple(t) for f in fragments for t in f.dataset_fold_lst_pp}),
            dataset_fold_lst_gt=sorted({tuple(t) for f in fragments for t in f.dataset_fold_lst_gt}),
        )


@dataclass
class _MethodBuild:
    """One method's build output for one slice of tasks (or all tasks, for in-memory runs)."""

    results: MethodResults
    repo_fragment: _RepoFragment | None = None


def process_results_in_memory(
    results_lst: list[BaselineResult | dict],
    *,
    task_metadata: TaskMetadataCollection,
    method_metadata: MethodMetadata | None = None,
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    suite: str | None = None,
    artifact_dir: str | Path | None = None,
    cache_raw: bool = False,
    cache_processed: bool = False,
    simulate_backend: Backend = "native",
    keep_repo: bool = False,
    verbose: bool = False,
) -> list[_MethodBuild]:
    """Process already-loaded raw results into per-method results (one ``_MethodBuild`` each).

    The per-method core shared by the in-memory entry point (all tasks at once) and the
    per-task workers (one (task, split) slice at a time): clean + rename the results, split
    them by method (unless pinned via ``method_metadata``), and per method optionally write
    the raw/processed artifact slices, build the in-memory repo, and simulate HPO /
    model-selection results.

    ``keep_repo`` retains each method's in-memory repo on its ``MethodResults`` — only for
    in-process use; workers must not ship repos (predictions) back to the driver.
    """
    log = _log_fn(verbose)
    results_lst = clean_raw(results_lst=results_lst)
    if method_metadata is not None and model_key is None:
        model_key = method_metadata.model_key
    results_lst = rename_results(
        results_lst=results_lst,
        name=name,
        name_prefix=name_prefix,
        name_suffix=name_suffix,
        model_key=model_key,
    )

    if method_metadata is not None:
        groups: list[tuple[MethodMetadata | None, list[BaselineResult]]] = [(method_metadata, results_lst)]
    else:
        grouped: dict[tuple, list[BaselineResult]] = {}
        for r in results_lst:
            r_metadata = MethodMetadata.from_raw(results_lst=[r])
            key = (r_metadata.method, r_metadata.suite, r_metadata.method_type)
            grouped.setdefault(key, []).append(r)
        if len(grouped) > 1:
            log(f"Found {len(grouped)} unique methods: {list(grouped)}")
            single_method_only = {"name": name, "method": method, "artifact_dir": artifact_dir}
            set_args = [k for k, v in single_method_only.items() if v is not None]
            if set_args:
                raise ValueError(
                    f"Arguments {set_args} apply to a single method, but the raw results contain "
                    f"{len(grouped)} methods: {list(grouped)}. Process the methods separately "
                    f"(e.g. filter via name_prefix_raw) or drop these arguments."
                )
        groups = [(None, group) for group in grouped.values()]

    builds: list[_MethodBuild] = []
    for group_metadata, group in groups:
        cur_metadata = group_metadata
        if cur_metadata is None:
            cur_metadata = MethodMetadata.from_raw(
                results_lst=group,
                method=method,
                suite=suite,
                artifact_dir=artifact_dir,
            )
        if cache_raw:
            log(f'\tCaching raw results to "{cur_metadata.path_raw}" ({len(group)} task results)')
            cur_metadata.cache_raw(results_lst=group)

        repo = cur_metadata.generate_repo(results_lst=group, task_metadata=task_metadata, cache=False)
        hpo_results, model_results = MethodSimulator(cur_metadata).generate_results(
            repo=repo,
            cache=False,
            backend=simulate_backend,
        )

        repo_fragment = None
        if cache_processed:
            repo.to_dir_task_data(path=cur_metadata.path_processed)
            repo_fragment = _RepoFragment.from_repo(repo)

        builds.append(
            _MethodBuild(
                results=MethodResults(
                    method_metadata=cur_metadata,
                    model_results=model_results,
                    hpo_results=hpo_results,
                    repo=repo if keep_repo else None,
                ),
                repo_fragment=repo_fragment,
            ),
        )
    return builds


def _process_task_group(
    *,
    file_paths_method: list[Path],
    task_metadata: TaskMetadataCollection,
    method_metadata: MethodMetadata | None = None,
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    suite: str | None = None,
    artifact_dir: str | Path | None = None,
    cache_raw: bool = False,
    cache_processed: bool = False,
) -> list[_MethodBuild]:
    """Worker: fully process one (task, split) group of ``results.pkl`` files."""
    results_lst = load_all_artifacts(
        file_paths=file_paths_method,
        engine="sequential",
        progress_bar=False,
    )
    return process_results_in_memory(
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
        cache_raw=cache_raw,
        cache_processed=cache_processed,
        simulate_backend="native",
        keep_repo=False,
        verbose=False,
    )


def finalize_method_builds(
    builds: list[_MethodBuild],
    *,
    task_metadata: TaskMetadataCollection,
    cache: bool = True,
    cache_processed: bool = False,
    cache_hpo_trajectories: bool = False,
    trajectories_backend: Backend = "ray",
    verbose: bool = True,
) -> list[MethodResults]:
    """Merge per-task builds per method and write the driver-side artifacts.

    Per method: concatenate the result frames (and reconcile per-task-inferred metadata) via
    :meth:`MethodResults.concat`; then, per the flags, cache ``metadata.yaml`` + results
    parquets, write the processed tier's context files over the task slices the workers already
    wrote, and generate + cache HPO trajectories (config methods only).
    """
    log = _log_fn(verbose)
    if not builds:
        raise ValueError("No results to process (all raw results were filtered out or none were found).")

    builds_per_method: dict[tuple[str, str], list[_MethodBuild]] = {}
    for build in builds:
        key = (build.results.method_metadata.method, build.results.method_metadata.suite)
        builds_per_method.setdefault(key, []).append(build)

    results_lst: list[MethodResults] = []
    for method_builds in builds_per_method.values():
        if len(method_builds) == 1:
            results = method_builds[0].results
        else:
            results = MethodResults.concat([b.results for b in method_builds])
        method_metadata = results.method_metadata

        if cache:
            log(f"Caching metadata and results to {method_metadata.path}...")
            results.cache()

        if cache_processed:
            fragments = [b.repo_fragment for b in method_builds]
            assert all(f is not None for f in fragments), (
                "cache_processed=True requires every build to carry a repo fragment."
            )
            fragment = _RepoFragment.merge(fragments)
            # Same construction as the per-task repos (EvaluationRepository.from_raw defaults),
            # so the written context matches what a single monolithic repo.to_dir would produce.
            from tabarena.repository.evaluation_repository import write_processed_context
            from tabarena.simulation.simulation_context import ZeroshotSimulatorContext

            zeroshot_context = ZeroshotSimulatorContext(
                df_configs=fragment.df_configs,
                df_baselines=fragment.df_baselines,
                df_metadata=task_metadata.to_legacy_df(),
                configs_hyperparameters=fragment.configs_hyperparameters,
                pct=False,
                score_against_only_baselines=False,
            )
            log(f"Writing processed context to {method_metadata.path_processed}...")
            write_processed_context(
                path=method_metadata.path_processed,
                zeroshot_context=zeroshot_context,
                dataset_fold_lst_pp=fragment.dataset_fold_lst_pp,
                dataset_fold_lst_gt=fragment.dataset_fold_lst_gt,
            )

        if cache_hpo_trajectories:
            if method_metadata.method_type == "config":
                log("\tGenerating and caching HPO trajectories...")
                # Prefer a retained in-memory repo; else load the (memmap) processed tier.
                if results.repo is not None:
                    repo = results.repo
                elif method_metadata.path_processed_exists:
                    repo = method_metadata.load_processed()
                else:
                    raise ValueError(
                        f"cache_hpo_trajectories=True requires the processed repo for "
                        f"method={method_metadata.method!r}; pass cache_processed=True (or run on a "
                        f"method whose processed tier is already cached).",
                    )
                MethodSimulator(method_metadata).generate_hpo_trajectories(
                    repo=repo,
                    backend=trajectories_backend,
                    cache=True,
                )
            else:
                log(
                    f"\tSkipping HPO trajectories (method_type={method_metadata.method_type!r}, requires 'config')",
                )

        results_lst.append(results)
    return results_lst


def process_raw_results(
    results_lst: list[BaselineResult | dict],
    *,
    task_metadata: TaskMetadataCollection | None = None,
    method_metadata: MethodMetadata | None = None,
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    suite: str | None = None,
    artifact_dir: str | Path | None = None,
    cache: bool = True,
    cache_raw: bool = False,
    cache_processed: bool = False,
    cache_hpo_trajectories: bool = False,
    backend: Backend = "ray",
    verbose: bool = True,
) -> list[MethodResults]:
    """In-memory entry point: process an already-loaded raw results list end to end."""
    reject_legacy_task_metadata(task_metadata)
    results_lst = clean_raw(results_lst=results_lst)
    if task_metadata is None:
        tids = list({r.task_metadata["tid"] for r in results_lst})
        task_metadata = fetch_task_metadata(tids=tids, verbose=verbose)

    builds = process_results_in_memory(
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
        cache_raw=cache_raw,
        cache_processed=cache_processed,
        simulate_backend=backend,
        keep_repo=True,
        verbose=verbose,
    )
    return finalize_method_builds(
        builds,
        task_metadata=task_metadata,
        cache=cache,
        cache_processed=cache_processed,
        cache_hpo_trajectories=cache_hpo_trajectories,
        trajectories_backend=backend,
        verbose=verbose,
    )


def process_path_raw(
    path_raw: str | Path | list[str | Path],
    *,
    task_metadata: TaskMetadataCollection | None = None,
    method_metadata: MethodMetadata | None = None,
    name: str | None = None,
    name_prefix: str | None = None,
    name_suffix: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    suite: str | None = None,
    artifact_dir: str | Path | None = None,
    name_prefix_raw: str | None = None,
    file_paths: list[str | Path] | None = None,
    cache: bool = True,
    cache_raw: bool = False,
    cache_processed: bool = False,
    cache_hpo_trajectories: bool = False,
    backend: Backend = "ray",
    num_cpus: int | None = None,
    verbose: bool = True,
) -> list[MethodResults]:
    """On-disk entry point: process a raw-results directory task-by-task in parallel.

    Each (task, split) group of ``results.pkl`` files is processed independently (in ray
    workers for ``backend="ray"``, in-process for ``"native"``); the driver only merges
    prediction-free frames, so memory stays flat regardless of run size.

    ``file_paths`` are pre-discovered ``results.pkl`` paths under ``path_raw``; passing them
    skips the (potentially expensive) recursive directory walk, e.g. when a preceding
    inspect/verify step already discovered them.
    """
    reject_legacy_task_metadata(task_metadata)
    log = _log_fn(verbose)
    if num_cpus is None:
        num_cpus = len(os.sched_getaffinity(0))

    if file_paths is None:
        log("Get results paths...")
        file_paths = fetch_raw_result_paths(
            path_raw=path_raw,
            name_pattern=name_prefix_raw,
            # The parallel directory walk runs on ray; keep "native" fully ray-free.
            num_workers=num_cpus if backend == "ray" else None,
        )
    elif name_prefix_raw is not None:
        raise ValueError(
            "Pass either pre-discovered `file_paths` (already filtered) or `name_prefix_raw` (a walk filter), not both."
        )
    file_paths = [Path(p) for p in file_paths]
    all_file_paths_by_task = group_file_paths_by_task(file_paths)

    if task_metadata is None:
        # Only integer (OpenML) task dirs can be fetched from OpenML; UserTask slugs
        # require an explicitly-provided ``task_metadata``.
        tids = list({int(_task_dir(k)) for k in all_file_paths_by_task if _task_dir(k).isdigit()})
        task_metadata = fetch_task_metadata(tids=tids, verbose=verbose)
    all_file_paths_by_task = filter_file_paths_by_task_metadata(all_file_paths_by_task, task_metadata)

    worker_kwargs = dict(
        task_metadata=task_metadata,
        method_metadata=method_metadata,
        name=name,
        name_prefix=name_prefix,
        name_suffix=name_suffix,
        model_key=model_key,
        method=method,
        suite=suite,
        artifact_dir=artifact_dir,
        cache_raw=cache_raw,
        cache_processed=cache_processed,
    )
    task_groups = list(all_file_paths_by_task.values())
    if backend == "ray":
        import ray

        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        payloads: list[list[_MethodBuild]] = ray_map_list(
            list_to_map=task_groups,
            func=_process_task_group,
            func_element_key_string="file_paths_method",
            num_workers=num_cpus,
            num_cpus_per_worker=1,
            func_put_kwargs=worker_kwargs,
            track_progress=True,
            tqdm_kwargs={"desc": "Processing Results"},
            ray_remote_kwargs={"max_calls": 0},
        )
    else:
        payloads = [_process_task_group(file_paths_method=group, **worker_kwargs) for group in task_groups]
    builds = [build for payload in payloads for build in payload]

    log("Merging results...")
    return finalize_method_builds(
        builds,
        task_metadata=task_metadata,
        cache=cache,
        cache_processed=cache_processed,
        cache_hpo_trajectories=cache_hpo_trajectories,
        trajectories_backend=backend,
        verbose=verbose,
    )
