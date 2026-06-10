"""``AbstractArenaContext`` — the arena-agnostic core shared by every arena context.

An "arena context" bundles a benchmark's task metadata and method metadata and exposes the
operations that depend only on them: building a scoped ``ExperimentBatchRunner``, computing a
leaderboard via ``compare``, subsetting results, and rendering a website leaderboard. None of
that is specific to TabArena — it works for any benchmark whose tasks/methods are described by
a :class:`TaskMetadataCollection` / :class:`MethodMetadataCollection`.

Each concrete arena fills in two hooks:

* :meth:`_resolve_task_metadata_preset` — turn a named preset (e.g. ``"tabarena"``) into a
  :class:`TaskMetadataCollection`.
* :meth:`_resolve_methods_preset` — turn a named methods preset into a ``list[MethodMetadata]``.

and may override the class-level :attr:`SUBSET_PREDICATES` and :attr:`_default_subsets` to
declare arena-specific subset filters. :class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext`
is the reference implementation (TabArena v0.1 presets + paper results + the full paper workflow);
``BeyondArenaContext`` subclasses it.
"""

from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.website.website_format import format_leaderboard

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import ExperimentBatchRunner
    from tabarena.models._method_metadata import MethodMetadata


class AbstractArenaContext(ABC):
    """Arena-agnostic base: task/method metadata + comparison/runner/leaderboard plumbing."""

    #: Subset-filter predicates available to `compare` / `make_experiment_batch_runner`, keyed by
    #: name. Each :class:`SubsetPredicate` declares the grid columns it needs (validated before
    #: it runs). Subclasses override to add arena-specific filters (size buckets, split regimes,
    #: ...). Read via :attr:`subset_predicates` so subclass overrides take effect.
    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # split-level filter: keeps split 0 == (fold 0, repeat 0). Evaluated on the task grid's
        # "split" column (see TaskMetadataCollection.task_grid); a results frame's "fold" is the
        # split, so this maps to fold == 0 there.
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: list[MethodMetadata] | str,
        task_metadata: str | TaskMetadataCollection,
        *,
        extra_methods: list[MethodMetadata] | None = None,
        include_unverified: bool = False,
        backend: Literal["ray", "native"] = "ray",
        fillna_method: str | None = None,
        calibration_method: str | None = None,
    ):
        # A TaskMetadataCollection is the single source of truth; the legacy `task_metadata`
        # DataFrame view is derived from it on demand (see the `task_metadata` cached_property).
        self.task_metadata_collection = self._resolve_task_metadata_collection(task_metadata)
        self.fillna_method = fillna_method
        self.calibration_method = calibration_method
        assert backend in ["ray", "native"]
        self.backend = backend
        self.engine = "ray" if self.backend == "ray" else "sequential"

        # A string selects a named preset (resolved by the concrete arena); otherwise an
        # explicit list of MethodMetadata is used as-is.
        if isinstance(methods, str):
            method_metadata_lst = self._resolve_methods_preset(methods, include_unverified=include_unverified)
        else:
            method_metadata_lst = list(methods)
        method_names = {m.method for m in method_metadata_lst}

        if extra_methods:
            for method_metadata in extra_methods:
                assert method_metadata.method not in method_names, f"{method_metadata.method} already in methods..."
                method_metadata_lst.append(method_metadata)
                method_names.add(method_metadata.method)

        self.method_metadata_collection: MethodMetadataCollection = MethodMetadataCollection(method_metadata_lst)

    # ------------------------------------------------------------------ arena-specific hooks
    @abstractmethod
    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        """Resolve a named task-metadata preset (e.g. ``"tabarena"``) to a collection."""

    @abstractmethod
    def _resolve_methods_preset(self, name: str, *, include_unverified: bool) -> list[MethodMetadata]:
        """Resolve a named methods preset (e.g. ``"tabarena"``) to a ``list[MethodMetadata]``."""

    def load_results(
        self,
        methods: list[str] | None = None,
        download_results: str | bool = "auto",
        methods_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load the cached results of this arena's methods (downloading on cache miss).

        These are the baseline/reference results ``compare`` compares new results against.
        A context constructed with no methods contributes none (empty DataFrame), so a
        self-contained arena's leaderboard is computed purely from ``new_results``.
        """
        if methods is None:
            methods = self.methods
        if methods_drop is not None:
            for method in methods_drop:
                if method not in methods:
                    raise AssertionError(
                        f"Specified '{method}' in `methods_drop`, but '{method}' is not present in methods: {methods}",
                    )
            methods = [method for method in methods if method not in methods_drop]
        if not methods:
            return pd.DataFrame()

        df_results_lst = []
        for method in methods:
            method_metadata = self.method_metadata(method=method)
            if isinstance(download_results, bool) and download_results:
                method_downloader = method_metadata.method_downloader()
                method_downloader.download_results()

            try:
                df_results = method_metadata.load_results()
            except FileNotFoundError as err:
                if isinstance(download_results, str) and download_results == "auto":
                    print(
                        f"Missing local results files for method! "
                        f"Attempting to download from s3 and retry... "
                        f'(download_results={download_results}, method="{method_metadata.method}")',
                    )
                    method_downloader = method_metadata.method_downloader()
                    method_downloader.download_results()
                    df_results = method_metadata.load_results()
                else:
                    print(
                        f"Missing local results files for method {method_metadata.method}! "
                        f"Try setting `download_results=True` to get the required files.",
                    )
                    raise err
            df_results_lst.append(df_results)

        return pd.concat(df_results_lst, ignore_index=True)

    # ------------------------------------------------------------------ metadata views
    def _resolve_task_metadata_collection(
        self,
        task_metadata: str | TaskMetadataCollection,
    ) -> TaskMetadataCollection:
        """Normalize the constructor input to a native ``TaskMetadataCollection``.

        Accepts an explicit ``TaskMetadataCollection`` or a named preset (delegated to
        :meth:`_resolve_task_metadata_preset`). A legacy DataFrame / ``list[TabArenaTaskMetadata]``
        is not accepted — wrap it before constructing the context
        (``TaskMetadataCollection.from_legacy_df(df)`` / ``TaskMetadataCollection(tasks)``) so the
        (lossy) legacy conversion stays an explicit, opt-in step at the call site.
        """
        if isinstance(task_metadata, TaskMetadataCollection):
            return task_metadata
        if isinstance(task_metadata, str):
            return self._resolve_task_metadata_preset(task_metadata)
        raise TypeError(
            f"task_metadata must be a preset name or a TaskMetadataCollection, got "
            f"{type(task_metadata).__name__}. Wrap a legacy DataFrame with "
            f"TaskMetadataCollection.from_legacy_df(df) or a list with TaskMetadataCollection(tasks).",
        )

    @functools.cached_property
    def task_metadata(self) -> pd.DataFrame:
        """Legacy one-row-per-dataset ``task_metadata`` DataFrame (back-compat bridge only).

        The sole remaining legacy-DataFrame surface on the context: nothing internal consumes
        it (the runner, ``compare``, and subset predicates all work off
        :attr:`task_metadata_collection` and its native :meth:`~TaskMetadataCollection.task_grid`).
        Kept as a convenience for external callers still on the legacy schema; derived from the
        collection via ``to_legacy_df()`` and cached (effectively immutable post-init).
        """
        return self.task_metadata_collection.to_legacy_df()

    @property
    def subset_predicates(self) -> dict[str, SubsetPredicate]:
        """Predicates available for subset filtering. Reads from
        ``type(self).SUBSET_PREDICATES`` so subclass overrides take effect.
        """
        return type(self).SUBSET_PREDICATES

    @property
    def _default_subsets(self):
        return [
            [],
            ["binary"],
            ["multiclass"],
            ["classification"],
            ["regression"],
        ]

    @property
    def methods(self) -> list[str]:
        return [m.method for m in self.method_metadata_collection.method_metadata_lst]

    def method_metadata(
        self,
        method: str,
        artifact_name: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
    ) -> MethodMetadata:
        return self.method_metadata_collection.get_method_metadata(
            method=method,
            artifact_name=artifact_name,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
        )

    def get_method_rename_map(self) -> dict[str, str]:
        method_rename_map = dict()
        method_metadatas = self.method_metadata_collection.method_metadata_lst
        for m in method_metadatas:
            if m.method_type == "config":
                display_name = m.display_name
                if display_name is not None:
                    if m.config_type in method_rename_map:
                        print(
                            f"WARNING: Multiple display_name values detected for the same config_type={m.config_type!r}"
                            f"\n\tdisplay_name 1: {method_rename_map[m.config_type]!r}"
                            f"\n\tdisplay_name 2: {display_name!r}",
                        )
                    method_rename_map[m.config_type] = display_name
        return method_rename_map

    def _method_rename_map_to_display_names(self) -> dict[str, str]:
        """Build a mapping ``"<config_type> (<subtype>)" -> "<display_name>
        (<subtype>)"`` covering every config method in this collection plus
        the bare ``method -> display_name`` mapping for baseline/portfolio
        methods. Used to switch the rendered ``method`` column from
        ``config_type``/``ag_key``-based codes to friendlier display names.
        """
        rename_map: dict[str, str] = {}
        suffixes = [" (default)", " (tuned)", " (tuned + ensemble)"]
        for m in self.method_metadata_collection.method_metadata_lst:
            if not m.display_name:
                continue
            if m.method_type == "config" and m.config_type and m.config_type != m.display_name:
                for suffix in suffixes:
                    rename_map[f"{m.config_type}{suffix}"] = f"{m.display_name}{suffix}"
            elif m.method_type in ("baseline", "portfolio") and m.method != m.display_name:
                rename_map[m.method] = m.display_name
        return rename_map

    def leaderboard_to_website_format(
        self,
        leaderboard: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        method_metadata_info = self.method_metadata_collection.info()
        method_metadata_info = method_metadata_info.rename(
            columns={
                "method": "ta_name",
                "artifact_name": "ta_suite",
            },
        )
        return format_leaderboard(
            df_leaderboard=leaderboard,
            method_metadata_info=method_metadata_info,
            **kwargs,
        )

    # ------------------------------------------------------------------ comparison / runner
    def compare(
        self,
        output_dir: str | Path,
        new_results: pd.DataFrame | None = None,
        ta_results: pd.DataFrame | None = None,
        only_valid_tasks: bool | str | list[str] = False,
        subset: str | list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
        datasets: list[str] | None = None,
        folds: list[int] | None = None,
        score_on_val: bool = False,
        average_seeds: bool = False,
        fillna: str | pd.DataFrame | None = "auto",
        calibration_method: str | None = "auto",
        remove_imputed: bool = False,
        tmp_treat_tasks_independently: bool = False,
        leaderboard_kwargs: dict | None = None,
        figure_file_type: str = "pdf",
        **kwargs,
    ) -> pd.DataFrame:
        """Compute the leaderboard comparing ``new_results`` against this arena's baselines.

        ``ta_results`` defaults to :meth:`load_results`; ``new_results`` (if given)
        are concatenated to them. ``fillna`` / ``calibration_method`` resolve ``"auto"`` to
        the context's settings.
        """
        # Deferred import: tabarena.nips2025_utils.compare imports TabArenaContext at module
        # level, which would be circular at import time.
        from tabarena.nips2025_utils.compare import compare, filter_to_valid_tasks

        if fillna == "auto":
            fillna = self.fillna_method
        if calibration_method == "auto":
            calibration_method = self.calibration_method

        if ta_results is None:
            ta_results = self.load_results(
                download_results="auto",
            )

        if new_results is not None:
            new_results = new_results.copy(deep=True)
            if "method_subtype" not in new_results.columns:
                new_results["method_subtype"] = np.nan

        df_results = pd.concat([ta_results, new_results], ignore_index=True) if new_results is not None else ta_results

        kwargs = kwargs.copy()
        if isinstance(only_valid_tasks, (tuple, np.ndarray)):
            only_valid_tasks = list(only_valid_tasks)
        if isinstance(only_valid_tasks, (str, list)):
            kwargs["only_valid_tasks"] = only_valid_tasks
        elif only_valid_tasks and new_results is not None:
            df_results = filter_to_valid_tasks(
                df_to_filter=df_results,
                df_filter=new_results,
            )

        # TODO: only methods that exist in runs
        #  Pair with (method, artifact_name)
        method_rename_map = self.get_method_rename_map()

        return compare(
            df_results=df_results,
            output_dir=Path(output_dir),
            task_metadata=self.task_metadata_collection,
            subset=subset,
            tasks=tasks,
            datasets=datasets,
            folds=folds,
            tabarena_context=self,
            fillna=fillna,
            calibration_framework=calibration_method,
            score_on_val=score_on_val,
            average_seeds=average_seeds,
            remove_imputed=remove_imputed,
            tmp_treat_tasks_independently=tmp_treat_tasks_independently,
            leaderboard_kwargs=leaderboard_kwargs,
            method_rename_map=method_rename_map,
            figure_file_type=figure_file_type,
            **kwargs,
        )

    def compare_per_dataset(
        self,
        output_dir: str | Path,
        new_results: pd.DataFrame | None = None,
        ta_results: pd.DataFrame | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        output_dir = Path(output_dir)
        if ta_results is None:
            ta_results = self.load_results(
                download_results="auto",
            )
        datasets = sorted(ta_results["dataset"].unique())
        if new_results is not None:
            new_datasets = sorted(new_results["dataset"].unique())
            datasets = sorted(datasets + [d for d in new_datasets if d not in datasets])

        outs = {}
        plot_tuning_kwargs = kwargs.pop("plot_tuning_kwargs", {})
        for dataset in datasets:
            plot_tuning_kwargs_dataset = copy.deepcopy(plot_tuning_kwargs)
            plot_tuning_kwargs_dataset["title"] = f"Dataset: {dataset}"
            outs[dataset] = self.compare(
                output_dir=output_dir / "per_dataset" / dataset,
                ta_results=ta_results,
                new_results=new_results,
                datasets=[dataset],
                plot_tuning_kwargs=plot_tuning_kwargs_dataset,
                **kwargs,
            )
        return outs

    def subset_results(
        self,
        df_results: pd.DataFrame,
        *,
        subset: list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
        datasets: list[str] | None = None,
        folds: list[int] | None = None,
    ) -> pd.DataFrame:
        from tabarena.nips2025_utils.compare import subset_tasks

        if subset is not None or datasets is not None or folds is not None or tasks is not None:
            df_results = subset_tasks(
                df_results=df_results,
                subset=subset,
                tasks=tasks,
                datasets=datasets,
                folds=folds,
                task_metadata_og=self.task_metadata_collection,
                predicates=self.subset_predicates,
            )
        return df_results

    def make_experiment_batch_runner(
        self,
        expname: str,
        *,
        subset: str | list[str] | None = None,
        datasets: list[str] | None = None,
        splits: list[int] | None = None,
        folds: list[int] | None = None,
        repeats: list[int] | None = None,
        dataset_fold_repeats: list[tuple[str, int, int]] | None = None,
        **kwargs,
    ) -> ExperimentBatchRunner:
        """Create an `ExperimentBatchRunner` over this context's `task_metadata`.

        The (dataset, fold, repeat) triplets the runner's `run_all` executes are
        determined by the following filters (all default None):

        - `subset`: predicate expressions (the same as `compare`, e.g. ``"lite"``,
          ``"small"``, ``"binary"``, ``"!tiny"``) applied to the task grid, which carries an
          explicit ``split`` column (``n_folds * repeat + fold``); ``"lite"`` keeps
          ``split == 0`` == ``(fold 0, repeat 0)``.
        - `datasets`: restrict to these dataset names.
        - `splits`: restrict to these split indices (``n_folds * repeat + fold``).
        - `folds` / `repeats`: restrict to these fold / repeat indices.
        - `dataset_fold_repeats`: an explicit list of triplets. When given, the result is
          the **intersection** of these and the triplets derived from `subset`/`datasets`
          (so user triplets that fall outside the subset/datasets selection are dropped).

        `subset`, `datasets`, and the split/fold/repeat filters compose (AND). Two
        combinations are disallowed: `splits` together with `folds`/`repeats`, and
        `dataset_fold_repeats` together with any of `splits`/`folds`/`repeats`.

        If no filters are given, `run_all` runs every split in `task_metadata`. Extra keyword
        arguments (``cache_mode``, ``debug_mode``, ...) are forwarded to `ExperimentBatchRunner`.
        """
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        if splits is not None and (folds is not None or repeats is not None):
            raise ValueError("Cannot specify `splits` together with `folds`/`repeats`.")
        if dataset_fold_repeats is not None and any(x is not None for x in (splits, folds, repeats)):
            raise ValueError("Cannot specify `dataset_fold_repeats` together with `splits`/`folds`/`repeats`.")

        derived = None
        if any(x is not None for x in (subset, datasets, splits, folds, repeats)):
            derived = self._subset_dataset_fold_repeats(
                subset=subset,
                datasets=datasets,
                splits=splits,
                folds=folds,
                repeats=repeats,
            )

        if dataset_fold_repeats is not None:
            if derived is not None:
                allowed = set(derived)
                dataset_fold_repeats = [t for t in dataset_fold_repeats if t in allowed]
            # else: no subset/datasets constraint -> use the user's triplets as-is.
        else:
            dataset_fold_repeats = derived

        # The collection is the single source of truth for what `run_all` executes: pre-filter
        # it to the selected triplets (when any filter was given) instead of passing a separate
        # "allowed triplets" channel to the runner.
        collection = self.task_metadata_collection
        if dataset_fold_repeats is not None:
            collection = collection.subset(dataset_fold_repeats)
        return ExperimentBatchRunner(
            expname=expname,
            task_metadata=collection,
            **kwargs,
        )

    def _subset_dataset_fold_repeats(
        self,
        subset: str | list[str] | None = None,
        datasets: list[str] | None = None,
        splits: list[int] | None = None,
        folds: list[int] | None = None,
        repeats: list[int] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Expand the task grid into (dataset, fold, repeat) triplets kept by the filters.

        Evaluates the `subset` predicate expressions on the native task grid
        (:meth:`TaskMetadataCollection.task_grid` — one row per ``(dataset, fold, repeat, split)``),
        then filters (AND) by an explicit `datasets` list and the `splits`/`folds`/`repeats` index
        lists. ``"lite"`` keys on the grid's ``split`` column, so it keeps ``split == 0`` (i.e.
        ``(fold 0, repeat 0)``).
        """
        from tabarena.nips2025_utils.compare import _evaluate_subset_expression

        if isinstance(subset, str):
            subset = [subset]

        grid = self.task_metadata_collection.task_grid()
        if subset:
            for expression in subset:
                mask = _evaluate_subset_expression(expression, grid, predicates=self.subset_predicates)
                grid = grid[mask.values]
        if datasets is not None:
            grid = grid[grid["dataset"].isin(datasets)]
        if splits is not None:
            grid = grid[grid["split"].isin(splits)]
        if folds is not None:
            grid = grid[grid["fold"].isin(folds)]
        if repeats is not None:
            grid = grid[grid["repeat"].isin(repeats)]
        return [
            (dataset, int(fold), int(repeat))
            for dataset, fold, repeat in zip(grid["dataset"], grid["fold"], grid["repeat"], strict=False)
        ]
